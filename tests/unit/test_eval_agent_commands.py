# SPDX-License-Identifier: Apache-2.0
"""Fixed headless command adapters for the isolated coding-agent evaluator."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from evals.run_agent import (
    AGENT_EXECUTABLES,
    _copy_project_for_verification,
    agent_command,
    agent_environment,
    agent_exec_command,
    isolated_agent_identity,
    isolated_verifier_identity,
    main,
    private_results_directory,
)


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        (
            "claude",
            [
                "claude",
                "-p",
                "complete the task",
                "--output-format",
                "json",
                "--permission-mode",
                "acceptEdits",
                "--allowedTools",
                "Bash",
                "Read",
                "Edit",
                "Write",
                "Glob",
                "Grep",
            ],
        ),
        (
            "codex",
            [
                "codex",
                "exec",
                "--sandbox",
                "workspace-write",
                "--config",
                'approval_policy="never"',
                "--skip-git-repo-check",
                "complete the task",
            ],
        ),
        (
            "copilot",
            [
                "copilot",
                "-p",
                "complete the task",
                "--allow-all-tools",
                "--allow-all-paths",
                "--silent",
            ],
        ),
        (
            "grok",
            [
                "grok",
                "--no-auto-update",
                "-p",
                "complete the task",
                "--output-format",
                "plain",
                "--always-approve",
            ],
        ),
    ],
)
def test_agent_command_uses_a_fixed_argument_vector(agent: str, expected: list[str]) -> None:
    assert agent_command(agent, "complete the task") == expected
    assert all(isinstance(argument, str) for argument in expected)


def test_all_supported_agent_names_have_one_executable() -> None:
    assert AGENT_EXECUTABLES == {
        "claude": "claude",
        "codex": "codex",
        "copilot": "copilot",
        "grok": "grok",
    }


def test_unknown_agent_is_rejected() -> None:
    with pytest.raises(SystemExit, match="unknown agent"):
        agent_command("shell", "complete the task")


@pytest.mark.parametrize("agent", sorted(AGENT_EXECUTABLES))
def test_agent_environment_keeps_only_local_runtime_and_selected_credentials(agent: str) -> None:
    source = {
        "HOME": "/sandbox/home",
        "PATH": "/sandbox/bin",
        "SIGNALQUARRY_CONFIG_DIR": "/sandbox/config",
        "SIGNALQUARRY_CACHE_DIR": "/sandbox/cache",
        "ANTHROPIC_API_KEY": "other-provider-key",
        "OPENAI_API_KEY": "other-provider-key",
        "CODEX_API_KEY": "must-not-pass",
        "XAI_API_KEY": "must-not-pass",
        "COPILOT_PROVIDER_API_KEY": "must-not-pass",
        "COPILOT_GITHUB_TOKEN": "must-not-pass",
        "GH_TOKEN": "must-not-pass",
        "GITHUB_TOKEN": "must-not-pass",
        "ANTHROPIC_AUTH_TOKEN": "must-not-pass",
        "CLAUDE_CODE_OAUTH_TOKEN": "must-not-pass",
        "ANTHROPIC_BASE_URL": "must-not-pass",
        "CODEX_HOME": "/host/home",
        "APCA_API_KEY_ID": "broker-key",
        "UNRELATED_SECRET": "must-not-pass",
    }
    assert agent_environment(agent, source, home=Path("/sandbox/run-home")) == {
        "HOME": "/sandbox/run-home",
        "TEMP": "/sandbox/run-home/tmp",
        "TMP": "/sandbox/run-home/tmp",
        "TMPDIR": "/sandbox/run-home/tmp",
        "PATH": "/sandbox/bin",
        "SIGNALQUARRY_CONFIG_DIR": "/sandbox/config",
        "SIGNALQUARRY_CACHE_DIR": "/sandbox/cache",
    }


def test_unknown_agent_environment_is_rejected() -> None:
    with pytest.raises(SystemExit, match="unknown agent"):
        agent_environment("shell", {"PATH": "/bin"}, home=Path("/sandbox/home"))


def test_agent_exec_uses_setpriv_without_a_shell() -> None:
    assert agent_exec_command(10001, 10001, ["claude", "-p", "task text"]) == [
        "/usr/bin/setpriv",
        "--reuid=10001",
        "--regid=10001",
        "--clear-groups",
        "--no-new-privs",
        "--",
        "claude",
        "-p",
        "task text",
    ]


def test_isolated_agent_requires_root_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.run_agent.os.geteuid", lambda: 1000)

    with pytest.raises(SystemExit, match="must run as root"):
        isolated_agent_identity()


def test_verifier_identity_must_differ_from_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr("evals.run_agent.os.geteuid", lambda: 0)
    monkeypatch.setattr("evals.run_agent.getpwnam", lambda name: SimpleNamespace(pw_uid=10001, pw_gid=10002))

    with pytest.raises(SystemExit, match="must be distinct and non-root"):
        isolated_verifier_identity(agent_uid=10001)


def test_result_directory_is_private_for_agent_uid(tmp_path: Path) -> None:
    results = private_results_directory(tmp_path / "results", agent_uid=10001)
    assert results.stat().st_mode & 0o077 == 0

    results.chmod(0o755)
    with pytest.raises(SystemExit, match="results directory must be private"):
        private_results_directory(results, agent_uid=10001)


def test_result_directory_rejects_agent_ownership_and_symlinks(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir(mode=0o700)
    with pytest.raises(SystemExit, match="results directory must be private"):
        private_results_directory(owned, agent_uid=os.getuid())

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(SystemExit, match="must not be a symlink"):
        private_results_directory(link, agent_uid=10001)


def test_verification_copy_is_bounded_read_only_and_skips_runtime_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    source = project / "src" / "demo.py"
    source.parent.mkdir(parents=True)
    source.write_text("answer = 42\n")
    (project / ".git").mkdir()
    (project / ".git" / "config").write_text("private history")
    (project / ".eval").mkdir()
    (project / ".eval" / "sqy.log").write_text("untrusted log")
    monkeypatch.setattr("evals.run_agent.os.chown", lambda *_args, **_kwargs: None)

    copied = _copy_project_for_verification(project, tmp_path / "verification")

    assert (copied / "src" / "demo.py").read_text() == "answer = 42\n"
    assert not (copied / ".git").exists()
    assert not (copied / ".eval").exists()
    assert stat.S_IMODE((copied / "src").stat().st_mode) == 0o555
    assert stat.S_IMODE((copied / "src" / "demo.py").stat().st_mode) == 0o444
    (copied / "src" / "demo.py").chmod(0o644)
    (copied / "src").chmod(0o755)
    copied.chmod(0o755)


def test_verification_copy_rejects_symlinks_and_hard_links(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret")
    (project / "link").symlink_to(outside)
    with pytest.raises(SystemExit, match="containing a symlink"):
        _copy_project_for_verification(project, tmp_path / "symlink-copy")

    (project / "link").unlink()
    os.link(outside, project / "hard-link")
    with pytest.raises(SystemExit, match="multiply linked file"):
        _copy_project_for_verification(project, tmp_path / "hardlink-copy")


@pytest.mark.parametrize("run_count", ["0", "-1"])
def test_eval_cli_rejects_nonpositive_run_count(monkeypatch: pytest.MonkeyPatch, run_count: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_agent.py", "--agent", "claude", "--wheel", "dist/signalquarry.whl", "--runs", run_count],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_codex_model_is_an_explicit_argument_without_changing_sandbox_policy() -> None:
    command = agent_command("codex", "complete the task", codex_model="gpt-5")
    assert command[-3:] == ["--model", "gpt-5", "complete the task"]
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert 'approval_policy="never"' in command


def test_official_copilot_commented_cache_keeps_urls_and_account_tokens(tmp_path: Path) -> None:
    from evals.subscription_auth import _oauth_tokens, _read_cache

    tmp_path.chmod(0o700)
    directory = tmp_path / ".copilot"
    directory.mkdir(mode=0o700)
    cache = directory / "config.json"
    cache.write_text(
        "// Official CLI configuration\n"
        '{"authTokens":{"https://github.com:example:github":{"token":"gho_test"}},'
        '"loggedInUsers":[{"host":"https://github.com","login":"example"}]}\n'
    )
    cache.chmod(0o600)
    payload = _read_cache(tmp_path, ".copilot/config.json", 10001)
    assert payload["loggedInUsers"][0]["host"] == "https://github.com"
    assert _oauth_tokens(payload["authTokens"]) == {"gho_test"}
    assert _oauth_tokens({"token": "github_pat_must_not_pass"}) == set()


def test_non_copilot_cache_does_not_accept_comments(tmp_path: Path) -> None:
    from evals.subscription_auth import _read_cache

    tmp_path.chmod(0o700)
    directory = tmp_path / ".codex"
    directory.mkdir(mode=0o700)
    cache = directory / "auth.json"
    cache.write_text("// Unsupported comment\n{}\n")
    cache.chmod(0o600)
    with pytest.raises(SystemExit, match="invalid account login cache"):
        _read_cache(tmp_path, ".codex/auth.json", 10001)
