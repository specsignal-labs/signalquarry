#!/usr/bin/env python3
"""Fail if files contain secrets, data files, or terms from a private rules file.

Stdlib only. The built-in rules catch credentials and committed market data.
Project-specific terms (identifiers, symbols, hostnames that must never be
published) come from a JSON rules file that is kept out of the scanned tree:

    python leak_scan.py [--rules RULES.json] [--json] PATH [PATH ...]

The rules file may also be supplied through the LEAK_SCAN_RULES_JSON
environment variable (for CI secrets). Its format:

    {"identifier_patterns": {"name": "regex", ...},
     "literal_terms": {"name": "case-insensitive text", ...},
     "line_patterns": {"name": "regex", ...},
     "exempt_prefixes": ["relative/path/prefix/", ...]}

Exit status: 0 clean, 1 findings, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

SECRET_PATTERNS: dict[str, str] = {
    "alpaca_key_id": r"\b(?:PK|AK)[A-Z0-9]{16,}\b",
    "aws_key_id": r"\bAKIA[0-9A-Z]{16}\b",
    "private_key_block": r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    "secret_assignment": r"(?i)\b(?:api[_-]?secret|secret[_-]?key|password)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]",
}
DENIED_SUFFIXES = (
    ".parquet",
    ".csv",
    ".feather",
    ".arrow",
    ".ipynb",
    ".pkl",
    ".pickle",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".jsonl.gz",
)
SKIPPED_DIRS = frozenset(
    {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build"}
)
MAX_TEXT_BYTES = 5_000_000


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    match: str


@dataclass(frozen=True)
class Rules:
    patterns: tuple[tuple[str, re.Pattern[str]], ...] = ()
    exempt_prefixes: tuple[str, ...] = ()
    sources: tuple[str, ...] = field(default_factory=tuple)


def load_rules(path: Path | None = None, environ: dict[str, str] | None = None) -> Rules:
    compiled = [(name, re.compile(pattern)) for name, pattern in SECRET_PATTERNS.items()]
    exempt: list[str] = []
    sources = ["builtin"]
    documents: list[tuple[str, dict]] = []
    if path is not None:
        documents.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
    env_value = (environ if environ is not None else os.environ).get("LEAK_SCAN_RULES_JSON", "").strip()
    if env_value:
        documents.append(("env:LEAK_SCAN_RULES_JSON", json.loads(env_value)))
    for source, document in documents:
        sources.append(source)
        for name, pattern in document.get("identifier_patterns", {}).items():
            compiled.append((name, re.compile(pattern)))
        for name, term in document.get("literal_terms", {}).items():
            compiled.append((name, re.compile(re.escape(term), re.IGNORECASE)))
        for name, pattern in document.get("line_patterns", {}).items():
            compiled.append((name, re.compile(pattern)))
        exempt.extend(document.get("exempt_prefixes", []))
    return Rules(tuple(compiled), tuple(exempt), tuple(sources))


def _iter_files(paths: list[Path]) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for base in paths:
        if base.is_file():
            files.append((base, base.name))
            continue
        for candidate in sorted(base.rglob("*")):
            relative = candidate.relative_to(base)
            if candidate.is_file() and not SKIPPED_DIRS.intersection(relative.parts):
                files.append((candidate, relative.as_posix()))
    return files


def scan(paths: list[Path], rules: Rules | None = None) -> list[Finding]:
    rules = rules if rules is not None else load_rules()
    findings: list[Finding] = []
    for path, shown in _iter_files(paths):
        if rules.exempt_prefixes and shown.startswith(rules.exempt_prefixes):
            continue
        if path.name.lower().endswith(DENIED_SUFFIXES):
            findings.append(Finding(shown, 0, "denied_file_type", path.suffix))
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192] or len(data) > MAX_TEXT_BYTES:
            continue
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
            for rule, pattern in rules.patterns:
                for match in pattern.finditer(line):
                    findings.append(Finding(shown, number, rule, match.group(0)[:80]))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--rules", type=Path, help="JSON rules file kept outside the scanned tree")
    parser.add_argument("--require-rules", action="store_true", help="fail unless project rules were loaded")
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    args = parser.parse_args(argv)
    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        print(f"LEAK_SCAN_PATH_MISSING:{','.join(missing)}", file=sys.stderr)
        return 2
    rules = load_rules(args.rules)
    if args.require_rules and len(rules.sources) < 2:
        print("LEAK_SCAN_RULES_MISSING", file=sys.stderr)
        return 2
    findings = scan(args.paths, rules)
    if args.json:
        print(
            json.dumps(
                {
                    "clean": not findings,
                    "rule_sources": list(rules.sources),
                    "findings": [asdict(item) for item in findings],
                },
                indent=2,
            )
        )
    else:
        for item in findings:
            print(f"{item.path}:{item.line}: {item.rule}: {item.match}")
        print("LEAK_SCAN_CLEAN" if not findings else f"LEAK_SCAN_FINDINGS count={len(findings)}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
