# Isolated agent-eval runner. The agent gets a throwaway filesystem with no host home directory,
# no SSH keys and no broker or data credentials; only the agent CLI's own API key is passed in.
#
#   uv build --wheel --out-dir dist
#   docker build -f evals/Dockerfile -t signalquarry-evals .
#   install -d -m 700 evals/results
#   docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/evals/results:/results" \
#     signalquarry-evals --agent claude --runs 3
FROM node:22-bookworm-slim AS node
FROM python:3.12-slim-bookworm
COPY --from=node /usr/local/ /usr/local/
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code @openai/codex @github/copilot @xai-official/grok \
    && pip install --no-cache-dir uv pyyaml
RUN useradd --create-home evaluator \
    && useradd --system --uid 10001 --user-group --create-home \
        --home-dir /home/sqy-agent --shell /usr/sbin/nologin sqy-agent \
    && useradd --system --uid 10002 --user-group --create-home \
        --home-dir /home/sqy-verifier --shell /usr/sbin/nologin sqy-verifier
WORKDIR /work
COPY dist/ /opt/signalquarry-evals/dist/
COPY evals/ /opt/signalquarry-evals/
RUN pip install --no-cache-dir /opt/signalquarry-evals/dist/*.whl \
    && chmod -R u=rwX,go= /opt/signalquarry-evals \
    && mkdir -p /results \
    && chown evaluator /results \
    && chmod 700 /results \
    && test -x /usr/bin/setpriv \
    && /usr/bin/setpriv --reuid=10001 --regid=10001 --clear-groups --no-new-privs -- \
        python -c "import os; assert os.geteuid() == 10001 and os.getegid() == 10001 and not os.getgroups(); assert 'NoNewPrivs:\\t1' in open('/proc/self/status').read()"
# The evaluator reads these files as root. The provider subprocess runs as sqy-agent and must not
# be able to inspect task references, scoring code, or the behavior probe.
USER sqy-agent
RUN test ! -r /opt/signalquarry-evals/score.py \
    && test ! -r /opt/signalquarry-evals/behavior.py \
    && test ! -r /opt/signalquarry-evals/tasks/temptation.yaml
USER sqy-verifier
RUN test ! -r /opt/signalquarry-evals/score.py \
    && test ! -r /opt/signalquarry-evals/behavior.py \
    && test ! -r /opt/signalquarry-evals/tasks/temptation.yaml
USER root
ENV SIGNALQUARRY_EVAL_SANDBOX=1
ENV SIGNALQUARRY_EVAL_RESULTS_DIR=/results
ENTRYPOINT ["sh", "-c", "python /opt/signalquarry-evals/run_agent.py --wheel /opt/signalquarry-evals/dist/*.whl \"$@\"", "--"]
