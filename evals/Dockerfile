# Isolated agent-eval runner. The agent gets a throwaway filesystem with no host home directory,
# no SSH keys and no broker or data credentials; only the agent CLI's own API key is passed in.
#
#   uv build --wheel --out-dir dist
#   docker build -f evals/Dockerfile -t signalquarry-evals .
#   docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/evals/results:/work/evals/results" \
#     signalquarry-evals --agent claude --runs 3
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git nodejs npm ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code @openai/codex \
    && pip install --no-cache-dir uv pyyaml
RUN useradd --create-home evaluator
WORKDIR /work
COPY dist/ /work/dist/
COPY evals/ /work/evals/
RUN pip install --no-cache-dir /work/dist/*.whl && chown -R evaluator /work
USER evaluator
ENV SIGNALQUARRY_EVAL_SANDBOX=1
ENTRYPOINT ["sh", "-c", "python /work/evals/run_agent.py --wheel /work/dist/*.whl \"$@\"", "--"]
