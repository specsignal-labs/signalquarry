# MCP adapter

Install the optional local adapter with `pip install "signalquarry[mcp]"` and
give your coding agent the `sqy-mcp` launch command. The server uses stdio;
it opens no listening port. Pass an explicit `project` path to tools when the
agent's working directory is outside the project.

For Claude Code, register the command with `claude mcp add signalquarry --
/absolute/path/to/sqy-mcp`. For VS Code with GitHub Copilot Agent mode, add a
local server entry to `.vscode/mcp.json`:

```json
{
  "servers": {
    "signalquarry": {
      "type": "stdio",
      "command": "/absolute/path/to/sqy-mcp"
    }
  }
}
```

Use the path to the `sqy-mcp` executable in the environment where you
installed SignalQuarry. Other MCP hosts use the same executable over stdio;
their configuration file format may differ. See the [MCP Python SDK host
guide](https://py.sdk.modelcontextprotocol.io/get-started/real-host/) for
current host examples.

The tools return the same `signalquarry.cli/v1` envelope as `sqy --json`:
inspect `status`, `reason_codes`, `evidence`, and `next_actions`. Tool names
follow the `sqy_` prefix. The adapter covers project creation, strategy and
factor checks, data fetch/list/verify, backtest, freeze, evaluation, trial and
holdout status, reports, and core paper operations. The CLI remains available
for the full command set.

`sqy_paper_run_once` can submit **paper** orders only after a human has armed
the deployment in an interactive terminal and the existing kernel checks pass.
Agents should call it only when the user explicitly requests a paper session.
Paper arming and trial-budget extension are not exposed as MCP tools. A sealed
holdout is never opened by `sqy_evaluate`; use the CLI's explicit holdout flow.
No live-trading origin or order path exists.

The adapter imports and runs project strategy or factor code for some tools.
Run untrusted projects in an isolated development environment. The MCP extra
does not change SignalQuarry's four mandatory runtime dependencies.
