"""Scaffold a new strategy family and register its first strategy.

    python tools/new_family.py momentum [--strategy momentum-a]

Creates families/<family>/ with a package, a starter strategy (to be replaced), a
default-deny publication.yaml, and adds the module to signalquarry.toml. Commit the
result on its own (one family per commit).
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = re.compile(r"^[a-z][a-z0-9-]{1,40}$")

STRATEGY = '''"""{title}: replace with the family's first hypothesis."""

from decimal import Decimal

from signalquarry.sdk import Ctx, Decision, Field, Params, strategy, ta


class P(Params):
    symbol: str = "SPY"
    period: int = Field(100, ge=20, le=400)
    weight: Decimal = Field(Decimal("0.5"), ge=0, le=1)


@strategy(params=P, lookback=lambda p: p.period)
def decide(ctx: Ctx, p: P) -> Decision:
    close = ctx.bars(p.symbol).close
    if close[-1] > ta.sma(close, p.period)[-1]:
        return Decision.target({{p.symbol: p.weight}}, "ABOVE_AVERAGE")
    return Decision.target({{}}, "BELOW_AVERAGE")
'''

SPEC = """schema: signalquarry.strategy/v1
id: {strategy_id}
family: {family}
version: 0.1.0
kind: equity_daily
hypothesis:
  statement: Replace with the hypothesis before the first evaluation on real data.
  falsification: Replace with what would prove the hypothesis wrong.
data:
  symbols: [SPY]
  feed: sip
account: {{model: cash, initial_cash: "100000"}}
execution: {{sizing: whole_shares, costs: {{bps: "5"}}}}
params: {{symbol: SPY, period: 100, weight: "0.5"}}
reason_codes:
  ABOVE_AVERAGE: Close above the moving average.
  BELOW_AVERAGE: Close below the moving average.
"""

PUBLICATION = """# Default-deny. Only a human edits this file.
schema: signalquarry.publication/v1
family: {family}
family_label: {title}
commercial: true
tier: category
backtest_results: deny
strategies:
  - id: {strategy_id}
    title: {title} strategy A
    summary: A systematic strategy under research; details are available under NDA only.
    assets: US equities
"""


TEST = '''"""The {family} family's strategies meet the SignalQuarry contract (the checks `sqy check` runs)."""

from pathlib import Path

from signalquarry.testing import conformance

LAB = Path(__file__).resolve().parents[3]


def test_{name}_conformance() -> None:
    conformance(project=LAB, strategy="{strategy_id}")
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("family")
    parser.add_argument("--strategy", help="first strategy id (default: <family>-a)")
    args = parser.parse_args()
    family = args.family
    if not SLUG.fullmatch(family):
        raise SystemExit("family must be lowercase letters, digits and dashes")
    strategy_id = args.strategy or f"{family}-a"
    config_path = ROOT / "signalquarry.toml"
    config = tomllib.loads(config_path.read_text())
    prefix = config["lab"]["package_prefix"]
    package = f"{prefix}_{family.replace('-', '_')}"
    module_dir = strategy_id.replace("-", "_")
    base = ROOT / "families" / family
    if base.exists():
        raise SystemExit(f"{base} exists")
    title = family.replace("-", " ").title()
    files = {
        base / "src" / package / "__init__.py": "",
        base / "src" / package / module_dir / "__init__.py": "",
        base / "src" / package / module_dir / "strategy.py": STRATEGY.format(title=title),
        base / "src" / package / module_dir / "strategy.yaml": SPEC.format(
            strategy_id=strategy_id, family=family
        ),
        base / "publication.yaml": PUBLICATION.format(family=family, title=title, strategy_id=strategy_id),
        base / "tests" / f"test_{family.replace('-', '_')}_conformance.py": TEST.format(
            family=family, strategy_id=strategy_id, name=family.replace("-", "_")
        ),
        base
        / "README.md": f"# {family}\n\nHypotheses, decisions and results for this family live in its ledger.\n",
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    module = f"{package}.{module_dir}.strategy"
    text = config_path.read_text()
    current = config["strategies"]["modules"]
    replacement = "modules = [" + ", ".join(f'"{m}"' for m in [*current, module]) + "]"
    config_path.write_text(re.sub(r"modules = \[[^\]]*\]", replacement, text, count=1))
    print(f"created families/{family} with strategy {strategy_id}; registered {module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
