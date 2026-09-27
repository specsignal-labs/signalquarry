# SPDX-License-Identifier: Apache-2.0
"""A paper deployment wired to the FakeBroker over a synthetic dataset."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, time
from decimal import Decimal
from pathlib import Path

import numpy as np

from signalquarry._internal.contracts.paper import PaperDeploymentV1
from signalquarry._internal.data.dataset import Dataset, truncated, with_pending_session
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.paper.brokers.fake import FakeBroker
from signalquarry._internal.paper.runner import Deployment, PaperKernel
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy
from tests.helpers import spec as make_spec


class Momentum(Params):
    lookback: int = 20
    weight: Decimal = Decimal("0.45")


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def rotate(ctx: Ctx, p: Momentum) -> Decision:
    """Hold whichever symbol has the stronger positive momentum; flat when neither does."""
    scores = {}
    for symbol in ("SYNA", "SYNB"):
        close = ctx.bars(symbol).close
        scores[symbol] = close[-1] / close[0] - 1
    best = max(sorted(scores), key=lambda s: scores[s])
    if not np.isfinite(scores[best]) or scores[best] <= 0:
        return Decision.target({}, "WAIT", state={"last": None})
    return Decision.target({best: p.weight}, "GO", state={"last": best})


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def hold_syna(ctx: Ctx, p: Momentum) -> Decision:
    """A constant target: any missing fill must be retried on the next session."""
    return Decision.target({"SYNA": p.weight}, "GO")


def parity_dataset(start: date = date(2023, 1, 2), end: date = date(2024, 12, 31)) -> Dataset:
    """SYNA and SYNB (with SYNB's 2-for-1 split in range), without dividends (Alpaca paper omits them)."""
    return replace(synthetic_dataset(start, end, symbols=("SYNA", "SYNB")), dividends=())


def paper_spec(**overrides):
    return make_spec(
        ("SYNA", "SYNB"),
        execution={"costs": {"bps": "0", "per_share": "0"}, **overrides.pop("execution", {})},
        **overrides,
    )


@dataclass
class Rig:
    dataset: Dataset
    broker: FakeBroker
    kernel: PaperKernel
    deployment: Deployment

    def loader(self, session: date) -> Dataset:
        return with_pending_session(truncated(self.dataset, self.dataset.index_of(session)), session)

    def at(self, session: date, at: time = time(9, 10)) -> None:
        self.broker.pre_open(session, at)

    def arm(self, session: date) -> None:
        self.at(session, time(9, 5))
        self.kernel.arm()

    def session(self, session: date):
        self.at(session)
        outcome = self.kernel.run_once()
        self.broker.open_session(session)
        return outcome


def rig(
    tmp_path: Path,
    *,
    dataset: Dataset | None = None,
    the_spec=None,
    submission: str = "enabled",
    config: dict | None = None,
    freeze_hash: str | None = "sha256:" + "f" * 64,
    configuration_hash: str = "sha256:" + "c" * 64,
    broker_class: type[FakeBroker] = FakeBroker,
    decide=rotate,
) -> Rig:
    data = dataset or parity_dataset()
    the_spec = the_spec or paper_spec()
    broker = broker_class(data, the_spec.account.initial_cash)
    deployment = Deployment(
        config=PaperDeploymentV1.model_validate(
            {
                "schema": "signalquarry.paper/v1",
                "alias": "demo",
                "strategy": "test-strategy",
                "submission": submission,
                "isolation": "none",  # tests of isolation opt in; in-process planning keeps the suite fast
                **(config or {}),
            }
        ),
        spec=the_spec,
        definition=definition_of(decide),
        params=Momentum(),
        configuration_hash=configuration_hash,
        freeze_hash=freeze_hash,
        state_dir=tmp_path / "paper" / "demo",
    )
    holder: dict[str, Rig] = {}
    kernel = PaperKernel(deployment, broker, lambda s: holder["rig"].loader(s), now=lambda: broker.now)
    holder["rig"] = Rig(data, broker, kernel, deployment)
    return holder["rig"]
