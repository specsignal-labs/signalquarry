# SPDX-License-Identifier: Apache-2.0
"""An options paper deployment wired to the FakeOptionsVenue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from pathlib import Path

from signalquarry._internal.contracts.paper import PaperDeploymentV1
from signalquarry._internal.data.dataset import Dataset, truncated, with_pending_session
from signalquarry._internal.paper.brokers.fake_options import FakeOptionsVenue, at
from signalquarry._internal.paper.options_runner import OptionsKernel
from signalquarry._internal.paper.runner import Deployment
from signalquarry.sdk import definition_of
from tests.options.wheel_reference import WheelParams, wheel, wheel_spec


@dataclass
class OptionsRig:
    dataset: Dataset
    venue: FakeOptionsVenue
    kernel: OptionsKernel
    deployment: Deployment

    def loader(self, session: date) -> Dataset:
        return with_pending_session(truncated(self.dataset, self.dataset.index_of(session)), session)

    def poll(self, session: date, moment: time):
        self.venue.now = at(session, moment)
        return self.kernel.poll()

    def arm(self, session: date) -> None:
        self.venue.now = at(session, time(9, 31))
        self.kernel_arm()

    def kernel_arm(self) -> None:
        from signalquarry._internal.paper import arm
        from signalquarry._internal.paper.journal import Journal
        from signalquarry._internal.paper.lease import RunLease

        d = self.deployment
        with RunLease(d.lease_path):
            journal = Journal.open(d.journal_path)
            token = arm.issue(
                alias=d.config.alias,
                strategy_id=d.spec.id,
                configuration_hash=d.configuration_hash,
                freeze_hash=d.freeze_hash or "",
                account_id=self.venue.account_id,
                broker=self.venue.name,
                journal_head=journal.head,
                now=self.venue.now,
                days=90,
            )
            arm.write(d.arm_path, token)
            journal.append(
                "armed",
                {
                    "token_hash": token["token_hash"],
                    "baseline_session": self.venue.now.date(),
                    "baseline_positions": {},
                },
                now=self.venue.now,
            )

    def day(self, session: date) -> list:
        outcomes = [self.poll(session, time(9, 35)), self.poll(session, time(15, 55))]
        self.venue.end_of_day(session)
        return outcomes


def options_rig(
    tmp_path: Path,
    *,
    dataset: Dataset,
    params: WheelParams | None = None,
    spec=None,
    cash: Decimal = Decimal(100000),
    config: dict | None = None,
) -> OptionsRig:
    spec = spec or wheel_spec(options={"underlyings": ["QQQ"], "spread_haircut": "0.000001"})
    venue = FakeOptionsVenue(dataset, cash)
    deployment = Deployment(
        config=PaperDeploymentV1.model_validate(
            {
                "schema": "signalquarry.paper/v1",
                "alias": "wheel",
                "strategy": "wheel-reference",
                "submission": "enabled",
                "isolation": "none",
                **(config or {}),
            }
        ),
        spec=spec,
        definition=definition_of(wheel),
        params=params or WheelParams(),
        configuration_hash="sha256:" + "c" * 64,
        freeze_hash="sha256:" + "f" * 64,
        state_dir=tmp_path / "paper" / "wheel",
    )
    holder: dict[str, OptionsRig] = {}
    kernel = OptionsKernel(deployment, venue, lambda s: holder["rig"].loader(s), now=lambda: venue.now)
    holder["rig"] = OptionsRig(dataset, venue, kernel, deployment)
    return holder["rig"]
