# SPDX-License-Identifier: Apache-2.0
"""A fake Alpaca market-data server speaking the real response format."""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time

from signalquarry._internal.data.alpaca import HttpResponse
from signalquarry._internal.data.dataset import MICRO, Dataset


def _price(micro: int) -> float:
    return micro / MICRO


@dataclass
class FakeAlpaca:
    dataset: Dataset
    page_size: int = 400
    fail_first: int | None = None  # HTTP status returned once before normal responses
    loop_tokens: bool = False
    calls: list[str] = field(default_factory=list)

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        self.calls.append(url)
        assert headers.get("APCA-API-KEY-ID") and headers.get("APCA-API-SECRET-KEY")
        if self.fail_first is not None:
            status, self.fail_first = self.fail_first, None
            return HttpResponse(status, {"retry-after": "0"}, b'{"message":"fail"}')
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == "https" and parsed.netloc == "data.alpaca.markets"
        params = {key: values[0] for key, values in urllib.parse.parse_qs(parsed.query).items()}
        if parsed.path == "/v2/stocks/bars":
            body = self._bars(params)
        elif parsed.path == "/v1/corporate-actions":
            body = self._actions(params)
        else:
            return HttpResponse(404, {}, b"{}")
        return HttpResponse(200, {}, json.dumps(body).encode())

    def _bars(self, params: dict[str, str]) -> dict:
        assert params["adjustment"] == "raw" and params["timeframe"] == "1Day"
        start, end = date.fromisoformat(params["start"]), date.fromisoformat(params["end"])
        rows = []
        for symbol in params["symbols"].split(","):
            item = self.dataset.series.get(symbol)
            if item is None:
                continue
            for i, session in enumerate(self.dataset.sessions):
                if item.present[i] and start <= session <= end:
                    rows.append(
                        (
                            symbol,
                            {
                                "t": datetime.combine(session, time(5), tzinfo=UTC)
                                .isoformat()
                                .replace("+00:00", "Z"),
                                "o": _price(int(item.micro["open"][i])),
                                "h": _price(int(item.micro["high"][i])),
                                "l": _price(int(item.micro["low"][i])),
                                "c": _price(int(item.micro["close"][i])),
                                "v": float(item.volume[i]),
                                "n": 100,
                                "vw": _price(int(item.micro["close"][i])),
                            },
                        )
                    )
        offset = int(params.get("page_token", "0"))
        chunk = rows[offset : offset + self.page_size]
        bars: dict[str, list] = {}
        for symbol, row in chunk:
            bars.setdefault(symbol, []).append(row)
        more = offset + self.page_size < len(rows)
        token = ("0" if self.loop_tokens else str(offset + self.page_size)) if more else None
        return {"bars": bars, "next_page_token": token}

    def _actions(self, params: dict[str, str]) -> dict:
        symbols = set(params["symbols"].split(","))
        start, end = date.fromisoformat(params["start"]), date.fromisoformat(params["end"])
        splits = [
            {"symbol": s.symbol, "new_rate": float(s.ratio), "old_rate": 1, "ex_date": s.ex_date.isoformat()}
            for s in self.dataset.splits
            if s.symbol in symbols and start <= s.ex_date <= end
        ]
        dividends = [
            {
                "symbol": d.symbol,
                "rate": float(d.amount),
                "ex_date": d.ex_date.isoformat(),
                "payable_date": d.pay_date.isoformat(),
            }
            for d in self.dataset.dividends
            if d.symbol in symbols and start <= d.ex_date <= end
        ]
        return {
            "corporate_actions": {"forward_splits": splits, "cash_dividends": dividends},
            "next_page_token": None,
        }
