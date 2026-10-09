# SPDX-License-Identifier: Apache-2.0
"""``sqy data``: explicit, hash-recorded market-data collection and diagnostics."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.data.action_observations import store_action_capture, verify_action_capture
from signalquarry._internal.data.alpaca import AlpacaDataClient, ProviderError, RawPage
from signalquarry._internal.data.credentials import load_data_credentials
from signalquarry._internal.data.library import (
    Library,
    LibraryError,
    build_dataset,
    dataset_from_manifest,
    make_manifest,
)
from signalquarry._internal.data.quality import assess
from signalquarry._internal.project.project import ProjectError, find_root, load_config, load_strategies
from signalquarry._internal.validation.evaluate import add_months
from signalquarry.api.envelope import Envelope
from signalquarry.api.evidence import family_seal, holdout_start_for
from signalquarry.api.resolve import Resolved, resolve

HISTORY_START = date(2016, 1, 1)  # Alpaca stock history begins in 2016
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.]{0,9}\Z")


def library_for(root: Path) -> Library:
    cache = Path(
        os.environ.get("SIGNALQUARRY_CACHE_DIR") or Path.home() / ".cache" / "signalquarry"
    ).expanduser()
    return Library(cache_dir=cache, manifest_dir=root / "data" / "manifests")


def _symbols_and_feed(
    root: Path, strategy_id: str | None, symbols: tuple[str, ...], feed: str | None
) -> tuple[tuple[str, ...], str]:
    if strategy_id:
        strategies = load_strategies(load_config(root))
        if strategy_id not in strategies:
            raise ProjectError("STRATEGY_NOT_FOUND", strategy_id)
        spec = strategies[strategy_id].spec
        wanted = set(spec.data.symbols) | ({spec.benchmark} if spec.benchmark else set[str]())
        return tuple(sorted(wanted | set(symbols))), feed or spec.data.feed
    return tuple(sorted({s.upper() for s in symbols})), feed or "sip"


def data_fetch(
    *,
    strategy_id: str | None = None,
    symbols: tuple[str, ...] = (),
    start: date | None = None,
    end: date | None = None,
    feed: str | None = None,
    project: Path | None = None,
    client: AlpacaDataClient | None = None,
    today: date | None = None,
) -> Envelope:
    envelope = Envelope(command="data fetch")
    try:
        root = find_root(project)
        wanted, chosen_feed = _symbols_and_feed(root, strategy_id, symbols, feed)
    except ProjectError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            [exc.code],
            exc.detail or exc.code,
        )
        return envelope
    if not wanted:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "usage",
            ["USAGE_INVALID"],
            "give --strategy or --symbols",
        )
        return envelope
    if chosen_feed not in ("sip", "iex"):
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            ["PROVIDER_UNAVAILABLE"],
            f"feed {chosen_feed} is not fetchable",
        )
        return envelope
    if client is None:
        credentials = load_data_credentials()
        if credentials is None:
            envelope.status, envelope.reason_codes = "unavailable", ["DATA_CREDENTIALS_MISSING"]
            envelope.summary = "no Alpaca market-data credentials"
            return envelope
        client = AlpacaDataClient(credentials.key_id, credentials.secret_key)
    last = end or (today or datetime.now(UTC).date()) - timedelta(days=1)
    first = start or HISTORY_START
    try:
        bar_pages = client.daily_bars(wanted, first, last, chosen_feed)
        action_pages = client.corporate_actions(wanted, first, last)
        dataset = build_dataset(bar_pages, action_pages, source=f"alpaca:{chosen_feed}")
    except (ProviderError, LibraryError) as exc:
        envelope.status = (
            "unavailable" if exc.code in ("PROVIDER_UNAVAILABLE", "DATA_CREDENTIALS_REJECTED") else "invalid"
        )
        envelope.reason_codes, envelope.summary = [exc.code], str(exc)
        return envelope
    library = library_for(root)
    for page in [*bar_pages, *action_pages]:
        library.store_page(page)
    manifest = make_manifest(
        provider="alpaca",
        feed=chosen_feed,
        symbols=wanted,
        start=first,
        end=last,
        bar_pages=bar_pages,
        action_pages=action_pages,
        dataset=dataset,
        fetched_at=datetime.now(UTC),
    )
    path = library.write_manifest(manifest)
    missing = [symbol for symbol, row in manifest["coverage"].items() if row["count"] == 0]
    envelope.summary = f"{manifest['dataset_id']}: {len(wanted)} symbols, {manifest['sessions']} sessions"
    envelope.data = {
        "dataset_id": manifest["dataset_id"],
        "manifest": str(path.relative_to(root)),
        "coverage": manifest["coverage"],
        "splits": manifest["splits"],
        "dividends": manifest["dividends"],
        "dataset_identity": manifest["dataset_identity"],
    }
    envelope.artifacts = [
        {
            "path": str(path.relative_to(root)),
            "sha256": manifest["manifest_hash"].removeprefix("sha256:"),
            "kind": "dataset_manifest",
        }
    ]
    if missing:
        envelope.warnings.append("DATASET_SYMBOLS_MISSING")
        envelope.data["missing_symbols"] = missing
    return envelope


def data_capture_actions(
    symbols: tuple[str, ...],
    start: date,
    end: date,
    *,
    project: Path | None = None,
    client: AlpacaDataClient | None = None,
    observed_at: datetime | None = None,
) -> Envelope:
    """Capture all action categories for audit without enabling their economics."""
    envelope = Envelope(command="data capture-actions")
    wanted = tuple(sorted(set(symbols)))
    if (
        not wanted
        or len(wanted) > 500
        or any(not _SYMBOL.fullmatch(symbol) for symbol in wanted)
        or start > end
    ):
        envelope.status, envelope.reason_codes, envelope.summary = (
            "usage",
            ["USAGE_INVALID"],
            "give 1–500 uppercase symbols and a valid --start/--end range",
        )
        return envelope
    try:
        root = find_root(project)
    except ProjectError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = "invalid", [exc.code], exc.detail
        return envelope
    if client is None:
        credentials = load_data_credentials()
        if credentials is None:
            envelope.status, envelope.reason_codes = "unavailable", ["DATA_CREDENTIALS_MISSING"]
            envelope.summary = "no Alpaca market-data credentials"
            return envelope
        client = AlpacaDataClient(credentials.key_id, credentials.secret_key)
    try:
        pages = client.corporate_actions(wanted, start, end)
        captured_at = observed_at or datetime.now(UTC)
        library = library_for(root)
        path, capture = store_action_capture(library, pages, observed_at=captured_at)
        observations = verify_action_capture(library, path)
    except (ProviderError, LibraryError) as exc:
        envelope.status = (
            "unavailable" if exc.code in ("PROVIDER_UNAVAILABLE", "DATA_CREDENTIALS_REJECTED") else "invalid"
        )
        envelope.reason_codes, envelope.summary = [exc.code], str(exc)
        return envelope
    except OSError:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "unavailable",
            ["CACHE_DIR_NOT_WRITABLE"],
            "corporate-action cache could not be written",
        )
        return envelope
    envelope.summary = f"captured {len(observations)} corporate-action observations"
    envelope.data = {
        "capture_hash": capture["capture_hash"],
        "observed_at": capture["observed_at"],
        "pages": len(pages),
        "observations": len(observations),
        "cache_record": str(path.relative_to(library.cache_dir)),
        "redistributable": False,
    }
    return envelope


def data_verify(*, project: Path | None = None) -> Envelope:
    envelope = Envelope(command="data verify")
    try:
        root = find_root(project)
    except ProjectError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = "invalid", [exc.code], exc.detail
        return envelope
    library = library_for(root)
    results: list[dict[str, Any]] = []
    for manifest in library.manifests():
        try:
            dataset_from_manifest(library, manifest)
            results.append({"dataset_id": manifest.get("dataset_id"), "ok": True})
        except (LibraryError, KeyError, ValueError) as exc:
            results.append({"dataset_id": manifest.get("dataset_id"), "ok": False, "error": str(exc)})
    for path, record in options_records(root):
        try:
            for reference in record["pages"]:
                library.load_page(reference)
            results.append({"options_record": str(path.relative_to(root)), "ok": True})
        except (LibraryError, KeyError, TypeError) as exc:
            results.append({"options_record": str(path.relative_to(root)), "ok": False, "error": str(exc)})
    for path in sorted((library.cache_dir / "corporate-actions").glob("*.json")):
        name = str(path.relative_to(library.cache_dir))
        try:
            observations = verify_action_capture(library, path)
            results.append({"action_capture": name, "observations": len(observations), "ok": True})
        except LibraryError as exc:
            results.append({"action_capture": name, "ok": False, "error": str(exc)})
    failed = [item for item in results if not item["ok"]]
    envelope.data = {"datasets": results}
    if failed:
        envelope.status, envelope.reason_codes = (
            "blocked",
            sorted({item["error"].split(":", 1)[0] for item in failed}),
        )
    envelope.summary = f"{len(results) - len(failed)} of {len(results)} data records verified"
    return envelope


def _quality_sufficiency(resolved: Resolved, rows: dict[str, Any]) -> dict[str, Any]:
    spec = resolved.strategy.spec
    selected = [rows[symbol] for symbol in spec.data.symbols]
    holdout = None if spec.kind == "options_single_leg" else family_seal(resolved.root, spec.family)
    years, folds = 0.0, 0
    if selected and all(row["present"] for row in selected):
        first = date.fromisoformat(max(row["first"] for row in selected))
        last = date.fromisoformat(min(row["last"] for row in selected))
        if first <= last:
            holdout = holdout or holdout_start_for(spec, last)
            pre_end = min(last, holdout - timedelta(days=1)) if holdout else last
            years = max(0.0, (pre_end - first).days / 365.25)
            cursor = add_months(first, spec.evaluation.walk_forward.train_months)
            while True:
                fold_end = add_months(cursor, spec.evaluation.walk_forward.test_months) - timedelta(days=1)
                if fold_end > pre_end:
                    break
                folds += 1
                cursor = fold_end + timedelta(days=1)
    return {
        "years": round(years, 2),
        "g1_years_ok": years >= 5,
        "walk_forward_folds": folds,
        "g2_folds_ok": folds >= 6,
        "holdout_start": holdout.isoformat() if holdout else None,
        "warm_up_sessions": resolved.strategy.definition.lookback(resolved.strategy.params),
    }


def data_quality(
    *, strategy_id: str | None = None, dataset_id: str | None = None, project: Path | None = None
) -> Envelope:
    """Assess a recorded dataset or a strategy's inputs and write a value-free report."""
    envelope = Envelope(command="data quality")
    if (strategy_id is None) == (dataset_id is None):
        envelope.status, envelope.reason_codes, envelope.summary = (
            "usage",
            ["USAGE_INVALID"],
            "give exactly one of --strategy and --dataset-id",
        )
        return envelope
    resolved = None
    if strategy_id is not None:
        resolved = resolve(envelope.command, strategy_id, project)
        if isinstance(resolved, Envelope):
            return resolved
        root, dataset, dataset_id = resolved.root, resolved.dataset, resolved.dataset_id
        wanted = set(resolved.strategy.spec.data.symbols)
        benchmark = resolved.strategy.spec.benchmark
        if benchmark is not None and benchmark in dataset.series:
            wanted.add(benchmark)
        report = assess(dataset, symbols=tuple(wanted), check_calendar=resolved.grade != "synthetic")
    else:
        try:
            root = find_root(project)
            library = library_for(root)
            matches = [item for item in library.manifests() if item.get("dataset_id") == dataset_id]
            if len(matches) != 1:
                raise LibraryError("DATA_MANIFEST_INVALID", "expected exactly one matching dataset")
            dataset = dataset_from_manifest(library, matches[0])
        except ProjectError as exc:
            envelope.status, envelope.reason_codes, envelope.summary = "invalid", [exc.code], exc.detail
            return envelope
        except LibraryError as exc:
            envelope.status, envelope.reason_codes, envelope.summary = "invalid", [exc.code], str(exc)
            return envelope
        except (OSError, KeyError, TypeError, ValueError) as exc:
            envelope.status, envelope.reason_codes, envelope.summary = (
                "invalid",
                ["DATA_MANIFEST_INVALID"],
                str(exc),
            )
            return envelope
        report = assess(dataset, check_calendar=dataset.source != "synthetic")
    assert dataset_id is not None
    if not dataset_id or Path(dataset_id).name != dataset_id or dataset_id in (".", ".."):
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            ["DATA_MANIFEST_INVALID"],
            "dataset id must be a file name",
        )
        return envelope
    path = root / "data" / "quality" / f"{dataset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    envelope.artifacts = [
        {"path": str(path.relative_to(root)), "kind": "data_quality", "sha256": file_sha256(path)}
    ]
    envelope.data = {
        "dataset_id": dataset_id,
        **{key: report[key] for key in ("dataset_identity", "ok", "findings", "sessions", "common_window")},
        "symbols": {
            symbol: {
                key: row[key] for key in ("present", "first", "last", "coverage", "longest_gap", "findings")
            }
            for symbol, row in report["symbols"].items()
        },
    }
    if resolved is not None:
        envelope.data["sufficiency"] = _quality_sufficiency(resolved, report["symbols"])
    findings = report["findings"]
    detail = f"{len(findings)} findings ({', '.join(findings)})" if findings else "no findings"
    envelope.summary = f"{dataset_id}: {len(report['symbols'])} symbols, {detail}"
    if findings:
        envelope.warnings.append("DATA_QUALITY_FINDINGS")
    return envelope


def data_ls(*, project: Path | None = None) -> Envelope:
    try:
        root = find_root(project)
    except ProjectError as exc:
        return Envelope(command="data ls", status="invalid", reason_codes=[exc.code], summary=exc.detail)
    library = library_for(root)
    rows = [
        {
            key: manifest.get(key)
            for key in ("dataset_id", "provider", "feed", "symbols", "start", "end", "sessions", "fetched_at")
        }
        for manifest in library.manifests()
    ]
    chains: dict[str, dict[str, Any]] = {}
    for _, record in options_records(root):
        at = str(record["recorded_at"])  # ISO-8601 UTC strings sort chronologically
        entry = chains.setdefault(str(record["underlying"]), {"records": 0, "first": at, "last": at})
        entry["records"] += 1
        entry["first"], entry["last"] = min(entry["first"], at), max(entry["last"], at)
    captures = [
        {"capture_hash": f"sha256:{path.stem}", "cache_record": str(path.relative_to(library.cache_dir))}
        for path in sorted((library.cache_dir / "corporate-actions").glob("*.json"))
    ]
    summary = f"{len(rows)} datasets" + (
        f"; option chains recorded for {', '.join(sorted(chains))}" if chains else ""
    )
    if captures:
        summary += f"; {len(captures)} corporate-action captures"
    return Envelope(
        command="data ls",
        summary=summary,
        data={"datasets": rows, "option_chains": chains, "action_captures": captures},
    )


class OptionContractSource(Protocol):
    def option_contracts(self, underlying: str, first: date, last: date) -> list[str]: ...


def data_probe_options(
    underlying: str,
    month: str,
    *,
    broker: OptionContractSource | None = None,
    client: AlpacaDataClient | None = None,
) -> Envelope:
    """Does Alpaca serve contracts that expired in ``month`` and daily bars for them?"""
    from datetime import date as _date

    from signalquarry._internal.data.alpaca import option_daily_bar_count
    from signalquarry._internal.data.credentials import load_paper_credentials

    envelope = Envelope(command="data probe")
    try:
        year, number = (int(part) for part in month.split("-"))
        first = _date(year, number, 1)
        last = _date(year + (number == 12), number % 12 + 1, 1) - timedelta(days=1)
    except ValueError:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "usage",
            ["USAGE_INVALID"],
            "--month is YYYY-MM",
        )
        return envelope
    if broker is None or client is None:
        from signalquarry._internal.paper.brokers.alpaca_paper import AlpacaPaperBroker

        keys = load_data_credentials() or load_paper_credentials("paper.probe")
        if keys is None:
            envelope.status, envelope.reason_codes = "unavailable", ["DATA_CREDENTIALS_MISSING"]
            envelope.summary = "the probe needs Alpaca keys (paper keys can read contracts and data)"
            return envelope
        broker = broker or AlpacaPaperBroker(keys.key_id, keys.secret_key)
        client = client or AlpacaDataClient(keys.key_id, keys.secret_key)
    try:
        symbols = broker.option_contracts(underlying.upper(), first, last)
        sample = symbols[: min(3, len(symbols))]
        counts = {
            symbol: option_daily_bar_count(client, symbol, first - timedelta(days=45), last)
            for symbol in sample
        }
    except ProviderError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = "unavailable", [exc.code], str(exc)
        return envelope
    except Exception as exc:  # noqa: BLE001 - broker errors carry their own code
        code = getattr(exc, "code", "PROVIDER_UNAVAILABLE")
        envelope.status, envelope.reason_codes, envelope.summary = "unavailable", [code], str(exc)
        return envelope
    available = bool(symbols) and any(count > 0 for count in counts.values())
    envelope.data = {
        "underlying": underlying.upper(),
        "month": month,
        "expired_contracts_listed": len(symbols),
        "sample_bar_counts": counts,
        "historical_option_bars": available,
    }
    envelope.summary = f"{underlying.upper()} {month}: {len(symbols)} expired contracts listed; " + (
        "daily bars available for sampled contracts" if available else "no daily bars for sampled contracts"
    )
    if not available:
        envelope.warnings.append("OPTIONS_HISTORY_UNAVAILABLE")
    return envelope


OPTIONS_RECORD_SCHEMA = "signalquarry.options-record/v1"


def options_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    import json

    directory = root / "data" / "options"
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*/*.json"))
        if directory.is_dir()
    ]


def data_record_options(
    underlying: str,
    *,
    max_dte: int = 60,
    width: float = 0.2,
    options_feed: str = "indicative",
    stock_feed: str = "iex",
    project: Path | None = None,
    client: AlpacaDataClient | None = None,
    now: datetime | None = None,
) -> Envelope:
    """Record today's option chain for one underlying: raw pages cached, a hash-only record committed.

    Alpaca's options history is short, so recording chains forward builds real quote history
    over time. The record in ``data/options/<UNDERLYING>/`` holds page hashes and request
    parameters only; the pages (with prices) stay in the cache like every other raw page.
    """
    import json

    from signalquarry._internal.canonical import to_canonical
    from signalquarry._internal.data.alpaca import (
        latest_stock_quote,
        option_snapshot_pages,
        quotes_from_snapshot_pages,
    )

    envelope = Envelope(command="data record")
    symbol = underlying.strip().upper()
    if not symbol.isalpha() or not 1 <= max_dte <= 400 or not 0 < width <= 1:
        envelope.status, envelope.reason_codes = "usage", ["USAGE_INVALID"]
        envelope.summary = "--underlying letters only; 1 <= --max-dte <= 400; 0 < --width <= 1"
        return envelope
    try:
        root = find_root(project)
    except ProjectError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = "invalid", [exc.code], exc.detail
        return envelope
    if client is None:
        credentials = load_data_credentials()
        if credentials is None:
            envelope.status, envelope.reason_codes = "unavailable", ["DATA_CREDENTIALS_MISSING"]
            envelope.summary = "recording option chains needs Alpaca market-data keys"
            return envelope
        client = AlpacaDataClient(credentials.key_id, credentials.secret_key)
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    try:
        quote = latest_stock_quote(client, symbol, stock_feed)
        if quote is None:
            raise ProviderError("PRICE_MISSING", symbol)
        spot = (quote[0] + quote[1]) / 2
        low, high = spot * Decimal(str(1 - width)), spot * Decimal(str(1 + width))
        pages: list[RawPage] = []
        for right in ("put", "call"):
            pages += option_snapshot_pages(
                client,
                symbol,
                right=right,
                expiration_from=moment.date(),
                expiration_to=moment.date() + timedelta(days=max_dte),
                strike_from=low.quantize(Decimal("0.01")),
                strike_to=high.quantize(Decimal("0.01")),
                feed=options_feed,
            )
    except ProviderError as exc:
        envelope.status = (
            "unavailable" if exc.code in ("PROVIDER_UNAVAILABLE", "DATA_CREDENTIALS_REJECTED") else "invalid"
        )
        envelope.reason_codes, envelope.summary = [exc.code], str(exc)
        return envelope
    library = library_for(root)
    for page in pages:
        library.store_page(page)
    contracts = len(quotes_from_snapshot_pages(pages))
    record = {
        "schema": OPTIONS_RECORD_SCHEMA,
        "underlying": symbol,
        "recorded_at": moment.isoformat().replace("+00:00", "Z"),
        "options_feed": options_feed,
        "max_dte": max_dte,
        "contracts": contracts,
        "pages": [{"endpoint": p.endpoint, "params": p.params, "sha256": p.sha256} for p in pages],
        "redistributable": False,
    }
    target = root / "data" / "options" / symbol / f"{moment.strftime('%Y%m%dT%H%M%SZ')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_canonical(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    envelope.summary = f"{symbol}: {contracts} option quotes recorded ({len(pages)} page(s)); record {target.relative_to(root)}"
    envelope.data = {"path": str(target.relative_to(root)), "contracts": contracts, "pages": len(pages)}
    if contracts == 0:
        envelope.warnings.append("OPTIONS_CHAIN_EMPTY")
    envelope.next_actions.append(
        {
            "command": "git add data/options",
            "why": "Commit the hash-only record; the cached pages stay local.",
        }
    )
    return envelope


def load_recorded_chains(root: Path) -> dict[tuple[str, date], dict[str, tuple[Decimal, Decimal]]]:
    """Recorded option chains by (underlying, New York session date): OCC symbol -> (bid, ask)."""
    from zoneinfo import ZoneInfo

    from signalquarry._internal.data.alpaca import quotes_from_snapshot_pages

    library = library_for(root)
    chains: dict[tuple[str, date], dict[str, tuple[Decimal, Decimal]]] = {}
    for _, record in options_records(root):
        at = datetime.fromisoformat(str(record["recorded_at"]).replace("Z", "+00:00"))
        session = at.astimezone(ZoneInfo("America/New_York")).date()
        pages = [library.load_page(reference) for reference in record["pages"]]
        quotes = {symbol: (bid, ask) for symbol, (bid, ask, _) in quotes_from_snapshot_pages(pages).items()}
        chains.setdefault((str(record["underlying"]), session), {}).update(quotes)
    return chains
