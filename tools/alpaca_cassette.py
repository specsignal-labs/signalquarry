# SPDX-License-Identifier: Apache-2.0
"""Private Alpaca HTTP cassette recording and strict offline replay.

The recorder is deliberately limited to allowlisted read-only Alpaca GET
requests. It writes response bodies to a private file outside every Git
worktree, omits request headers, and redacts credentials. Replay has no
delegate or network fallback and can drive both Alpaca transport protocols.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, unquote_plus, urlsplit

from signalquarry._internal.data.alpaca import DATA_ORIGIN, HttpResponse
from signalquarry._internal.data.credentials import redact, secret_values
from signalquarry._internal.paper.brokers.alpaca_paper import PAPER_ORIGIN

CASSETTE_SCHEMA = "signalquarry.alpaca-http-cassette/v2"
ALLOWED_ORIGINS = frozenset((DATA_ORIGIN, PAPER_ORIGIN))
SAVED_RESPONSE_HEADERS = frozenset(("content-type", "retry-after"))
MAX_CASSETTE_BYTES = 64 * 1024 * 1024
PAPER_READ_ONLY_PATHS = frozenset(("/v2/assets", "/v2/calendar", "/v2/clock", "/v2/options/contracts"))
DATA_READ_ONLY_PATHS = frozenset(("/v1/corporate-actions", "/v1beta1/options/bars", "/v2/stocks/bars"))


class CassetteError(ValueError):
    """Cassette is invalid, unsafe to save, or does not match the replayed request."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_document_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _url_contains_credential(url: str, values: set[str]) -> bool:
    decoded = (url, unquote(url), unquote_plus(url))
    return any(value and any(value in candidate for candidate in decoded) for value in values)


def _check_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
    except (TypeError, ValueError) as exc:
        raise CassetteError("invalid request URL") from exc
    if (
        origin not in ALLOWED_ORIGINS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise CassetteError("cassette URL must use a fixed Alpaca HTTPS origin")


def _check_recordable_get(url: str) -> None:
    _check_url(url)
    parsed = urlsplit(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin == DATA_ORIGIN:
        allowed = parsed.path in DATA_READ_ONLY_PATHS or bool(
            re.fullmatch(r"/v1beta1/options/snapshots/[^/]+", parsed.path)
            or re.fullmatch(r"/v2/stocks/[^/]+/quotes/latest", parsed.path)
        )
    else:
        allowed = origin == PAPER_ORIGIN and parsed.path in PAPER_READ_ONLY_PATHS
    if not allowed:
        raise CassetteError("recorder path is not on the read-only Alpaca allowlist")


def _body_hash(body: bytes | None) -> str | None:
    return None if body is None else _sha256(body)


def _is_git_path(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path, *path.parents))


def _secure_target(path: Path) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise CassetteError("cassette target must not be a symlink")
    target = requested.resolve()
    if _is_git_path(target):
        raise CassetteError("cassette files must be stored outside Git worktrees")
    return target


@dataclass(frozen=True)
class CassetteEntry:
    method: str
    url: str
    request_body_sha256: str | None
    status: int
    headers: dict[str, str]
    body: bytes
    body_sha256: str

    @classmethod
    def create(
        cls,
        method: str,
        url: str,
        request_body: bytes | None,
        response: HttpResponse,
    ) -> CassetteEntry:
        body = bytes(response.body)
        return cls(
            method=method.upper(),
            url=url,
            request_body_sha256=_body_hash(request_body),
            status=response.status,
            headers={key.lower(): value for key, value in response.headers.items()},
            body=body,
            body_sha256=_sha256(body),
        )

    def validate(self) -> None:
        if not isinstance(self.method, str) or not isinstance(self.url, str):
            raise CassetteError("invalid cassette request")
        _check_url(self.url)
        if self.method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise CassetteError("invalid cassette method")
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise CassetteError("invalid cassette response status")
        if not isinstance(self.body, bytes):
            raise CassetteError("invalid cassette response body")
        if self.request_body_sha256 is not None and (
            not isinstance(self.request_body_sha256, str)
            or len(self.request_body_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.request_body_sha256)
        ):
            raise CassetteError("invalid request body hash")
        if not isinstance(self.body_sha256, str) or self.body_sha256 != _sha256(self.body):
            raise CassetteError("cassette response body hash mismatch")
        if not isinstance(self.headers, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or key.lower() not in SAVED_RESPONSE_HEADERS
            for key, value in self.headers.items()
        ):
            raise CassetteError("cassette contains an unapproved response header")

    def document(self) -> dict[str, Any]:
        self.validate()
        return {
            "method": self.method,
            "url": self.url,
            "request_body_sha256": self.request_body_sha256,
            "status": self.status,
            "headers": dict(sorted(self.headers.items())),
            "body_base64": base64.b64encode(self.body).decode("ascii"),
            "body_sha256": self.body_sha256,
        }

    @classmethod
    def from_document(cls, document: object) -> CassetteEntry:
        if not isinstance(document, dict):
            raise CassetteError("cassette entry must be an object")
        expected = {
            "method",
            "url",
            "request_body_sha256",
            "status",
            "headers",
            "body_base64",
            "body_sha256",
        }
        if set(document) != expected:
            raise CassetteError("cassette entry has missing or unapproved fields")
        try:
            body = base64.b64decode(document["body_base64"], validate=True)
            headers = document["headers"]
            method = document["method"]
            url = document["url"]
            status = document["status"]
            body_hash = document["body_sha256"]
            request_hash = document["request_body_sha256"]
            if (
                not isinstance(method, str)
                or not isinstance(url, str)
                or type(status) is not int
                or not isinstance(body_hash, str)
                or (request_hash is not None and not isinstance(request_hash, str))
            ):
                raise CassetteError("invalid cassette entry fields")
            if not isinstance(headers, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
            ):
                raise CassetteError("invalid cassette response headers")
            entry = cls(
                method=method,
                url=url,
                request_body_sha256=request_hash,
                status=status,
                headers={key.lower(): value for key, value in headers.items()},
                body=body,
                body_sha256=body_hash,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, CassetteError):
                raise
            raise CassetteError("invalid cassette entry") from exc
        entry.validate()
        return entry


@dataclass(frozen=True)
class Cassette:
    entries: tuple[CassetteEntry, ...]
    recording_kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.recording_kind, str) or self.recording_kind not in {
            "synthetic",
            "private-provider",
        }:
            raise CassetteError("recording_kind must be synthetic or private-provider")
        for entry in self.entries:
            if not isinstance(entry, CassetteEntry):
                raise CassetteError("cassette entries must be CassetteEntry values")
            entry.validate()
            if self.recording_kind == "private-provider":
                if entry.method != "GET":
                    raise CassetteError("private-provider cassettes may contain only GET requests")
                _check_recordable_get(entry.url)

    def dumps(self) -> bytes:
        document = {
            "schema": CASSETTE_SCHEMA,
            "recording_kind": self.recording_kind,
            "entries": [entry.document() for entry in self.entries],
        }
        document["cassette_sha256"] = _sha256(_canonical_document_bytes(document))
        raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(raw) > MAX_CASSETTE_BYTES:
            raise CassetteError("cassette exceeds configured size limit")
        return raw

    @classmethod
    def loads(cls, raw: bytes) -> Cassette:
        if not isinstance(raw, bytes):
            raise CassetteError("cassette must be supplied as bytes")
        if len(raw) > MAX_CASSETTE_BYTES:
            raise CassetteError("cassette exceeds configured size limit")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise CassetteError("cassette is not valid JSON") from exc
        expected = {"schema", "recording_kind", "entries", "cassette_sha256"}
        if not isinstance(document, dict) or set(document) != expected:
            raise CassetteError("cassette has missing or unapproved fields")
        if document.get("schema") != CASSETTE_SCHEMA:
            raise CassetteError("unsupported cassette schema")
        claimed_hash = document.get("cassette_sha256")
        if (
            not isinstance(claimed_hash, str)
            or len(claimed_hash) != 64
            or any(char not in "0123456789abcdef" for char in claimed_hash)
        ):
            raise CassetteError("cassette checksum is invalid")
        recording_kind = document.get("recording_kind")
        if not isinstance(recording_kind, str):
            raise CassetteError("cassette recording_kind is invalid")
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise CassetteError("cassette entries must be a list")
        payload = dict(document)
        del payload["cassette_sha256"]
        cassette = cls(tuple(CassetteEntry.from_document(item) for item in entries), recording_kind)
        if claimed_hash != _sha256(_canonical_document_bytes(payload)):
            raise CassetteError("cassette checksum mismatch")
        return cassette

    @classmethod
    def read(cls, path: Path, *, require_private: bool = True) -> Cassette:
        target = _secure_target(path)
        if target.stat().st_size > MAX_CASSETTE_BYTES:
            raise CassetteError("cassette exceeds configured size limit")
        if require_private:
            mode = stat.S_IMODE(target.stat().st_mode)
            if mode & 0o077:
                raise CassetteError("private cassette permissions must be owner-only")
        return cls.loads(target.read_bytes())


@dataclass(frozen=True)
class AlpacaReadOnlyTransport:
    """Join the data and paper GET transports for one ordered read-only cassette."""

    data: Any
    paper: Any

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        _check_recordable_get(url)
        return self.data.get(url, headers)

    def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> HttpResponse:
        if method != "GET" or body is not None:
            raise CassetteError("combined Alpaca cassette transport is read-only")
        _check_recordable_get(url)
        return self.paper.request(method, url, headers, body)


@dataclass
class RecordingAlpacaTransport:
    """Record allowlisted GET responses to a private cassette; mutations are refused."""

    delegate: Any
    path: Path
    recording_kind: str
    credentials: tuple[str, ...] = ()
    max_bytes: int = MAX_CASSETTE_BYTES
    _entries: list[CassetteEntry] = field(default_factory=list, init=False, repr=False)
    _recorded_bytes: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = _secure_target(self.path)
        if self.recording_kind not in {"synthetic", "private-provider"}:
            raise CassetteError("recording_kind must be synthetic or private-provider")
        if self.max_bytes <= 0:
            raise CassetteError("max_bytes must be positive")

    def _record(self, url: str, call: Callable[[], HttpResponse]) -> HttpResponse:
        _check_recordable_get(url)
        secrets = set(self.credentials) | secret_values()
        if _url_contains_credential(url, secrets):
            raise CassetteError("refusing to record a request URL containing credentials")
        response = call()
        try:
            response_text = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CassetteError("market-data response is not UTF-8") from exc
        safe_body = redact(response_text, secrets).encode("utf-8")
        safe_headers = {
            key.lower(): redact(value, secrets)
            for key, value in response.headers.items()
            if key.lower() in SAVED_RESPONSE_HEADERS
        }
        entry = CassetteEntry.create(
            "GET",
            url,
            None,
            HttpResponse(response.status, safe_headers, safe_body),
        )
        entry.validate()
        if self._recorded_bytes + len(safe_body) > self.max_bytes:
            raise CassetteError("cassette exceeds configured size limit")
        self._entries.append(entry)
        self._recorded_bytes += len(safe_body)
        return response

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        return self._record(url, lambda: self.delegate.get(url, headers))

    def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> HttpResponse:
        if method != "GET" or body is not None:
            raise CassetteError("cassette recorder is read-only")
        return self._record(url, lambda: self.delegate.request(method, url, headers, body))

    def cassette(self) -> Cassette:
        return Cassette(tuple(self._entries), self.recording_kind)

    def save(self) -> Path:
        target = _secure_target(self.path)
        parent = target.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        mode = stat.S_IMODE(parent.stat().st_mode)
        if mode & 0o077 or mode & 0o300 != 0o300:
            raise CassetteError("cassette directory must be owner-only and writable")
        raw = self.cassette().dumps()
        if len(raw) > self.max_bytes:
            raise CassetteError("cassette exceeds configured size limit")
        fd, temporary = tempfile.mkstemp(prefix=".sqy-cassette-", dir=parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return target

    def __enter__(self) -> RecordingAlpacaTransport:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._entries:
            self.save()


@dataclass
class ReplayTransport:
    """Strict offline adapter for both Alpaca transport protocols."""

    cassette: Cassette
    _next: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        for entry in self.cassette.entries:
            entry.validate()

    @classmethod
    def read(cls, path: Path) -> ReplayTransport:
        return cls(Cassette.read(path))

    def _response(self, method: str, url: str, body: bytes | None) -> HttpResponse:
        _check_url(url)
        if self._next >= len(self.cassette.entries):
            raise CassetteError(f"cassette exhausted at request {self._next + 1}")
        entry = self.cassette.entries[self._next]
        if (
            entry.method != method.upper()
            or entry.url != url
            or entry.request_body_sha256 != _body_hash(body)
        ):
            raise CassetteError(f"cassette request mismatch at entry {self._next + 1}")
        self._next += 1
        return HttpResponse(entry.status, dict(entry.headers), entry.body)

    def get(self, url: str, headers: dict[str, str]) -> HttpResponse:
        del headers  # Authentication headers are intentionally not part of cassette matching.
        return self._response("GET", url, None)

    def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> HttpResponse:
        del headers
        return self._response(method, url, body)

    def assert_complete(self) -> None:
        remaining = len(self.cassette.entries) - self._next
        if remaining:
            raise CassetteError(f"{remaining} cassette entr{'y' if remaining == 1 else 'ies'} unused")
