# 0013. Corporate-action identity and effective time

Status: proposed (2026-09-27)

## Context

The current `Dataset` and equity ledger use a ticker as the position key. That
works for ordinary splits and cash dividends, but a symbol can change without
changing the asset, or an action can retire one asset and deliver another. A
missing bar cannot distinguish either case from a delisting. The current Alpaca
adapter therefore rejects unmodelled action types and symbol-changing splits
(ADR 0004). This proposal defines the model needed before that guard can be
relaxed.

Alpaca's corporate-action search filters by *process date*, not necessarily by
the economic effective date. The feed can be revised or populated late. Its
mandatory-action guide says a symbol-only change can keep the same asset, while
a CUSIP change can deactivate the old asset and create a new one. An asset
snapshot alone does not reconstruct the symbol map that was visible in the
past. These are two distinct questions: what happened economically and what
was known at a decision cutoff.

## Proposed decision

1. **Separate identity from display symbol.** The normalized security key is
   a provider namespace plus a provider asset ID, when available. A CUSIP can
   corroborate the key but is not required and must not be published when its
   license forbids it. A symbol is a dated alias of exactly one security key.
   If a source lacks a stable ID or a unique, nonoverlapping alias mapping for
   the relevant interval, identity resolution is unavailable; do not join two
   price histories merely because their ticker strings match. Reused tickers
   are different assets. Alias record IDs are scoped to their provider; a
   revision of one record cannot silently change its asset key. Overlapping
   observations that assign one ticker to different assets, or one asset to
   different tickers, fail identity resolution.
2. **Record three clocks.** Every normalized event records its economic effective
   date, provider process date, and first observed UTC timestamp, with raw-page
   hashes and the normalization version. A later revision is a new observation,
   not an overwrite of an earlier frozen observation. For a decision at time
   `t`, only the version observed by `t` can affect the information set. A
   retrospective, finalized backtest may use later resolved economics, but its
   evidence must say it is retrospective. It cannot claim a point-in-time
   universe from a current feed or current asset list alone.
3. **Apply explicit lifecycle transitions.** A same-asset rename changes only
   the symbol alias, never shares or cash. A split changes shares by its exact
   ratio and, when it also changes symbol, updates the alias in the same
   transition. A worthless removal terminates the old position at zero
   proceeds only when the event explicitly establishes that outcome. A cash
   merger terminates old shares and credits the stated cash consideration. A
   stock merger terminates old shares and creates shares in a *specified
   successor asset* at the stated ratio. A mixed merger does both. A spin-off
   creates a separately identified successor holding; it is not a dividend or
   a gap in the parent's series. Cash and stock components, fractional-share
   treatment, payment/settlement timing, and any fees must be explicit before
   a transition is executable. Missing or conflicting terms are unsupported.
4. **Preserve conservation and ordering.** Each transition has a stable event
   ID, source/target asset keys, effective date, consideration, and provenance.
   It may be applied once. Multiple same-day events require an explicit order
   or a verified commutative result. Each holding must have a valid price or a
   resolved terminal payoff at valuation time; a missing bar must not imply a
   zero value. This model must be shared by backtest and pre-open planning,
   with a parity test over held positions across the transition.
5. **Keep paper reconciliation broker led.** The paper broker's positions,
   activities, and asset identities are authoritative for what actually
   settled. A corporate action may explain position drift only after its
   specific activity, asset mapping, amounts, and dates reconcile to the
   journaled expectation. Unmatched or pending events halt before new orders;
   the kernel does not manufacture a trade or silently repair a journal. An
   observed rename alone never authorizes an order for a successor symbol.

## Rollout and acceptance

ADR 0004's fail-closed guard remains in force until each event family has a
versioned schema, synthetic golden ledger cases (including zero holdings,
reused symbols, late revisions, and ambiguous consideration), replay from
sanitized real provider cassettes, and paper reconciliation tests. An event
family is enabled independently after its exact provider fields and timing
are verified. Existing `DatasetManifestV1` identities and results remain
unchanged; a new version must carry any lifecycle and observation manifests.
The factor universe may use only a historical asset/alias snapshot with a
recorded observation cutoff, or report retrospective coverage explicitly.

The initial internal transition model accepts only fully specified synthetic
terms with fees explicitly confirmed zero. It models rename, split, worthless
removal, cash merger, stock merger and mixed merger. A noninteger share outcome
is allowed only when the event explicitly retains fractional shares; otherwise
it stops. It rejects successor identities from a different provider namespace
until a cross-provider mapping is verified. It records cash consideration as a
dated receivable. Provider parsing,
valuation, paper activity reconciliation and spin-offs remain separate work;
no Alpaca action family is enabled by the internal model alone.

## Consequences

This proposal adds an identity and event layer before factor discovery. It
does not enable any new Alpaca action today. The current symbol-keyed engine
and paper runner continue to reject unsupported actions until the schema,
ledger, and reconciliation work is implemented and reviewed.

## Provider references

Alpaca's Corporate Actions API reference, Mandatory Corporate Actions guide,
and 2026-06-23 reverse-split `new_symbol` changelog describe the provider
behavior used here.
