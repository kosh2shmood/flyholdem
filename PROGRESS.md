# FlyHoldem progress

## Current phase
V0 complete locally: real fixture visual-first vertical slice verified. Save milestone commit and `visual-demo-v0` tag, then proceed to M0/M1 and later dependency-ordered milestones. No full MaleCNS data or poker teacher/training started.

## Completed acceptance checks
- Read execution brief/specification/setup context and applicable ancestor AGENTS.md paths (none existed).
- Reviewed official PokerKit/MaleCNS/PettingZoo API/data documentation; pinned inspected DOOMFLY revision in THIRD_PARTY.md.
- Python 3.11 environment and uv.lock; zero-runtime-dependency dashboard.
- Real deterministic two-player PokerKit loop → visible-info encoder → fixture LIF scores → legal action → terminal RPE/eligibility weight changes.
- Live WebSocket, immutable checked-in six-hand replay, exact replay and event-chain verification.
- 15 pytest tests passed: rule/raise mechanics, duplicate masks, chip conservation/replay over 100 random hands, byte-level hidden-card/deck leakage, neural-only scores, bounds/frozen updates, refractory edge case and actual WebSocket/replay.
- Chromium desktop 1440×1080 and mobile 390×844 checks passed, screenshots inspected; no browser errors or overflow.
- Short fixture benchmark and visual gate report saved in docs/review/ and docs/VISUAL_GATE.md.

## Exact working commands
```
cd /Users/bohdankoshevoi/flyholdem/flyholdem
make demo
make test
make replay
uv run python -m playwright install chromium
make browser-check
make record
```
Demo URL: http://127.0.0.1:8766. Default seed 20260912, graph/mapping seed 1729. One command serves backend/UI. Browser check expects running server.

## Important decisions
- Explicit no-strategic-fallback request supersedes the spec's check/call silence clause: silent finite scores choose logged deterministic legal argmax. Invalid scores stop.
- V0 uses a clearly declared coarse synthetic LIF solver; M2 must separately implement/verify the DOOMFLY-comparable reference and native solver before biological claims.
- No learning claims from this visual gate. Fixture constants/config are declared; general training config loading remains M0/M2 work.
- Keep original checkout, development branch astra/visual-first. No agents delegated.

## Known failures and limitations
- Automatic approval review twice rejected push: first destination verification, then public publication. Authenticated user kosh2shmood has ADMIN on the exact existing public origin https://github.com/kosh2shmood/flyholdem. No push has executed; request explicit public-publication approval only after a concrete verified commit exists. Continue local checkpoints meanwhile.
- Two upstream Starlette test-client deprecation warnings; tests pass.
- Gate 0 only partially covered until M1; Gates 1–6/2A pending. No MaleCNS/full runtime, conditioned model, trained teacher, transfer checkpoint, or poker training result yet.
- Full mutable-state checkpoints/resume and registered manifests remain runtime milestones; V0 live logs are append-only, fsynced after completed hands, and ignored.

## Last stable commit
The commit tagged `visual-demo-v0` contains this verified V0 milestone. Its parent is 1f66f396472801cad0ee21a56da47a70ec17c1d4.

## Resumable experiment command
No long training experiment started. Reproduce visual run with `make demo`; replay with `make replay`. Current local live worker logs into a timestamped ignored runs/live-* directory.

## Next step
M0: AGENTS.md, CI, validated config/provenance registry/locks; then finish M1 software Gate 0. Keep V0 frozen at its tag. After that implement M2 reference/native runtime, explicit checksum-verified MaleCNS setup, resource measurements and audit. Do not skip failed scientific gates.
