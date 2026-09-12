# FlyHoldem progress

## Current phase
M0 complete; proceed to M1 software Gate 0. V0 commit 2bd2d3c and visual-demo-v0 are pushed. Official source registry/provenance checks, import policy, repository contract and data-free CI are implemented. No MaleCNS source download or teacher/poker training started.

## Completed acceptance checks
- Read execution brief/specification/setup context and applicable ancestor AGENTS.md paths (none existed).
- Reviewed official PokerKit/MaleCNS/PettingZoo API/data documentation; pinned inspected DOOMFLY revision in THIRD_PARTY.md.
- Python 3.11 environment and uv.lock; zero-runtime-dependency dashboard.
- Real deterministic two-player PokerKit loop → visible-info encoder → fixture LIF scores → legal action → terminal RPE/eligibility weight changes.
- Live WebSocket, immutable checked-in six-hand replay, exact replay and event-chain verification.
- 15 pytest tests passed: rule/raise mechanics, duplicate masks, chip conservation/replay over 100 random hands, byte-level hidden-card/deck leakage, neural-only scores, bounds/frozen updates, refractory edge case and actual WebSocket/replay.
- Chromium desktop 1440×1080 and mobile 390×844 checks passed, screenshots inspected; no browser errors or overflow.
- Short fixture benchmark and visual gate report saved in docs/review/ and docs/VISUAL_GATE.md.

- M0: source-lock validation and mismatch rejection tests pass; exact expected data bytes total 1,109,008,094. Explicit resumable fetch retains the pre-existing trusted hashes.
- M0: config/source/graph/encoder/decoder/binary/runtime identity comparison rejects mismatched checkpoints. CI uses locked Python packages and uv 0.12.13.

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
- Resolved publication restriction: user explicitly approved publishing checked milestones on astra/visual-first and tags to public kosh2shmood/flyholdem. V0 branch and tag pushed successfully. Earlier automatic-review rejections are historical; do not ask again.
- Two upstream Starlette test-client deprecation warnings; tests pass.
- Gate 0 only partially covered until M1; Gates 1–6/2A pending. No MaleCNS/full runtime, conditioned model, trained teacher, transfer checkpoint, or poker training result yet.
- Full mutable-state checkpoints/resume and registered manifests remain runtime milestones; V0 live logs are append-only, fsynced after completed hands, and ignored.

## Last stable commit
V0: 2bd2d3c2fc6158248c4f633690b5d170add14b32 (visual-demo-v0), pushed. M0 is the following `chore: lock provenance and add fixture CI` checkpoint; see Git log for its exact hash.

## Resumable experiment command
No long training experiment started. Reproduce visual run with `make demo`; replay with `make replay`. Current local live worker logs into a timestamped ignored runs/live-* directory.

## Next step
Finish M1 software Gate 0: full golden hands (ties, short raises, uneven/side pots), deterministic paired seat-symmetry evidence, state replay serialization and opponent policies. Then M2 reference/native numerical runtime and checksum-verified full MaleCNS import/audit, measuring memory before any concurrent full jobs. Do not skip failed scientific gates.
