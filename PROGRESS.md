# FlyHoldem progress

## Current phase
M1 software Gate 0 passed locally; committing/pushing the coherent rules and CI correction checkpoint. Animated fly checkpoint 6aa7895 is already pushed. Next is M2. V0 commit 2bd2d3c and visual-demo-v0 are pushed. Official source registry/provenance checks, import policy, repository contract and data-free CI are implemented. No MaleCNS source download or teacher/poker training started.

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

- Animated fly: 18 core tests still pass; WebGL 2 / Three.js 186 browser checks cover all six real gestures, live/replay card correspondence, replay-state reset, pause and camera/table controls. No browser errors.

- M1: 29 tests passed, including showdown/ties/side pots/short all-in reopening/private checkpoint continuation. Fixed 2,512-hand software symmetry check passed: independent mean -0.151 BB/hand, SE 0.256678, declared interval [-0.921035, 0.619035], exact paired-seat residual zero.

## Exact working commands
```
cd /Users/bohdankoshevoi/flyholdem/flyholdem
make demo
make test
make replay
uv run python -m playwright install chromium
make browser-check
make record
uv run python -m flyholdem.experiments.software_gate
uv run python scripts/avatar_check.py
```
Demo URL: http://127.0.0.1:8766. Default seed 20260912, graph/mapping seed 1729. One command serves backend/UI. Browser check expects running server.

## User steering
The user requested a visible animated fly like the Doom/döner examples, with its cards face up to the viewer and hands moving for each action. Implemented an original Three.js poker scene with a DOOMFLY-informed avatar; all six check/call/fold/half-pot/pot/all-in gestures passed Chromium checks against actual live/replay events. Cards and event hashes match; both hands move for all-in; pause freezes pose; table toggle/camera reset/mobile layout pass. Screenshot inspection passed. `docs/review/avatar/avatar-check.json` records evidence; compact screenshots are tracked and the full capture/video remains ignored and reproducible. This is an illustrative event-driven character, not a biological motor simulation.

## Important decisions
- Explicit no-strategic-fallback request supersedes the spec's check/call silence clause: silent finite scores choose logged deterministic legal argmax. Invalid scores stop.
- V0 uses a clearly declared coarse synthetic LIF solver; M2 must separately implement/verify the DOOMFLY-comparable reference and native solver before biological claims.
- No learning claims from this visual gate. Fixture constants/config are declared; general training config loading remains M0/M2 work.
- Keep original checkout, development branch astra/visual-first. No agents delegated.

## Known failures and limitations
- Resolved publication restriction: user explicitly approved publishing checked milestones on astra/visual-first and tags to public kosh2shmood/flyholdem. V0 branch and tag pushed successfully. Earlier automatic-review rejections are historical; do not ask again.
- CI M0 tests and browser checks passed, but setup-uv cleanup failed because the background `uv run` server retained a cache lock. Fix: start the installed executable directly and always terminate it with a shell trap. Fixed in the animated-fly commit; CI cleanup now succeeds. The next CI run exposed last-bit floating differences between ARM/macOS and x86/Linux in re-simulated event hashes. Resolution preserves exact within-runtime regeneration, exact recorded playback/hash integrity, and exact categorical/spike behavior with 1e-12 cross-platform diagnostic tolerance. Verify next pushed run.
- Two upstream Starlette test-client deprecation warnings; tests pass.
- Gate 0 passed locally at M1; Gates 1–6/2A pending. No MaleCNS/full runtime, conditioned model, trained teacher, transfer checkpoint, or poker training result yet.
- Full mutable-state checkpoints/resume and registered manifests remain runtime milestones; V0 live logs are append-only, fsynced after completed hands, and ignored.

## Last stable commit
V0: 2bd2d3c2fc6158248c4f633690b5d170add14b32 (visual-demo-v0), pushed. M0: fa873a6, pushed. Animated fly: 6aa7895, pushed. M1 is the following `feat: complete deterministic poker rules gate` checkpoint; see Git log for its exact hash.

## Resumable experiment command
No long training experiment started. Reproduce visual run with `make demo`; replay with `make replay`. Current local live worker logs into a timestamped ignored runs/live-* directory.

## Next step
Verify corrected CI for this rules checkpoint. M2: implement and compare pure-Python and native sparse LIF references with DOOMFLY-comparable .1 ms timing/refractory semantics, complete hash-locked atomic state checkpoints, then explicitly fetch/verify MaleCNS, stream-import all retained nodes/edges and measure memory/throughput. No full-graph training before controllability/conditioning gates. Keep the working avatar/dashboard functional.
