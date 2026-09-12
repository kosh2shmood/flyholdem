# FlyHoldem progress

## Current phase
Card-readability revision passed local browser checks and is ready to save/push. Larger 3D cards, matched fly/community insets and visible opponent chip piles now work on desktop/mobile. Previous side-seating checkpoint 80a32fd passed CI (run 34677267958). All three official MaleCNS files have downloaded and independently matched the committed byte counts/SHA-256 expectations. Import/preparation code is a tested draft, not yet run on the full data; save its own checkpoint/config before starting the import. No teacher or poker training started.

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

- M1 corrected CI passed: https://github.com/kosh2shmood/flyholdem/actions/runs/34676112475.
- M2 numerical subset: four native/Python tests and three atomic-checkpoint tests pass. Independent Brian2 2.10.1 oracle: two tests pass in isolated Python 3.12.14. The app remains Python 3.11.16. Runtime source/binary identity locked; latest-three-plus-best retention and tamper/mismatch rejection verified.

## Exact working commands
```
cd /Users/bohdankoshevoi/flyholdem/flyholdem
make demo
make test
make replay
uv run python -m playwright install chromium
make browser-check
make record
make build-kernel
make test-oracle
uv run python -m flyholdem.experiments.software_gate
uv run python scripts/avatar_check.py
```
Demo URL: http://127.0.0.1:8766. Default seed 20260912, graph/mapping seed 1729. One command serves backend/UI. Browser check expects running server.

## Latest readability evidence
- All six fly gestures and both real opponent gestures still pass Chromium checks. Community inset matches actual events for 0/3/4/5 dealt cards. Both players show chips iff their balance is positive. Full models remain in the desktop/mobile camera frame; no page errors or horizontal overflow.
- Existing 36 core tests pass; four additional importer-draft tests also pass (40 total with the optional data packages installed). The two independent Brian2 tests remain in their separate passing oracle environment.
- Data download verified: annotations 14,483,314 bytes, neurotransmitters 43,282,834 bytes, edges 1,051,241,946 bytes. Exact digests match data-provenance/malecns_v1/source.lock.json. Raw data stays ignored.

## Current visual revision evidence
- Core: 36 passed, two Brian2 tests intentionally skipped in the base environment; both run and pass in the separate Python 3.12 oracle, including CI.
- Browser: all six fly gestures plus actual opponent check/call, matching action/event hashes and cards, opponent privacy including mucked cards, pause for both players, table toggle, camera reset, desktop 1440×1080 and mobile 390×844 full silhouette bounds. No browser errors.
- Cards have distinct single-sided front/back materials and a 0.12 scene-unit curve. Both printed faces have positive orientation toward the fly and default camera. Screenshot inspection corrected all-in card placement to the felt.
- Compact evidence: docs/review/avatar/avatar-check.json and three screenshots. Full video and per-gesture screenshots remain generated/ignored.
- User explicitly permits Python upgrades and other visualization/runtime languages when useful. Current Three.js viewer and isolated Python 3.12 numerical oracle are retained; no artificial Python-only constraint.

## Latest user steering
The user wants the fly moved to the side of the table so its whole body is visible, its cards oriented toward itself and slightly bent toward the spectator for readability, and an abstract opponent across the table. Both actors must make physical-looking gestures based on their actual poker actions. Implement after the current numerical/CI issues are resolved (they now are).

## Earlier user steering
The user requested a visible animated fly like the Doom/döner examples, with its cards face up to the viewer and hands moving for each action. Implemented an original Three.js poker scene with a DOOMFLY-informed avatar; all six check/call/fold/half-pot/pot/all-in gestures passed Chromium checks against actual live/replay events. Cards and event hashes match; both hands move for all-in; pause freezes pose; table toggle/camera reset/mobile layout pass. Screenshot inspection passed. `docs/review/avatar/avatar-check.json` records evidence; compact screenshots are tracked and the full capture/video remains ignored and reproducible. This is an illustrative event-driven character, not a biological motor simulation.

## Important decisions
- Explicit no-strategic-fallback request supersedes the spec's check/call silence clause: silent finite scores choose logged deterministic legal argmax. Invalid scores stop.
- V0 uses a clearly declared coarse synthetic LIF solver; M2 must separately implement/verify the DOOMFLY-comparable reference and native solver before biological claims.
- No learning claims from this visual gate. Fixture constants/config are declared; general training config loading remains M0/M2 work.
- Keep original checkout, development branch astra/visual-first. No agents delegated.

## Known failures and limitations
- Resolved publication restriction: user explicitly approved publishing checked milestones on astra/visual-first and tags to public kosh2shmood/flyholdem. V0 branch and tag pushed successfully. Earlier automatic-review rejections are historical; do not ask again.
- CI M0 tests and browser checks passed, but setup-uv cleanup failed because the background `uv run` server retained a cache lock. Fix: start the installed executable directly and always terminate it with a shell trap. Fixed in the animated-fly commit; CI cleanup now succeeds. The next CI run exposed last-bit floating differences between ARM/macOS and x86/Linux in re-simulated event hashes. Resolution preserves exact within-runtime regeneration, exact recorded playback/hash integrity, and exact categorical/spike behavior with 1e-12 cross-platform diagnostic tolerance. Resolved: corrected M1 CI succeeded.
- Two upstream Starlette test-client deprecation warnings; tests pass.
- Gate 0 passed locally at M1; Gates 1–6/2A pending. No MaleCNS/full runtime, conditioned model, trained teacher, transfer checkpoint, or poker training result yet.
- Atomic native-state checkpoint infrastructure is verified; full experiment/controller/game integration remains a runner milestone; V0 live logs are append-only, fsynced after completed hands, and ignored.

## Last stable commit
V0: 2bd2d3c2fc6158248c4f633690b5d170add14b32 (visual-demo-v0), pushed. M0: fa873a6, pushed. Animated fly: 6aa7895, pushed. M1: c2f2ffe, pushed and CI green. Numerical: 8d6f69c, pushed and CI green (run 34676666552). Side seating: 80a32fd, pushed and CI green. The next visual checkpoint is `feat: clarify community cards and both chip stacks`.

## Resumable experiment command
No long training experiment started. Reproduce visual run with `make demo`; replay with `make replay`. Current local live worker logs into a timestamped ignored runs/live-* directory.

## Next step
Save/push the verified card-readability revision. Then finalize/importer audit tests and the exact runtime config, commit them, and run `make prepare-malecns`. Sources are verified and remain in ignored connectome_data/malecns_v1. Hardware measured: 16 GiB RAM, 77 GiB free before download. Run one full-data/runtime worker at a time; measure real peak memory and throughput before conditioning. No biological or poker learning gate has passed yet.
