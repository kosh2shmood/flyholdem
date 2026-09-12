# FlyHoldem

A play-money-only, inspectable poker experiment. **Current release: fixture visual prototype.**

## Run the real fixture demo

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 22+ (syntax checks), and Git. `uv` installs project-local dependencies and selects Python 3.11; it does not replace the system Python.

```bash
make demo
```

Open **http://127.0.0.1:8766**. A real two-player PokerKit game autoplays continuously. The default 3D fly camera shows an articulated fly holding its actual cards face up, tapping to check, pushing chips for calls/raises and sweeping both hands forward for all-in. Use Table view for the overhead layout. The fly's scores come from a synthetic LIF circuit. The opponent is a labeled calling station. Watch cards, pot, stacks, action trail, exact encoded inputs, legal masks, neural activity, and terminal reinforcement. Pause freezes the display; the live process continues. The replay control uses the checked-in compact log. The speed control applies to recorded replay.

**FIXTURE / VISUAL PROTOTYPE / NOT A FULL-CONNECTOME RESULT.** The fixture contains 126 synthetic cells and 1,920 existing synthetic edges. Its layout is schematic. Weight changes and wins are not evidence of learning.

```bash
make test                  # data-free rules, leakage, neural, replay, WebSocket tests
make replay                # backend streams the checked-in recorded truth
make record                # regenerate six deterministic fixture hands
uv run python -m playwright install chromium
make browser-check         # while make demo is running; docs/review/ screenshots
uv run python scripts/avatar_check.py  # six actual action gestures + WebGL screenshots/video
```

Locked dependencies live in `uv.lock`. The UI uses plain browser APIs plus pinned Three.js 0.186.0 (MIT) from `ui/package-lock.json`. `make demo` installs it with `npm ci` and serves it locally; the dashboard makes no third-party asset requests. No separate frontend build is needed; one backend serves UI and WebSocket.

## Status and evidence

- V0: a working live/replay fixture vertical slice, with automated browser evidence.
- Software Gate 0: passed locally at M1; 29 tests and the 2,512-hand registered symmetry check. Machine-readable evidence is in `docs/review/software-gate0.json`.
- Gates 1–6 and teacher Gate 2A: pending. No conditioning, teacher transfer, or poker learning claim.
- No MaleCNS data has been downloaded for V0. The full-graph runtime and both experiment modes are subsequent milestones.

See [PROGRESS.md](PROGRESS.md), [visual gate report](docs/VISUAL_GATE.md), [protocol](docs/PROTOCOL.md), [model card](docs/MODEL_CARD.md), and [browser evidence](docs/review/browser-check.json). Live logs are written to ignored, timestamped `runs/live-*/events.jsonl` directories, flushed at every completed hand. Hash chains detect modifications; seeds regenerate the loop exactly on the same locked runtime. Cross-architecture numerical comparison uses absolute 1e-12 tolerance for floating diagnostics with exact cards/actions/spike counts; recorded playback is byte-exact everywhere. V0 does not yet resume a running neural state from a checkpoint.

## Scope of the intended full experiment

“A simulated network using the wiring of one reconstructed male fruit-fly central nervous system controls an engineered heads-up no-limit Hold'em interface.”

That wording describes the intended full-data system, not this synthetic fixture. Do not call this a living fly, consciousness, faithful whole-brain emulation, or proven learner. The remaining implementation follows `flyholdem_build_spec.md`; scientific gates cannot be replaced by a visually compelling demonstration.

Original code is MIT licensed. Data and dependencies retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md). No real-money service, account, or execution integration is included.
