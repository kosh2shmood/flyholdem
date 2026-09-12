# FlyHoldem

A play-money-only, inspectable poker experiment. **Current build: fixture and native-connectome dashboards; full-graph conditioning passed, poker learning remains unvalidated.**

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
- Gate 1: circuit and full quarter-strength controllability passed; earlier failures remain documented.
- Gate 2: full-graph conditioning passed five independent seeds, retention and erasure. This demonstrates engineered cue-dependent readout suppression, not poker skill.
- Gate 2A: pending full-hand teacher/corpus qualification. Local exact-cue transfer failed at 77.8%; the subsequent bounded-edge surrogate confirmed at 94.5%, with actual teacher-removal inference verified. A separate tabular 10 BB shove/fold teacher passed its own confirmation. Full 20 BB candidates v1–v8 failed; V9 tests the exact terminal fold value in the conventional teacher.
- Human heads-up play: implemented and browser-verified with private cards and legal actions. Poker learning Gates 3–6 remain pending; no strategic fly-learning claim.
- Native dashboard: all 166,700 retained neurons rendered, annotated positions distinguished from missing-coordinate grid, actual frozen neural scores drive the table. Full data remain local and checksum-verified.
- Current local suite: 143 passed / 2 isolated-oracle skips; fixture/live/replay/avatar, native desktop/mobile and private human-play browser checks pass.

See [PROGRESS.md](PROGRESS.md), [visual gate report](docs/VISUAL_GATE.md), [protocol](docs/PROTOCOL.md), [model card](docs/MODEL_CARD.md), and [browser evidence](docs/review/browser-check.json). Live logs are written to ignored, timestamped `runs/live-*/events.jsonl` directories, flushed at every completed hand. Hash chains detect modifications; seeds regenerate the loop exactly on the same locked runtime. Cross-architecture numerical comparison uses absolute 1e-12 tolerance for floating diagnostics with exact cards/actions/spike counts; recorded playback is byte-exact everywhere. V0 does not yet resume a running neural state from a checkpoint.

## Scope of the intended full experiment

“A simulated network using the wiring of one reconstructed male fruit-fly central nervous system controls an engineered heads-up no-limit Hold'em interface.”

That wording describes the native full-data controller. The default fixture remains a separately labeled synthetic prototype. Do not call this a living fly, consciousness, faithful whole-brain emulation, or proven learner. The remaining implementation follows `flyholdem_build_spec.md`; scientific gates cannot be replaced by a visually compelling demonstration.

Original code is MIT licensed. Data and dependencies retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md). No real-money service, account, or execution integration is included.

## Official data preparation

After the fixture demo, `make fetch-malecns` explicitly downloads and verifies the three pinned source files. `make prepare-malecns` installs the locked data extra, accounts for all retained/excluded objects and contacts, then compiles full and circuit CSR arrays. `make audit-malecns` verifies all hashes/counts and runs the registered sequential resource smoke checks. Data and generated reports remain ignored under `connectome_data/` and `runs/data-audit/`. `make test-data` exercises import/preparation with tiny synthetic data; it never downloads MaleCNS.

## Native full/circuit dashboard

After checksum-verified data preparation and a passing controllability registration, use the existing locked data environment:

```sh
.venv/bin/flyholdem serve --mode full --port 8767
.venv/bin/flyholdem serve --mode circuit --port 8767
```

These commands freeze the original registered weights. To inspect an exported frozen model, add `--model runs/conditioning-full-confirm-v1-model` (the locally exported full conditioning model) or another matching native model directory. The model is not shipped in Git. This is explicitly unvalidated poker evaluation; it delivers no reinforcement and changes no weights. The default fixture still runs with `make demo` on port 8766.

The native point cloud uses annotated soma positions for 139,662 neurons; the remaining 27,038 occupy a separately labeled schematic grid. Every retained neuron is represented; the 25,582,938 edges are simulated but are not all drawn. Filter input/readout/dopamine or recent activity, orbit/zoom, and click a cell to inspect its annotation. Source switching retains correct fixture/native labels and geometry. The evidence panel reports separate experiments, not a learning claim about the current table.

```sh
.venv/bin/python scripts/native_browser_check.py --url http://127.0.0.1:8767 --output runs/native-browser-check
```

Native replay uses `--mode full --replay <complete-native-log.jsonl>` (or circuit); it checks the recorded graph and population registration before displaying activity. Native live streams are recorded in ignored timestamped runs/live-*/events.jsonl directories. For experiment commands, checkpoints, results and explicit failures, see docs/CONDITIONING.md, docs/TRANSFER.md, docs/TEACHER.md and PROGRESS.md. Environment installation commands may change installed extras; do not run setup/sync while an experiment is active.


## Play a private heads-up match

Open either dashboard and choose **Play against fly**. Your hand stays readable beside the community-card strip, the fly's cards remain private until shown at showdown, and both rigs animate their actual actions. Buttons reflect server-validated legal wagers. Deal next hand alternates the button and resets both stacks to 20 BB; reload keeps the current match while the server is running. Select Live or Recorded replay to leave the match. The current native model has confirmed cue learning but no validated poker strategy. Details and browser-check commands are in docs/UI.md.


## Unified research commands and reports

Run `.venv/bin/flyholdem --help` for registered conditioning, exact-cue distillation, controllability, conventional teacher training/export/evaluation and guarded corpus commands. Completed workflows generate local Markdown, standalone HTML and JSON reports. To inspect existing evidence without reexecuting a model, use `.venv/bin/flyholdem report --run runs/exact-surrogate-circuit-confirm-v1`. The reporter rejects inconsistent/corrupt modern hash chains and explicitly marks the limited verification of older logs. See [command reference](docs/COMMANDS.md) for exact commands, resume conventions and current scope. General poker curriculum runners remain pending; these commands do not imply their gates passed.
