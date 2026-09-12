# FlyHoldem progress

## Current phase
M4 local plasticity and the registered conditioning runner are implemented; 52 tests pass and two isolated-oracle tests skip in the app environment. Full and circuit Gate 1 passed, with all full retained edges at the frozen quarter-strength configuration. No conditioning result exists yet. First execute the committed smoke/recovery check, then the ordered development learning-rate candidates, then independent confirmation only after development qualifies. Exact protocol: docs/CONDITIONING.md and configs/conditioning.yaml.

M2 runtime and M3 controllability evidence are complete and published through a895d0e. The public live scene remains the honestly labeled synthetic fixture. The original build, teacher/corpus/transfer, poker optimization and human play interface remain required; the goal is active.

## Latest user steering — implemented
- Three.js fly at the side of the table, fully visible, with curved cards oriented toward itself and the spectator.
- Faceless abstract opponent; both players make actual committed-action hand gestures.
- Larger 3D cards, readable fly/community card insets and clear chip piles for both players. Community slots follow preflop/flop/turn/river; chips vanish at zero balance.
- Python upgrades and other languages/runtimes explicitly permitted when useful. Current app Python 3.11.16, isolated numerical oracle Python 3.12.14, C++17 kernel and Three.js 0.186.0.

## Gate status
- V0 visual gate: passed, committed/pushed/tagged visual-demo-v0.
- Gate 0 rules/software: passed. 2,512-hand fixed software check, independent mean -0.151 BB/hand, SE 0.256678, exact paired-seat residual zero. This is not a learning result.
- M2 engineering: passed (no scientific learning gate implied).
- Gate 1: passed for circuit v1 and full quarter-strength v2; full baseline and half-strength failures preserved.
- Gates 2, 2A, 3, 4, 5, 6: pending. Do not skip dependent scientific gates. Publish negative results if they fail and continue valid independent engineering.

## Latest verification
- 41 tests passed with optional data tools installed; two Brian2 tests intentionally skipped in that environment and both pass separately via make test-oracle.
- Browser: all six fly gestures, opponent check/call, exact live/replay card/event correspondence, opponent privacy, community inset across 0/3/4/5 cards, both balances/chip visibility, pause for both rigs, camera/table controls, desktop 1440×1080 and mobile 390×844 full model bounds, no errors/overflow.
- Latest data/visual CI passed: https://github.com/kosh2shmood/flyholdem/actions/runs/34677926827 (0dc8111); readability CI also passed on 0b05d88.
- Full: 166,700 neurons / 25,582,938 edges / 124,177,617 contacts. Circuit: 4,505 / 1,018,725 / 2,794,887.
- Full graph hash 70c9daefc6185272d0e001700d75aaa8718648c02d9aa59bbadbe6b4cd9e4c06.
- Circuit graph hash d85be4e7fe4f78b2747f859b575a5c48ef4d85f8f7e6eab24dd7e80cbb463f94.
- Independent second normalization and preparation: all reports and every file hash identical.
- Registered sparse-stimulus smoke: full peak RSS 835,305,472 bytes, 100 ms neural stimulus in 0.07118 s; circuit peak 150,847,488 bytes, 0.001183 s. Complete-state continuation exact. These are short resource measurements, not training limits or learning evidence.
- Actual compact reports: docs/RUNTIME.md and docs/review/malecns/. Hardware: 16 GiB RAM; roughly 77 GiB free before data download.

## Exact working commands
```
cd /Users/bohdankoshevoi/flyholdem/flyholdem
make demo
make replay
make record
make test
make test-data
make test-oracle
make browser-check
uv run python scripts/avatar_check.py
make fetch-malecns
make prepare-malecns
make audit-malecns
make build-kernel
uv run python -m flyholdem.experiments.software_gate
```
Demo: http://127.0.0.1:8766. One command serves backend/UI. Browser checks expect the server running. Current live worker uses ignored timestamped runs/live-* logs. No training process is active yet.

## Registered M2 run and reproducibility
Code/config fixed at 04b86b2; execution commit 0dc8111, clean tree. Config configs/runtime.yaml. Every benchmark manifest records config/source/binary/environment hashes and the exact 64 KC stimulus IDs, seed 1729, 50 ms baseline, 100 ms stimulus at drive 12. Official source bytes total 1,109,008,094 and all exact digests match data-provenance/malecns_v1/source.lock.json. Outputs are ignored under connectome_data/malecns_v1 and runs/data-audit. A second generation remains under connectome_data/malecns_v1/reproduction. Run only one full worker until the actual learning workload is measured.

## Engineering decisions and boundaries
- PokerKit is the only rules engine. Neural counts are the fly's sole action-score source. Explicit user prohibition on strategic fallbacks supersedes the spec's check/call-silence clause: silent finite scores use logged deterministic legal argmax; invalid scores stop.
- Fixture LIF is a synthetic 1 ms Euler visual mechanism. Full/circuit use the separately verified analytic 0.1 ms DOOMFLY-comparable solver. Keep these identities distinct.
- Neuron selection and biological parameter tuning may use annotation, connectivity, controllability and conditioning only, never poker profit.
- Card/deck/teacher mutation leakage tests remain byte-exact on a locked runtime. Recorded playback preserves original bytes/hash chains everywhere. Re-simulation is byte-identical within a runtime; ARM/macOS vs x86/Linux floating diagnostics use 1e-12 tolerance with exact categorical/spike results and independently verified chains.
- Source/binary/environment mismatch refuses checkpoint resume. Native atomic generations retain latest three plus best, fsync and verify all arrays. Future runner must additionally checkpoint every encoder/decoder/plasticity/RNG/game variable every five minutes and on shutdown.
- No agents delegated. Keep original checkout and astra/visual-first history; no squash/rewrite.

## Last stable commits and publication
V0 2bd2d3c / visual-demo-v0; M0 fa873a6; first avatar 6aa7895; rules c2f2ffe; numerical 8d6f69c; side seating 80a32fd; card readability 0b05d88; importer 04b86b2; registered M2 execution 0dc8111. All pushed. M2 evidence ab9a658 is pushed. Next checkpoint is feat: register neural controllability experiments.

User explicitly approved publication of checked milestones and tags to public kosh2shmood/flyholdem on astra/visual-first. Earlier automatic-review publication rejections are resolved; do not request approval again. Generated data/checkpoints/large logs/secrets stay ignored.

## Known limitations
No native graph poker controller or biological learning result yet. The full neural rendering, preregistration, conditioning, conventional teacher, corpus/transfer, curricula, integrated long-run recovery and final experiment bundle remain. Two upstream Starlette deprecation warnings are non-failing. Earlier CI uv-cache lock and cross-architecture hash issues are resolved.

## Next step
Commit/push this actual M2 evidence, then implement M3 fixed symbolic population mapping, annotation/connectivity-selected disjoint readout ensembles, candidate/calibration artifacts and the controllability gate. Record black/no-input and shuffled-input controls. Freeze valid preregistration before any poker evaluation. Then proceed to M4 conditioning and teacher work under the registered gates.

## Next registered M3 run
Configuration: configs/controllability.yaml. Circuit command: `uv run --frozen --extra data python -m flyholdem.interface.preregister --mode circuit --output runs/controllability-circuit-v1`. Add `--resume` after graceful interruption, using unchanged code/config/environment. Full-mode follow-up uses `--mode full --output runs/controllability-full-v1` only after inspecting circuit engineering/controls. Each run records exact commit/config/source/graph/encoder/decoder/binary/environment identities before its first trial. No poker information or profit is inspected during this registration. Candidate/selected artifacts, append-only trials, checkpoints and results stay ignored in runs/; publish compact evidence after checking it.

## Active M3 checkpoint
Circuit run runs/controllability-circuit-v1 passed Gate 1 on 529358f: selected gain 12 and 32 control cells, all five actions 20/20 on the predeclared new seeds, shuffled inputs 20% overall and black input zero spikes. All 237 symbolic channels have nonempty fixed projection sets and all readout cells have recorded direct KC paths. Full run runs/controllability-full-v1 is now executing the same committed protocol; no outcome is assumed. Resume command adds --resume to the registered full-mode command. No learning claim follows from controllability.

## M3 v1 completed result and next diagnosis
Circuit Gate 1 passed; full Gate 1 failed all nine registered calibration candidates (225 trials / 312.3 s). See docs/CONTROLLABILITY.md and docs/review/controllability/. Circuit resume preserved trial and mapping bytes and control outcomes. V1 full failure omitted supplementary black/shuffled controls because they were only implemented after calibration success; add separately labeled failure baselines in the follow-up, preserving original logs. Next: pre-register global synaptic gain sensitivity at unchanged retained topology to address recurrent response saturation, using only controllability. Do not proceed to dependent full poker training on a failed Gate 1.

## Expanded user objective
After completing the original specification and its engineering/scientific gates, continue improving the fly's poker strength using stronger teachers, connectome-constrained transfer and other justified methods. Compare improvements on held-out fixed opponents and preserve the information boundary and honest biological/engineering labels. The goal remains active and is not complete at an engineering or visual checkpoint.

## Registered recovery sequence
Before execution, commit configs/controllability_full_{half,quarter,tenth}.yaml and the unchanged-topology scaling implementation. First supplement v1 full failure controls with `uv run --frozen --extra data python -m flyholdem.interface.preregister --mode full --config configs/controllability.yaml --failure-reference runs/controllability-full-v1 --output runs/controllability-full-v1-controls`. Then run the ordered global-scale candidates half/quarter/tenth, stopping only at the first valid full confirmation. Initial command: `uv run --frozen --extra data python -m flyholdem.interface.preregister --mode full --config configs/controllability_full_half.yaml --output runs/controllability-full-half-v2`. Other paths replace half with quarter or tenth. All include manifests, trial checkpoints and --resume support. Four population tests pass, including preservation of every edge/sign under uniform scale.

The expanded goal also includes a play-chip heads-up interface for the user to play against the strongest validated fly after building and improving the policy. Preserve the animated table; enforce human-turn legality and session state, and the fly must not see the human's private cards. This interface is required work, not an optional suggestion.

## M3 complete — current valid full configuration
Full Gate 1 passed at registered global scale 0.25: all five actions 20/20 confirmation; shuffled input 20% overall and black input silent. Input gain 12, controlled-cell count 64. Every retained neuron/edge and transmitter sign remains; base contact gain 0.275 is uniformly scaled to 0.06875. Half-scale failed one action at 17/20 and remains published; the registered sequence stopped before tenth-scale. V1 failure baselines were separately supplied with unchanged historical-manifest links.
Frozen full artifact: runs/controllability-full-quarter-v2/preregistration.json, SHA-256 e3e2b48df9e946d800cec63b5ef88164190c3ffa1cf451b4284a99df5cc26f23. Use load_controller() to validate the base graph and reconstruct the exact scaled initial weights. Circuit v1 remains its separately labeled base-gain configuration. Both passing mappings are frozen; no poker results were used. Next: M4 local eligibility/plasticity, two-cue conditioning and matched controls, plus the independent self-play teacher/corpus pipeline. No conditioning or learned poker claim exists yet.

## Registered M4 execution checkpoint
Local eligibility selects all existing annotated KC-to-MBON edges only; bounded ratios preserve topology/signs. PAM/PPL1 stimulation is explicitly an engineering proxy. Conditioning uses two shared-path cue groups with frozen readouts, matched frozen and shuffled-earned-reward controls, held-out seeds, retention and restoration checks. Hash-chained journals and complete state checkpoints support exact tail re-execution; invalid partial operations cannot overwrite the preceding valid checkpoint. Seven new data-free tests pass; complete suite 52 passed / 2 skipped. PyTorch 2.14.0 is now a locked optional teacher dependency and CPU import was verified; no teacher implementation or learning result is implied.
Before the first long experiment, commit this exact code/config. Smoke: `.venv/bin/python -m flyholdem.experiments.conditioning --profile smoke --learning-rate 0.03 --output runs/conditioning-full-smoke-v1`. Development: `.venv/bin/python -m flyholdem.experiments.conditioning --profile development --learning-rate 0.03 --output runs/conditioning-full-lr003-v1`. Add --resume for the same output after interruption. Registered candidate order 0.03, 0.1, 0.3; independent confirmation is blocked until a candidate meets development criteria. No concurrent full workers.

## M4 smoke/recovery verified — development next
Commit 3093ac0 implements the registered protocol. Full smoke completed: 61,210 existing plastic edges, 17.63 s, peak RSS 722.2 MB, no learning claim. Stop/resume at operation 30 and replay of a logged tail from the best checkpoint both reproduced the uninterrupted 173-operation journal byte-for-byte. Compact evidence: docs/review/conditioning/full-{smoke,recovery}-v1.json. Next run the already registered development candidate 0.03 with three seeds / 400 training trials per control / 32 held-out trials per cue. Command and resume: `.venv/bin/python -m flyholdem.experiments.conditioning --profile development --learning-rate 0.03 --output runs/conditioning-full-lr003-v1` (add --resume after interruption). Record this evidence commit before execution.
