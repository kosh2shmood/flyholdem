# Exact-cue transfer integrity

Status: circuit local transfer confirmation failed its registered fidelity threshold. Development and actual teacher-removal checks passed; no complete Gate 2A pass. This is one component of Gate 2A; the conventional poker teacher must independently pass its opponent suite, and corpus checks must pass before the complete gate can pass. No poker transfer or strategic learning is claimed.

The exact two-state teacher gives one-hot probabilities for two declared actions. Its only input is the visible cue token. During training, twice the difference between the selected action probability and the uniform-legal mean supplies the signed teaching signal. The already registered local eligibility rule subtracts a past-only signal baseline and changes only existing KC-to-MBON edges within 0.1–2.0 of their original signed strengths. The teacher is queried after the neural action, only in training. Evaluation scores come from the native circuit alone.

The registered circuit uses its original passing controllability mapping, 32 cells per cue chosen from existing shared paths, 15% amplitude jitter and rate 0.3 carried from conditioning. No poker performance selected these parameters. Matched arms use learning, frozen weights and a permutation of the learned arm's actual teaching signals. Held-out accuracy must reach 0.8 mean / 0.7 minimum seed, improve by at least 0.15 against frozen/shuffled/restored weights and retain within 0.05. Confirmation requires paired one-sided sign-flip p ≤ 0.05 over five new seeds. All criteria and seed lists are in configs/exact_transfer.yaml before execution.

The sensory cue driver is separately pinned alongside the frozen native runtime. It accepts cell stimuli, not teacher targets. Export reads actual best post-training checkpoint weights and validates unchanged nonplastic edges, bounds and signs. The removal check copies the actual teacher package, reproduces the run's held-out decisions, deletes teacher and target files, and requires byte-identical frozen decisions with no teacher/learning/experiment imports. A changed cue driver is rejected. Exact-cue checkpoint selection is explicit and is not poker seed selection.

```sh
.venv/bin/python -m flyholdem.experiments.exact_transfer --profile smoke --learning-rate 0.3 --output runs/exact-transfer-circuit-smoke-v1
.venv/bin/python -m flyholdem.experiments.exact_transfer --profile development --learning-rate 0.3 --output runs/exact-transfer-circuit-dev-v1
```

Both commands support --resume with unchanged code/config/environment. Confirmation requires a passing matching development result and --development-reference. The full conditioning confirmation is checked by checksum before any transfer run. Checkpoints retain latest three plus best, occur at least every five minutes and on shutdown, and support exact logged-tail replay.

```sh
.venv/bin/python -m flyholdem.experiments.export_exact --run runs/exact-transfer-circuit-dev-v1 --output runs/exact-transfer-circuit-dev-v1-model
.venv/bin/python scripts/exact_inference_check.py --run runs/exact-transfer-circuit-dev-v1 --model runs/exact-transfer-circuit-dev-v1-model --output runs/exact-transfer-circuit-dev-v1/inference-audit.json
```

A failed local transfer gate is preserved and blocks a local-transfer success claim. It permits the next explicitly nonbiological, bounded-edge surrogate method under a separately registered protocol; it never permits a policy bypass.

## Development result and actual inference removal check

Execution 1a232a5: development completed in 173.24 s / 134.1 MB RSS. Learned accuracy was 0.84375, 0.84375, 0.78125 over the three registered seeds (mean 0.8229167). Frozen/restored mean was 0.5520833; shuffled-signal mean 0.5416667. Retention decisions were unchanged. The registered development criteria passed; this does not substitute for the reserved confirmation seeds.

The 40-trial smoke completed in 6.31 s / 139.9 MB. Stop/resume at operation 30 reproduced its 173-operation journal byte for byte (SHA-256 2dca8a3e4889d3b94475d9cb2e7fa33b2dc542fddd9f83f80916385d662c16a3). The actual exported development model has SHA-256 1c4d40c30ea186577c9933aacd22e99c6d3188ec512c8c641234fb906cff5704. All 64 held-out decisions exactly reproduced the actual training run, remained byte-identical after deleting teacher/corpus files, and loaded no teacher, learning or experiment modules. Decision bytes SHA-256: 27b096091d9199766c43b9b0f7a76591638341bc4f631c8eda54d641502b1ee5. A changed cue driver was rejected. Generated evidence stays local in ignored runs/.

Freeze this source in ignored runs/exact-runtime-v1 (including the lockfile/config and references to immutable data/native binary). Confirmation uses the same source and selected 0.3 rate with five reserved seeds:

```sh
PYTHONPATH="$PWD/runs/exact-runtime-v1/src" .venv/bin/python -m flyholdem.experiments.exact_transfer --config runs/exact-runtime-v1/configs/exact_transfer.yaml --profile confirmatory --learning-rate 0.3 --development-reference runs/exact-transfer-circuit-dev-v1/result.json --output runs/exact-transfer-circuit-confirm-v1
```

Add --resume after interruption. Export/audit must use that same frozen runtime. Independent checkout implementation may continue; the snapshot, native binary, source data and installed environment must remain unchanged.

## Independent confirmation — failed fidelity threshold

The five reserved seeds finished in 341.67 s / 139.2 MB. Learned accuracies were 0.796875, 0.8046875, 0.765625, 0.7265625 and 0.796875: mean 0.778125, below the registered 0.8 requirement. Frozen/restored mean was 0.5421875 and shuffled mean 0.5546875. Both improvements had positive paired-seed bootstrap bounds and one-sided sign-flip p=0.03125; retention and exact erasure passed. These positive mechanism checks do not override the failed threshold. Confirmation is recorded as **fail**; no confirmatory model is exported and the full transfer gate stays pending. Generated result: runs/exact-transfer-circuit-confirm-v1/result.json, manifest SHA-256 973b95cbc5943cca6366dabb78652978bb4472dff5f1d9f5a971ff29055482b5.

The next permitted transfer method is a separately registered, explicitly nonbiological surrogate optimization restricted to bounded parameters on existing preregistered edges. It must retain the native simulation as the only inference score source and preserve this negative result.

## Registered surrogate follow-up

The next method is explicitly nonbiological and inspired by the general surrogate-gradient approach reviewed by [Neftci, Mostafa and Zenke (2019)](https://arxiv.org/abs/1901.09948). It is a project-specific direct-readout rate approximation, not an implementation of complete recurrent backpropagation. Actual native LIF scores are used in the forward loss and are always the sole action scores. The approximate backward Jacobian of a multiplier on an existing KC-to-readout edge is its original signed weight times the measured filtered presynaptic count, multiplied by tau_syn/tau_pre, divided by the fixed threshold/rest distance and readout ensemble size. Derivatives through recurrence, upstream activity and hard spike thresholds are omitted and explicitly reported. No trainable adapter, output decoder, new edge or alternative inference score is added.

Projected SGD minimizes legal teacher cross-entropy using that declared Jacobian, with temperature 1 and gradient norm cap 1. All non-direct or illegal-readout derivatives are zero; original multiplicative bounds remain 0.1–2.0. It delivers no dopamine pulse. This preserves the same native circuit/cues/readouts and can only qualify if actual native held-out decisions pass the gate. A finite-difference test verifies the derivative of the declared linearization; that test does not establish its fidelity to the full recurrent dynamics.

Config configs/exact_transfer_surrogate.yaml pins the failed local confirmation, unchanged sensory registration, and an ordered rate sequence 0.1, 1.0, 10.0. Development seeds 75101–75103 and confirmation seeds 75201–75205 are new. The shuffled control permutes exact teacher cue labels, preserving their full multiset; it is labeled shuffled-teacher. The original 80% accuracy, 70% minimum seed, 15-point control improvement, retention and paired-significance criteria remain unchanged. No poker performance selects any parameter.

```sh
.venv/bin/python -m flyholdem.experiments.exact_transfer --config configs/exact_transfer_surrogate.yaml --profile smoke --learning-rate 0.1 --output runs/exact-surrogate-circuit-smoke-v1
.venv/bin/python -m flyholdem.experiments.exact_transfer --config configs/exact_transfer_surrogate.yaml --profile development --learning-rate 0.1 --output runs/exact-surrogate-circuit-lr01-v1
```

Add --resume after interruption. If development fails, run the next already registered rate in a separate output directory, stopping at the first qualifying rate before independent confirmation. The optimizer is stateless projected SGD; complete neural state, local activity observer, schedule and journal position retain the existing atomic recovery contract. Actual model export and teacher-removal checks remain required. No surrogate experiment result exists at registration.

## Surrogate development qualified at the first registered rate

Rate 0.1 completed in 148.20 s. Held-out learned accuracy was 0.859375, 0.9375 and 0.96875 across the three development seeds (mean 0.921875), versus 0.53125 frozen/restored and 0.546875 shuffled-teacher. Retention and exact erasure passed. The first candidate qualifies; rates 1.0 and 10.0 are not run. Preserve the original local-rule confirmation failure. This is still an exact-cue circuit result, not poker skill.

Freeze 0.1 and run the five reserved confirmation seeds using the unchanged surrogate-runtime-v1 snapshot:

```sh
PYTHONPATH="$PWD/runs/surrogate-runtime-v1/src" .venv/bin/python -m flyholdem.experiments.exact_transfer --config runs/surrogate-runtime-v1/configs/exact_transfer_surrogate.yaml --profile confirmatory --learning-rate 0.1 --development-reference runs/exact-surrogate-circuit-lr01-v1/result.json --output runs/exact-surrogate-circuit-confirm-v1
```

Add --resume after interruption. Export and actual teacher-removal checks use the same snapshot. Confirmation has not yet run at this checkpoint; Gate 2A remains pending independent teacher/corpus qualification as well.


## Surrogate independent confirmation and actual teacher removal

The first registered surrogate rate, 0.1, passed all five reserved confirmation seeds in 324.24 s / 134.4 MiB peak RSS. Learned accuracies were 0.9453125, 0.9453125, 0.96875, 0.921875 and 0.9453125: mean **0.9453125**. Frozen/restored mean was 0.5515625; shuffled-teacher mean was 0.521875. Paired improvements were 0.39375 (95% seed bootstrap interval 0.3859375–0.4046875) and 0.4234375 (0.3890625–0.459375). Each one-sided paired sign-flip p was 0.03125. Retention exactly reproduced post-training decisions; weight restoration exactly reproduced initial decisions. The original local-rule failure remains unchanged.

An independent audit recomputed all 9,865 journal hashes, actual legal argmax choices, per-seed accuracy, retention/restoration and shuffled-target multisets. Every training record used actual native forward scores, no dopamine, and bounded ratios. Cue 0 accuracy was 100%, with 49.6875% silent legal readouts; cue 1 accuracy was 89.0625%, with 10.3125% silent readouts. Thus the result still partly uses the fixed silent-tie rule, while cue 1 also produces active winning output. It does not establish arbitrary five-action poker control or faithful biological learning.

The actual confirmed model exported from the best exact-cue checkpoint (seed 75203) has SHA-256 c3876173e83c739738706ee4bd2e0a28c7854b934e63fc67ffbaba90b9e6defb. All **128** actual held-out decisions reproduced after deleting the teacher and corpus, with no teacher/learning/experiment imports. Decision bytes SHA-256: 6fb3340bc2982622e3d2d458530143f9c1b950a5d30e246e807e0ad88815d099. Mutating the separately pinned cue driver was rejected. Manifest SHA-256: f33ef9ae87f3646256fac72693144cb0b76c63f6d1760cc2cb7a18a3573ee1e7; frozen source hash: 0ba41f497c48a2ffad355f97433691664845b4449dc7edd8236844c08f90b988.

Generated evidence/model remain local in runs/exact-surrogate-circuit-confirm-v1 and its sibling -model directory. The small exact-transfer and actual teacher-removal components now pass. **Whole Gate 2A remains pending** independent conventional poker-teacher validation and disjoint corpus validation. No poker-learning claim follows.
