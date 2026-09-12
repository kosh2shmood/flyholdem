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
