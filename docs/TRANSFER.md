# Exact-cue transfer integrity

Status: implementation verified, circuit experiment pending. This is one component of Gate 2A; the conventional poker teacher must independently pass its opponent suite, and corpus checks must pass before the complete gate can pass. No poker transfer or strategic learning is claimed.

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
