# Conventional teacher and disconnected student runtime

Status: implemented and tested; no validated poker teacher, corpus, transfer result or poker learning claim exists yet. The live table still runs the explicitly labeled fixture.

## Teacher algorithm

The conventional control uses neural fictitious self-play, following [Heinrich and Silver (2016)](https://arxiv.org/abs/1603.01121). Two independent agents maintain Q/target networks, transition replay, and an average policy trained on a uniform reservoir of their own best-response behavior. Each agent samples its best-response/average-policy mixture once per hand. This implementation is an experiment in the project's five-action no-limit abstraction; the paper's results do not establish this implementation's strength or equilibrium convergence.

The initial development config uses 20 BB stacks, 25,000 hands, two 128-unit hidden layers, deterministic CPU PyTorch, a 10% anticipatory mixture, 50,000 replay entries and 200,000 reservoir entries per agent. Exploration decays from 1.0 to 0.1 over 20,000 hands. Q learning receives zero intermediate reward and terminal net chips divided by the initial stack. A transition bootstraps only at the same player's next decision, never on the opponent's private observation. The average policy learns only from that agent's best-response episodes. Frozen teacher targets are the equal probability mixture of both average policies.

The teacher and native fly share the 237-channel representation. The native first-action pulse is derived from the acting player's public relative history, including its first decision after an opponent action. An inconsistent external flag is rejected. Legacy fixture encoding is preserved so the V0 replay remains unchanged.

## Information and validation

Canonical states strictly accept the acting-player observation allowlist. Own hole-card order and the simultaneous flop order are normalized; turn/river order, suits and the complete visible action history are preserved. Hidden fields are rejected. Numeric values, card uniqueness, masks and history are validated. Canonical IDs, rather than deal IDs, determine corpus splits.

A teacher is unavailable for distillation until frozen evaluation passes the complete random/calling-station/tight-aggressive/equity-bucket suite on independent paired deals. Development and confirmation use separate fixed seeds. The initial criterion requires a positive lower bootstrap bound for every opponent, adjusted across the suite. This may fail; training alone never authorizes use as a teacher. Actual private-hole, future-deck and teacher-label mutations must preserve information bytes and target probabilities.

Corpus collection mixes teacher, random and calling-station behavior to cover streets and states. It caps samples per declared stratum: street, position, coarse visible strength proxy, SPR, call price, legal pattern and recent public history shape. The proxy is not hidden equity. Targets depend only on canonical visible states. Repeated states must have identical targets, and train/validation/test IDs cannot overlap. A hash-chained complete-hand collection cache supports exact resume; incomplete hands are reconstructed from their independent seeds.

## Recovery and frozen policies

Teacher checkpoints use numeric arrays and a typed JSON tensor tree, including models, target models, optimizer tensors, replay/reservoir contents and counters, all RNG state, mixture state and the completed-hand cursor. No pickle is used. Checkpoints occur every five minutes and on graceful shutdown at complete hands; exact tail replay verifies previously logged operations. A trained model remains `trained-unvalidated` until the separate held-out checks pass.

Small frozen teacher exports pin inference code, runtime versions, tensors and source training checkpoint. Frozen **fly** exports are separate: they store bounded weights on declared existing graph edges, the fixed population/readout registration, graph/binary identity and inference code identity. They contain no trainable input adapter or decoder. The teacher, corpus, learning and experiment code is absent from the fly's required source scope. Historical training checkpoints still enforce their original whole-source/environment identity; this is a separate frozen-inference contract, not a silent weakening of old checkpoints.

An integration test copies a minimal native runtime, loads a bounded synaptic fixture, records four decisions, deletes the actual teacher package and corpus, then verifies byte-identical decisions. Changing required encoder code is rejected. This proves engineering separation; Gate 2A still requires the actual trained-student transfer evidence.

## Commands

Use the locked environment with `uv sync --frozen --extra data --extra teacher`, then `make build-kernel`. Avoid changing installed extras during a running experiment because its environment identity is strict.

```sh
.venv/bin/python -m flyholdem.teacher.training --config configs/teacher_nfsp.yaml --output runs/teacher-nfsp-v1
.venv/bin/python -m flyholdem.teacher.export --run runs/teacher-nfsp-v1 --output runs/teacher-nfsp-v1-policy
.venv/bin/python -m flyholdem.teacher.evaluation --policy runs/teacher-nfsp-v1-policy --config configs/teacher_evaluation.yaml --profile development --output runs/teacher-nfsp-v1-development
```

Only a qualifying development policy should advance to the separate confirmatory seeds using `--profile confirmatory` and a new output path. A matching passing confirmation is required for:

```sh
.venv/bin/python -m flyholdem.teacher.corpus --policy runs/teacher-nfsp-v1-policy --validation runs/teacher-nfsp-v1-confirmation/result.json --config configs/corpus.yaml --output runs/teacher-nfsp-v1-corpus
```

Training, evaluation and corpus commands support `--resume` with unchanged config/source/environment. Public CLI consolidation and actual local/surrogate transfer runners remain subsequent work. Optional teacher tests skip when PyTorch is absent; the core fixture/native separation tests require no full dataset.
