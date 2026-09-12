# Registered native poker curricula

`curriculum run` now combines complete-hand native learning, independent training seeds, matched controls, frozen exports, isolated evaluation, conventional references and reverified statistical endpoints. This is an implemented experiment runner, not a passed poker-learning gate. Full 20 BB teachers v1–v9 failed; no full Gate 2A certificate, full teaching corpus or poker-trained fly currently exists. The live human opponent remains the cue-conditioned, poker-unvalidated model.

## Prerequisites and fixed choices

The public runner first reverifies the complete Gate 2A certificate. It refuses missing or failed evidence before loading a native player or creating a run. It accepts only the exact checked-in `configs/poker_curricula.yaml` protocol. Each later stage requires a passing confirmation of the preceding stage in the same learning mode. Each confirmation requires its matching passing development run, the same Gate 2A inputs and the same preceding models. Cyclic references, changed protocols and fixture-only evidence are rejected.

The four stages are genuine PokerKit 10 BB shove/fold, controlled check/call-prefix 20 BB river, full-hand 20 BB and full-hand 100 BB. Development has three independent exploration seeds; confirmation has five distinct seeds. All stage/profile training and evaluation deal ranges are disjoint. Original random, station and tight-aggressive opponents train the network; the unchanged equity-bucket opponent is held out. All four original opponents appear in every evaluation.

Stage 3 starts from the original full conditioning-approved registration. Subsequent stages carry each preceding final confirmatory model by fixed ordinal position: first three for development, all five for confirmation. There is no best-profit checkpoint or seed selection. Stages 5 and 6 also include all preceding confirmatory models as frozen native opponents, with only the current seat learning. Their parameters and snapshots are fixed throughout a run.

All biological input populations, projection, readouts, gains, timing, KC-to-MBON eligibility parameters, bounds and the local rate come from the verified conditioning evidence. The explicitly nonbiological surrogate and its rate come from verified exact-transfer evidence. Neither comes from poker profit. Terminal reward is `tanh(net_bb / initial_stack_bb)`, with a baseline using only past hands in the same canonical position. This stack-relative scale and the exploration schedule are fixed before poker development begins.

`bio-plastic` uses terminal local eligibility in every stage. `distilled-connectome` uses the declared direct-readout-rate surrogate in stages 3/4 and terminal-only local fine-tuning in stages 5/6. Its label remains distinct throughout fine-tuning. Stage 3 requires the separately confirmed small tabular teacher; stage 4 uses the full Gate 2A teacher. Online targets are requested only after the actual neural action commits, and only for canonical IDs in the train split. Validation/test IDs receive no target or supervision event. The reproduced offline corpus remains a prerequisite; online trajectories may visit additional canonical train IDs.

## Complete experiment

For each seed, the runner evaluates initial weights, then trains and evaluates plastic, frozen, shuffled-reward or shuffled-teacher, shuffled-encoder and shuffled-connectome arms. It also evaluates the trained model after an unreinforced native retention interval and after restoring the stage's exact starting weights. All phases use identical held-out deals and seats. Frozen training uses the matched exploration schedule without teacher or reinforcement input. Terminal controls permute actual completed plastic-arm returns within canonical position; teacher controls permute target mass within legal actions. Input and topology controls retain their explicit fixed seeds and labels.

Synthetic topology exports have their own prepared graph, preserving source/destination degrees, signs and edge strengths. A model's persisted edges include inherited changes plus its current eligible subset; metadata distinguishes those sets. Inherited changes do not become newly trainable edges merely because they must be serialized.

A conventional reference plays the same held-out curriculum/deals in a separate labeled match. It is replayed against its exact policy probabilities and PokerKit outcomes. This comparison does not create another teacher qualification and is never mixed into fly scores. A 20 BB-qualified reference evaluated at 100 BB remains a reference measurement, not proof of 100 BB teacher strength.

Each native evaluation runs in a minimal copied package and separate process with teacher imports denied. Original project training files, explicit teacher/corpus paths, prior runs and the current parent training run are unavailable, including when source lives in a separate snapshot. The allowed prepared graph, copied model, locked Python environment and evaluation output remain accessible. This is verified Python import/file isolation, not an operating-system sandbox for arbitrary code. No teacher label, teacher action or policy adapter supplies a fly inference score.

Complete native hands are fsynced to hash-chained journals. Training checkpoints preserve weights, membrane/queue state, eligibility, past baselines, RNG and progress at five-minute/shutdown boundaries. Completed training resumes without rewriting its recorded result. Model exports are atomic; the outer phase journal verifies existing completed children and resumes partial children. Only the registered final checkpoint is exported as the trained model. Full private trajectories remain on disk; endpoint aggregation retains compact fields plus whole-record fingerprints to bound parent memory.

## Decision criteria

The primary endpoint is trained-minus-initial BB/100 hands on paired deals/seats/opponents, with a 95% bootstrap across independent training seeds. The same seed analysis compares every control, erased weights, retained improvement and the held-out opponent. All required lower bounds must be positive. Confirmation also requires one-sided exact sign-flip p ≤ 0.05 across seeds. Hands are never treated as independent training replicates.

Retention must reproduce all complete frozen records exactly after the no-learning interval, and erasure must reproduce initial records and exact initial weights. Shove/fold additionally requires monotonic opening-shove frequency over four fixed visible-equity buckets, sufficient observations per bucket and a positive top-minus-bottom span. Equity is a deterministic 256-sample canonical hand-class estimate used only by the scoring process. It never enters native observations. River/full-hand criteria require at least three actions used at ≥1% and no action above 95%.

`curriculum verify` recomputes prerequisites, full schedules, visible information, legal native argmax, encoded hashes, actual fixed-opponent actions and PokerKit settlement. It checks numeric model bytes and final training weights, matched reward arrays, past-only baselines, retention/erasure, source copies and all endpoint statistics. It does not rerun historical native learning, prove an unrecorded biological mechanism or cryptographically attest execution. The ordinary report command checks artifact chains and presents recorded statistics; it does not replace this stronger verifier.

## Commands and recovery

These commands intentionally cannot run until the full teacher/corpus prerequisite exists:

```sh
.venv/bin/flyholdem curriculum run --stage 3 --profile development --learning-mode bio-plastic --gate2a runs/gate2a.json --output runs/poker-bio-stage3-development
.venv/bin/flyholdem curriculum run --stage 3 --profile development --learning-mode distilled-connectome --gate2a runs/gate2a.json --output runs/poker-distilled-stage3-development
.venv/bin/flyholdem curriculum run --stage 3 --profile confirmatory --learning-mode bio-plastic --gate2a runs/gate2a.json --development runs/poker-bio-stage3-development --output runs/poker-bio-stage3-confirmation
.venv/bin/flyholdem curriculum run --stage 4 --profile development --learning-mode bio-plastic --gate2a runs/gate2a.json --previous runs/poker-bio-stage3-confirmation --output runs/poker-bio-stage4-development
.venv/bin/flyholdem curriculum verify --run runs/poker-bio-stage3-confirmation --require-pass
.venv/bin/flyholdem report --run runs/poker-bio-stage3-confirmation
```

Resume with the same arguments and `--resume RUN` in place of `--output RUN`. `--stop-after N` stops at a complete outer phase for recovery checks. SIGINT/SIGTERM stop at the underlying complete-hand boundary. An interrupted phase resumes from its saved child checkpoint; corrupt or truncated evidence is rejected rather than silently erased.

Commit exact source/config before a long run. A source snapshot must include the complete configs, lock and native binary, plus checksum-bound access to the prepared graph and already verified prerequisite artifacts. Keep the original runtime unchanged for recovery. Stop the advancing full spectator and measure resource use before executing a full graph with snapshot opponents. The runner executes phases sequentially and records parent/child RSS peaks separately; these peaks are not a measurement of simultaneous system memory. Data, checkpoints, generated reports and large logs remain local and ignored.

## Engineering verification

Real seven-neuron native fixtures exercise two independent terminal-learning seeds, all controls, isolated evaluations, report generation and interruption after three phases; recovered training/evaluation records are byte-identical. A separate fixture exercises the full distilled phase path and canonical train-only targets. Tests also cover exact conventional-policy reference replay, altered-score rejection after journal rehashing, inherited-edge serialization, seed-level statistics, and prerequisite/confirmation ordering. These synthetic numerical checks cannot qualify a full-connectome curriculum.


## Exact recorded activity for training inspection

Future training journals retain the actual complete readout-window spike vector in a compact, lossless sparse form, captured before action commitment or teaching. Sorted nonzero index/count pairs use fixed little-endian int32 packing, zlib level 6 and base64 inside the existing per-hand hash chain. Original native dtype, neuron count, total spikes and the existing full-vector checksum are preserved. The training manifest records the codec runtime and actual neuron count. A bounded decoder rejects malformed, duplicate, out-of-range, negative, truncated, trailing or checksum-inconsistent data, including excess decompression output. Silent vectors remain exactly silent. No activity is reconstructed from a policy estimate.

The curriculum auditor reconstructs the original native bytes and rederives each training readout rate/score from the registered ensembles, baseline and timing. After auditing, only the two matched-reward scheduling scalars are retained between control arms; bulky replay data stays on disk. Actual frozen hands still match the existing evaluator, and complete training/checkpoint recovery remains exact in both learning modes. A synthetic 166,700-entry sparse vector verifies compact storage without claiming a full-graph experiment.

This supplies exact activity for a subsequent training-replay adapter. It does not yet add a `serve --run` training viewer. Existing spectator/human-play streams are unchanged. Historical frozen source snapshots retain their own format; they are not silently migrated. No full-graph poker learning run has started.


## Recorded training decisions bound to actual poker and action RNG

Added a solver-free recorded-hand audit used by the curriculum verifier. It reconstructs the actual seeded PokerKit hand and river setup, verifies the correct pre-action visible observation and exact encoded hash, restores each lossless native spike vector, reproduces rates/scores, checks the legal mask/silence/source labels, and replays the continuous registered action RNG at the scheduled temperature. It checks every committed action, action count, private replay checkpoint and final payoff. Original scripted opponents are replayed exactly; frozen snapshot opponents bind their own visible-observation hashes, reported legal argmax and snapshot identity. The latter checks reported scores, not unavailable historical opponent spike vectors. Existing source, matched-control, teacher-target and weight-continuity checks remain in the enclosing verifier.

Fifteen added integration cases cover both seats, all four curricula and three teaching/learning methods; continuous RNG state matches the original player after both hands. Native solver and teacher calls are explicitly forbidden during audit. Changed observations are rejected even with recomputed input hashes; changed sampled actions, commitment amounts, temperatures, private decks, missing decisions and rehashed frozen-opponent input records are rejected. All 23 focused replay/curriculum checks passed in 43.96 seconds. Complete regression suite: 208 passed / 2 isolated-oracle skips in 80.85 seconds. Frozen inference identity remains 1b2c99883afc6e6e30b65e8cf9aafef456953177c10caa443a8d62d531da09ca. This is record-consistency verification, not independent historical neural execution or a scientific learning claim.

V11 continues the fixed 100,000-traversal schedule from immutable code 62bf21f / source d16984f8442b6fb8fec62f962ff4992d74dc3bedbca69ee7da017249a60ed48a. Last inspected 3,600 traversals. The exact --resume command above remains valid. V10 failed the original full suite; full confirmation deals remain unused, Gate 2A and Gates 3–6 remain pending. The remaining training visualization task is a viewer adapter that consumes these audited records, preserves both player seats/private-card boundaries and labels historical teaching methods honestly; no training-run viewer is claimed yet.


Completed native curriculum arms can now be viewed with `serve --training-run <experiment>/<seed>/<arm>/training --graph <exact-prepared-graph> --start-hand 0 --hands 16 --port 8768`. The service audits the complete record, replays actual saved activity and preserves card privacy without a neural worker or live player. See [TRAINING_REPLAY.md](TRAINING_REPLAY.md) for source requirements, labels, resource limits and checked fixture examples. This does not change any scientific gate.
