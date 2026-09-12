# Poker strength: current limitation and next candidates

The visible full-connectome fly is a frozen two-cue conditioning model, not a trained poker policy. Its action choice is dominated by folding and moving all-in. Improving its poker strategy now takes priority over further dashboard polish.

## Measured live behavior

A read-only audit of the current live spectator log verified 30,472 complete hash-linked events, containing 7,606 settled hands and 7,630 fly decisions against the calling station. The sampled prefix ended at journal head `8f2fe43afa14014c381179378c7e3ae04eb1e2bae273eedf231db52f99358455`. The frozen model manifest SHA is `9e0f94abdd1cce92fc93c16dbbe9597d99456b01c3d09076784d2e51f8cf0259`.

| Action | Decisions |
|---|---:|
| Fold | 3,804 |
| Check / call | 0 |
| Half-pot raise | 23 |
| Pot raise | 0 |
| All-in | 3,803 |

Folds and all-ins account for 99.7% of decisions. There were no silent decisions: this behavior came from the cue-trained neural scores. All 23 flop decisions were all-ins; the remaining decisions were preflop. These are descriptive spectator counts against one opponent, not a paired evaluation or evidence of poker learning. The private human interface uses the same frozen model but has a separate match history.

## What must improve

The five-action interface already supports checking, calling and two smaller raise sizes. The strategy must learn when to use them. The existing registered river and full-hand criteria require at least three actions at 1% or more and no single action above 95%, together with profitable held-out performance relative to matched controls. Action variety alone cannot establish competent play.

The visible model will be replaced only by an actually trained, verified native model. A teacher or hand-written rule will not substitute for the fly's neural scores. Teacher qualification, corpus verification, native distillation, disconnected inference and the later poker/control gates remain required. Both learning modes keep their scientific labels.

## Fixed training and separate final-strategy candidate

V11 is completing the previously registered 100,000-traversal population-response run. Its original frozen source, sampled deals, opponents and historical-average export remain unchanged. Only the fixed final checkpoint may be exported; the original development suite must then run and independently verify.

V12 is a separately registered extraction from that same completed final numeric state. At each information set it normalizes the positive final accumulated regrets over legal actions. If every legal regret is nonpositive, it uses the existing legal uniform prior. The rule does not choose a checkpoint, temperature, opponent, action override or mixture from poker results. Unseen states keep the declared uniform prior. Its manifest explicitly distinguishes it from V11's time-averaged strategy.

The [MCCFR paper](https://www.cs.cmu.edu/~kwaugh/publications/nips09b.pdf) defines regret matching from positive cumulative regrets and explains the role of averaging. We use the fixed final regret-matching strategy as an empirical candidate. The average-strategy guarantees do not establish that this final strategy is better, converged or an equilibrium. The hypothesis is that removing the early training average may improve finite-run play; it can also make it worse.

Operational order: finish V11 and record its original average-policy development outcome first. If that fails, evaluate V12 on the same original development suite. If V11 passes, proceed with its existing confirmation and native-training path. Do not spend the reserved full confirmation on V12 without its own passing, independently verified development result. No full confirmation has yet been used.

## Extraction and provenance

`configs/teacher_final_regret_v12.yaml` pins the exact V11 training manifest, source, configuration identity and final 100,000 iterations. Export checks the complete journal chain, every population/deal RNG draw, total nodes/leaves, final checkpoint arrays, restored sampling RNG and actual information-set count. A partial run is rejected before policy output is created. Historical training is read from its pinned artifacts; it is never resumed or migrated through the new implementation.

The exported numeric policy includes the exact final regret tensor and its original checkpoint checksum. Loading verifies source/runtime identities, every tensor checksum, complete legal information keys and exact recomputation of every probability from those regrets. Rehashing a changed probability file cannot substitute another strategy. Frozen inference needs neither the training run nor PyTorch.

Freeze the checked new source/configs and existing equity binary into a separate `runs/teacher-runtime-v12` before using these commands. Do not modify `runs/teacher-runtime-v11` or the earlier downstream snapshot.

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v12/src" .venv/bin/python -m flyholdem.cli teacher export-final-regret --config runs/teacher-runtime-v12/configs/teacher_final_regret_v12.yaml --output runs/teacher-final-regret-v12-policy
PYTHONPATH="$PWD/runs/teacher-runtime-v12/src" .venv/bin/python -m flyholdem.cli teacher evaluate --policy runs/teacher-final-regret-v12-policy --config runs/teacher-runtime-v12/configs/teacher_evaluation.yaml --profile development --output runs/teacher-final-regret-v12-development
PYTHONPATH="$PWD/runs/teacher-runtime-v12/src" .venv/bin/python -m flyholdem.cli teacher verify-evaluation --run runs/teacher-final-regret-v12-development --policy runs/teacher-final-regret-v12-policy --allow-development
```

Publish the final policy identity before development. Resume evaluation replaces `--output` with `--resume` for the same directory. Confirmation retains the existing required `--development-reference`. Generated logs, tensors and model checkpoints remain local and ignored.

At this registration milestone, V12 has no exported model, poker result, corpus or teacher qualification. The visible fly remains cue-trained. Full Gate 2A and Gates 3–6 remain pending.


Engineering verification: 26 focused checks passed in 8.66 seconds; the complete suite passed 248 tests with two isolated-oracle skips in 94.52 seconds. The actual active V11 run was refused before output creation. These checks establish extraction integrity, not strategic strength.


V12 conventional runtime is now frozen and preflighted at runs/teacher-runtime-v12. Code commit 0a1a6c7cc9d7a602796db4c789875c8356853687; source SHA 38b9e758a439054a701f22917a2bc5516b0d84f885d517eab8a5a813d7dfea0f; file ledger 13e329b06e516bd64ae63126cafc0484c198e09cda997c868e440d91c7c9b17c. Its complete file ledger and original evaluation YAML 5fb344fad4c27a5a5556764db0d0c43b1f64177ccadafc68b3d81453f52ca9c1 verified. The separate existing equity binary is 80846bbc3357982b6310ef05c3c8f9921621c0caa9db1d0c17a1c54cc43207ea; it was copied, not rebuilt. A fresh process from this exact runtime passed real tiny numeric extraction/evaluation/qualification tests in 2.06 seconds, with Torch absent before and after. It also rejected active V11 before policy output creation. No full model or held-out trial was produced by these preflight checks.
