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

The checked source/configs and copied existing equity binary are ready in `runs/teacher-runtime-v12-reviewed`; its exact identities and successful preflight are recorded below. Do not modify `runs/teacher-runtime-v11` or the earlier downstream snapshot.

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v12-reviewed/src" .venv/bin/python -m flyholdem.cli teacher export-final-regret --config runs/teacher-runtime-v12-reviewed/configs/teacher_final_regret_v12.yaml --output runs/teacher-final-regret-v12-policy
PYTHONPATH="$PWD/runs/teacher-runtime-v12-reviewed/src" .venv/bin/python -m flyholdem.cli teacher evaluate --policy runs/teacher-final-regret-v12-policy --config runs/teacher-runtime-v12-reviewed/configs/teacher_evaluation.yaml --profile development --output runs/teacher-final-regret-v12-development
PYTHONPATH="$PWD/runs/teacher-runtime-v12-reviewed/src" .venv/bin/python -m flyholdem.cli teacher verify-evaluation --run runs/teacher-final-regret-v12-development --policy runs/teacher-final-regret-v12-policy --allow-development
```

Publish the final policy identity before development. Resume evaluation replaces `--output` with `--resume` for the same directory. Confirmation retains the existing required `--development-reference`. Generated logs, tensors and model checkpoints remain local and ignored.

At this registration milestone, V12 has no exported model, poker result, corpus or teacher qualification. The visible fly remains cue-trained. Full Gate 2A and Gates 3–6 remain pending.


Engineering verification: 26 focused checks passed in 8.66 seconds; the complete suite passed 248 tests with two isolated-oracle skips in 94.52 seconds. The actual active V11 run was refused before output creation. These checks establish extraction integrity, not strategic strength.


V12 conventional runtime is now frozen and preflighted at runs/teacher-runtime-v12. Code commit 0a1a6c7cc9d7a602796db4c789875c8356853687; source SHA 38b9e758a439054a701f22917a2bc5516b0d84f885d517eab8a5a813d7dfea0f; file ledger 13e329b06e516bd64ae63126cafc0484c198e09cda997c868e440d91c7c9b17c. Its complete file ledger and original evaluation YAML 5fb344fad4c27a5a5556764db0d0c43b1f64177ccadafc68b3d81453f52ca9c1 verified. The separate existing equity binary is 80846bbc3357982b6310ef05c3c8f9921621c0caa9db1d0c17a1c54cc43207ea; it was copied, not rebuilt. A fresh process from this exact runtime passed real tiny numeric extraction/evaluation/qualification tests in 2.06 seconds, with Torch absent before and after. It also rejected active V11 before policy output creation. No full model or held-out trial was produced by these preflight checks.


## Independent review and actual-play reproduction

Independent review found that the first V12 loader bound its regret values to the final checkpoint but did not separately bind the key mapping and legal masks. Swapping two valid information keys with the same legal mask, then rehashing the exported key file, could reassign the probabilities to different states. Export and loading now preserve and verify the original checkpoint hashes for all four defining arrays: key offsets, key bytes, legal masks and regrets. The original regret checksum remains checked. A regression uses an actual tiny trained table to reject the key swap; missing/partial checkpoint bindings are also rejected.

The earlier `runs/teacher-runtime-v12` snapshot is retained as historical preflight evidence. Do not use it for actual policy exports. Use the now-frozen reviewed runtime named in the commands above. The method, fixed final iteration and extraction configuration are unchanged. Neither V11's original runtime nor any existing native model was altered.

`scripts/audit_teacher_play.py` independently reruns every complete recorded pair with the loaded policy, actual PokerKit rules, original opponents and exact evaluation RNGs. It requires every payoff and five-action count to reproduce, then separates check from call and reports actions by street and opponent. The CLI accepts the original registered suite only. This is a read-only replay of already evaluated deals, not an additional trial or qualification gate. Its report includes the audit script hash, full decision-trace hash and original evaluation identities.

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v11/src" .venv/bin/python scripts/audit_teacher_play.py --run runs/teacher-external-regret-v11-development --policy runs/teacher-external-regret-v11-policy --output runs/teacher-external-regret-v11-play-audit.json
```

Run that only after V11's original evaluation finishes. For V12 use its reviewed runtime and corresponding completed evaluation/policy paths. Existing policy/runtime compatibility checks remain enforced.

The audit was exercised on V10's unchanged failed development result: all 512 paired records and 1,948 decisions reproduced. Actual actions were 89 folds, 423 checks, 109 calls, 244 half-pot raises, 820 pot raises and 263 all-ins. The decision trace is `b442acaa3248115e6aec5db3efcbfee9c53a807c565d734d8716472b4fc9bac7`; audit script SHA is `e272b0e67668f7fa1387fdbca4ba2296f45560d50fbe0d4e7850e90514bbcef3`. V10 remains unqualified: action variety does not establish profitable play against every opponent, and these conventional actions do not describe the visible native fly.

Eleven focused extraction tests passed in 4.88 seconds. Seven new audit tests passed in 6.46 seconds. Expected action counts come from instrumenting committed PokerKit payments in the original evaluator, covering both numeric policy kinds, all streets and both seats. Coherently rehashed false payoffs/action counts pass the older offline statistical consistency check but fail actual replay. Artifact/source immutability, no-Torch numeric execution and the public registration boundary are also checked.

Complete integration verification for the checkpoint-binding fix and actual-play audit: 257 tests passed, with two isolated-oracle skips and two existing dependency deprecations, in 99.59 seconds. No browser-facing code, neural policy parameters, active runtime or environment changed.


## Reviewed execution runtime ready

The reviewed V12 runtime is frozen at `runs/teacher-runtime-v12-reviewed` from checked commit `a97506f6dbad5bd3ea33c5439a3b59dcbc41c199`. Source SHA `f70a256a49f9e80274290f1835cb1f9f20d84abb9e573dceff95f4fdb3dd373a`; file ledger `8392816361bf644c0a3469349283ca7e906cf8972aa7ccd9c7bdeeb561c542b2`; policy implementation `f6aa18842893bfceb4783ebbac827697cf957c18caf6a230e231491c0d9f1d98`. Extraction configuration remains `a3bd05a204bb8603ffd3e0e5946e3ff27bc2214619c97c2e4832e593c68d356a` and the original evaluation configuration remains `5fb344fad4c27a5a5556764db0d0c43b1f64177ccadafc68b3d81453f52ca9c1`. The copied existing equity binary remains `80846bbc3357982b6310ef05c3c8f9921621c0caa9db1d0c17a1c54cc43207ea`. All frozen files verified. Four actual tiny extraction/evaluation/replay checks passed in 7.11 seconds through this exact runtime, including key-rebinding refusal; Torch was absent before and after, and original V11 source was unchanged. No full model or new held-out trial was produced by preflight. Use this reviewed runtime for conditional V12 execution; retain the older snapshot only as history.

The checkpoint-binding/play-audit milestone passed CI [34714885981](https://github.com/kosh2shmood/flyholdem/actions/runs/34714885981), including the fixture, oracle and browser jobs. The complete local suite remains 257 passed / 2 isolated-oracle skips in 99.59 seconds. These checks establish engineering integrity, not poker strength.


## V11 fixed final average exported before development

Original V11 completed all 100,000 traversals, 13,858,172 counterfactual nodes and 5,275,792 terminal branches, producing 306,943 information sets. Its resumed invocation took 11270.364090 seconds; peak RSS was 545,734,656 bytes. The policy contains 74,592,670 disk bytes. These are training/resource measurements, not held-out poker results. The original immutable runtime's complete final-table export and an independent repeated provenance/checkpoint/array/RNG audit passed; recorded node and terminal totals also matched the complete journal. No Torch loaded.

The frozen original average policy SHA is `52bd9fbf298031415bf03f99072c4a144539d0f4f620452ee03f2aaf65f22df7`. Training result `4852e3905060ea000ec6663b5486c6b6236680ae2008454bc781cec82c2dde6f`; manifest `1276bb4b27357c9b33a9ed08ae3e048e6ac0ec896d3a82f6c110b783a99b88a3`; complete journal `8969685cc3337ed22a62b128ed2975d40f3eae5f4d9959c1267614742a6c47e9`; journal head `fb7911e24214ad47b4eb911d9c4f89526a03d07a6e59c6d26716b36b1f5ecb6e`. Original training source remains `d16984f8442b6fb8fec62f962ff4992d74dc3bedbca69ee7da017249a60ed48a`. Only the registered final iteration was exported, and its identity is recorded before development. No V12 policy has been exported.

After publishing this identity, run the original development suite:

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v11/src" .venv/bin/python -m flyholdem.cli teacher evaluate --policy runs/teacher-external-regret-v11-policy --config runs/teacher-runtime-v11/configs/teacher_evaluation.yaml --profile development --output runs/teacher-external-regret-v11-development
PYTHONPATH="$PWD/runs/teacher-runtime-v11/src" .venv/bin/python -m flyholdem.cli teacher verify-evaluation --run runs/teacher-external-regret-v11-development --policy runs/teacher-external-regret-v11-policy --allow-development
PYTHONPATH="$PWD/runs/teacher-runtime-v11/src" .venv/bin/python scripts/audit_teacher_play.py --run runs/teacher-external-regret-v11-development --policy runs/teacher-external-regret-v11-policy --output runs/teacher-external-regret-v11-play-audit.json
```

Evaluation resume replaces `--output` with `--resume` for that same directory. Original confirmation is still unused and requires independently verified passing development. If V11 development fails, preserve its result and proceed with the reviewed V12 fixed-final candidate; otherwise use the existing V11 confirmation/downstream path. No full teacher qualification, full corpus, Gate 2A certificate or native poker model exists yet. Gates 3–6 remain pending, and the full goal stays active.


## V11 failed; reviewed V12 final policy frozen before development

The original V11 average policy completed all 128 paired deals per original opponent and failed the unchanged full development suite:

| Opponent | BB/hand | Suite-adjusted interval |
| --- | ---: | --- |
| random | +2.576172 | [1.083459, 4.097754] |
| calling-station | +3.568359 | [2.699695, 4.446313] |
| tight-aggressive | +0.314453 | [-0.438977, 1.031775] |
| equity-bucket | -0.515625 | [-1.595703, 0.639673] |

Only random and calling station have positive lower confidence bounds. Complete-journal and statistical recomputation passed, as did all 32 hidden-information probes. Result SHA `30856a567f37c83b4a89adc2f47622ef5ce5a8756c43fbf6d71e67a458749eb8`; evaluation manifest `16efe2b48dd6503c6e3c066fe77d85242643552c74a72f5924707b0ccb411245`; paired journal `8ad199317e7cb8b2b61fabffb260984f8dabbcbdbc3dd7d612b1071fd605ea3e`. V11 remains barred from teaching and did not use full confirmation.

Actual PokerKit replay reproduced all 512 paired records and 2,226 decisions: 105 folds, 575 checks, 236 calls, 309 half-pot raises, 800 pot raises and 201 all-ins. The decision trace is `7680a956e2fb23a089a505a4be9781477acd161a94a435f2e46f5d615a43e1b4`. This is real action variety in a conventional candidate, not a stronger native fly or sufficient poker qualification.

Following the already registered conditional order, reviewed V12 was extracted from that same completed 100,000-step final checkpoint. Its frozen policy SHA is `6b14ca2d9b1dd559fb5de1f8649611a38de488d5eae2f73dc190f97a0367440a`; 306,943 information sets; 86,871,895 disk bytes. All original checkpoint keys, legal masks and regret bytes matched their pinned hashes. A fresh load exactly rederived every action distribution and reverified the complete historical checkpoint/provenance/RNG; no Torch loaded. Final checkpoint metadata SHA `bced70b7cbb4e0b750b5216aa15a960491dbd9d2802c51eca4513371fc2c8725`; regret file SHA `773be6a96db31f28ac1858d14ad5e92003aa96dfe78c1d34d7d20eb1af2d1250`; extraction YAML `a3bd05a204bb8603ffd3e0e5946e3ff27bc2214619c97c2e4832e593c68d356a`; implementation `f6aa18842893bfceb4783ebbac827697cf957c18caf6a230e231491c0d9f1d98`. Source remains reviewed `f70a256a49f9e80274290f1835cb1f9f20d84abb9e573dceff95f4fdb3dd373a` from a97506f. This identity is recorded before V12 development; no policy temperature, checkpoint or action override was selected from the V11 outcome.

After this identity is published, run:

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v12-reviewed/src" .venv/bin/python -m flyholdem.cli teacher evaluate --policy runs/teacher-final-regret-v12-policy --config runs/teacher-runtime-v12-reviewed/configs/teacher_evaluation.yaml --profile development --output runs/teacher-final-regret-v12-development
PYTHONPATH="$PWD/runs/teacher-runtime-v12-reviewed/src" .venv/bin/python -m flyholdem.cli teacher verify-evaluation --run runs/teacher-final-regret-v12-development --policy runs/teacher-final-regret-v12-policy --allow-development
PYTHONPATH="$PWD/runs/teacher-runtime-v12-reviewed/src" .venv/bin/python runs/teacher-runtime-v12-reviewed/scripts/audit_teacher_play.py --run runs/teacher-final-regret-v12-development --policy runs/teacher-final-regret-v12-policy --output runs/teacher-final-regret-v12-play-audit.json
```

Evaluation resume replaces `--output` with `--resume` for the same directory. Full confirmation remains unused and requires a passing independently verified development result for this exact V12 policy. No qualified full teacher/corpus, full Gate 2A certificate or native poker model exists. The visible fly remains cue-conditioned; Gates 3–6 and the full goal remain pending.
