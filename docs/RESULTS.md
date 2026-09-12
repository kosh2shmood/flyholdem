# FlyHoldem — current measured results

Status: development, 12 September 2026. The application works with fixture, circuit and full-connectome controllers. Full-graph cue conditioning is confirmed; poker learning remains unvalidated. Conventional teacher candidates through V12 failed the full development suite. The independently checked [V13 self-play candidate](SELF_PLAY_TEACHER.md) is registered before training; no V13 model or result exists yet. This is an interim evidence index, not a completed scientific release.

> A simulated network using the wiring of one reconstructed male fruit-fly central nervous system controls an engineered heads-up no-limit Hold'em interface.

## What you can use

The live/recorded dashboard connects genuine PokerKit hands to actual neural scores, legal actions and activity. The side-seated fly and abstract opponent animate their actions. Large face-up hand and community-card panels show the cards clearly, and chip piles follow each player's balance. Human heads-up play keeps cards private, validates legal actions and restores a match after a browser reload while the server is running. A separate [audited training replay](TRAINING_REPLAY.md) now displays completed native curriculum arms using their actual saved spikes, original teaching labels and seat-correct cards, without executing a neural solver. It is verified with numerical fixtures; no full-graph poker arm exists yet.

The native display represents all 166,700 retained neurons and simulates 25,582,938 retained edges. Anatomical positions and the separate missing-coordinate grid are labeled. The current full model is cue-conditioned and frozen; neither the animation nor its demo chip return demonstrates poker learning. The evidence panel shows independent historical control comparisons.

## Gates and evidence

| Gate | Current result | Meaning |
| --- | --- | --- |
| V0 / Gate 0 | Passed | Real fixture live/replay UI, rules, leakage and registered seat-symmetry checks. |
| Gate 1 | Passed | Circuit and full quarter-strength registrations can select all five action outputs; each passed 20/20 direct confirmation trials. |
| Gate 2 | Passed in full graph | Two-cue local conditioning across five independent seeds, matched controls, retention and erasure. |
| Gate 2A | Pending | Small cue surrogate transfer and teacher removal passed; a qualified full-hand conventional teacher/corpus is still missing. |
| Gates 3–6 | Pending | Shove/fold, river, full 20 BB and 100 BB environments and gated multi-seed runners are implemented. Actual full-graph poker training has not run. |

All cue comparisons below are held-out mean accuracies across five independent seeds. The intervals describe the paired improvement over frozen weights, in percentage points.

| Registered cue experiment | Learned | Frozen | Shuffled control | Improvement over frozen, 95% interval | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| Full graph, local conditioning | 82.34% | 47.19% | 50.00% | +35.16 points [34.22, 36.09] | Passed |
| Circuit, local teacher-advantage transfer | 77.81% | 54.22% | 55.47% | +23.59 points [20.78, 26.25] | Failed the registered 80% fidelity threshold |
| Circuit, existing-edge surrogate transfer | 94.53% | 55.16% | 52.19% | +39.38 points [38.59, 40.47] | Small exact-cue component passed |

Each paired one-sided sign-flip test gave p=0.03125. Retention reproduced the post-training decisions; restoring initial weights reproduced the initial decisions. The local-rule failure is preserved. The surrogate is explicitly nonbiological, uses actual native forward scores, and changes only bounded parameters on registered existing edges. Its backward approximation omits recurrent and threshold derivatives. All 128 tested frozen-model decisions were byte-identical after copied teacher/corpus files were removed.

These tasks concern two engineered cues. The observed conditioning mechanism includes competing-readout suppression and silent legal ties. See [conditioning](CONDITIONING.md), [transfer](TRANSFER.md) and the [model card](MODEL_CARD.md) for limitations and exact registrations.

## Poker teacher results

Full 20 BB candidates V1–V12 failed the unchanged original development suite. Positive returns against an individual opponent do not pass that suite. V9's terminal fold-value correction produced identical recorded paired outcomes to V8. The reserved full confirmation deals remain unused, and no full-hand teacher corpus or poker-trained student exists.

A separate tabular CFR teacher passed its own 10 BB shove/fold confirmation against all four original opponents. It is qualified only for that restricted subgame. It cannot substitute for the full-hand teacher gate. See the [small-game report](SHOVE_FOLD_TEACHER.md).

V10 is a separately registered external-sampling tabular regret response to the stationary uniform original opponent population. Its 30,000-traversal schedule, seeds, abstraction and final-checkpoint rule were fixed before execution. It is a conventional candidate with no self-play equilibrium or fly-learning claim. The first 128 traversals completed in 11.78 seconds at 63.3 MB peak process RSS; the same run then continued. The fixed final run completed 30,000 traversals and 152,391 information sets at 284.5 MB peak RSS. Frozen policy SHA is `762b180986a3983e8a43f4744c6d89cf11299e16b611df6aba2e8b62a4206955`. Development returned +1.843750 / +2.480469 / +0.328125 / −0.013672 BB/hand against random / station / tight-aggressive / equity-bucket. Only random and station had positive adjusted lower bounds, so V10 failed. All 512 paired records reproduced exactly in a subsequent coverage audit. V11 changed only the fixed horizon to 100,000 traversals and started from the same initialization. Its first 30,000 full journal records reproduced V10 byte for byte; final training completed with 306,943 information sets. See the [teacher history](TEACHER.md) and [V10 protocol](EXTERNAL_REGRET_TEACHER.md).

V11's fixed average policy `52bd9fbf298031415bf03f99072c4a144539d0f4f620452ee03f2aaf65f22df7` failed the original 128 paired development deals per opponent: random +2.576172 BB/hand [1.083459, 4.097754], station +3.568359 [2.699695, 4.446313], tight-aggressive +0.314453 [−0.438977, 1.031775], equity-bucket −0.515625 [−1.595703, 0.639673]. These are suite-adjusted intervals; only random and station have positive lower bounds. Independent qualification recomputation and replay of all 512 actual paired records passed, preserving the negative outcome. All 32 information-boundary checks passed.

That replay counted 2,226 teacher decisions: 105 folds, 575 checks, 236 calls, 309 half-pot raises, 800 pot raises and 201 all-ins. The teacher uses all five abstract actions, but this variety does not establish poker strength or change the cue-trained live fly. The separate V12 final-positive-regret policy is now exported, SHA `6b14ca2d9b1dd559fb5de1f8649611a38de488d5eae2f73dc190f97a0367440a`; it also failed the original development suite. The detailed negative result is preserved in [POKER_STRENGTH.md](POKER_STRENGTH.md). Neither candidate can teach.

Full teacher confirmation now requires a verified passing development run for the same frozen policy. Confirmation, recovery and downstream corpus/Gate 2A checks bind and reverify its exact artifacts. The real failed V9 result was rejected before policy inference, confirmation deals or output creation.

## Validation and resources

The latest complete local suite passed 257 tests, with two isolated-oracle skips, in 99.59 seconds. CI separately runs the scalar/Brian2 oracle, data-free import checks, and fixture live/replay/avatar/private-play browser checks. Native desktop/mobile checks also passed locally. The new published control panel was inspected at 1440×1080 and 390×844 with no horizontal overflow; all nine displayed accuracies match the original result files.

| Recorded experiment | Work | Wall time | Peak process RSS |
| --- | --- | ---: | ---: |
| Gate 0 seat/rules check | 2,512 actual hands | 1.69 s | Not recorded in that result |
| Full conditioning confirmation | 9,865 registered operations | 1,004.48 s | 717.6 MB |
| Circuit surrogate confirmation | 9,865 registered operations | 324.24 s | 141.0 MB |

MB here means decimal megabytes. These are particular runs on the current host, not forecasts of later poker training. Large source data, prepared graphs, models and complete logs remain local and ignored. Immutable source/config/binary/environment records and complete-state checkpoints support exact same-runtime recovery.

## Commands

Run these from the repository root. Setup/data installation commands can change environment extras; use them outside active experiments.

```sh
make demo
make test
make fetch-malecns
make prepare-malecns
make audit-malecns
make build-kernel
```

`make demo` starts the labeled fixture at `http://127.0.0.1:8766`. After data preparation and the required registration, the locally exported full conditioning model can be viewed or played against with:

```sh
.venv/bin/flyholdem serve --mode full --model runs/conditioning-full-confirm-v1-model --port 8767
```

The frozen native model is not shipped in Git. Exact experiment command forms, historical recovery instructions and prerequisite requirements are in [COMMANDS.md](COMMANDS.md), [GATES.md](GATES.md) and [POKER_TRAINING.md](POKER_TRAINING.md). Generate a standalone report for a recorded run with:

```sh
.venv/bin/flyholdem report --run runs/exact-surrogate-circuit-confirm-v1
```

Actual native poker training remains guarded by full Gate 2A and preceding curriculum evidence. The missing prerequisite is a scientific result, not an unavailable credential or asset; more training does not guarantee that it will pass.

## Locate the recorded evidence

| Evidence | Local run directory | Result SHA-256 |
| --- | --- | --- |
| Full conditioning | `runs/conditioning-full-confirm-v1` | `8fec88ae335b4940f97c09a3c8e5f6e026b1059dd8444b29f5ce63102409cca5` |
| Failed local cue transfer | `runs/exact-transfer-circuit-confirm-v1` | `5110abe179cbb6dcecb13fc2d6a9527ed287b27808366607faa4c7528b37a0ab` |
| Circuit surrogate transfer | `runs/exact-surrogate-circuit-confirm-v1` | `3a9a11fe172c6aa1fa659d3c0b7e154b9ed94dcbdd44b1e7f6b30d80e0175f4e` |
| Failed V11 full teacher | `runs/teacher-external-regret-v11-development` | `30856a567f37c83b4a89adc2f47622ef5ce5a8756c43fbf6d71e67a458749eb8` |
| Failed V10 full teacher | `runs/teacher-external-regret-v10-development` | `9aa3b092046284baf4f3801f62af956d57e7f394816855675584b73d4e3ca37f` |
| Failed V9 full teacher | `runs/teacher-potential-boundary-v9-development` | `0fd06b781bfc89d5be2e26d310c0839934c95882d025df246a9d01d5e29c3931` |

Each named run contains its `manifest.json` and `result.json`; modern experiment journals have complete hash chains. Per-run reports preserve failures and scope. Artifact consistency checks do not claim external signatures or historical neural-training reexecution. Source, commands, exact hashes and chronological negative findings remain in the tracked `PROGRESS.md`.

Native training and its replay now keep detailed hand records on disk, with streamed audits and bounded retained windows. In a synthetic 64 MiB journal benchmark, indexed write/resume used 322,314 peak traced Python bytes versus 68,535,459 with resident records, with byte-identical files. These storage figures exclude the neural graph/state; see [training replay resources](TRAINING_REPLAY.md).

A future conventional teacher runtime can use [exact PokerKit traversal reuse](POKER_TREE_PERFORMANCE.md). A small prototype workload ran about twice as fast with identical recorded and numeric results; the final helper reproduced all first 128 original training records. V11 completed through its original frozen source.

Twenty focused corpus/regret/qualification checks and a fresh process with no Torch module loaded also passed after correcting tabular corpus backend provenance. These integration fixtures do not constitute a qualified teacher or scientific corpus.


The current spectator model uses folds or all-ins on 99.7% of its audited live decisions. This is an observed limitation, not competent full-hand strategy. See [POKER_STRENGTH.md](POKER_STRENGTH.md) for the descriptive counts and separately registered final-strategy teacher candidate. The visible model has not yet been upgraded.
