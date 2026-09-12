# Downstream execution snapshots

Current path: V13 self-play is training, with its separately checked downstream runtime recorded in the final section. V11 and V12 failed development; their earlier conditional sections are retained as history and do not authorize teaching.

`runs/poker-runtime-v11` prepares later corpus, Gate 2A verification and native curriculum work without changing V11's active training. It combines orchestration from commit `52016bfde9efae1611a6969a961f02d3d98f5da5` with the seven exact policy implementation files from V11's original runtime. Their source checks are preserved, not bypassed. The snapshot is local and ignored; its `SOURCE.json` records every copied source/config/binary hash and each explicitly linked prerequisite.

The original training snapshot stays at `runs/teacher-runtime-v11`, source `d16984f8442b6fb8fec62f962ff4992d74dc3bedbca69ee7da017249a60ed48a`. Use that original runtime for V11 resume, fixed-final export and the already registered development/confirmation evaluation. Do not substitute the downstream snapshot into its training command.

## Execution identity

| Item | SHA-256 |
| --- | --- |
| Composed source | `217828684efa5b9bf54dd4bf6bfd58d193f6e1813722156c8915a580b886e0ab` |
| Original policy implementation | `857a8bedcea999e2ca6be6b41d2183d2030c44fdec838f71846735d8acb237ff` |
| Copied file ledger | `ca10a3e67d931b4ff05a6078464de0968d869a7fd07b2812a44b327387c661fa` |
| Original evaluation YAML | `5fb344fad4c27a5a5556764db0d0c43b1f64177ccadafc68b3d81453f52ca9c1` |
| Corpus YAML | `5e2c85c9b588bae55fa2f7bd50651fa7000d041288df1a18a84e86464e265241` |
| Native curriculum YAML | `3c4d239ab4e7384e288080366618a3f4ef39668ba5e53cbd07ed8f151b23142a` |
| Existing native binary copy | `44f5ce277154c953b213f41105b0d7f61e40af0cdd8d685289609c10b06be897` |
| Existing equity binary copy | `80846bbc3357982b6310ef05c3c8f9921621c0caa9db1d0c17a1c54cc43207ea` |

The snapshot links ten named prerequisite directories: both prepared graphs, both controllability registrations, the full conditioning result, local-transfer failure, surrogate-transfer result and exported model, and the restricted shove/fold policy and confirmation. These are hash-checked local links, not duplicated datasets or operating-system write protection. Normal gate verification repeats the required artifact checks.

## Preflight evidence

The actual frozen V10 policy loaded through both original and composed runtimes with identical implementation identity. Thirty-two decisions on training-deal check/call trajectories produced the same probability fingerprint, `8e580b8d10f4173d6b1d75e863056776832e8d6e7fd7d7ee80e70160c257cda2`. Neither process loaded Torch. V10 remains failed and unqualified; this is source compatibility evidence only.

Both prepared graphs passed every prepared-file and graph-identity check: circuit 4,505 neurons / 1,018,725 edges; full 166,700 / 25,582,938. The frozen native inference identity still matches `1b2c99883afc6e6e30b65e8cf9aafef456953177c10caa443a8d62d531da09ca`. The copied native build verified without recompilation.

Complete cue evidence reverified the existing full conditioning pass, surrogate-transfer pass and preserved local-transfer failure. The separate 10 BB teacher loaded and its original confirmation plus 32 hidden-information checks passed again. Its qualification remains limited to that subgame. An actual two-hand numerical training replay produced nine viewer events with native-worker construction forbidden. Attempting Gate 2A with V10's development result was rejected before certificate creation. No native training worker, new held-out evaluation, full corpus or Gate 2A certificate was created.

## Conditional next commands

These commands require a genuinely passing V11 confirmation, which does not exist at this checkpoint:

```sh
PYTHONPATH="$PWD/runs/poker-runtime-v11/src" .venv/bin/python -m flyholdem.cli teacher export-corpus \
  --policy runs/teacher-external-regret-v11-policy \
  --validation runs/teacher-external-regret-v11-confirmatory/result.json \
  --config runs/poker-runtime-v11/configs/corpus.yaml \
  --output runs/teacher-external-regret-v11-corpus

PYTHONPATH="$PWD/runs/poker-runtime-v11/src" .venv/bin/python -m flyholdem.cli teacher verify-qualified-corpus \
  --policy runs/teacher-external-regret-v11-policy \
  --validation-run runs/teacher-external-regret-v11-confirmatory \
  --corpus runs/teacher-external-regret-v11-corpus

PYTHONPATH="$PWD/runs/poker-runtime-v11/src" .venv/bin/python -m flyholdem.cli gate certify-transfer \
  --policy runs/teacher-external-regret-v11-policy \
  --confirmation runs/teacher-external-regret-v11-confirmatory \
  --corpus runs/teacher-external-regret-v11-corpus --output runs/gate2a-v11.json
```

Corpus resume replaces `--output RUN` with `--resume RUN`. Before any actual native curriculum, verify the certificate, stop the advancing full spectator, record the experiment identity and measure resource use at a complete phase boundary. Use the same downstream `PYTHONPATH` with the existing [curriculum commands](POKER_TRAINING.md), its frozen config and `runs/gate2a-v11.json`. The first stage still requires the full teacher/transfer gate; the small-game reference alone does not unlock it.


## Conditional V12 downstream runtime

`runs/poker-runtime-v12` now contains the exact reviewed V12 source, configurations and equity binary, plus a separately verified copy of the existing native binary and the same ten explicitly hash-checked prerequisite links. It makes no policy-module substitutions. The original reviewed conventional runtime remains the path for V12 extraction/evaluation; this downstream copy is for corpus, Gate 2A and later native orchestration if V12 qualifies.

| Item | SHA-256 |
| --- | --- |
| Checked code commit | `a97506f6dbad5bd3ea33c5439a3b59dcbc41c199` |
| Source, identical to reviewed V12 | `f70a256a49f9e80274290f1835cb1f9f20d84abb9e573dceff95f4fdb3dd373a` |
| Final-regret policy implementation | `f6aa18842893bfceb4783ebbac827697cf957c18caf6a230e231491c0d9f1d98` |
| Copied file ledger | `b1350b32a5b5ccf42f6f5141306e99b309a695223e9ea3f08a7ba606f9d75708` |
| Original evaluation YAML | `5fb344fad4c27a5a5556764db0d0c43b1f64177ccadafc68b3d81453f52ca9c1` |
| Corpus YAML | `5e2c85c9b588bae55fa2f7bd50651fa7000d041288df1a18a84e86464e265241` |
| Native curriculum YAML | `3c4d239ab4e7384e288080366618a3f4ef39668ba5e53cbd07ed8f151b23142a` |
| Existing native binary copy | `44f5ce277154c953b213f41105b0d7f61e40af0cdd8d685289609c10b06be897` |

Preflight passed in 3.25 seconds. It compared every copied reviewed file, verified the complete ledger, linked prerequisite checksums and both prepared graphs, and rechecked the existing cue and restricted-teacher evidence. It exercised a tiny final-regret policy's actual corpus export and replay without Torch: 83 rows from 17 collection hands, train/validation/test splits 64/12/7, and street counts 38/25/11/9. Targets and complete trajectories reproduced, and completed resume left the files unchanged. Temporary fixture qualification substitutions were confined to that engineering check and restored before exercising the real rejection path. Actual failed V10 development was refused before policy loading, native work or certificate creation. These checks are engineering integration evidence, not a full teacher qualification, scientific corpus or native poker-learning result.

The following commands are conditional on V12 independently passing its original development suite and its subsequent reserved confirmation. V11's original average-policy evaluation still comes first; this runtime does not change candidate order.

```sh
PYTHONPATH="$PWD/runs/poker-runtime-v12/src" .venv/bin/python -m flyholdem.cli teacher export-corpus \
  --policy runs/teacher-final-regret-v12-policy \
  --validation runs/teacher-final-regret-v12-confirmatory/result.json \
  --config runs/poker-runtime-v12/configs/corpus.yaml \
  --output runs/teacher-final-regret-v12-corpus

PYTHONPATH="$PWD/runs/poker-runtime-v12/src" .venv/bin/python -m flyholdem.cli teacher verify-qualified-corpus \
  --policy runs/teacher-final-regret-v12-policy \
  --validation-run runs/teacher-final-regret-v12-confirmatory \
  --corpus runs/teacher-final-regret-v12-corpus

PYTHONPATH="$PWD/runs/poker-runtime-v12/src" .venv/bin/python -m flyholdem.cli gate certify-transfer \
  --policy runs/teacher-final-regret-v12-policy \
  --confirmation runs/teacher-final-regret-v12-confirmatory \
  --corpus runs/teacher-final-regret-v12-corpus --output runs/gate2a-v12.json
```

Use this same downstream source, frozen curriculum registration and `runs/gate2a-v12.json` for dependent native work only after the certificate verifies. Stop the advancing full spectator before starting any full native worker, record exact execution identities and measure the first phase's resource use. No native learning rate, encoder, readout or biological mapping parameter changes follow from teacher poker results.


## Conditional V13 self-play downstream runtime

`runs/poker-runtime-v13` is now prepared and preflighted from the exact immutable V13 source, with no policy-module overrides. It includes separate copies of the existing equity/native binaries and the same ten individually pinned prerequisite links. Original V13 training, fixed-final export and qualification continue through `runs/teacher-runtime-v13`. This downstream copy grants no teaching or native-training permission.

| Item | SHA-256 |
| --- | --- |
| Implementation commit | `107fe26ea26a527648f2ac9cec84a6bf6e9f5db2` |
| Exact V13 source | `46a10cd749bfbc0c17c1de7189ed5ec0df8a99ee231199c1969f53be02ed82e5` |
| Self-play policy implementation | `69b486631ae346f29758b864f4d7211d0814483d4f9ba605eb598764f346f232` |
| Complete copied file ledger | `cd101db732e3550b6fe835dde10596d6838066a44bc9b591483ab97f42c750c2` |
| Downstream SOURCE record | `4a8363016f3bc26ef6b807460e6f74f9d8103e9ab4a4774795fa029e6681b771` |
| Corpus configuration | `5e2c85c9b588bae55fa2f7bd50651fa7000d041288df1a18a84e86464e265241` |
| Native curriculum configuration | `3c4d239ab4e7384e288080366618a3f4ef39668ba5e53cbd07ed8f151b23142a` |
| Existing native binary | `44f5ce277154c953b213f41105b0d7f61e40af0cdd8d685289609c10b06be897` |

The preflight actually trained a four-iteration/eight-traversal engineering self-play fixture, interrupted/resumed it, exported its final numeric average and verified both roles' original checkpoint-array bindings. Completed resume was unchanged, partial export was refused, and all 32 information-boundary probes passed. Actual collection and journal replay reproduced 32 prescribed engineering hands and 161 canonical target rows: train/validation/test 120/24/17, streets 72/49/25/15, positions 82/79. Targets matched the core's role averages; canonical splits were disjoint. These temporary artifacts were explicitly unqualified and removed after verification.

Both original and downstream runtimes loaded the real V12 final-regret policy and identically recomputed its existing failed 512-record development summary. This was not another actual-play replay or new held-out trial. The unchanged real corpus-export, qualified-corpus-verification and Gate 2A functions all refused V12 before output creation. No qualification substitutions were used. The positive guarded corpus export/resume path remains conditional on actual V13 confirmation and was not exercised by this preflight; the existing integration suite separately covers its engineering behavior.

Every copied file and all ten previously pinned prerequisite sets were unchanged after preflight. Native inference identity remains `1b2c99883afc6e6e30b65e8cf9aafef456953177c10caa443a8d62d531da09ca`. No native worker, teacher-removal experiment, new scientific evaluation or certificate was created; no environment or binary was rebuilt. This bounded check did not repeat every full graph payload checksum or cue statistical recomputation. Actual Gate 2A must repeat its complete prerequisite verification and teacher removal when a full teacher exists.

Only after the exact final V13 policy independently passes original development and the reserved confirmation may these commands run:

```sh
PYTHONPATH="$PWD/runs/poker-runtime-v13/src" .venv/bin/python -m flyholdem.cli teacher export-corpus \
  --policy runs/teacher-self-play-regret-v13-policy \
  --validation runs/teacher-self-play-regret-v13-confirmatory/result.json \
  --config runs/poker-runtime-v13/configs/corpus.yaml \
  --output runs/teacher-self-play-regret-v13-corpus

PYTHONPATH="$PWD/runs/poker-runtime-v13/src" .venv/bin/python -m flyholdem.cli teacher verify-qualified-corpus \
  --policy runs/teacher-self-play-regret-v13-policy \
  --validation-run runs/teacher-self-play-regret-v13-confirmatory \
  --corpus runs/teacher-self-play-regret-v13-corpus

PYTHONPATH="$PWD/runs/poker-runtime-v13/src" .venv/bin/python -m flyholdem.cli gate certify-transfer \
  --policy runs/teacher-self-play-regret-v13-policy \
  --confirmation runs/teacher-self-play-regret-v13-confirmatory \
  --corpus runs/teacher-self-play-regret-v13-corpus --output runs/gate2a-v13.json
```

Corpus resume replaces --output with --resume for the same run. Full native curriculum work requires a verified certificate, frozen cue-selected parameters, stopped advancing full spectator, recorded execution identity and a measured native resource boundary. V13 training is still in progress; no full teacher, qualified corpus or new native poker model exists. Full Gate 2A and Gates 3–6 remain pending.
