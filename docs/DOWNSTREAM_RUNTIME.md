# V11 downstream execution snapshot

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
