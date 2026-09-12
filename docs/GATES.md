# Evidence prerequisites

Gate 2A is implemented as an evidence check, not a manually set pass flag. `configs/gate2a.yaml` pins the existing full conditioning result, failed local cue transfer, subsequent circuit surrogate transfer and actual exported model. It also pins the original full 20 BB teacher evaluation YAML. No certificate exists yet because every completed full teacher candidate has failed.

```sh
.venv/bin/flyholdem gate certify-transfer --policy POLICY --confirmation CONFIRMATION_RUN --corpus CORPUS --output runs/gate2a-certificate.json
.venv/bin/flyholdem gate verify-transfer --certificate runs/gate2a-certificate.json
```

Certification first requires the entire confirmed full teacher suite and reproduces the qualified canonical corpus's complete PokerKit collection. It then rederives the registered cue operation schedules, native-score argmax choices, correctness labels, independent-seed summaries, bootstrap intervals, sign-flip tests, retention and erasure. It checks the failed-local-before-surrogate sequence and runs actual frozen inference before and after deleting copied teaching artifacts. Verification of an existing certificate repeats these dependencies; a stored flag or changed checksum cannot replace them. This is reproducible local artifact consistency, not a cryptographic signature or a rerun of historical native training.

The separate confirmed 10 BB tabular teacher cannot substitute for the full 20 BB teacher. A Gate 2A certificate would authorize subsequent registered experiments, not assert that a fly learned poker. Gates 3–6 still require their own curriculum, matched-control and held-out evidence. The gated multi-seed poker orchestrator is implemented; its public entry point reverifies this certificate and preceding-stage evidence before native poker training. See POKER_TRAINING.md.

## Actual checks

Full conditioning and circuit surrogate transfer each had all 9,865 operations independently rechecked: mean held-out accuracies remain 82.34375% and 94.53125%, respectively. The failed local transfer's 9,865 operations also recomputed exactly at 77.8125%, still below its 80% threshold. Five CI checks use actual tiny native decisions in explicitly frozen evidence and reject rehashed changes to summary statistics, stimuli, argmax choices and evaluation learning events. They grant no scientific pass.

The actual surrogate model c3876173e83c739738706ee4bd2e0a28c7854b934e63fc67ffbaba90b9e6defb reproduced its selected checkpoint's 128 recorded post-training decisions. After deleting copied teacher source and cue corpus, every decision byte remained identical: SHA 6fb3340bc2982622e3d2d458530143f9c1b950a5d30e246e807e0ad88815d099. Neither subprocess imported teacher, learning or experiment modules. The isolated package's path was checked, preventing fallback to the project training package. The chosen checkpoint seed, 75203, came from the original cue validation; it was not selected using poker outcomes. Generated audit results remain local.

```sh
.venv/bin/flyholdem gate verify-cues --run runs/conditioning-full-confirm-v1 --config configs/conditioning.yaml
.venv/bin/flyholdem gate verify-cues --run runs/exact-surrogate-circuit-confirm-v1 --config configs/exact_transfer_surrogate.yaml
.venv/bin/flyholdem gate verify-cues --run runs/exact-transfer-circuit-confirm-v1 --config configs/exact_transfer.yaml --allow-failed
.venv/bin/flyholdem gate verify-removal --model runs/exact-surrogate-circuit-confirm-v1-model --graph connectome_data/malecns_v1/prepared-circuit --transfer-run runs/exact-surrogate-circuit-confirm-v1
```

The actual failed V6 teacher was rejected by both the certification function and public CLI before any biological model or nonexistent corpus was loaded. No rejected certificate file was created.


Full teacher qualification now additionally requires a confirmation manifest bound to the same policy's passed development run. The result, manifest, complete paired journal and registered configuration are reverified; missing or changed development evidence prevents both confirmation execution and downstream Gate 2A qualification. No full confirmation result predates this guard.
