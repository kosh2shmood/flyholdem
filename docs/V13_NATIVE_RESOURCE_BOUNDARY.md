# V13 first native resource boundary

This resource assessment uses existing measured records and frozen source, supplemented by the actual numerical fixture signal check below. No full-native worker, poker training or qualification procedure ran for this assessment. All full-native commands below remain conditional on an independently qualified final V13 teacher, reproduced corpus and verified `runs/gate2a-v13.json`. Use the unchanged `runs/poker-runtime-v13` snapshot/source `46a10cd749bfbc0c17c1de7189ed5ec0df8a99ee231199c1969f53be02ed82e5`; stop the advancing full spectator first.

## Existing measurements, not forecasts

| Local record | Observed work and elapsed time | Recorded memory |
| --- | --- | --- |
| `runs/data-audit/full-benchmark.json` | 166,700 neurons / 25,582,938 edges; 100 ms sparse stimulus took 0.071177 s; exact state continuation | Peak RSS 835,305,472 bytes; mutable state 120,835,540 bytes |
| `runs/conditioning-full-smoke-v1/result.json` | 173 full cue operations, including 40 training trials: 17.631 s | Peak RSS 722,190,336 bytes |
| `runs/conditioning-full-confirm-v1/result.json` | 9,865 full cue operations: 1,004.483 s | Peak RSS 717,553,664 bytes |
| `runs/native-full-frozen-baseline-v1/result.json` | 256 actual frozen 20 BB hands / 220 neural decisions: 17.322 s | RSS absent; report explicitly has null peak RSS |

The poker baseline uses the same native binary hash `44f5ce277154c953b213f41105b0d7f61e40af0cdd8d685289609c10b06be897` and the conditioning-approved full registration, but a different model, curriculum and historical orchestration source. Its 0.067663 s/hand implies **69.287 s for 1,024 hands only by naive linear scaling**. This is a rough size comparison, not an end-to-end Gate 3 estimate: prerequisite verification, reference replay, graph loading, isolated-process startup, export/audit/checkpoint I/O and different neural activity must be measured. Cue and sparse-stimulus rates do not establish learning-hand throughput.

## Actual registered first workload

Frozen `configs/poker_curricula.yaml` and `src/flyholdem/experiments/curriculum_protocol.py` prescribe Gate 3 `shove-fold-10bb-v1`: three development seeds 81001/81002/81003, 10,000 training hands per arm, paired training starts 50000000/50100000/50200000, and evaluation deals 150000000–150000127 against all four original opponents with both seats.

`--stop-after 1` counts **outer phases**, not hands. It first reverifies Gate 2A (including actual corpus and teacher-removal dependencies), then runs and reproduces the separately confirmed small-teacher reference: 1,024 recorded conventional hands plus their 1,024-hand replay. The first native phase is seed 81001 **initial evaluation**, 128 pairs × four opponents × two seats = **1,024 native hands**, with no poker weight training. The shove/fold subgame gives at most one neural decision per hand; all 512 button-seat hands require a decision, and the nonbutton seat acts if the opponent shoves, so there are 512–1,024 native decisions. Each uses the fixed 50/300/100/50 ms baseline/stimulus/readout/commit schedule: 0.5 simulated seconds per decision.

The next phase is **plastic training plus its evaluation**: 10,000 training hands, then another 1,024 isolated evaluation hands. Each training decision is observed in 5 ms eligibility bins (100 bins per full decision). Bio-plastic adds its fixed 200 ms terminal dopamine pulse per hand; the separate distilled mode uses its registered surrogate and small teacher. These learning, recording and checkpoint costs are not measured by the frozen poker baseline. Do not estimate them simply as 10,000 × frozen-hand time.

For scale, a complete development run in **one** mode has eight evaluation phases per seed (initial, plastic, retention, erased, frozen, reward/teacher shuffle, encoder shuffle, topology shuffle): 24,576 native evaluation hands and 150,000 training hands across five arms × three seeds, plus reference/prerequisite/audit work. Both learning modes remain separate; this is not a one-phase job.

The parent keeps a full player, graph and initial weight copy while the isolated evaluation child loads another full graph. Only the child advances then, but **simultaneous parent+child memory is unmeasured**. The three old single-process RSS peaks do not bound it. Frozen native checkpoints also preserve roughly the recorded 121 MB mutable state per generation, before learning traces, logs and exports; measure actual disk growth rather than treating model export size as run storage.

## Exact conditional probe and recovery

From `/Users/bohdankoshevoi/flyholdem/flyholdem`, after verifying prerequisites and stopping the advancing full spectator, record the fixed execution identities and launch only one mode/process tree. The following uses the original first bio-plastic run path; a separately scheduled distilled run uses `--learning-mode distilled-connectome` and `runs/poker-distilled-stage3-development`.

```sh
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/poker-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli curriculum run --config runs/poker-runtime-v13/configs/poker_curricula.yaml --stage 3 --profile development --learning-mode bio-plastic --gate2a runs/gate2a-v13.json --output runs/poker-bio-stage3-development --stop-after 1
```

Observe the actual curriculum Python PID and its isolated child; do not signal the timer/shell or an assumed PID. Once child progress and complete `hands.jsonl` records exist, send **one SIGINT to that verified Python parent**, wait for both processes to finish/reap, and inspect persisted status/checkpoints before resuming. Sample parent+child RSS together and system memory pressure during the overlap; save per-invocation wall time/RSS and disk usage in projectless scratch. The interrupted outer result omits the complete-run resource fields, so external measurement is necessary. Do not change registered deal counts, config or methods to make a smoke run.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/poker-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli curriculum run --config runs/poker-runtime-v13/configs/poker_curricula.yaml --stage 3 --profile development --learning-mode bio-plastic --gate2a runs/gate2a-v13.json --resume runs/poker-bio-stage3-development --stop-after 1
```

Inspect `runs/poker-bio-stage3-development/result.json`, `phases.jsonl`, and `81001/initial/evaluation/{isolation-result.json,evaluation/result.json,evaluation/hands.jsonl,evaluation/checkpoints/latest.json}`. A mid-phase stop has outer status interrupted / zero completed phases and a child complete-hand checkpoint; a last-hand race may instead complete the initial phase. `--stop-after 1` bounds that race to the initial phase. Resume preserves the same child/model/config, finishes/re-audits its complete 1,024 hands, then stops after phase one. It does not start plastic training.

After that measured boundary, the first learning-resource probe is the **same run** resumed with `--stop-after 2`; this phase bound includes all 10,000 plastic hands and their evaluation. To measure a short training prefix without changing registration, send a single observed parent SIGINT while `81001/plastic/training/hands.jsonl` is advancing, then inspect its complete-hand checkpoint/result. The public CLI exposes no deterministic `--stop-after-hands`: private `_train_arm(..., stop_after=N)` and `evaluate_disconnected(..., stop_after=N)` accept hand limits, but calling private executors would bypass the public orchestration and is not proposed here. No unrestricted continuation is justified before actual learning throughput, peak memory, checkpoint cost and disk growth are measured.

## Signal semantics and evidence limits

Frozen `src/flyholdem/experiments/disconnected_evaluation.py:57–65` launches the child, installs SIGINT/SIGTERM forwarding with `process.send_signal(signum)`, and remains in `process.wait()`; it waits for/reaps the child rather than intentionally leaving it behind. After the child evaluation has installed its handler, `evaluate.py:113–140` sets a stop flag, finishes/fsyncs the current hand, writes its interrupted result, and saves complete native state/RNG in `finally`. `disconnected_worker.py` then writes the matching isolation result and exits normally. `poker_curriculum.py:167–180` propagates an interrupted child to an interrupted parent result. During a training arm, `poker_training.py:85–115` similarly stops/checkpoints at a complete hand. The five-minute checkpoint rule also retains exact journal-tail recovery.

Existing `tests/integration/test_disconnected_evaluation.py:36` proves a real isolated fixture stops at 7/16 hands and resumes to byte-identical hands; `test_poker_training.py:22` verifies stop at 5/12 hands and exact full state/RNG/journal recovery; `test_poker_curriculum.py:52` verifies complete phase recovery. These tests use deterministic stop arguments. Send a signal only after observed child hands so startup/default-handler behavior is not confused with the installed complete-hand handler. No new implementation is required by the inspected normal-path logic; unexpected nonzero child exits must stop the workflow and be inspected.

## Actual OS-signal fixture check

A separate seven-neuron/ten-edge fixture check used the exact V13 downstream source and existing binary. A scratch-only trace observer, without source edits or function replacement, identified the actual wrapper/child PIDs after both forwarding handlers were installed. Authoritative process inspection confirmed child 87179 under wrapper 87177; seven post-fsync progress hands and thirteen complete journal rows were observed before sending exactly one SIGINT to that wrapper. The actual forwarding handler returned, the child saved a truthful interrupted result/checkpoint at fourteen complete hands, and both processes were reaped.

Same-artifact resume and an uninterrupted same-config fixture baseline each completed 256 hands, with byte-identical journals, all 168 actual neural decisions, all fourteen final native state arrays and controller RNG. The existing import/file teacher boundary held; graph/model/config and frozen source/binaries remained unchanged. Torch was absent and temporary fixtures were removed. Elapsed 4.403250 seconds. Engineering deal seeds were 17101–17132, bootstrap seed 17102; no scientific run was accessed or modified. An initial sandbox-denied process-inspection attempt was cleaned up and separately recorded before the successful authorized run.

Local projectless harness `work/mccfr/check-v13-fixture-sigint.py` SHA `7bf71831720a16bd74c377879cc23a8787a76dac5a5c44384f017838d6abac38`; actual report `work/mccfr/v13-fixture-sigint-check.json` SHA `d43389d707529565551b60dbaa5704cd4ce7615815c0d9339b00b156de8d19bd`. Complete journal SHA `0628fdf0f9835bf59c3fb0b3a1d6ec22c238381f0e66803273a00190838654d0`; decision trace `92ee3215e24b527592198520569901bed8348855921ea9a04f3ae63bb5afd050`. This actual check covers the wrapper and its isolated fixture child. It does not execute the guarded outer full curriculum, a learning-arm OS signal, full-native hand duration or simultaneous full graph RSS; those limits remain explicit.
