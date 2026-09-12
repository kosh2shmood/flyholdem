# Commands and recorded reports

Run from the repository root in the existing locked environment. The `.venv/bin/flyholdem` entry point now exposes the existing registered workflows. Module-level commands remain available for historical runtime snapshots. New commands do not relax any gate or source/config/binary/environment identity check.

| Task | Command |
|---|---|
| Fixture dashboard | `.venv/bin/flyholdem serve --port 8766` |
| Native dashboard / human play | `.venv/bin/flyholdem serve --mode full --model runs/conditioning-full-confirm-v1-model --port 8767` |
| Controllability | `.venv/bin/flyholdem preregister --mode circuit --config configs/controllability.yaml --output runs/new-registration` |
| Local conditioning | `.venv/bin/flyholdem train --config configs/conditioning.yaml --profile development --learning-rate 0.3 --output runs/new-conditioning` |
| Exact-cue surrogate transfer | `.venv/bin/flyholdem distill --config configs/exact_transfer_surrogate.yaml --profile development --learning-rate 0.1 --output runs/new-exact-transfer` |
| Conventional self-play | `.venv/bin/flyholdem teacher train --config configs/teacher_nfsp_double_v4.yaml --output runs/new-teacher` |
| Frozen conventional export | `.venv/bin/flyholdem teacher export --run runs/new-teacher --output runs/new-policy` |
| Paired teacher development | `.venv/bin/flyholdem teacher evaluate --policy runs/new-policy --config configs/teacher_evaluation.yaml --output runs/new-teacher-development` |
| Guarded teaching corpus | `.venv/bin/flyholdem teacher export-corpus --policy runs/new-policy --validation runs/passing-confirmation/result.json --config configs/corpus.yaml --output runs/new-corpus` |
| Canonical corpus audit | `.venv/bin/flyholdem teacher verify-corpus --corpus runs/new-corpus` |
| Frozen native paired baseline | `.venv/bin/flyholdem evaluate --config configs/frozen_poker_evaluation.yaml --model runs/conditioning-full-confirm-v1-model --output runs/native-full-frozen-baseline-v1` |
| Recorded evidence report | `.venv/bin/flyholdem report --run runs/exact-surrogate-circuit-confirm-v1` |

These are command forms, not instructions to rerun historical experiments with changed source. Use new output directories and commit the exact source/config before starting a long experiment. Use `--resume RUN_DIRECTORY` with the same command/config and unchanged runtime to continue. The old module commands retain their boolean `--resume` convention. Omitting `--output` creates a timestamped ignored run directory. Confirmation uses `--profile confirmatory` and the runner's matching `--development-reference`; do not use reserved seeds before the registered prerequisite passes. Never run a second full-graph experiment while the native full dashboard is advancing.

The top-level `train` currently runs the actual conditioning protocol and `distill` the actual small exact-cue local/surrogate protocol. Frozen paired native evaluation and four PokerKit curriculum environments now exist (see CURRICULA.md). The gated multi-seed poker runner is available through `curriculum run` (see POKER_TRAINING.md); these commands do not claim that dependent poker gates are passed. Teacher evaluation uses an explicitly exported `--policy` so the chosen model identity is concrete. Corpus export still refuses an unconfirmed or failed teacher. Current teacher candidates are unqualified, so the corpus example intentionally cannot run until qualification exists.

## Reports

Successful command completion for controllability, conditioning, exact transfer and conventional training/evaluation writes a report beside the result. `report --run RUN --output DIRECTORY` also generates it later without importing a teacher, loading a connectome model or changing original artifacts. Outputs: `REPORT.md`, `report.html` and `report.json`. HTML is standalone, printable and escaped; no external script/service is used. Reports preserve failed/interrupted/development-only labels and the original experiment's scope.

The reporter streams and verifies every modern journal row's index, preceding hash and content hash; checks the result's recorded head/count, manifest digest and any preregistration digest; and rejects truncated/nonfinite/mutated/inconsistent evidence. Historical controllability logs predate chained journals and are explicitly limited to file-checksum/JSON-structure verification. A source commit in the original run manifest describes the execution checkout; the separate source hash identifies immutable experiment source bundles. Reports label these as execution commit and source hash.

This is artifact verification and presentation, not a new statistical evaluation. Accuracy tables show seed min/max ranges; teacher tables preserve the registered suite-adjusted bootstrap intervals. Full paired evidence, hashes, runtime, graph/learning labels and scope are retained in JSON and Markdown. Exact-cue, conditioning and failed teacher reports were generated from actual completed runs and checked in Chromium at desktop/mobile sizes. New generated reports remain ignored/local.

Historical runs use their frozen source snapshots for resume/export/evaluation. For example, v4's active runtime is `runs/teacher-runtime-v4`; its CLI predates this consolidation, so continue it using the documented `python -m flyholdem.teacher.training` command in TEACHER.md. The offline reporter may read those completed artifacts from the current checkout because it checks their bytes without reexecuting their models.

Teacher qualification can be independently recomputed with `flyholdem teacher verify-evaluation --run RUN --policy POLICY`. It requires the original registered confirmatory suite and complete evidence. Add `--allow-development` only to inspect development evidence; it cannot authorize a teacher. Corpus export performs this check automatically and repeats the actual policy information-boundary test.

A separate small-game conventional reference uses `flyholdem teacher train-shove-fold --config configs/shove_fold_teacher.yaml --output RUN` and `teacher export-shove-fold --run RUN --output POLICY`. Resume training with `--resume RUN`. Counts are complete three-branch chance traversals, and exports remain unvalidated. This workflow cannot qualify the full-hand teacher. Details: SHOVE_FOLD_TEACHER.md.

The small-game evaluator is `flyholdem teacher evaluate-shove-fold --policy POLICY --training-run TRAINING_RUN --config configs/shove_fold_teacher_evaluation.yaml --output RUN`. Confirmation additionally requires `--profile confirmatory --development-reference DEVELOPMENT_RUN`. It checks matching completed training, disjoint deals and the original complete opponent suite. Its small-game qualification is separate from the full teacher gate.

A separately registered best-response extraction uses `flyholdem teacher export-best-response --config configs/teacher_best_response_v6.yaml --output POLICY`. It checks the exact completed source/checkpoint and emits a distinctly labeled conventional greedy-Q mixture. The existing `teacher evaluate` and guarded corpus commands recognize it without treating it as an NFSP historical average. Prior average-policy exports retain their original source/runtime checks.


To qualify a completed full-hand corpus from its measured teacher, use `.venv/bin/flyholdem teacher verify-qualified-corpus --corpus CORPUS --policy POLICY --validation-run CONFIRMATION_RUN`. This repeats the registered collection and compares actual targets and split bytes. It rejects an unqualified teacher before collection replay; structural `verify-corpus` alone does not authorize training.


Gate prerequisite commands are documented in GATES.md. `gate certify-transfer` requires complete confirmed full-teacher/corpus evidence and actual native cue transfer/removal; `gate verify-transfer` repeats every check. `gate verify-cues --run RUN --config PROTOCOL` rederives registered confirmation statistics; add `--allow-failed` only for read-only negative audits. These commands do not claim that the pending poker learning gates passed.


`evaluate --disconnected` requires `--model` and runs the frozen native evaluator in a separate minimal package/process. Teacher import and original project training-file access are actively denied and checked. The outer run preserves its immutable runtime; actual hand evidence is under `evaluation/`. Resume with `--resume OUTER_RUN`, retaining the same config/model. This remains a baseline workflow and grants no poker-learning claim.


`curriculum run --stage 3 --profile development --learning-mode bio-plastic --gate2a CERTIFICATE --output RUN` starts the registered full native poker experiment only after all prerequisites reverify. Use `distilled-connectome` for the separately labeled path. Confirmation additionally requires `--development RUN`; stages 4–6 require `--previous CONFIRMED_PRECEDING_STAGE`. Resume replaces `--output` with `--resume RUN`. `curriculum verify --run RUN --require-pass` recomputes the complete endpoint and prerequisite chain; ordinary `report --run RUN` displays recorded BB/100 seed statistics. The full teacher is still unqualified, so no actual poker curriculum run is authorized yet. Details and resource scheduling: POKER_TRAINING.md.
