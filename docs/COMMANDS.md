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

The top-level `train` currently runs the actual conditioning protocol and `distill` the actual small exact-cue local/surrogate protocol. Frozen paired native evaluation and four PokerKit curriculum environments now exist (see CURRICULA.md). General poker curriculum learning/control runners remain in implementation; these commands do not claim that dependent poker gates are passed. Teacher evaluation uses an explicitly exported `--policy` so the chosen model identity is concrete. Corpus export still refuses an unconfirmed or failed teacher. Current teacher candidates are unqualified, so the corpus example intentionally cannot run until qualification exists.

## Reports

Successful command completion for controllability, conditioning, exact transfer and conventional training/evaluation writes a report beside the result. `report --run RUN --output DIRECTORY` also generates it later without importing a teacher, loading a connectome model or changing original artifacts. Outputs: `REPORT.md`, `report.html` and `report.json`. HTML is standalone, printable and escaped; no external script/service is used. Reports preserve failed/interrupted/development-only labels and the original experiment's scope.

The reporter streams and verifies every modern journal row's index, preceding hash and content hash; checks the result's recorded head/count, manifest digest and any preregistration digest; and rejects truncated/nonfinite/mutated/inconsistent evidence. Historical controllability logs predate chained journals and are explicitly limited to file-checksum/JSON-structure verification. A source commit in the original run manifest describes the execution checkout; the separate source hash identifies immutable experiment source bundles. Reports label these as execution commit and source hash.

This is artifact verification and presentation, not a new statistical evaluation. Accuracy tables show seed min/max ranges; teacher tables preserve the registered suite-adjusted bootstrap intervals. Full paired evidence, hashes, runtime, graph/learning labels and scope are retained in JSON and Markdown. Exact-cue, conditioning and failed teacher reports were generated from actual completed runs and checked in Chromium at desktop/mobile sizes. New generated reports remain ignored/local.

Historical runs use their frozen source snapshots for resume/export/evaluation. For example, v4's active runtime is `runs/teacher-runtime-v4`; its CLI predates this consolidation, so continue it using the documented `python -m flyholdem.teacher.training` command in TEACHER.md. The offline reporter may read those completed artifacts from the current checkout because it checks their bytes without reexecuting their models.

Teacher qualification can be independently recomputed with `flyholdem teacher verify-evaluation --run RUN --policy POLICY`. It requires the original registered confirmatory suite and complete evidence. Add `--allow-development` only to inspect development evidence; it cannot authorize a teacher. Corpus export performs this check automatically and repeats the actual policy information-boundary test.

A separate small-game conventional reference uses `flyholdem teacher train-shove-fold --config configs/shove_fold_teacher.yaml --output RUN` and `teacher export-shove-fold --run RUN --output POLICY`. Resume training with `--resume RUN`. Counts are complete three-branch chance traversals, and exports remain unvalidated. This workflow cannot qualify the full-hand teacher. Details: SHOVE_FOLD_TEACHER.md.
