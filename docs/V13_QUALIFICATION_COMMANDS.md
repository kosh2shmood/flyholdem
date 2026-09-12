# V13 final policy and qualification commands

Prepared command sequence; final export and qualification have not run. The training worker is still active. Run from `/Users/bohdankoshevoi/flyholdem/flyholdem`. Keep the original frozen teacher runtime `runs/teacher-runtime-v13`, source `46a10cd749bfbc0c17c1de7189ed5ec0df8a99ee231199c1969f53be02ed82e5`, and existing environment unchanged.

1. **Complete and verify the one registered run.** Wait for `runs/teacher-self-play-regret-v13` to finish exactly 100,000 synchronous iterations / 200,000 player traversals. Reverify the complete journal, final numeric checkpoint/RNG, configuration and frozen source bindings before export. Preserve the original V1–V12 negatives. Export only the registered final sampled average, with original role mapping and legal-uniform unseen/zero-mass prior; no intermediate candidate, current-regret extraction, temperature or alternate checkpoint.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/teacher-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli teacher export-self-play-regret --run runs/teacher-self-play-regret-v13 --output runs/teacher-self-play-regret-v13-policy
```

2. **Publish the checked final identity before development.** Independently load and verify the exported numeric arrays and original final-checkpoint bindings. Record/publish the SHA-256 of `runs/teacher-self-play-regret-v13-policy/manifest.json`, its original training manifest/result/journal/checkpoint identities and frozen runtime identity. The policy remains unqualified. Do not start development until this identity is published; use this exact policy unchanged for every later step.

3. **Run and independently verify the original development suite, then reproduce actual play.** This is 128 paired deals per original opponent (random, calling-station, tight-aggressive, equity-bucket), 512 paired records total, registered seeds 2000000–2000127 at 20 BB.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/teacher-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli teacher evaluate --policy runs/teacher-self-play-regret-v13-policy --config runs/teacher-runtime-v13/configs/teacher_evaluation.yaml --profile development --output runs/teacher-self-play-regret-v13-development
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/teacher-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli teacher verify-evaluation --run runs/teacher-self-play-regret-v13-development --policy runs/teacher-self-play-regret-v13-policy --allow-development
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/teacher-runtime-v13/src" .venv/bin/python -B runs/teacher-runtime-v13/scripts/audit_teacher_play.py --run runs/teacher-self-play-regret-v13-development --policy runs/teacher-self-play-regret-v13-policy --output runs/teacher-self-play-regret-v13-play-audit.json
```

Inspect `runs/teacher-self-play-regret-v13-development/{manifest.json,result.json,paired-deals.jsonl,report/REPORT.md}` and the audit. Require **all four suite-adjusted bootstrap lower bounds strictly above zero**, `passes_fixed_suite: true`, all 32 hidden-hole/future-deck/teacher-label boundary checks, independent complete-journal/statistical verification and exact reproduction of all 512 actual paired records. A successful `verify-evaluation --allow-development` exit proves consistency, **not a pass**: it also verifies a truthful failed result. Development always leaves `allowed_as_teacher: false`, including when it passes. Record/publish the result and actual action counts; action variety grants no qualification.

**If development fails any requirement, stop the dependent sequence: preserve the failure; do not use full confirmation, export a qualified corpus, issue Gate 2A or teach the fly.**

4. **Only after independently verified passing development, run the original reserved confirmation.** Use the same frozen policy and the required development reference. This is 512 paired deals per original opponent, 2,048 paired records total, seeds 3000000–3000511.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/teacher-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli teacher evaluate --policy runs/teacher-self-play-regret-v13-policy --config runs/teacher-runtime-v13/configs/teacher_evaluation.yaml --profile confirmatory --development-reference runs/teacher-self-play-regret-v13-development --output runs/teacher-self-play-regret-v13-confirmatory
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/teacher-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli teacher verify-evaluation --run runs/teacher-self-play-regret-v13-confirmatory --policy runs/teacher-self-play-regret-v13-policy
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/teacher-runtime-v13/src" .venv/bin/python -B runs/teacher-runtime-v13/scripts/audit_teacher_play.py --run runs/teacher-self-play-regret-v13-confirmatory --policy runs/teacher-self-play-regret-v13-policy --output runs/teacher-self-play-regret-v13-confirmatory-play-audit.json
```

Again require all four strictly positive adjusted lower bounds, 32 boundary checks, independent verification, exact actual-play reproduction and `allowed_as_teacher: true` in the **confirmation evidence**. Publish the verified outcome. A failed confirmation blocks all dependent teaching work; do not change policy or repeat the suite to seek a pass.

5. **Only a qualifying confirmation unlocks the existing V13 downstream path.** Recheck the separately frozen/published `runs/poker-runtime-v13` identity and prerequisite evidence before use. Its bounded engineering preflight does not substitute for actual qualification. Then export/reproduce the real corpus and request the original Gate 2A certificate; its verifier must repeat actual graph/cue/model/teacher-removal prerequisites.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/poker-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli teacher export-corpus --policy runs/teacher-self-play-regret-v13-policy --validation runs/teacher-self-play-regret-v13-confirmatory/result.json --config runs/poker-runtime-v13/configs/corpus.yaml --output runs/teacher-self-play-regret-v13-corpus
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/poker-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli teacher verify-qualified-corpus --policy runs/teacher-self-play-regret-v13-policy --validation-run runs/teacher-self-play-regret-v13-confirmatory --corpus runs/teacher-self-play-regret-v13-corpus
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/runs/poker-runtime-v13/src" .venv/bin/python -B -m flyholdem.cli gate certify-transfer --policy runs/teacher-self-play-regret-v13-policy --confirmation runs/teacher-self-play-regret-v13-confirmatory --corpus runs/teacher-self-play-regret-v13-corpus --output runs/gate2a-v13.json
```

Evaluation/corpus recovery replaces `--output RUN` with `--resume RUN`, retaining every other argument, exact runtime/config/policy and confirmation development reference. Never run a duplicate worker. Export and audit outputs must be new; do not overwrite completed artifacts. Confirmed teacher/corpus/Gate 2A evidence still does not establish native poker learning: native curriculum execution needs its own resource boundary and Gates 3–6.
