# Conventional teacher and disconnected student runtime

Status: implemented and tested; no validated poker teacher, corpus, transfer result or poker learning claim exists yet. The live table still runs the explicitly labeled fixture.

## Teacher algorithm

The conventional control uses neural fictitious self-play, following [Heinrich and Silver (2016)](https://arxiv.org/abs/1603.01121). Two independent agents maintain Q/target networks, transition replay, and an average policy trained on a uniform reservoir of their own best-response behavior. Each agent samples its best-response/average-policy mixture once per hand. This implementation is an experiment in the project's five-action no-limit abstraction; the paper's results do not establish this implementation's strength or equilibrium convergence.

The initial development config uses 20 BB stacks, 25,000 hands, two 128-unit hidden layers, deterministic CPU PyTorch, a 10% anticipatory mixture, 50,000 replay entries and 200,000 reservoir entries per agent. Exploration decays from 1.0 to 0.1 over 20,000 hands. Q learning receives zero intermediate reward and terminal net chips divided by the initial stack. A transition bootstraps only at the same player's next decision, never on the opponent's private observation. The average policy learns only from that agent's best-response episodes. Frozen teacher targets are the equal probability mixture of both average policies.

The original teacher and native fly share the 237-channel representation. The native first-action pulse is derived from the acting player's public relative history, including its first decision after an opponent action. An inconsistent external flag is rejected. Legacy fixture encoding is preserved so the V0 replay remains unchanged.

## Information and validation

Canonical states strictly accept the acting-player observation allowlist. Own hole-card order and the simultaneous flop order are normalized; turn/river order, suits and the complete visible action history are preserved. Hidden fields are rejected. Numeric values, card uniqueness, masks and history are validated. Canonical IDs, rather than deal IDs, determine corpus splits.

A teacher is unavailable for distillation until frozen evaluation passes the complete random/calling-station/tight-aggressive/equity-bucket suite on independent paired deals. Development and confirmation use separate fixed seeds. The initial criterion requires a positive lower bootstrap bound for every opponent, adjusted across the suite. This may fail; training alone never authorizes use as a teacher. Actual private-hole, future-deck and teacher-label mutations must preserve information bytes and target probabilities.

Corpus collection mixes teacher, random and calling-station behavior to cover streets and states. It caps samples per declared stratum: street, position, coarse visible strength proxy, SPR, call price, legal pattern and recent public history shape. The proxy is not hidden equity. Targets depend only on canonical visible states. Repeated states must have identical targets, and train/validation/test IDs cannot overlap. A hash-chained complete-hand collection cache supports exact resume; incomplete hands are reconstructed from their independent seeds.

## Recovery and frozen policies

Teacher checkpoints use numeric arrays and a typed JSON tensor tree, including models, target models, optimizer tensors, replay/reservoir contents and counters, all RNG state, mixture state and the completed-hand cursor. No pickle is used. Checkpoints occur every five minutes and on graceful shutdown at complete hands; exact tail replay verifies previously logged operations. A trained model remains `trained-unvalidated` until the separate held-out checks pass.

Small frozen teacher exports pin inference code, runtime versions, tensors and source training checkpoint. Frozen **fly** exports are separate: they store bounded weights on declared existing graph edges, the fixed population/readout registration, graph/binary identity and inference code identity. They contain no trainable input adapter or decoder. The teacher, corpus, learning and experiment code is absent from the fly's required source scope. Historical training checkpoints still enforce their original whole-source/environment identity; this is a separate frozen-inference contract, not a silent weakening of old checkpoints.

An integration test copies a minimal native runtime, loads a bounded synaptic fixture, records four decisions, deletes the actual teacher package and corpus, then verifies byte-identical decisions. Changing required encoder code is rejected. This proves engineering separation; Gate 2A still requires the actual trained-student transfer evidence.

## Commands

Use the locked environment with `uv sync --frozen --extra data --extra teacher`, then `make build-kernel`. Avoid changing installed extras during a running experiment because its environment identity is strict.

```sh
.venv/bin/python -m flyholdem.teacher.training --config configs/teacher_nfsp.yaml --output runs/teacher-nfsp-v1
.venv/bin/python -m flyholdem.teacher.export --run runs/teacher-nfsp-v1 --output runs/teacher-nfsp-v1-policy
.venv/bin/python -m flyholdem.teacher.evaluation --policy runs/teacher-nfsp-v1-policy --config configs/teacher_evaluation.yaml --profile development --output runs/teacher-nfsp-v1-development
```

Only a qualifying development policy should advance to the separate confirmatory seeds using `--profile confirmatory` and a new output path. A matching passing confirmation is required for:

```sh
.venv/bin/python -m flyholdem.teacher.corpus --policy runs/teacher-nfsp-v1-policy --validation runs/teacher-nfsp-v1-confirmation/result.json --config configs/corpus.yaml --output runs/teacher-nfsp-v1-corpus
```

Training, evaluation and corpus commands support `--resume` with unchanged config/source/environment. Public CLI consolidation and actual local/surrogate transfer runners remain subsequent work. Optional teacher tests skip when PyTorch is absent; the core fixture/native separation tests require no full dataset.

## Initial 25,000-hand result — validation failed

The registered run on 3f3069a completed in 112.16 s. Frozen policy SHA-256: 14b3e116392149159bc689889f0e73a83214af3538246d03d244c92bfeab744f. The actual target information-boundary check passed 32 private-data mutations. Development results over 128 paired deals per opponent were +0.635 BB/hand against random, −0.264 against calling station, −0.049 against tight-aggressive and −2.422 against equity-bucket. None had a positive suite-adjusted lower confidence bound, so the policy is barred from teaching and no corpus was generated. This is a negative teacher result, not a fly result. Exact machine-readable evidence remains local in ignored runs/teacher-nfsp-v1 and runs/teacher-nfsp-v1-development.

Register a longer 250,000-hand development run in configs/teacher_nfsp_250k.yaml. Only duration and progress-print cadence change; architecture, seeds, optimization and information boundary remain fixed. Start from the same initialization, preserving the old checkpoint contract rather than silently changing its configured horizon. Use a separate run directory and the same held-out development suite; confirmatory teacher seeds remain untouched. Resume command: `.venv/bin/python -m flyholdem.teacher.training --config configs/teacher_nfsp_250k.yaml --output runs/teacher-nfsp-250k-v1 --resume`.

## Longer teacher result and visible-card representation v2

The 250,000-hand v1 run completed in 1,253.76 s. Its first 25,000 complete hand events exactly matched the shorter run. Frozen policy SHA-256: 578c49b5c6189908943ad7c364a93433ffba35325bb63b2f1f08c07d584a079a. Development return was +1.791 BB/hand against random, +0.154 against station, −0.670 against tight-aggressive and −2.070 against equity-bucket. Only the random comparator had a positive suite-adjusted lower bound. The teacher remains unvalidated; no confirmatory seeds or distillation corpus were used. Generated evidence stays local in runs/teacher-nfsp-250k-v1-development.

The next registered development candidate, configs/teacher_nfsp_cards_v2.yaml, retains the 250,000-hand horizon, seeds, network hidden widths and optimizer settings. It adds deterministic visible-card structure to the conventional teacher: rank counts, suit counts, straight coverage, visible made-hand category/rank, own-card participation, and public pot/price ratios. Its 354 inputs start with the exact original 237 channels. All additional features are functions of the same canonical player information set; they never enter the fly's frozen encoder or corpus observations. There is no opponent-card input or future-card sampling. This is a teacher representation experiment, not a change to the biological interface.

Both versions pass exact complete-checkpoint recovery. V2 passes actual hidden-card/future-deck invariance and frozen-export probability identity. The complete suite passes 68 tests with 2 isolated-oracle skips. Source snapshots under ignored runs/teacher-runtime-v1 and runs/teacher-runtime-v2 preserve historical loader identities. Execute v2 with PYTHONPATH="$PWD/runs/teacher-runtime-v2/src" so later independent source work cannot alter the running teacher. The snapshot includes the committed src, uv.lock and registered configuration; the installed environment stays fixed. Resume the same command with --resume. Frozen export/evaluation must also use that snapshot.

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v2/src" .venv/bin/python -m flyholdem.teacher.training --config configs/teacher_nfsp_cards_v2.yaml --output runs/teacher-nfsp-cards-v2
```

## Visible-card teacher v2 — validation failed

The registered 250,000-hand v2 run completed in 1,410.12 s. Policy SHA-256: 9ec0bc011ec48ad84191ee25c0084bfece4a3d56414dc3913914bc736a06bf77. Held-out development return was +0.977 / +0.348 / −0.523 / −1.619 BB per hand against random/station/TAG/equity. None had a positive suite-adjusted lower confidence bound. The 32 actual private-data invariance checks passed, but strength validation failed. Confirmation seeds remain unused and no corpus exists. Exact evidence is local in runs/teacher-nfsp-cards-v2-development. Additional visible-card features alone did not qualify this teacher; a stronger training procedure is required. The next candidate should address its opponent distribution and value learning under a separately committed protocol, rather than treating longer training or visible activity as success.

## Registered visible-equity teacher v3

The next teacher representation adds a deterministic 256-sample estimate against uniformly sampled unknown opponent cards/runouts, its square, and its difference from the visible call price. The estimator accepts only the teacher's two own cards and current public board. Samples are hypothetical; the actual opponent cards, deck, game object and hand counters are unavailable. A visible-card hash seeds each calculation, so repeated canonical states have identical features. The 357 inputs retain the exact original 237 channels as their prefix. None of these added features enter the fly's observation or corpus observation fields.

The conventional utility uses a separate original C++ hand evaluator for speed. PokerKit remains the only dealing, betting and settlement engine. The utility's rank ordering/ties matched PokerKit on 4,500 independent five/six/seven-card hands plus targeted straight/flush/quads/full-house/two-pair cases; another 1,800 sampled hands are covered in CI. On this machine, 1,000 calls of 256 samples took 0.0281 s in the isolated prototype. This benchmark establishes utility throughput, not poker policy strength. Source, compiler flags and actual binary hash are pinned in v3 training and frozen-policy identities; existing native LIF binaries and the fly's inference runtime are unchanged. Old teacher versions use their preserved source snapshots.

Config configs/teacher_nfsp_equity_v3.yaml retains the v2 250,000-hand horizon, source/deal seeds, 128-unit network hidden widths, anticipatory mixture, replay/reservoir sizes and optimizer settings. Only representation and its explicitly pinned optional evaluator change. The original held-out development suite remains fixed; confirmation seeds are still unused. Full suite: 83 passed / 2 isolated-oracle skips, including exact complete training recovery, native-evaluator agreement, private-data invariance and byte-identical frozen exports for v3.

Commit this configuration before training. Snapshot src, uv.lock and config into ignored runs/teacher-runtime-v3 and preserve the separate equity binary under its runs/build reference. Execute:

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v3/src" .venv/bin/python -m flyholdem.teacher.training --config runs/teacher-runtime-v3/configs/teacher_nfsp_equity_v3.yaml --output runs/teacher-nfsp-equity-v3
```

Resume adds --resume; export/evaluation also use this PYTHONPATH. Do not rebuild the equity utility or change the installed environment while the experiment is active. No result or teacher qualification is assumed at registration.


## Visible-equity v3 result — development failed

V3 completed 250,000 hands in 1,674.43 s from the immutable teacher-runtime-v3 snapshot. Frozen policy SHA-256: 6d86d320863d3aadc6f6765492b846d5c6b14316032bb418f2538e24a7f2e3ce. The same 128 paired development deals per opponent returned +1.1796875 BB/hand against random, −0.34765625 against calling station, −0.25 against tight-aggressive and −1.587890625 against equity-bucket. None had a positive suite-adjusted lower bound. This is a failed development result; no reserved confirmation seeds or teaching corpus are used. The 32 hidden-hole/future-deck/teacher-label mutation decisions passed exactly.

Evaluation manifest SHA-256: 9066cd8afbb172300e065d578dcdd2b0b38cbea815aa8fc0b21a9d2c34a1ca75. Training journal head: db0a465be116f4f397a767d3e268764da0b5bbe4c406e24ebd6c0175275fda75; evaluation journal head: 1e50990c95581dc2791226cb0d616537187fb5a7309e8f728212c10b4b8bedfd. Full generated records remain local under runs/teacher-nfsp-equity-v3 and sibling policy/development directories. Visible equity alone did not solve teacher training. Next diagnose continuation-value estimates and average-policy behavior before registering another conventional candidate; never select fly populations or biological parameters from these poker results.


## Registered v4 Double DQN candidate

V4 changes only the NFSP best-response bootstrap from a target-network maximum to Double DQN: choose the legal next action with the online Q network and evaluate that action with the frozen target network. Terminal continuation values are exactly zero. This follows [van Hasselt, Guez and Silver (2015)](https://arxiv.org/abs/1509.06461), which addresses maximization overestimation in deep Q-learning. It is a controlled algorithm candidate, not a claim that the v3 failure has been diagnosed conclusively. A training-deal-only inspection found weak separation in v3's average-policy behavior across visible equity buckets; it used no reserved confirmation deals or profit-based biological tuning.

Config configs/teacher_nfsp_double_v4.yaml keeps v3's 357 visible features, 250k hands, 71001 seed, network/optimizers/memories, exploration, 20 BB stacks and original four-opponent validation suite. Average-policy export remains the fixed equal mixture of both agents; no evaluation-time Q override is introduced. V1–v3 remain preserved in immutable runtime bundles. Exact short-run interruption/resume and target-choice/legality/terminal tests pass. No v4 poker result exists at registration.

After committing, copy src, uv.lock, the config and the unchanged teacher equity binary/metadata into ignored runs/teacher-runtime-v4. Run from that immutable source:

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v4/src" .venv/bin/python -m flyholdem.teacher.training --config runs/teacher-runtime-v4/configs/teacher_nfsp_double_v4.yaml --output runs/teacher-nfsp-double-v4
```

Add --resume after interruption. Export/evaluation must use the same PYTHONPATH. Do not change the snapshot, installed environment or copied binary while it runs. A passing development suite is required before reserved confirmation and corpus creation; no fly learning claim follows from conventional teacher training.


## Double DQN v4 result — development failed

The controlled v4 candidate completed 250,000 hands in 1,620.01 s and failed the complete original development suite: +1.064453125 / −0.216796875 / −0.263671875 / −1.80859375 BB per hand against random/station/TAG/equity. No suite-adjusted lower bound was positive. Hidden-information mutation checks passed all 32 decisions. The frozen average policy SHA-256 is 372e5ee1a74b83750ad945dbe62c2fc6dcd316ca00cada307aa2c3138402cf9a; evaluation manifest bf99368c38b7d64937e05ba2f85932a8383a41825b2832b05bdef79057401395. All generated records remain local. No confirmation or corpus is allowed. Double DQN alone did not solve teacher robustness; the next candidate will require a separately registered training-population change, preserving all original evaluation opponents and reserved deals.


## Registered v5 fixed-policy-prior NFSP candidate

V5 changes the conventional training population after v1–v4 failed robustness validation. Every eight-hand cycle contains four NFSP self-play hands followed by one hand against each original fixed opponent: random, calling station, TAG and equity-bucket. The fixed opponent alternates seats on successive cycles, so each learned agent sees both positions. Only the active learned agent receives transitions, supervised best-response samples and optimizer updates on fixed-opponent hands. No opponent identity or internals enter the canonical neural-network input. The same NFSP episode mixture and uniform average-policy reservoir remain; the exported policy is still the equal average-probability mixture of the two learned agents.

This is explicitly **NFSP with a fixed policy prior**, not vanilla NFSP or a complete PSRO implementation. It takes the policy-mixture robustness motivation from [Lanctot et al. (2017)](https://arxiv.org/abs/1711.00832), while keeping a declared fixed prior rather than solving an empirical meta-game. No equilibrium claim is made. The entire original opponent suite remains in held-out evaluation; only training deals overlap earlier development training. Reserved confirmation deals remain unused. No biological mapping or plasticity parameter changes.

Config configs/teacher_nfsp_population_v5.yaml registers 500,000 hands, the original 71001 initialization/deal seed sequence, v4 Double DQN, 357 visible features, the same networks/optimizers/exploration and 20 BB stacks. Half the hands are self-play; half are equally split among all four original opponents, with no performance-based weighting or opponent selection.

To keep the equity-bucket training opponent practical, a teacher-only helper preserves its exact visible-state hash, ordered unknown cards, Python RNG draws, thresholds and action priority, replacing only repeated PokerKit rank comparisons with the already-tested batched C++ rank utility. It matched exact equity on 120 CI states and complete action trajectories for all four opponents; a separate 200-state benchmark matched every value and took 0.03758 s versus 3.28761 s (87.5× on this machine). The independent evaluation opponent retains its original PokerKit implementation. No shared native LIF or equity binary is rebuilt.

Complete population-run stop/resume journals and frozen export identity passed; fixed-opponent transitions cannot enter learned-agent memories. Commit source/config before execution, then freeze src/lock/config and a separate copied equity binary in runs/teacher-runtime-v5:

```sh
PYTHONPATH="$PWD/runs/teacher-runtime-v5/src" .venv/bin/python -m flyholdem.teacher.training --config runs/teacher-runtime-v5/configs/teacher_nfsp_population_v5.yaml --output runs/teacher-nfsp-population-v5
```

Add --resume after interruption. Export and validation use the same PYTHONPATH. This candidate remains unvalidated until the unchanged development and reserved confirmation suites pass. Generated data/checkpoints/logs remain local. Preserve every prior failure.

## Recomputed qualification before corpus export

Corpus export now requires the complete evaluated run: result.json, manifest.json and its paired-deal journal. It verifies the exact registered opponent order, deal seeds/counts, policy tensors and stack, recomputes every bootstrap interval and summary, and rejects inconsistent qualification flags. The loaded policy also repeats the actual hidden-hole/future-deck/label boundary test before collection. This checks local artifact consistency; it is not a cryptographic signature or a substitute for measured validation.

Audit a development candidate without authorizing it:

```sh
.venv/bin/flyholdem teacher verify-evaluation --run runs/teacher-nfsp-double-v4-development --policy runs/teacher-nfsp-double-v4-policy --allow-development
```

Omit `--allow-development` to require passing confirmation. The real v4 development evidence recomputed exactly (512 paired deals), and remains failed/unapproved. Seven new numerical-artifact checks cover altered intervals/flags, schedule substitutions, tensor changes, missing opponents and truncation. These synthetic test artifacts are not poker results.

A separate small tabular CFR teacher has now passed development and confirmation for the restricted 10 BB shove/fold subgame. It does not replace the full 20 BB NFSP qualification. See SHOVE_FOLD_TEACHER.md for its registered algorithm, independent paired returns and explicit scope. Full-hand v5 population training remains in progress.

V5 population training completed all 500,000 hands in 2,901.88463 seconds, using the unchanged frozen source fcbf967. Final policy manifest c7f922e82890e7722ad43d9a73740f7e95ac62b775a4200aba54c52e478bca7b remains unvalidated. Training manifest d3324b5fd51335d9f743d71d6698e2ed7084dc7296441c26379eabbdcf198697 and journal head ab0c4b211e457210ff766a43b06039283384e87ae12124464dfba9bb1858cc0f identify the completed run. Next evaluate the original development suite; confirmation remains untouched.

V5 failed the unchanged full development suite: random +0.568359, station +0.404297, TAG −0.283203 and equity −1.480469 BB/hand, with no positive adjusted lower bound. Its 32-state private-information invariance passed. Recomputed all 512 paired-deal summaries independently; result SHA ddfe482d3163c0f53b562fea532c4375c33f3deea0dbb70c1810ad0b15f63743, evaluation manifest 22d60152a3b2303cf91089e4fa496648487c19794873cb6158ad415ffdd6aee4. Full confirmation deals remain unused and no full-hand corpus is permitted.

Next controlled conventional candidate: separately export the two final learned Double-DQN best-response components as an equal mixture of their legal greedy policies. This tests a different existing component, without retraining or selecting a biological parameter. It must be explicitly labeled as a population-trained best-response mixture, not NFSP's time-average policy or an equilibrium claim, and independently pass the same full suite before any teaching. The candidate implementation/configuration must be committed before evaluation.

## V6: final best-response components, without retraining

V6 extracts the two final Double-DQN Q heads from the completed V5 checkpoint. Each component chooses its own legal greedy action; the candidate distribution is their equal mixture (one action with probability 1, or two with 0.5 each). This is a population-trained best-response mixture, not the historical NFSP average or an equilibrium claim. The distinction follows NFSP's separate best-response and average-policy components; the existing Double DQN value update is unchanged.

The new extraction protocol pins V5's training manifest, frozen source and completed 500,000-hand boundary. It checks the full original source hash, exact original feature/network/codec files, environment, feature binary identity, checkpoint hashes and complete training journal before exporting numeric Q tensors. A separate policy module and explicit loader preserve every previous frozen average-policy source. Its original private-information, paired-game, complete-suite and corpus-qualification checks apply unchanged, with an accurate sampling label. There is no change to fly inference, biological parameters or held-out full-hand confirmation seeds.

```sh
.venv/bin/flyholdem teacher export-best-response --config configs/teacher_best_response_v6.yaml --output runs/teacher-best-response-v6-policy
.venv/bin/flyholdem teacher evaluate --policy runs/teacher-best-response-v6-policy --config configs/teacher_evaluation.yaml --profile development --output runs/teacher-best-response-v6-development
```

Commit and freeze the candidate source/config with a copied verified teacher-equity binary before these operations; execute through that snapshot's PYTHONPATH. Extraction is not resumable and writes a new policy directory; evaluation resumes with `--resume RUN`. No candidate is qualified until the unchanged full development and confirmation suites pass. Three new checks verify exact saved-Q tensor bytes, distinct Q/average components, legal greedy mixtures, invalid values, private invariance, corrupted tensors and rederived full-suite results/sampling identity.
