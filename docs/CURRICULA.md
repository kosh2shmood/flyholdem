# Poker curriculum environments and frozen evaluation

The curriculum layer now implements genuine PokerKit games for four declared environments. It does not implement a substitute rules engine or a strategic fallback.

| Environment | Initial state | Legal decisions |
|---|---|---|
| shove-fold-10bb-v1 | Genuine 10 BB heads-up preflop deal, 1/2 blinds | Fold or commit the whole remaining stack |
| river-20bb-v1 | Uniform standard-deck deal; both seats check/call through the turn; 4-chip pot, 38-chip stacks at the river | All five canonical nonduplicate actions |
| hu-20bb-v1 | Genuine 20 BB full-hand deal | All five canonical nonduplicate actions |
| hu-100bb-v1 | Genuine 100 BB full-hand deal | All five canonical nonduplicate actions |

The shove/fold mask retains canonical action identities. The opening shove is action 4; facing an opponent's shove, action 1 represents the full-stack call. It never adds a duplicate action 4 when PokerKit's abstraction already represents that wager as a call. Limps and partial raises are prohibited in this subgame. River's forced prefix defines its initial state and remains in public history; it is not a policy fallback. Uniform independent standard-deck ranges are explicitly registered. Nonuniform controlled ranges are not implemented at this checkpoint.

Only the legal mask changes in the canonical neural observation. No curriculum tag, equity, teacher label, opponent private hand or future deck is added. Serialized private checkpoints contain the exact curriculum registration, setup actions and private PokerKit hand; restore verifies the registration and replays every policy action under its original mask. Byte-level hidden-hole/future-deck invariance and all four exact continuation checks pass.

## Frozen evaluation

`flyholdem evaluate` now measures an explicit native frozen model, or the original weights under the selected registration, on fixed seat-swapped deals. It imports no teacher module. Both seats share the same canonical deal for each pair; the neural player resets to fresh rest for each hand and uses zero exploration. Its actual five neural scores remain the only policy source. Each committed action is checked against the legal argmax, no reward is delivered and the final weights must equal their initial hash. Existing core frozen-inference files remain unchanged.

Each completed hand is immediately written/fsynced in a hash-chained journal. It records canonical neural information, scores/masks, selected action, actual native count hash, private replay checkpoint and outcome. Full neural state, controller RNG and progress are checkpointed every five minutes and on graceful shutdown; latest three generations are retained. Resume verifies source/config/graph/binary/environment identity and reexecutes any logged tail exactly. Incomplete seat pairs are excluded from paired statistics and counted separately. The complete four-opponent suite is fixed in its original order; per-opponent intervals apply the same suite adjustment as the conventional teacher evaluation.

The registered first full baseline is configs/frozen_poker_evaluation.yaml: 32 paired deals per opponent (256 hands total), development seeds beginning 5,100,000, bootstrap seed 5,190,000 and 5,000 resamples. It measures the existing full cue-conditioned model, whose poker strategy is unvalidated. No teacher or curriculum confirmation seed is used. This is an independent diagnostic baseline, not a passed learning gate or policy-selection criterion. Do not select biological parameters from these returns.

```sh
.venv/bin/flyholdem evaluate --config configs/frozen_poker_evaluation.yaml --model runs/conditioning-full-confirm-v1-model --output runs/native-full-frozen-baseline-v1
```

Commit exact source/config before running. Stop the advancing full native spectator first; no concurrent full-graph workers. Use `--resume runs/native-full-frozen-baseline-v1` with the same frozen runtime after interruption. A snapshot of source/lock/config plus immutable data/binary references permits independent checkout work during execution. Generated logs, checkpoints, results and reports stay ignored/local. Core rollout and recovery tests use a clearly labeled seven-neuron native numerical fixture without downloading MaleCNS.

Actual poker learning/control runners and Gates 3–6 remain pending. Implementing a 100 BB environment does not permit 100 BB learning before Gate 5. Whole Gate 2A is still awaiting a validated conventional teacher/corpus; its small exact-transfer/removal components have passed. Current baseline evaluation does not bypass those requirements.


## First full frozen baseline completed

The registered 256-hand / 32-pair-per-opponent run completed in 17.32 s with no weight changes. Mean BB/hand: random +0.7734375 (adjusted interval −0.515625 to +2.390625), station +2.25 (−2.125 to +6.3125), TAG +0.3671875 (+0.0234375 to +1.2890625), equity +1.046875 (+0.09375 to +2.4765625). This small development sample is a baseline only. Across 220 neural decisions, the model selected fold 149 times and all-in 71 times, with no other action. Action collapse and cue-only prior training rule out interpreting these returns as demonstrated poker learning or a Gate 3–5 pass. No policy or biological parameter was selected from these results.

Model SHA-256 9e0f94abdd1cce92fc93c16dbbe9597d99456b01c3d09076784d2e51f8cf0259; result manifest cebed8d2bdcfafa19572fcf09f33000220d54b923e4b0b5d5545c71431ba9522; source hash 0053df4112958f6c900c529b4109f99687824db78cc49b59958a79dd9dd9f8e1. Generated evidence: runs/native-full-frozen-baseline-v1, with local reports. The full dashboard was restored after the one full worker completed.

## Poker learning observer engineering

A separate training observer now wraps the unchanged native controller, accumulating the registered five-millisecond eligibility bins. Its returned counts, scores, masks and chosen actions are byte-identical to observer-free native decisions when weights are frozen. Terminal local chip reward, post-action local teacher advantage and the separately labeled bounded-edge surrogate are implemented. Targets cannot enter the controller's observation or decoder, and frozen/bio-plastic calls reject teacher targets. Complete neural/eligibility/baseline/RNG state restores exactly. This is numerical-fixture engineering evidence; no actual poker training/control run or learning gate has executed yet.

## Complete-hand learning mechanism

`learning.poker_rollout.poker_hand` now connects the verified observer to complete PokerKit hands. Exactly one seat learns; the other uses its original fixed policy and sees only its own observation. The native-selected action is committed before a teacher distribution is requested. Frozen rollouts accept no teacher, reward override or exploration and assert unchanged weights. Bio-plastic terminal reward records the real net chip result separately from a matched shuffled-reward override and uses the canonical button/nonbutton past-only baseline. Distillation records original targets and any explicitly seeded permutation within legal actions; this preserves target mass/entropy while breaking action-label association.

All four curricula reproduce the existing frozen evaluator's full settled hand and neural decision/count records on the numerical native fixture. Local terminal, local teacher and surrogate learning reproduce state-restored continuation exactly. Ten new integration checks cover these paths, actual teaching order, control bookkeeping and rejected input misuse. Full suite 133 passed / 2 isolated-oracle skips. This callable mechanism does not itself authorize an unvalidated teacher or issue a gate claim. The registered multi-seed experiment/control orchestrator and prerequisite certification remain subsequent work; no biological poker learning has been executed.

## Matched controls prepared

Frozen training arms may now use the same registered exploration temperature without receiving learning/teacher/reward inputs; evaluation still forces frozen weights and zero exploration. A deterministic topology null permutes postsynaptic endpoints while preserving every source degree, destination degree, sign, original weight and selected starting checkpoint weight. It starts from fresh dynamics, records a distinct graph hash and an explicit synthetic-control label, and cannot load under the original connectivity registration. Reward controls permute completed plastic-arm terminal returns independently within canonical position, preserving each position's reward multiset.

Two additional integration checks verify these invariants, deterministic reconstruction, unchanged original state, graph-binding rejection and matched exploration. Full suite 145 passed / 2 isolated-oracle skips. These are prepared mechanisms; multi-seed gated orchestration and actual biological poker runs remain pending.


## Terminal fine-tuning and frozen neural opponents

PokerLearningPlayer now supports terminal-local-eligibility with the distilled-connectome label: it retains the origin of transferred weights but accepts only the actual terminal chip reward. It rejects teacher targets and uses the same declared local rule, past-only position baseline and bounded existing edges as the strict terminal path. Actual fixture hands and complete checkpoint continuation are identical to that terminal mechanism, apart from the explicit mode label. This enables later fine-tuning; it is not evidence that poker distillation occurred.

FrozenNeuralOpponent.load loads a hash-pinned native snapshot. The opponent consumes only its own canonical observation, uses the actual deterministic legal native argmax, resets dynamics per hand and checks unchanged weights at hand boundaries. Complete opponent decisions and hashes are logged. The rollout rejects shared mutable weights with the learning player, preserving one learning seat per match. Native fixture matches and recovery pass; no concurrent full-graph opponent experiment has been run. General gated multi-seed orchestration remains subsequent work.


## Isolated frozen evaluation

Add `--disconnected` to the frozen `evaluate` command when evaluating an exported model. A minimal copied package includes the exact frozen inference code, PokerKit wrapper, curriculum and evaluator, with a separate verified native binary copy and numeric model. It excludes teacher code and disables Python bytecode caches. A fresh subprocess actively rejects teacher imports and reads of the original project's training files; both denials are checked before native inference. The graph is the only prepared-data link, and the existing Python environment supplies locked dependencies. The isolated source/config/model/graph/binary identities are checked again on resume.

The worker preserves the complete-hand append-only journal and atomic neural checkpoints, including interruption forwarding. CI verifies that all decision/journal bytes match ordinary native evaluation and exact recovery after seven of sixteen fixture hands. A separate 16-hand check with the actual exported circuit model also matched exactly while teacher access was denied. Both are frozen integration checks; neither is evidence of poker learning. No extra full-graph worker was launched for these checks.

```sh
.venv/bin/flyholdem evaluate --config configs/frozen_poker_evaluation.yaml --model runs/conditioning-full-confirm-v1-model --disconnected --output runs/disconnected-full-baseline
```

Resume by replacing `--output` with `--resume` naming the same outer run directory. The actual evaluation and report live under `evaluation/`; `isolation.json` pins runtime inputs and `isolation-result.json` records enforced denials. Use resource-aware scheduling before launching another full graph alongside the dashboard.


## Resumable complete training arms

The private experiment executor now runs complete paired-seat training schedules around PokerLearningPlayer. It binds the initial weights/RNG, actual plasticity and controller registrations, teacher identity, control reward array and opponent versions. Every settled hand is fsynced into a hash chain. Atomic checkpoints contain all native state, eligibility traces, past-only baselines and controller RNG, at five-minute intervals and graceful shutdown. Recovery from an older complete-hand boundary reexecutes and verifies any journal tail. The only retained selected checkpoint is the registered final training boundary; it is not chosen by poker profit.

Seven added native integration checks cover strict/local/surrogate/terminal-distilled learning, frozen and shuffled controls, exact state/journal recovery, missing control labels and changed configuration. These low-level fixture executions grant no teacher or biological gate authorization and have no public ungated training CLI. The top-level curriculum orchestrator must verify prerequisite certificates before calling them and combine their trained models with isolated evaluation and registered endpoints.


Synthetic topology controls now export a distinct prepared graph and an explicit shuffled-control mode, preserving their actual endpoint permutation and original scaled edge strengths. The exported null graph uses scale one because the registered global gain is already present in its stored weights. Loading its frozen model against the original connectome is rejected. An encoder-row shuffle retains the exact row multiset, gains and readouts. A native integration check proves exact reloaded count/score/action bytes and rejects original-graph substitution. This enables honest isolated control evaluation; it grants no learning result.
