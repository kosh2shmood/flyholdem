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
