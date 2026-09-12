# Exact PokerKit reuse in conventional training

Future external-regret teacher runs reuse PokerKit's evaluated hands within each counterfactual traversal. This reduces repeated work on the same cards across betting branches. PokerKit still performs all dealing, legal actions, hand comparison and settlement; no alternative rules engine or strategic policy is introduced.

The helper copies every mutable game field while sharing the locked PokerKit 0.7.5 immutable standard-card values. It gives only the copied training tree a bounded 256-entry cache of `StandardHighHand.from_game` results, keyed by the exact ordered hole-card and board tuples. Results retain the original `StandardHighHand` type and selected card combination. Invalid hands still use PokerKit's existing exception path. The cache is cleared when the traversal exits, including on failure. The original game and public evaluation engine are untouched.

This uses the installed package's immutable `Card`/hand contracts and [PokerKit's documented hand-evaluation API](https://pokerkit.readthedocs.io/en/stable/evaluation.html). It does not approximate hand values, reorder equivalent cards or expose private rules-engine data to a policy observation. Future numeric policy manifests bind the helper's source along with their existing runtime identity.

## Measured effect

A warm-cache prototype comparison used the first 64 registered training traversals, with unchanged initialization, sampling, actual PokerKit deals and original four-opponent mixture. Timing order was ordinary / candidate / candidate / ordinary while the separate V11 process remained active. Both ordinary trials took 8.76 and 8.66 seconds; both candidate trials took 4.27 and 4.28 seconds, approximately a twofold speedup on this small workload. Every trial traversed 7,989 nodes and produced identical complete-row and numeric-table fingerprints. This is not a measurement of the duration of a future long run.

The earlier immutable-card-copy-only candidate improved the same workload by about nine percent; the larger improvement comes from avoiding repeated PokerKit combination searches. No poker-profit result was used to choose this execution change.

The final helper then reexecuted the first 128 original V10/V11 training records. Every complete value matched, including the trace of actions, counterfactual values, regrets and average-strategy updates. There were 2,383 information sets and the original prefix journal head was `20cb181d8bf6e1130bbcf6082ae40a5e6908f46716aab17cef5f9b8f266dd9a6`; original prefix file SHA `6fe1de304408cfa25103e6390697eac4b053e92af78455f59d5ec36e5bc54395`. No held-out evaluation was run for this performance check.

## Verification and recovery

Differential tests cover both seats against all four original opponents, fresh and restored decks, complete traversal records and every numeric table array. Separate tests fork every legal action at each street, compare subsequent observations/actions/settlement, and verify that parent and sibling state remain independent. Explicit all-in aces-versus-kings and royal-board split cases reproduce PokerKit. Cache tests verify original result types and clearing between traversals. The 22 focused rules, traversal, source-binding, export and recovery checks passed in 11.99 seconds. The complete suite passed 237 tests with two isolated-oracle skips in 140.47 seconds.

Existing V10 and V11 snapshots are immutable. V11 continues through its originally registered source to 100,000 traversals; this optimization must not be substituted into its resume command. Future runs must commit and freeze their own source/config/binaries before training. The optimization is an engineering result, with no new teacher qualification, poker-learning claim or changed scientific gate.

V11 policy export, evaluation and any later corpus/gate work must retain its exact policy implementation files. A later experiment runtime can combine updated orchestration with those explicitly hash-bound frozen policy modules; it must record that composed source identity and must not weaken the policy loader's source checks.
