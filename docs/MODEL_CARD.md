# FlyHoldem model card

**Version:** Native spectator and private heads-up interface; conditioning and small exact transfer confirmed. **Status:** development. **Money:** play chips only.

## What works

A synthetic fixture, the 4,505-neuron mushroom-body circuit and the full retained 166,700-neuron MaleCNS graph produce actions through five fixed neural spike readouts in genuine PokerKit heads-up Hold'em. Native topology contains 25,582,938 retained edges. Full controllability passed at uniform 0.25 synaptic scale (signed contact gain 0.06875), preserving topology and transmitter signs. Circuit retains its separate base-gain registration. Population mappings were selected only on annotations/connectivity/controllability, never poker profit.

The native live/recorded dashboard renders every retained neuron and shows actual action scores/activity. Anatomical soma coordinates and the explicitly schematic missing-coordinate grid are distinguished. The side-seated fly and abstract opponent use original illustrative action gestures, readable hand/community insets and balance-driven chip piles. Human play isolates each private information set, freezes an independent neural player, validates every action on the server and supports browser reload while the server remains running.

## Confirmed evidence and limits

Full local conditioning passed five independent seeds: 82.3% versus 47.2% frozen and 50.0% shuffled, with retention and erasure. Its mechanism is competing-readout suppression and partly silent legal tie selection. It establishes learned selection for two engineered cues only, not poker strategy. See CONDITIONING.md.

Local exact-cue transfer qualified in development but **failed** confirmation at 77.8%, below 80%. A separately registered, explicitly nonbiological direct-readout surrogate subsequently passed circuit confirmation: 94.5% versus 55.2% frozen and 52.2% shuffled-teacher. It changes only bounded multipliers on 10,450 existing registered KC-to-readout edges, uses actual native forward scores and adds no adapter/decoder/bypass. The backward approximation omits recurrent and threshold derivatives. Silent ties still contribute. All 128 actual exported-model decisions remained byte-identical after teacher/corpus deletion, and changed cue source was rejected. See TRANSFER.md.

The 25k/250k v1, visible-card v2, visible-equity v3 and Double DQN v4 full 20 BB NFSP candidates failed their complete held-out suite. V5 population training and the separately extracted V6 final Q-component mixture also failed the same suite. V7 public-stack potential training also failed the full suite. V8’s final Q-component mixture also failed. V9’s exact terminal fold-value correction produced identical paired-deal records and also failed; no full teacher is qualified. A separate tabular CFR reference passed confirmation against all four original opponents in the restricted 10 BB shove/fold subgame; it is qualified only for that subgame, not the full-hand teacher gate. No validated full-hand teacher/corpus or poker-trained student exists yet. Whole Gate 2A and poker Gates 3–6 remain pending. The full model shown in the dashboard and available for human play is the cue-conditioned model, **unvalidated for poker**.

## Reproducibility and intended use

Inspect the neural/game information boundary, reproduce registered experiments and play against the frozen development controller. Do not treat demo chips won, activity or weight movement as evidence of poker learning. Native numerical checks cover scalar Python/Brian2 parity and complete-state continuation. Experiment journals, atomic checkpoints and frozen runtime bundles preserve source/config/binary/environment identities. See RUNTIME.md and PROGRESS.md for exact commands and negative results. Models/data/logs remain local and ignored; uv.lock pins the environment.

The default fixture is a synthetic Euler prototype with schematic engineering roles, distinct from the native 0.1 ms solver and reconstructed cells. Human sessions are local single-user play-chip games, not a real-money or production service. Neither gesture animation nor surrogate optimization is claimed as faithful biological motor/learning dynamics.

## Public wording

“A simulated network using the wiring of one reconstructed male fruit-fly central nervous system controls an engineered heads-up no-limit Hold'em interface.”

This describes the native full-data controller, not the synthetic fixture. No living fly, consciousness, faithful whole-brain emulation or proven poker learner is claimed.


The first 256-hand full-model baseline completed with unchanged weights and only fold/all-in actions. Its small-sample positive returns are explicitly not a poker-learning claim. Four genuine PokerKit curriculum environments, isolated paired evaluation and the gated multi-seed learning/control runner are implemented. Only numerical-fixture orchestration has executed; actual full-graph poker experiments still require a passing full teacher/corpus and preceding curriculum gates. See CURRICULA.md.
