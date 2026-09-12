# FlyHoldem: implementation-ready build specification

Date: 2026-09-12

## 1. Goal

Build a reproducible, play-money-only experiment in which a simulation constrained by the MaleCNS v1.0 fruit-fly connectome receives the information available to a poker player, chooses actions in heads-up no-limit Texas Hold'em, and changes a declared subset of existing synapses through dopamine-gated plasticity.

Ship two explicitly labeled training modes over the same poker, connectome, encoder, decoder, logging, evaluation, and visualization stack. `bio-plastic` uses only the local dopamine-gated rule. `distilled-connectome` may learn from a stronger conventional poker teacher, but the teacher must be absent at inference and the strategy must be stored in declared trainable parameters on existing connectome edges.

The public-facing experience should make the experiment fun to watch while the research layer makes it difficult to mistake activity, weight changes, or occasional wins for learning.

The defensible claim, before results exist, is:

> A simulated network using the wiring of one reconstructed male fruit-fly central nervous system controls an engineered heads-up no-limit Hold'em interface. Poker observations are converted to neural stimulation, selected neural activity is converted to legal poker actions, and an experimental dopamine-gated rule changes existing synapses.

Do not describe this as a living fly, a consciousness, a faithful whole-brain emulation, or a proven poker learner. Upgrade the claim only after the preregistered controls pass.

## 2. Product decision record

These defaults are fixed for the first build so implementation can begin without further product questions.

| Decision | MVP choice | Reason |
| --- | --- | --- |
| Game | Heads-up no-limit Texas Hold'em, cash-game semantics | Smallest standard NLH setting and a well-defined two-player zero-sum experiment |
| Stakes | 1/2 blinds, 20 BB effective stacks initially | Shorter hands and a learnable curriculum; extend to 100 BB after the core gate passes |
| Money | Play chips only | The system is a research/demo environment, not a real-money poker bot |
| Action space | Fold; check/call; raise half pot; raise pot; all-in | Five outputs are practical for a neural BCI and match the useful PettingZoo/RLCard abstraction |
| Poker engine | PokerKit as the canonical rules engine behind a small PettingZoo-style wrapper | Correct, inspectable NLH mechanics with room for richer observations and later bet sizing |
| Reference environment | PettingZoo `texas_holdem_no_limit_v6` | Use for smoke tests and action-space comparisons, not as the only source of game state |
| Connectome | MaleCNS v1.0, exact source files pinned by hashes | It is the dataset used by the current Doom experiment and includes the brain plus ventral nerve cord |
| Input | Symbolic state encoded as deterministic sparse neural stimulation | Auditable, fast, and does not make unvalidated claims about fly vision |
| Output | Five fixed, preregistered neural readout ensembles plus a legal-action mask | Neural activity chooses the action; the engine only prevents illegal actions |
| Learning | Reward-modulated eligibility traces on a declared set of existing mushroom-body synapses | Gives delayed chip outcomes a plausible three-factor learning mechanism without rewiring the graph |
| Training modes | `bio-plastic` and `distilled-connectome` | Preserves a strict biological experiment while also attempting a strategically capable version |
| Teacher | Locally trained conventional NFSP/Deep-CFR-style policy using the identical information and five-action abstraction | Provides action distributions and values without leaking hidden cards or remaining in the runtime player |
| UI | Web dashboard with poker table, neural activity, action decision, plasticity, and matched controls | The visualization should explain both the spectacle and the evidence |
| Reuse strategy | Extract/adapt the generic MaleCNS importer, simulator, checkpointing, and audit ideas from DOOMFLY under its MIT license; retain attribution and third-party notices | Starts from the most relevant current implementation while keeping poker-specific code clean |

## 3. What is being built

```text
PokerKit hand
    -> information-set observation (never opponent hole cards or future cards)
    -> deterministic feature encoder
    -> sparse stimulus applied to declared sensory/association populations
    -> MaleCNS graph simulation
    -> five fixed neural readout scores
    -> legal-action mask and seeded exploration
    -> poker action
    -> terminal chip result
    -> appetitive/aversive dopamine pulse
    -> eligibility-gated updates to existing declared synapses
    -> next hand, with neural memory and weights preserved

Every event -> append-only audit log -> live dashboard + offline evaluation
```

Run the same interface in three modes:

1. `fixture`: a tiny synthetic graph for unit tests and CI.
2. `circuit`: a mushroom-body-focused subgraph for rapid conditioning experiments. Label it as a subgraph in all outputs.
3. `full`: the complete retained MaleCNS v1.0 graph for the actual demonstration and confirmatory evaluation.

The reduced modes are development tools. Results from them must never be presented as full-connectome results.

## 4. Game environment

### 4.1 Canonical rules

Use PokerKit for dealing, blinds, betting, all-ins, pots, showdown, and settlement. Wrap it in an agent-environment interface with deterministic seeding, serialization, legal-action masks, and replayable action histories.

The initial game is heads-up, 1/2 blinds, 40-chip starting stacks (20 BB), alternating button, no rake, and a fresh stack each hand. After the full-hand learning gate passes, add a 100 BB configuration. Keep the underlying wrapper player-count-aware, but do not begin six-max training in the MVP.

### 4.2 Abstract actions

Expose exactly five neural actions:

| ID | Meaning | Engine translation |
| --- | --- | --- |
| 0 | Fold | Fold when facing any wager; remain legal when checking is available only if the configured rules permit open-folding |
| 1 | Check/call | Check for zero to call; otherwise call, capped by the remaining stack |
| 2 | Raise half pot | A raise sized from the pot after calling; clamp upward to the minimum legal raise and downward to all-in |
| 3 | Raise pot | A pot-sized raise after calling; clamp to legal bounds |
| 4 | All-in | Bet or raise to the acting player's full remaining stack |

Specify the exact pot-after-call formula in code and test it with golden hands. If two abstract actions translate to the same legal wager, keep the lowest action ID and mask the duplicates so a single engine action does not receive multiple neural labels.

### 4.3 Observation contract

The encoder may receive only the acting player's information set:

- two private hole cards;
- visible board cards;
- street;
- button/position;
- pot, own stack, opponent stack, current contribution, amount to call, minimum raise, maximum raise, and stack-to-pot ratio;
- five-bit legal-action mask;
- action history for the current hand, with actor, action type, and wager expressed in big blinds and pot fractions;
- hand/decision counters used for logging, never for policy input unless explicitly declared in a curriculum.

It must not receive opponent hole cards, undealt cards, deck order, showdown result before termination, opponent policy internals, hand equity calculated with hidden information, or labels from a teacher during evaluation.

Add an automated leakage test that mutates hidden opponent cards and future deck order while holding the visible information set constant; encoded neural input must remain byte-identical.

## 5. Neural interface

### 5.1 Connectome and dynamics

Pin the three official MaleCNS v1.0 tables used by DOOMFLY: neuron annotations, neurotransmitter predictions, and connectome weights. Store source URLs, byte sizes, SHA-256 hashes, import policy, and the resulting graph counts in a lock file. Refuse to train when a source, prepared graph, simulator binary, or configuration hash differs from the checkpoint manifest.

Begin with the same declared leaky integrate-and-fire approximation as DOOMFLY for comparability, implemented in a sparse native kernel with a small pure-Python reference. Treat neuron constants, transmitter signs, global gains, and direct stimulation as model assumptions. Validate the native solver against the reference on small graphs and refractory edge cases.

The simulator API must support:

- deterministic reset and seeding;
- stimulus schedules with exact virtual-time onset, duration, and dose;
- activity aggregation by cell ID, cell type, superclass, and neuropil;
- fixed-weight and plastic modes;
- atomic, hashed checkpoints containing every mutable state variable;
- fast replay from an append-only hand log;
- intervention tests that stimulate one population and trace reachable readouts.

### 5.2 Symbolic encoder

Use a deterministic, versioned population code. Do not render the table into pixels for the MVP.

Recommended first mapping:

- separate 52-channel sparse codes for hole cards and board cards;
- one-hot codes for street and position;
- overlapping radial-basis population codes for each continuous scalar;
- categorical sparse codes for the last eight actions, including actor and size bucket;
- one channel per legal action;
- a begin-decision pulse and a begin-hand pulse.

Map these channels through a fixed seeded sparse projection into a declared input population. Start with direct Kenyon-cell stimulation in `engineered-kc-v1` because it provides a clean conditioning test. Add `pn-kc-v1`, which stimulates projection neurons upstream of Kenyon cells, only after the MVP passes. The UI and run manifest must always display which encoder is active.

Store the channel-to-cell mapping as a versioned artifact. Generate it once from a public seed, check it for balanced fan-in and fan-out, and freeze it before any poker results are inspected.

Suggested decision timing, configurable but fixed within a registered experiment:

- 50 ms baseline;
- 300 ms state stimulus;
- 100 ms readout window;
- action commit;
- 50 ms inter-decision interval;
- 200 ms terminal reinforcement pulse after the hand settles.

### 5.3 Neural decoder

Produce one score for each of the five abstract actions from five disjoint neural ensembles. Candidate ensembles should be selected once using only annotation, connectivity, baseline responsiveness, and reachability from the encoder population. Do not select them by poker profit.

The selection command must emit a preregistration artifact containing cell IDs, classes, sides, selection criteria, graph paths, baseline rates, normalization constants, and the seed. Freeze this artifact before learning runs.

For each decision:

1. Convert ensemble spike counts in the readout window to baseline-normalized rates.
2. Convert the five rates to scores using a fixed, declared transform.
3. Apply the legal-action mask.
4. During training, sample with a seeded softmax temperature schedule.
5. During evaluation, freeze weights and use the registered deterministic or low-temperature policy.
6. If every legal score is silent or invalid, use check/call and log `decoder_fallback=true`.

No external poker policy may replace or override the neural scores in a core experiment. An external policy is allowed only as a labeled upper-bound control.

## 6. Learning modes

### 6.1 `bio-plastic`

Use an explicitly experimental three-factor rule on a preregistered set of existing Kenyon-cell-to-MBON synapses. Never create edges to improve performance.

For each eligible synapse, maintain a decaying eligibility trace derived from presynaptic and postsynaptic activity. At a reinforcement event, update the synapse by a bounded amount proportional to:

```text
reward prediction error * eligibility trace * learning rate
```

Use separate declared appetitive and aversive dopamine populations where the data supports their annotation. The initial scalar reward is the terminal net chip result in big blinds, clipped or transformed by a fixed `tanh` scale. Subtract a running baseline estimated only from past hands with the same curriculum and position. Log the raw result, normalized reward, baseline, dopamine dose, eligible synapse count, total absolute update, and clipped updates.

Initial configurable ranges to test before preregistration:

- eligibility decay: 2-10 seconds of virtual neural time;
- learning rate: logarithmic sweep around the DOOMFLY-adapted scale, selected on conditioning tasks rather than poker profit;
- weight bounds: 0.1-2.0 times each original connection strength;
- reward scale: `tanh(net_bb / scale_bb)` with `scale_bb` selected on the curriculum and frozen for confirmatory runs.

Early curricula may use teacher or equity shaping, but these signals must be separate fields, visibly labeled, disabled in the full-hand confirmatory experiment, and impossible to access during evaluation.

Weight movement is a mechanism check. It is not a learning result.

### 6.2 `distilled-connectome`

This mode exists to answer a different question: can a strategy learned by a stronger conventional poker system be transferred into the connectome-constrained network and then executed by that network alone?

Keep the encoder, action ensembles, legal mask, poker environment, and information boundary identical to `bio-plastic`. The preferred transfer sequence is:

1. Use teacher action probabilities or counterfactual action values as a training signal on generated information sets.
2. First attempt transfer through the same reward-modulated local rule, using the teacher advantage for the fly's sampled action as the dopamine signal.
3. If local plasticity cannot reach the registered fidelity gate, allow surrogate-gradient optimization of bounded multiplicative parameters on a preregistered subset of existing connectome edges.
4. Never add recurrent edges, inject teacher logits into the decoder, train an observation-to-action bypass, or place poker logic in the UI/game adapter.
5. Fine-tune the transferred policy with terminal chip returns against the registered opponent mixture and frozen historical snapshots.
6. Freeze all learning, remove the teacher process and teacher artifacts from the runtime path, and evaluate the connectome player alone.

If a trainable input adapter or output decoder is explored as an ablation, label it `adapter-policy-control`; it is not the core distilled-connectome result because the strategy could reside outside the connectome.

Report both the strict mode and distilled mode even if one fails. Never blend their results or describe surrogate-gradient updates as biological dopamine learning.

### 6.3 Poker teacher and distillation corpus

Build the teacher locally so the project does not depend on a commercial solver, scraping, or a third-party poker account. Use the same PokerKit wrapper, information-state schema, stack depths, action abstraction, and legal-action rules as the fly.

Recommended teacher ladder:

1. Exact or tabular strategies for the two-cue, shove/fold, and small river subgames where tractable.
2. A conventional PyTorch policy trained by neural fictitious self-play, Deep CFR, or a similarly appropriate imperfect-information self-play method for the five-action heads-up abstraction.
3. A population of frozen checkpoints and exploitative scripted opponents, rather than a single continuously moving opponent.

The teacher must first beat the fixed opponent suite and outperform the random/rule baselines on held-out paired deals. It does not need to be claimed as GTO. Record its algorithm, source commit, config, seeds, checkpoint hash, action abstraction, and evaluation report.

Generate a versioned distillation corpus stratified across street, position, hand-strength bucket, stack-to-pot ratio, amount-to-call bucket, legal-action pattern, and action-history shape. Each record contains only:

- canonical acting-player information-state ID;
- visible observation and action history;
- legal-action mask;
- teacher probability for each legal abstract action;
- optional teacher action values or advantages;
- teacher/config/checkpoint hashes and generation seed.

Do not store opponent hole cards, future cards, deck order, or raw simulator state. For repeated samples of the same canonical information state, require identical teacher targets within numerical tolerance. Split train/validation/test by canonical information-state ID, not by row, so duplicated states cannot cross splits.

Distillation objectives may include legal-action cross-entropy/KL divergence, value-weighted imitation, and teacher-advantage dopamine pulses. Poker-profit fine-tuning follows distillation. At evaluation, the teacher executable and corpus must be unavailable to the runtime process; an integration test should prove that removing them does not change fly actions for a frozen checkpoint.

Call the result strategically learned only if it achieves all of the following:

- held-out policy fidelity better than the initial/frozen connectome;
- positive paired performance improvement against unseen fixed opponents;
- sensible response curves for position, hand strength, price, and aggression;
- persistence with the teacher disconnected;
- loss or material reduction of the improvement when trained edge parameters are restored to their initial values;
- no observation leakage or policy bypass.

## 7. Curriculum and gates

Do not start with unrestricted 100 BB poker. Each stage unlocks the next.

### Gate 0: software and rules

- Golden tests cover blinds, button order, all abstract action translations, minimum raises, all-ins, split/side pots even though the MVP is heads-up, showdown, ties, and deterministic replay.
- Random-vs-random produces symmetric returns under seat swapping within sampling error.
- Hidden-card leakage test passes.
- The fixture graph runs in CI without downloading MaleCNS.

### Gate 1: neural controllability

- Every input channel activates a nonempty declared input set.
- Each action ensemble has a path from the input population in the prepared graph.
- Direct controlled stimuli can make each legal action win the decoder at least 95% of trials without changing the mapping.
- Black/no-input and shuffled-input baselines are recorded.

### Gate 2: elementary conditioning

Train a two-cue/two-action task with immediate positive and negative reinforcement, unrelated to poker. Require held-out performance above the frozen-weight and shuffled-reward controls across independent seeds. Verify retention after a no-training interval and loss or reduction of the effect after restoring initial weights.

This gate answers the prerequisite question: can the chosen encoder, plastic synapses, and decoder express learned action selection at all?

### Gate 2A: teacher and transfer integrity

- The conventional teacher beats the registered random and rule-based baselines on held-out, seat-swapped paired deals.
- Teacher targets are functions of canonical information states and pass the hidden-card/future-deck invariance test.
- Corpus splits contain no overlapping canonical information-state IDs.
- A small exact/tabular curriculum can be distilled into the fixture or circuit player above the frozen-network baseline.
- Removing all teacher files and processes from the inference environment leaves a frozen fly checkpoint's decisions byte-for-byte reproducible.

### Gate 3: preflop shove/fold NLH

Use genuine heads-up NLH dealing and settlement at 10 BB, but restrict decisions to fold or all-in against fixed opponent policies. Curriculum labels or exact equity may shape training. Evaluation uses only visible cards/state and terminal chip reward.

Require a monotonic relationship between hand strength buckets and shove frequency, improved paired return versus the same network with frozen weights, and generalization to held-out deals and at least one held-out opponent.

### Gate 4: single-street river poker

Deal to the river from controlled ranges and allow fold, check/call, half-pot, pot, and all-in. This introduces bluffing, value betting, and wager sizing while keeping the horizon short. Train against a mixture of frozen scripted opponents.

Require better paired return than frozen, shuffled-reward, and shuffled-connectome controls, without action collapse.

### Gate 5: full-hand 20 BB heads-up NLH

Use all four streets, the five-action abstraction, terminal chip reward only, and a training mixture of fixed opponents plus frozen snapshots of earlier neural policies. Do not let both seats update during a single confirmatory match.

### Gate 6: 100 BB and self-play

Only after Gate 5 passes, increase stacks to 100 BB, lengthen eligibility/credit assignment as needed based on conditioning evidence, and add controlled self-play. Keep a frozen-opponent evaluation suite so improvement cannot be explained solely by co-adaptation.

## 8. Opponents and evaluation

Create a deterministic suite with seat-swapped, paired deals:

- uniform random legal action;
- check/call-heavy calling station;
- tight-aggressive rule bot using only its own information set;
- equity-bucket bot that uses Monte Carlo equity available to that bot, never exposed to the fly;
- frozen snapshots of earlier fly policies;
- a conventional learned policy as a labeled reference upper bound, not part of the fly controller.

Use the conventional learned policy as the teacher in `distilled-connectome` training only after its own held-out evaluation passes. The teacher may never act for the fly during evaluation or live public play.

Use separate development and confirmatory seeds. Lock confirmatory seeds and opponent parameters before running them. Evaluation always freezes learning and exploration unless the registered protocol says otherwise.

Primary endpoint:

- paired difference in big blinds per 100 hands between trained and initial/frozen weights on identical deals, seats, and opponents, with a bootstrap 95% confidence interval across independent training seeds.

Secondary endpoints:

- return by opponent, seat, street, stack-to-pot ratio, and hand-strength bucket;
- action distribution and illegal/duplicate/fallback rate;
- showdown rate, fold-to-bet, aggression factor, and bet-size distribution;
- policy stability and retention;
- divergence of activity and weights from baseline;
- wall-clock and neural-time throughput.

Minimum confirmatory controls:

1. learning on;
2. weights frozen;
3. reward labels shuffled within matched hands;
4. encoder channels shuffled after preregistration;
5. degree/weight-preserving connectome shuffle where computationally practical;
6. restored initial weights after successful training;
7. conventional small policy with the same observation/action abstraction as an upper bound.

Do not announce learning unless the trained condition improves the primary endpoint with its confidence interval above zero on held-out seeds/opponents, survives the reward-shuffle and frozen controls, retains the effect, and loses or materially reduces it when learned weights are erased. Publish negative results.

## 9. Visualization and spectator experience

The dashboard has five synchronized panels:

### Poker table

Show hole cards only for the fly and only when appropriate for the viewer mode, board, pot, stacks, position, action history, current legal actions, chosen action, and the opponent's revealed cards at showdown. Provide a delayed/public mode if hidden opponent cards are ever shown to researchers.

### Decision scope

Show the exact observation channels sent to the encoder, five raw neural scores, baseline-normalized scores, legal mask, temperature, sampled/selected action, and any fallback. This is the most important explanatory panel.

### Brain

Use a WebGL point-cloud/centroid view for the whole graph, with optional skeleton detail on selected populations. Color current activity by cell class/neuropil; highlight encoder cells, plastic synapses, dopamine populations, and the five readout ensembles. Avoid implying that all 25.6 million edges are being rendered live.

### Plasticity

Show delivered positive/negative reinforcement pulses, eligibility magnitude, number of changed synapses, weight-ratio histogram, largest changes, and a comparison with the initial checkpoint.

### Evidence

Show paired learning curves with confidence bands, returns by opponent, action frequencies, conditioning-gate status, current run manifest/hash, and side-by-side learning-on versus frozen or shuffled controls. Label development runs, confirmatory runs, and failed gates plainly.

Transport live events from the Python process over WebSocket. Store the canonical stream as versioned JSONL or Parquet and make the UI a consumer of recorded truth, not a second source of experimental logic.

## 10. Repository layout

```text
flyholdem/
  AGENTS.md
  README.md
  LICENSE
  THIRD_PARTY.md
  pyproject.toml
  Makefile
  .env.example
  configs/
    fixture.yaml
    conditioning.yaml
    shove_fold.yaml
    river.yaml
    hu_20bb.yaml
    hu_100bb.yaml
  data-provenance/
    malecns_v1/
      source.lock.json
      IMPORT_POLICY.md
  src/flyholdem/
    cli.py
    poker/
      engine.py
      actions.py
      observation.py
      opponents.py
      replay.py
    connectome/
      registry.py
      import_malecns.py
      prepare.py
      audit.py
    neural/
      reference.py
      kernel/
      simulator.py
      checkpoint.py
    interface/
      encoder.py
      decoder.py
      preregister.py
    learning/
      eligibility.py
      dopamine.py
      curriculum.py
    teacher/
      infoset.py
      policy.py
      self_play.py
      corpus.py
      distill.py
    experiments/
      runner.py
      controls.py
      evaluate.py
      reports.py
    server/
      app.py
      events.py
  ui/
    package.json
    src/
  tests/
    unit/
    integration/
    numerical/
    golden/
  docs/
    CLAIMS.md
    PROTOCOL.md
    MODEL_CARD.md
    UI.md
  outputs/                 # ignored except compact example reports
```

## 11. Command contract

The completed project should expose stable commands resembling:

```bash
make setup
make test
make fetch-malecns
make prepare-malecns
make audit-malecns
make build-kernel

flyholdem preregister --config configs/conditioning.yaml
flyholdem train --config configs/conditioning.yaml
flyholdem evaluate --run RUN_ID --controls frozen,shuffled-reward
flyholdem teacher train --config configs/teacher_hu_20bb.yaml
flyholdem teacher evaluate --run TEACHER_RUN_ID
flyholdem teacher export-corpus --run TEACHER_RUN_ID
flyholdem distill --mode distilled-connectome --teacher TEACHER_RUN_ID
flyholdem train --config configs/shove_fold.yaml
flyholdem train --config configs/hu_20bb.yaml --resume RUN_ID
flyholdem serve --run RUN_ID --port 8766
flyholdem report --run RUN_ID
```

Every training/evaluation command writes a run directory containing:

- immutable config and preregistration;
- git commit and dirty-tree status;
- Python/Node/compiler/platform versions;
- source, graph, binary, encoder, decoder, and initial-checkpoint hashes;
- all RNG seeds and opponent versions;
- append-only event data;
- atomic checkpoints with two retained generations;
- a compact machine-readable result and a human-readable report.

## 12. Implementation milestones

### M0 — repository and provenance

Create the package, CI, configs, docs, source registry, license attribution, and fixture graph. Port only generic, understood pieces from DOOMFLY. Exit when fixture tests and data-lock tests pass.

### M1 — poker engine

Implement PokerKit wrapper, actions, observation, deterministic opponents, replay, golden hands, and leakage tests. Exit when Gate 0 passes.

### M2 — connectome runtime

Implement/import the pinned MaleCNS graph, reference solver, native sparse kernel, audit, checkpoint, and interventions. Exit when graph counts are reproducible and small-graph numerical tests pass.

### M3 — neural BCI

Implement and preregister encoder/readout mappings, decision timing, legal masking, event schema, and replay. Exit when Gate 1 passes without poker-result-based neuron selection.

### M4 — plasticity, teacher, and conditioning

Implement eligibility traces, dopamine events, bounded updates, matched controls, retention/erasure checks, the conventional teacher, canonical information states, leak-free corpus export, both transfer paths, and reports. Tune biological parameters only on conditioning. Exit when Gates 2 and 2A pass or publish structured negative results explaining the failures.

### M5 — poker curricula

Implement shove/fold, river, 20 BB, and their opponent/evaluation suites in sequence. Never skip a failed gate. Keep all shaping separate and disable it for final full-hand evaluation.

### M6 — dashboard

Build the synchronized table, decision, brain, plasticity, and evidence panels. It must support both live and recorded runs and visually distinguish fixture/circuit/full modes and development/confirmatory status.

### M7 — packaging

Add Docker or reproducible local setup, smoke data, one-command demo, operational docs, model card, claims page, and a compact public result bundle that does not redistribute third-party data incorrectly.

## 13. Definition of done for the first public release

- A new user can run the fixture demo from a clean checkout with one documented command.
- A properly provisioned machine can fetch, hash, prepare, audit, and run MaleCNS v1.0 with no manual data edits.
- A full hand can be replayed exactly from logged events and seeds.
- The neural simulation is the sole source of the fly's five action scores.
- The legal-action adapter is deterministic and fully tested.
- Plasticity changes only declared existing edges and complete state survives checkpoint/resume.
- Both `bio-plastic` and `distilled-connectome` modes exist and are impossible to confuse in configs, logs, UI, or reports.
- The trained conventional teacher is absent from the fly's inference dependency graph and runtime filesystem test.
- Gates 0-2A pass; poker gates are shown as passed, failed, or pending rather than implied.
- At least frozen-weight and shuffled-reward controls can run on identical deals.
- The dashboard explains one decision end to end and shows the current evidence state.
- The README, model card, UI labels, and public copy use the defensible claim language.

## 14. Practical compute envelope

Plan for Python 3.11, a C++17 compiler, Node.js for the dashboard, at least 32 GB of RAM for comfortable full-graph work, and tens of gigabytes of free disk for source data, prepared graphs, checkpoints, and audit logs. A GPU is optional unless a conventional learned-policy baseline or GPU kernel is added. Do not run multiple full-graph jobs concurrently until memory and throughput have been measured.

Keep unit tests data-free. Downloads must be explicit, resumable, checksum-verified, and excluded from Git. Use the circuit mode for iteration and reserve the full graph for preregistered smoke and confirmatory runs.

## 15. Failure modes to guard against

- Mistaking wins, nonzero actions, changing weights, or persistent membrane state for learning.
- Choosing neurons, gains, seeds, or opponents after seeing poker profit.
- Leaking opponent cards, deck order, or teacher/equity labels into evaluation.
- Letting the legal-action wrapper or a fallback policy make strategic choices.
- Calling a cropped circuit a whole-brain run.
- Comparing different deals, seats, stack depths, or opponents and calling the difference learning.
- Training both players while using their changing head-to-head return as the only metric.
- Rendering impressive neural activity while the readout has no causal path from the input.
- Logging only aggregates, making exact replay and audit impossible.
- Redistributing MaleCNS or third-party assets without their license and attribution requirements.

## 16. Sources selected for implementation

- [DOOMFLY repository](https://github.com/nftechie/doomfly) — the closest current codebase; MIT-licensed original code, MaleCNS importer/simulator, audit, checkpointing, UI, and negative results.
- [DOOMFLY live training protocol](https://github.com/nftechie/doomfly/blob/main/docs/doom-live-training.md) — exact example of declaring stimulation, plasticity, checkpoints, and validation limits.
- [Official MaleCNS v1.0 downloads](https://male-cns.janelia.org/download/) — authoritative data files and CC-BY status.
- [Google Research MaleCNS announcement](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/) — project context.
- [PokerKit simulation documentation](https://pokerkit.readthedocs.io/en/stable/simulation.html) — canonical NLH rules engine.
- [PettingZoo no-limit Hold'em environment](https://pettingzoo.farama.org/_modules/pettingzoo/classic/rlcard_envs/texas_holdem_no_limit/) — reference five-action abstraction and legal mask.
- [FlyVis repository](https://github.com/TuragaLab/flyvis) — example of a connectome-constrained, task-optimized neural model with explicit scientific scope.
- [DOOMFLY/Pong audit discussion](https://prismix.dev/news/60aae45d859b) — recent failure analysis that motivates path, ablation, and learning-control gates.
