# Protocol and decision record

## V0 scope

The fixture is a synthetic, explicitly engineered 126-node graph: 96 KC-like input nodes, five disjoint four-node action readouts, and ten synthetic positive/negative dopamine indicator nodes. All 1,920 KC→readout edges exist from initialization. The 120 drawn edges are a fixed sample, not the full topology.

The graph and channel mapping seed is 1729, fixed independently of poker results. No cell, edge, gain, opponent or run seed has been selected by poker profit. V0 is a transport/mechanism demonstration, not a conditioning or learning experiment. Initial default demo seed is 20260912. `configs/fixture.yaml` records V0's constants; V0 does not yet load general training configs.

## Visible information

PokerKit 0.7.5 owns the rules. The engine remaps its HU seat 0 (big blind) and seat 1 (button/small blind) into physical viewer seats. Each hand resets play-chip stacks to 40, alternates button, uses 1/2 blinds and no rake. An isolated Python Random(seed) shuffles a complete explicit 52-card deck before dealing. PokerKit's internally randomized deck is replaced before any card is dealt.

The encoder receives only a dictionary constructed by an explicit allowlist: own hole cards, public board, street, relative position, visible betting scalars, legal mask and relative-actor action history. Physical seat IDs, hand counters, RNG/deck state, opponent hidden cards and opponent policy objects are absent. The logged seed belongs to replay provenance, not the encoder. Mutation tests alter hidden hole cards, undealt order and artificial teacher/internal fields and compare canonical observation bytes and little-endian encoded bytes.

Raise target = own street contribution + additional call + ceil(fraction × (pot + additional call)), clamped to PokerKit minimum/maximum. Fraction is 0.5 or 1.0. Duplicate wagers retain the lowest abstract ID; open folds are disabled. Fold, call and raises use PokerKit methods exclusively.

## V0 neural assumptions

Python/NumPy LIF uses 1 ms Euler ticks, membrane tau 20 ms, synaptic tau 5 ms, reset 0, threshold 1, two complete refractory ticks, and next-tick synaptic delivery. Incoming currents during refractory are discarded. **This coarse fixture solver is not the future DOOMFLY-comparable full-graph solver.** The later M2 reference/native solver must separately verify the registered dynamics and precise refractory schedule.

Each decision has 50 ms baseline, 300 ms stimulus, 100 ms readout under the same maintained stimulus, and 50 ms inter-decision decay. The five scores are mean readout spike rates divided by a fixed 100 Hz scale. Zero baseline rate is a declared fixture property. Evaluation chooses deterministic legal argmax with lowest-index ties; the visual development demo samples from masked softmax at temperature 0.65. There is no check/call or strategic fallback. This resolves the specification's silence fallback clause in favor of the explicit execution request against strategy substitution. Invalid scores fail closed rather than invoking an opponent/control policy.

Fixed symbolic channels use card codes, four street channels, relative position, eight five-center RBF scalar codes, last-eight-action categories, legal bits and decision/hand-start indicators. Two input cells per channel are mapped with a balanced seeded permutation. The hand-start indicator is currently defined by an empty visible history; a first-decision-per-agent pulse is deferred to the preregistered interface milestone.

## Local fixture plasticity

Presynaptic traces decay over 20 ms. Existing-edge eligibility decays over 5,000 ms and accumulates 0.001 × presynaptic trace × postsynaptic spike each tick. Terminal reward is tanh(net big blinds / 10). The running baseline is computed only from past hands at the same position. The RPE multiplies eligibility, learning rate 0.015 and original edge strength. The updated strength is clipped to 0.1–2.0 times initial strength; no edges are added. Positive/negative synthetic dopamine cells receive a 200 ms dose following the update. This scalar gate is an experimental engineered mechanism, not a measured dopamine circuit.

Eligibility and membrane state persist across hands. Reports distinguish eligibility, changed edges and behavior. V0 only establishes nonzero mechanism changes. It provides no matched learning controls or statistically supported performance claim.

## Event and UI contract

Canonical sorted JSON events form a SHA-256 chain. The same stream drives live WebSocket and recorded playback. A decision stores its pre-action visible information and post-action table; the viewer labels the decision as pre-action information. The opponent is a conventional calling-station control and never supplies fly scores. Its hidden observation is not logged. Fly cards are preserved for display even when PokerKit mucks a losing hand. Opponent cards appear only if revealed at showdown.

The neural activity view reports the latest measurement window; reinforcement replaces it with the actual dopamine-period spike counts. Controls and conditioning labels remain pending. Display pause does not pause the simulation. Live files are exclusively created in unique run directories, append-only and fsynced after each terminal reinforcement; mutable full-state checkpoints are subsequent runtime work.

## Reproducibility across runtime architectures

Recorded event playback preserves original bytes and hash chains on every platform. Re-simulation is required to be byte-identical on the same locked runtime; the test regenerates two complete streams and compares bytes. ARM/macOS and x86/Linux NumPy transcendental implementations can differ in the last floating-point bit, which also changes encoded-input and event hashes. The checked-in fixture is therefore additionally compared across platforms with exact cards/actions/spike counts/masks and absolute 1e-12 tolerance for floating diagnostics, with each chain independently verified. No tolerance applies to hidden-information mutation tests within a runtime. Training checkpoint resume must reject different recorded runtime/binary identities rather than claiming cross-architecture bit identity.

## M1 rules completion

The underlying hand adapter is aware of 2–10 players for settlement testing, while the registered neural observation explicitly rejects non-heads-up use. Golden cases now cover showdown, royal-board ties, three-way main/side pots, uncalled excess and short all-ins that do not reopen betting. Private hand serialization includes the initial deck and action sequence and verifies the reconstructed public-state hash. This payload is checkpoint-only and forbidden in observations, viewer events and teacher corpus rows.

Fixed software-validation seed/config: `configs/software_gate.yaml`. Run `uv run python -m flyholdem.experiments.software_gate`. The endpoint is independent random-play return within a predeclared three-standard-error bound plus exact paired physical-seat symmetry. This checks the rules adapter; it is not a learned-policy evaluation.
