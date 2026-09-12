# Prompt for Astra

Paste the following into a new Codex task running GPT-6 Astra at high or xhigh reasoning, with the folder containing `flyholdem_build_spec.md` as its workspace.

---

Build the FlyHoldem project described in `flyholdem_build_spec.md` end to end. Treat that document as the product specification and decision record. The result must be a reproducible, play-money-only experiment in which the MaleCNS v1.0 connectome simulation receives a poker player's information set and produces the five heads-up no-limit Hold'em action scores. Implement both registered learning modes: `bio-plastic`, which uses the declared dopamine-gated local plasticity mechanism, and `distilled-connectome`, which learns a stronger strategy from a conventional poker teacher and then acts with the teacher fully disconnected.

Work autonomously and persist through implementation, tests, numerical verification, the conditioning experiment, the poker curricula, the live/recorded dashboard, and the final documentation. Begin by inspecting the workspace, all applicable `AGENTS.md` files, the cited upstream repositories and licenses, and the current official APIs for the selected dependencies. Make routine implementation decisions yourself. Ask me only if a missing credential, unavailable licensed asset, destructive action, or a choice that would materially change the scientific claim prevents progress.

## Visual-first execution order

Treat the first deliverable as a visual demonstration, before full-connectome downloads, optimized neural kernels, teacher training, or long experiments. Build a thin, real end-to-end vertical slice using the deterministic fixture graph and a small PokerKit heads-up game loop. It must include:

- one documented command that starts the backend and web dashboard;
- a continuously playable or autoplaying two-player Hold'em table;
- a clearly labeled fixture-brain visualization with live neural activity;
- the five action scores, legal mask, selected action, cards, pot, stacks, and action history;
- visible reinforcement/plasticity events using the implemented fixture learning path;
- live WebSocket operation and deterministic replay from a checked-in compact example log;
- browser verification at a desktop viewport, with screenshots saved as review artifacts;
- explicit `FIXTURE / VISUAL PROTOTYPE / NOT A FULL-CONNECTOME RESULT` labeling.

Do not use a static mock for the core loop: table events, neural activity, action selection, and the dashboard must be connected. Fixture data is allowed and must be labeled. Do not begin the full MaleCNS or poker-training milestones until this visual gate runs successfully, its browser checks pass, and it is committed. Once the gate passes, continue through the remaining specification automatically.

## Progress persistence

At the beginning, create and push a working branch named `astra/visual-first` if GitHub authentication is available. Never commit directly generated connectome data, virtual environments, secrets, large run logs, or model checkpoints.

Maintain a tracked `PROGRESS.md` containing the current phase, completed acceptance checks, exact working commands, test results, important decisions, known failures, last stable commit, resumable experiment command, and next step. Update it whenever a meaningful unit of work completes and before starting a long-running command.

Create a verified Git checkpoint after each stable milestone. Run the relevant checks first, update `PROGRESS.md`, commit the coherent change, and push the branch. Use descriptive commits such as `feat: complete fixture visual demo` and `feat: add deterministic poker engine`. If pushing is temporarily unavailable, keep committing locally, record the push failure in `PROGRESS.md`, and continue; push the accumulated commits when access returns. Do not rewrite or squash previously pushed progress while the build is active.

For training processes, write append-only event logs, flush after every completed hand, save atomic resumable checkpoints at least every five minutes and on graceful shutdown, and retain the latest three plus the best registered validation checkpoint. These generated files stay outside Git in ignored run directories. Before each long experiment, commit the exact config and record the commit/config/source hashes and resume command in `PROGRESS.md`.

Create the tag `visual-demo-v0` after the visual-first gate passes and the tag `conditioning-v0` after the first valid conditioning gate. Push tags when remote access is available.

Use PokerKit as the canonical rules engine and expose the five-action abstraction in the spec. Use the official MaleCNS v1.0 data with exact hashes and an explicit import policy. You may adapt generic code and ideas from the MIT-licensed DOOMFLY repository, but preserve attribution, third-party notices, provenance, and a clean separation between upstream-derived and original code. Never commit the downloaded connectome or large generated artifacts.

After the visual-first gate, implement milestones M0 through M7 in their dependency order and enforce Gates 0 through 6. Do not skip a failed scientific gate to create a more impressive learning claim. If a scientific gate fails, keep the working system, preserve the negative result and controls, diagnose the failure, and continue with any engineering work that remains valid; do not fabricate a successful learning claim. Tune encoder, decoder, and plasticity parameters only on the designated controllability and conditioning tasks, then freeze preregistration artifacts before poker evaluation. Never select neurons, seeds, gains, or opponents based on poker profit.

The neural simulation must be the sole source of the fly's action scores. The game layer may translate abstract actions and mask illegal or duplicate actions, but it may not substitute a strategic fallback. No opponent hole cards, future cards, deck order, teacher labels, equity labels, or opponent internals may enter the fly observation during evaluation. Add explicit byte-level leakage tests.

Build a conventional self-play poker teacher over exactly the same canonical information states, legal masks, stack configurations, and five abstract actions. Use NFSP, Deep CFR, or another justified imperfect-information self-play method; validate it before using it as a teacher. Export a versioned, stratified corpus of teacher action distributions and optional advantages with no hidden opponent cards or future deck data. Distill first through teacher-advantage dopamine signals; if that cannot pass the registered transfer gate, add a clearly labeled surrogate-gradient path that adjusts only bounded parameters on preregistered existing connectome edges. Do not add an observation-to-action bypass or allow a trainable adapter/decoder to carry the core strategy. Prove with an integration test that deleting the teacher and corpus from the runtime environment does not change frozen fly decisions.

Keep a tiny fixture graph for CI, a clearly labeled mushroom-body circuit mode for iteration, and the full retained MaleCNS graph for the actual experiment. Implement a pure-Python numerical reference plus an optimized sparse kernel and verify them on small graphs, including refractory semantics. Every run must be deterministic from its recorded seeds, replayable, checkpointable, hash-locked to its sources/config/binaries, and accompanied by a machine-readable result and concise human report.

Build the dashboard as an honest scientific spectator experience. It must synchronize the poker table, exact encoded information, five neural action scores, legal mask and selected action, brain/population activity, dopamine and synaptic changes, and matched learning/control evidence. It must run against both a live WebSocket event stream and a recorded log. Label fixture/circuit/full mode, `bio-plastic`/`distilled-connectome`, teacher-connected training/teacher-disconnected evaluation, and development/confirmatory status prominently.

Use the command contract and repository layout from the spec unless a concrete technical constraint requires a documented deviation. Provide one-command fixture setup and demo, explicit checksum-verified MaleCNS setup, tests that do not require the full dataset, and resource-aware full-graph commands. Run the relevant tests and demos yourself. Do not stop at scaffolding, pseudocode, a plan, or mocked core behavior.

At completion, give me:

1. a concise statement of what works and which gates passed, failed, or remain pending;
2. exact commands for the fixture demo, data setup, training, evaluation, dashboard, and report generation;
3. test and benchmark results;
4. the location of the primary report, model card, run manifest, and any negative-result reports;
5. any remaining blockers that require hardware time or an external asset rather than more implementation.

The public wording must remain: “A simulated network using the wiring of one reconstructed male fruit-fly central nervous system controls an engineered heads-up no-limit Hold'em interface.” Do not call it a living fly, consciousness, faithful whole-brain emulation, or proven learner unless the corresponding preregistered evidence exists.

---

## Suggested Codex settings

- Model: GPT-6 Astra
- Reasoning: high for implementation; xhigh if asking it to complete the full graph, learning experiments, and dashboard in one task
- Workspace: a local Git repository on the machine that will run the full simulation
- Permissions: network access for official source/dependency downloads and write access to the project directory
- Hardware: 32 GB RAM recommended for comfortable full-graph work; 16 GB can be used cautiously for fixture/circuit work and one measured full-graph process; reserve tens of gigabytes of disk for data, logs, and checkpoints

The official OpenAI model guidance says Astra is intended for long, multistep software-engineering workflows and benefits from explicit instructions about follow-through, testing scope, and when to ask questions: https://developers.openai.com/api/docs/guides/latest-model
