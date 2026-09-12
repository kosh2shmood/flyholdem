# Audited native training replay

The dashboard can replay a completed native curriculum training arm using the actual saved readout spikes, scores, sampled actions and PokerKit hands. It does not run a neural solver or query a teacher. A recording does not qualify a teacher, demonstrate poker learning or pass a curriculum gate.

From the repository root, after the corresponding gated experiment has produced a complete arm:

```sh
.venv/bin/python -m flyholdem.cli serve \
  --training-run runs/poker-bio-stage3-development/81001/plastic/training \
  --graph connectome_data/malecns_v1/prepared-full \
  --start-hand 0 --hands 16 --port 8768
```

Open `http://127.0.0.1:8768/`. The graph and learning labels come from the verified recording; omit `--mode`, `--model`, `--preregistration` and `--replay`. The arm must retain its matching curriculum parent at `<experiment>/<seed>/<arm>/training`. Use that arm's actual prepared control graph when replaying a shuffled-connectome condition.

`--start-hand` is zero-based. `--hands` selects up to 128 consecutive hands; the displayed cumulative return covers that selected segment. The service streams and audits the complete arm before exposing any selected events. Only the requested hand records stay in memory; matched shuffled-reward controls retain compact payoff/seat scalars. Startup still reads and verifies every hand, including those outside the display range, so work grows with log length. A future full-graph run must measure that startup time on its real records.

The audit checks parent/config and complete-child bindings, the matched schedule, weight continuity, teacher-target split/control requirements, seeded PokerKit trajectories and original scripted opponents. Each native decision must reproduce its visible observation, exact encoded hash, lossless spike bytes, rates/scores and continuous registered action RNG at the scheduled temperature. Frozen snapshot opponents bind their own visible-observation hashes, reported legal argmax and snapshot identity. Those opponent scores are checked as records; the viewer does not reconstruct unavailable historical opponent spike vectors or independently repeat historical learning.

The exact prepared graph and scaled graph identity must match the recording. Full/circuit views use hash-bound anatomical annotations; native numerical fixtures are explicitly synthetic. Graph buffers are built without allocating a native neural worker. Old training logs that retain only spike hashes are rejected rather than assigned guessed activity.

Both original player seats map to the same visual fly seat. The fly's cards and public board remain readable in the separate card strips. Opponent cards stay hidden until actual showdown revelation; folded opponent cards remain hidden. Private checkpoints, deck order, deal seeds and future outcomes are not sent in pre-settlement viewer events. The river subgame's fixed check/call setup is labeled separately from the displayed neural-action trail; its real history remains in the exact information-set inspector.

The interface labels recorded training, its original development/confirmation profile, matched arm, actual opponent, stack depth, learning mode and teacher connectivity. Local updates show recorded dopamine/eligibility metrics. Surrogate updates are explicitly nonbiological, with dopamine and eligibility marked as unused where appropriate. Recorded dopamine pulses contain population/total spike summaries, not cell-level pulse vectors; the UI says so and does not fabricate a pulse animation. These metrics describe recorded updates, not independently validated learning.

Live human play is disabled for a historical recording, and its API refuses to construct a player. The existing live fixture/native dashboards still support private human matches. `Fixture example` switches to the original checked-in visual replay; `Training record` returns to the historical arm.

## Verification

Sixteen integration tests cover both seats, all four curricula, every matched control arm with its exact graph, local and surrogate recordings, exact spike reconstruction, hidden/folded cards, action/settlement mapping, range and graph rejection, graph buffers, WebSocket events and live-player refusal. Native solver construction/advance and teacher queries are explicitly forbidden during replay tests. The complete suite passed 231 tests with two isolated-oracle skips in 166.84 seconds. JavaScript syntax checks passed.

Actual CUA browser inspection used two small native numerical fixture recordings: eight river hands for local terminal learning and eight for surrogate teaching. Desktop 1440×1080 and mobile 390×844 checks showed readable hand/community cards, correct stacks, separate method labels, disabled play and no horizontal overflow. These fixtures are engineering checks, not full-graph poker experiments. Generated logs remain local and ignored. No actual full-graph poker training exists yet; full Gate 2A and Gates 3–6 remain pending.


## Detailed-log memory

Native training now retains journal offsets and hashes in memory, loading detailed historical hands from disk only for checkpoint-tail comparison or inspection. Appended bytes, per-hand fsync and the original chain format are unchanged. Resume streams all rows, rejects incomplete final writes and rechecks indexed bytes when accessed. The viewer's full-arm audit similarly streams detailed records and releases unselected hands. A changed file during consumption is rejected.

A synthetic write-and-resume benchmark used 65,536-byte payloads per record. The 128-record log occupied 8,418,230 bytes and peaked at 286,128 bytes of traced Python allocations. The 1,024-record log occupied 67,348,222 bytes: indexed recording/resume peaked at 322,314 Python bytes and 29,229,056 process-RSS bytes, versus 68,535,459 Python bytes and 128,811,008 RSS bytes with resident records. The two large files had identical bytes and SHA e4c4e300323ea8a86f63d0ab75d3fc89a914e0e3c534ccc3ef1328031c65336c. These are storage measurements, not full neural training memory or poker evidence.

Tests additionally verify exact checkpoint-tail replay, byte equality, corrupt/truncated/reordered journals, changes after indexing or during streamed reads, and rejection of missing hands, changed payouts, discontinuous weights or wrong final weights outside the displayed range. The complete original hand/score/RNG/control audit remains required.
