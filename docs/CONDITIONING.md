# Elementary conditioning protocol v1

Status: full retained graph passed Gate 2 on five independent confirmation seeds. The learned mechanism is engineered cue discrimination by readout suppression; poker learning is untested. Gate 1's full model uses all 166,700 neurons and all 25,582,938 retained edges at the previously frozen global synaptic scale 0.25. No poker result is involved in this protocol.

## Registered experiment

`configs/conditioning.yaml` fixes the candidate order, development and confirmation seeds, trial counts, controls and success criterion before execution. Two disjoint groups of 64 KCs are selected from cells with existing paths to both frozen action-0 and action-1 ensembles. Rank by the smaller original connection strength to either ensemble, select the top 128, and partition with seed 64001. Cue identity determines target 0 or 1. Both cues receive the same fixed gain and independent ±15% per-cell amplitude noise. This is an engineered non-poker conditioning input, distinct from the fixed 237-channel poker population code.

Only existing annotated KC-to-MBON connections can change. Ratios remain between 0.1 and 2.0 of their original signed strengths. The native LIF solver uses 0.1 ms integration; local eligibility aggregates spikes in registered 5 ms bins. Presynaptic traces decay over 20 ms and eligibility over 5 s; the coincidence coefficient is 0.001. The ordered learning rates are 0.03, 0.1 and 0.3. Five-millisecond observation boundaries can change lazy-state floating rounding slightly relative to one uninterrupted call; the fixed bin schedule is part of run identity. Numerical tests verify equal small-graph spike counts and a 1e-4 voltage/conductance tolerance against the uninterrupted call.

Each trial resets membrane, conductance, refractory state, delayed queue and eligibility, preserving weights. Stimulation follows the frozen 50/300/100 ms baseline/stimulus/readout timing. Masked neural argmax over actions 0 and 1 is the entire policy. Correct/incorrect decisions earn +1/−1, with a past-only running reward mean across both cues subtracted. Reward times local eligibility changes weights. A separate 200 ms native stimulation pulse targets annotated PAM or PPL1 cells. These labels are engineering valence proxies; the scalar gate is not a claim that measured dopamine receptor physiology was simulated. Pulses do not introduce a target-specific neural drive.

The plastic arm, frozen arm and shuffled-reward arm share initial weights, cue schedule and stimulus seeds. The shuffled arm receives an independently permuted copy of the plastic arm's actual earned reward sequence, preserving its reward multiset. Evaluation disables updates, uses independent stimulus seeds, and has balanced cue counts. The same evaluation stimuli are replayed before training, after training, after a 5 s no-training interval, and after restoring initial weights. Erased and frozen decisions must exactly equal initial decisions.

Development uses three seeds, 400 training trials per arm and 32 held-out trials per cue. Stop at the first candidate reaching mean accuracy 0.8, every seed at least 0.7, mean improvement at least 0.15 over both controls and erased weights, and retention loss at most 0.05. Confirmation requires that saved development result and uses five new seeds with 64 held-out trials per cue. It additionally requires one-sided paired-seed sign-flip p ≤ 0.05 against each comparator. Report each seed, paired bootstrap intervals and every failed candidate. These comparisons establish only this conditioning task, never poker skill.

## Recovery and provenance

Each run locks protocol, source, environment, binary, graph, cue, readout and plastic-edge identities. A hash-chained append journal is flushed/fsynced after each complete trial or state transition. Atomic checkpoints contain all native state, weights, eligibility and past-reward baselines at least every 5 min and on graceful shutdown. Stimulus RNG is independently derived from recorded seed/phase/trial; cue schedules and shuffled rewards are reconstructed deterministically. Keep latest three plus best held-out conditioning checkpoint; this best snapshot does not select seeds for poker.

Resume verifies checkpoint/journal alignment, skips the saved prefix and exactly re-executes any logged tail. Divergence is an error. An unexpected failure during an incomplete operation preserves the preceding valid checkpoint. Truncated/corrupt journals are detected rather than silently edited.

## Commands

Run from the repository, using the locked app environment with data tools and the verified native kernel:

```sh
.venv/bin/python -m flyholdem.experiments.conditioning --profile smoke --learning-rate 0.03 --output runs/conditioning-full-smoke-v1
.venv/bin/python -m flyholdem.experiments.conditioning --profile development --learning-rate 0.03 --output runs/conditioning-full-lr003-v1
```

Subsequent registered candidates use `0.1` / `0.3` and distinct output paths, only if the previous development candidate fails. Confirmation adds `--profile confirmatory --development-reference runs/conditioning-full-lr003-v1/result.json` with the selected learning rate and a new output path. Resume adds `--resume` to the original command without changing source/config/environment. `--stop-after N` is available for deterministic recovery verification; remove it when resuming.

No `conditioning-v0` tag until a valid confirmation passes.

## Smoke and recovery evidence

The 40-trial smoke profile completed in 17.63 s with peak RSS 722,190,336 bytes. It selected 61,210 existing KC-to-MBON edges; plastic post-test accuracy was 0.5, frozen 0.5 and shuffled reward 0.625 on only eight held-out trials. These are engineering smoke measurements, not a conditioning claim or a tuning selection. The original weights restored every initial decision exactly.

A second run stopped after operation 30, resumed, and produced the same 173-operation / 170,537-byte journal as the uninterrupted run (SHA-256 bb9ce424289968bf162de7136dc53937c4f0c2b3f0ae1a57fc7e3d8a72b3cd5a). A further recovery from the earlier best checkpoint exactly re-executed the already logged tail without changing a byte. Full neural arrays, eligibility and past-reward means were restored. Compact evidence is in docs/review/conditioning; complete runs remain ignored.

## Development candidate 0.03 — did not qualify

Commit 067d35d executed 4,767 operations in 508.79 s at peak RSS 725,778,432 bytes. Held-out plastic accuracy across seeds 65101/65102/65103 was 0.5000/0.53125/0.546875 (mean 0.52604); frozen 0.484375/0.453125/0.4375 (mean 0.45833); shuffled reward 0.46875/0.421875/0.515625 (mean 0.46875). These improvements are below the registered accuracy and effect-size criteria. Retention preserved post-training decisions and restoring weights exactly restored initial decisions. No confirmation or learning claim follows. Continue the predeclared 0.1, then 0.3 sequence if needed, without changing cues, biological mapping or reward rule.

## Development sequence completed — rate 0.3 selected

The 0.1 candidate failed: plastic accuracies 0.578125/0.71875/0.59375 (mean 0.63021), shuffled 0.4375/0.515625/0.453125, with the same frozen control as before. Runtime 506.28 s. The 0.3 candidate qualified: 0.796875/0.890625/0.84375 (mean 0.84375), versus frozen mean 0.45833 and shuffled mean 0.4375. Runtime 497.28 s, peak RSS 726,220,800 bytes. Retention preserved outcomes and restoring weights exactly restored the initial decisions for all seeds. Both candidates used unchanged neural conditioning code on commit 64a3471. All negative candidates remain recorded.

Freeze learning rate 0.3 for five independent confirmation seeds 65201–65205, 400 training trials per arm and 64 held-out trials per cue. The development result SHA is verified through --development-reference. No Gate 2 pass or conditioning-v0 tag exists until confirmation completes. New teacher, canonical-state and frozen-inference engineering does not alter the conditioning trial mathematics or stimuli.

Confirmation command: `.venv/bin/python -m flyholdem.experiments.conditioning --profile confirmatory --learning-rate 0.3 --development-reference runs/conditioning-full-lr03-v1/result.json --output runs/conditioning-full-confirm-v1` (add --resume after interruption on the same source/config/environment).

## Gate 2 confirmation passed

The registered confirmation on source checkpoint 3f3069a completed 9,865 operations in 1,004.48 s with peak RSS 717,553,664 bytes. All 166,700 neurons and 25,582,938 retained edges remained; 61,210 existing KC-to-MBON edges could change within their fixed 0.1–2.0 bounds.

| Seed | Plastic accuracy | Frozen / erased | Shuffled reward | Retained |
|---|---:|---:|---:|---:|
| 65201 | 0.8203125 | 0.4687500 | 0.5468750 | 0.8203125 |
| 65202 | 0.8046875 | 0.4609375 | 0.4843750 | 0.8046875 |
| 65203 | 0.8359375 | 0.5000000 | 0.4453125 | 0.8359375 |
| 65204 | 0.8281250 | 0.4687500 | 0.5703125 | 0.8281250 |
| 65205 | 0.8281250 | 0.4609375 | 0.4531250 | 0.8281250 |
| Mean | **0.8234375** | **0.4718750** | **0.5000000** | **0.8234375** |

Each seed used 400 training trials per arm and 128 balanced held-out trials. Paired improvement over frozen/erased was 0.3515625 (seed-bootstrap 95% interval 0.3421875–0.3609375); over shuffled rewards 0.3234375 (0.2765625–0.3703125). Each one-sided paired-seed sign-flip test gave p=0.03125. All registered accuracy, effect, retention and confirmation criteria passed. Every individual retained decision matched post-training, and every erased/frozen decision matched the initial network.

An independent journal audit verified all hash links, neural-only legal argmax, target/correctness fields, the exact shuffled earned-reward multisets and past-only RPE baselines. Journal SHA-256: 3e84f9408246e65e1256d7b151cbafbb1a2e4f74cbfd5d4f5f025223a24bae72. Run manifest SHA-256: d38469805e1c0e1617b249f91e24eb4a28fd3cba5acd908802bad55d64292cc0. Exact generated result/audit/checkpoint files remain local under ignored runs/conditioning-full-confirm-v1.

### Scope of the learned mechanism

Action-0 readout cells had zero firing in these held-out cue trials. Correct action-0 responses arose when plasticity suppressed the competing action-1 readout sufficiently for the fixed lowest-legal-index tie rule to select 0. There were 39–43 such silent-readout decisions per seed; the input/network itself was stimulated. This is learned cue-dependent suppression in an engineered binary decoder. Independently active learned output patterns and poker skill remain untested. The unchanged decoder contains no strategic fallback or teacher. The qualifying conditioning parameters are now frozen for subsequent experiments; no poker returns selected them.
