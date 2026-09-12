# External-sampling conventional teacher candidate V10

V10 is a separately registered tabular response to a stationary mixture of the four original scripted opponents. It is untrained and unqualified at registration. It replaces bootstrapped neural Q estimates with counterfactual values from actual PokerKit settlements. It is a conventional teacher candidate, never the fly's action controller. All v1–v9 negative results and the unchanged full-hand qualification suite remain intact.

## Algorithm and information

On each iteration, draw one fixed opponent uniformly, deal one actual 20 BB PokerKit hand, and alternate the updating seat. Enumerate every legal action of that seat, including actions with zero current probability. Sample the opponent's decisions. Use genuine terminal net BB to compute each action's counterfactual value, then accumulate value-minus-current-policy-value regret. Positive regrets determine the next strategy; zero positive regret gives the declared legal uniform prior. The game and opponent state are copied at branches, preserving the sampled chance history and sampled opponent RNG state.

This follows the external-sampling approach described by [Lanctot et al. (2009)](https://www.cs.cmu.edu/~kwaugh/publications/nips09b.pdf). The implementation is original; [OpenSpiel's reference implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/external_sampling_mccfr.py) was inspected for comparison, without adding an OpenSpiel dependency or copying its implementation. Only one player learns against fixed opponents; this is not a self-play equilibrium computation or a GTO claim.

Average strategy accumulation uses the updating player's own prefix reach. The fixed opponent/chance sampling factor is constant in expectation at each information set and cancels in its normalized long-run average. This stationary-population construction must not be silently reused for learning opponents or presented as the standard two-learning-player averaging procedure.

The card abstraction has eight fixed visible-equity buckets per street, using the existing deterministic teacher-only 64-sample C++ estimator. Keys retain all earlier street buckets, complete public actions/paid chips, position, pot/stacks/prices and legal mask. Earlier private buckets are recomputed only from the current own hole cards and visible board prefixes. No opponent hole card, actual unrevealed card, hand seed, opponent identity, network output or teacher label enters the key. Revisited abstract nodes within a single traversal are rejected. This coarsens card information; finite training has no optimality guarantee in unrestricted Hold'em.

The frozen policy contains numeric key offsets/bytes, legal masks and reach-weighted average probabilities. Unseen information sets use the registered legal uniform prior. Stack domain is part of the key: an untrained 100 BB state cannot alias a learned 20 BB state. Out-of-training stacks therefore receive that prior, without a strength claim. This conventional prior never substitutes for a fly's neural scores.

## Registered execution

`configs/teacher_external_regret_v10.yaml` fixes 30,000 complete sampled-chance traversals, seed 91001, deals 1,000,000–1,029,999, uniform original opponent mixture, alternating seats, eight buckets and 64 hypothetical equity samples. The 2,000,000-information-set and 100,000-node-per-traversal ceilings fail explicitly; they never prune actions or introduce a fallback strategy. No reward shaping, conventional neural network, fly parameter change or intermediate profit selection is used.

The trainer rejects overlap with original development, confirmation and information-boundary deals before creating an output. The exact evaluation YAML hash, equity backend, source, environment and all configuration are part of the runtime identity. Every completed traversal records its initial private replay checkpoint, sampled opponent/seed, node/terminal counts and a digest covering the complete counterfactual traversal and updates. Five-minute/shutdown atomic checkpoints retain the entire numeric regret/average table, key order, RNG and complete-traversal boundary. Recovery must reproduce every journal tail hash. Completed recovery does not rewrite the final result. Only the fixed final iteration may be exported; output creation is atomic.

Training values are counterfactual values conditional on a sampled tree, not held-out BB/hand estimates. Numeric exports remain unqualified. The original full suite still requires positive adjusted lower bounds against all four opponents; no full confirmation deals may be used before development passes. The evaluator continues to use original PokerKit opponents and original deal/seat pairing. A new explicit policy loader adds this conventional type without altering prior NFSP/Q/potential policy modules or frozen fly inference.

```sh
.venv/bin/flyholdem teacher train-regret --config configs/teacher_external_regret_v10.yaml --output runs/teacher-external-regret-v10 --stop-after 128
.venv/bin/flyholdem teacher train-regret --config configs/teacher_external_regret_v10.yaml --resume runs/teacher-external-regret-v10
.venv/bin/flyholdem teacher export-regret --run runs/teacher-external-regret-v10 --output runs/teacher-external-regret-v10-policy
.venv/bin/flyholdem teacher evaluate --policy runs/teacher-external-regret-v10-policy --config configs/teacher_evaluation.yaml --profile development --output runs/teacher-external-regret-v10-development
```

Commit source/config first and use a frozen runtime for these commands. The first 128 traversals are the beginning of the same registered run, used to measure resources before continuation. Do not change the environment or shared binaries while the native dashboard or an experiment is running. The existing separate verified equity binary can be copied into the new runtime. Generated checkpoints, data and logs stay local.

## Engineering checks before registration

Eight untrained traversal-cost probes covered both seats and all four original opponents, totaling 1,307 nodes in 0.87 seconds and 54.9 MB peak process RSS on the current host. A separate bounded core check traversed 2,608 nodes / 996 terminal branches over sixteen updates, recovered the exact numeric state and trace hashes, and passed 32 private-information probes. These are narrow resource/arithmetic checks, not a forecast or poker-strength result.

Integration tests cover exact two-branch PokerKit regret arithmetic and own-reach average, full-population sampled training/recovery, stable completed results, atomic numeric export/reload, corruption rejection, visible history/bucket recall, original full qualification recomputation, and reserved-seed refusal. Long-run scaling and strategic performance remain unmeasured at this registration milestone. The full teacher/corpus prerequisite remains unfulfilled until real held-out evidence qualifies a candidate.


## Execution update

V10 training runs from the unchanged c569620 source snapshot. Its 128-traversal resource boundary completed in 11.780402 seconds, at 63,258,624-byte peak RSS, before the same fixed schedule resumed. No held-out result exists yet. Current evaluation additionally requires a verified, hash-bound `--development-reference` for confirmation, and tabular evaluation no longer requires optional PyTorch. Freeze the current checked evaluation source/config separately before V10 development, then use that same evaluation snapshot for confirmation only if development passes. Export the training model from its original training snapshot; the frozen policy's unchanged implementation identity is checked by the evaluator. Do not mutate or restart the active training snapshot to pick up unrelated CLI/guard changes.
