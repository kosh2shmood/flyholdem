# Small tabular shove/fold reference

This conventional reference is limited to the registered 10 BB, two-decision shove/fold subgame. It is not the full 20 BB teacher, a fly controller, or an equilibrium certificate. No biological parameter is selected or updated.

The method follows counterfactual regret minimization with chance sampling: [Zinkevich et al.](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf) and [Lanctot et al.](https://bowlingmh.github.io/papers/09nips-mccfr.pdf). Each iteration samples one deterministic PokerKit deal and traverses all three terminal branches: button folds, button shoves/big blind folds, and shove/call showdown. PokerKit alone supplies every terminal payoff. Both players use their pre-update regret-matched strategies for that traversal.

The table has two public decision nodes, 169 own-card rank/suitedness classes and two actions. Canonical observations must exactly match the registered public node, including stack, history, position and legal mask. Full-hand/river states and hidden fields are rejected. The original 237-channel fly encoder is unaffected. All 1,326 combinations map to 13 six-combination pairs, 78 four-combination suited classes and 78 twelve-combination offsuit classes.

For sampled button terminal payoffs (fold, declined shove, showdown), button action values are fold and the big-blind-strategy-weighted shove return. Big-blind action values are the negatives of the latter two terminal payoffs. Button regret uses action value minus current value; big-blind regret additionally uses the button shove reach probability. Neither player has an earlier own action, so their strategy averages must not be weighted by opponent reach. Chance visitation frequencies supply the sample weighting. No finite-run exploitability guarantee is claimed.

Configuration `configs/shove_fold_teacher.yaml` registers 200,000 sampled deals from seed 6000000. It selects no policy by poker return. Checkpoints contain every regret, strategy accumulator, visitation count and completed traversal; source/config/environment identities and hash-chained journals permit exact replay from an older checkpoint. Checkpoints are atomic at five-minute intervals and graceful shutdown, with three latest retained. Exported numeric probability tables remain explicitly unvalidated and pin their implementation/runtime.

```sh
.venv/bin/flyholdem teacher train-shove-fold --config configs/shove_fold_teacher.yaml --output runs/shove-fold-teacher-v1
.venv/bin/flyholdem teacher train-shove-fold --config configs/shove_fold_teacher.yaml --resume runs/shove-fold-teacher-v1
.venv/bin/flyholdem teacher export-shove-fold --run runs/shove-fold-teacher-v1 --output runs/shove-fold-teacher-v1-policy
```

Long execution uses an immutable source/config/lock snapshot, as other experiments do. Training and exported artifacts stay ignored/local. The CLI writes the ordinary verified report and labels its count as sampled deals traversed through all three branches. A separate held-out small-game evaluation is still required before this reference can teach; it cannot satisfy the unchanged full 20 BB teacher gate.

Verification: eight checks cover exact class multiplicities, actual settlement, counterfactual reach weighting, two analytically solved toy games, hidden-hole/future-deck invariance, out-of-domain rejection, source/tensor identity, exact old-checkpoint replay, and real CLI training/export/report. A 100-traversal engineering throughput check took 0.20116 seconds with peak RSS 53,641,216 bytes. This is a resource measurement, not poker-strength evidence.

## Completed training and registered held-out evaluation

The first 200,000-traversal run completed in 427.01477 seconds with minimum 531 visits per card class/position. Training manifest 4f2d9b1e048c037230085e1722d8ea54afd65da75083d72b99338f406a630d2b, journal head 21ff5c436106e3132fe860d767c1f3f386fff9546b30a25567b8137c5a7ffe4e, frozen policy manifest 8a378ff881fc1bf11ae65f00687d271cafc05b847ef8dc7b117de8a91af5e1b1. No strength or equilibrium claim follows from training completion.

Configuration `configs/shove_fold_teacher_evaluation.yaml` registers the unchanged four original fixed policies, constrained by the small game's legal masks. Development uses 512 seat-swapped paired deals per opponent from seed 6400000; confirmation uses 2048 from 6500000. The 200,000 training deals start at 6000000; 32 actual information-boundary probes use the two decision nodes of 16 separate deals from 6300000. All ranges are disjoint and checked against the policy's matching completed training manifest/journal.

The same 5,000-resample Bonferroni suite bootstrap requires every opponent's lower confidence bound above zero. Confirmation requires independently recomputed passing development evidence for the same frozen table. The result's `allowed_as_teacher` remains false regardless: only a separately named `allowed_as_small_game_teacher` may become true after its own confirmation. Neither can satisfy the full 20 BB teacher prerequisite. Frozen policies and deterministic per-pair RNG permit recovery at any complete paired deal.

```sh
.venv/bin/flyholdem teacher evaluate-shove-fold --policy runs/shove-fold-teacher-v1-policy --training-run runs/shove-fold-teacher-v1 --config configs/shove_fold_teacher_evaluation.yaml --output runs/shove-fold-teacher-v1-development
```

Add `--profile confirmatory --development-reference runs/shove-fold-teacher-v1-development` only after passing development, and choose a distinct output. Replace `--output RUN` with `--resume RUN` after interruption. Two additional tests cover real paired-game recovery, recomputed summaries and rejected overlap/mismatched training; full suite 143 passed / 2 isolated-oracle skips. Evaluation has not run at this registration checkpoint.

Development passed all four adjusted lower bounds on 512 paired deals each: random +0.355469 BB/hand [0.077625, 0.622452], calling station +0.456055 [0.068478, 0.843512], tight-aggressive +0.330566 [0.225586, 0.433963], equity-bucket +0.276855 [0.079102, 0.472656]. Private-information checks passed and summaries were recomputed from the complete journal. Result SHA-256 735383b3fb45e1f63de219a67907bf791e49698a68dd4b149dd648ebda2eeb0b; manifest 59cb41ad6bbf3bf041d7784c7c9585ad8d8a47f2b5624af1e306b2828f5ea457. This qualifies the fixed table for its predeclared small-game confirmation, not for teaching yet. No parameters or checkpoint changed.
