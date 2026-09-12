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
