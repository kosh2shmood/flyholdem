# FlyHoldem model card

**Version:** M4 conditioning/teacher engineering checkpoint; live poker remains the fixture prototype. **Status:** development. **Money:** play chips only.

## What exists

A fixed synthetic LIF graph acts through five neural spike readouts in a real PokerKit heads-up game. A bounded eligibility/RPE mechanism updates existing synthetic synapses. The live/replay dashboard shows actual canonical events, exact visible information, activity and plasticity.

The full retained MaleCNS graph and the declared 4,505-cell circuit now import, audit and run in the native solver. Circuit and full controllability passed their registered checks. The full passing configuration uses a uniform 0.25 synaptic scale relative to the initial model (signed contact gain 0.06875); earlier full failures remain published. No topology was removed. A second preparation reproduced every artifact hash; the independent numerical oracle and complete-state continuation checks passed. See docs/RUNTIME.md for measured resources. The viewer uses original illustrative player rigs and displays canonical public cards and balances.

## What does not yet exist

A native population controller and bounded local rule now exist; the live poker backend remains the fixture. Conditioning development reached 84.4% versus 45.8% frozen and 43.8% shuffled controls; independent confirmation is pending. No validated poker teacher, actual distilled-connectome checkpoint or confirmatory poker evaluation exists. No confirmatory learning claim is supported. The synthetic DAN/KC/MBON labels describe engineering roles, not anatomically reconstructed cells. V0's Euler solver and schematic coordinates are development assumptions.

## Intended use

Inspect the neural/game information boundary and mechanism during software development. Run data-free tests and replay the checked-in log. Do not interpret chips won, spike activity or changing weights as evidence of strategic learning or biological validity.

## Reproducibility

Python 3.11, dependency pins in uv.lock, fixed graph/mapping seed 1729, default demo seed 20260912. Recorded truth: examples/fixture-demo.jsonl. Review: docs/review/browser-check.json and screenshots. The separate 0.1 ms full/circuit solver has passed scalar Python and Brian2 parity checks. Native conditioning checkpoints include weights, eligibility and reward baselines; interrupted and logged-tail continuation reproduced the entire journal exactly. Teacher optimizer/replay recovery and teacher-disconnected native fixture inference tests pass. Integrated native poker learning/recovery remains subsequent work.

## Claims

The intended full-data public wording remains: “A simulated network using the wiring of one reconstructed male fruit-fly central nervous system controls an engineered heads-up no-limit Hold'em interface.” This statement is not a claim about the synthetic V0 fixture. No living fly, consciousness, faithful whole-brain emulation, or proven learning is claimed.
