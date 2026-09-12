# Visual-first gate report

**V0 PASS — fixture visual prototype only.** No full-connectome or learning claim.

- One command: `make demo`, backend and dashboard at http://127.0.0.1:8766.
- Actual PokerKit 0.7.5 HU NLH; continuous autoplay, alternating button, play chips.
- Synthetic 126-cell LIF circuit produces all five fly scores. No strategic fallback.
- Cards, pot, stacks, history, raw recorded neural activity, normalized scores, legal mask, selection and exact encoder input are connected through canonical events.
- Implemented eligibility/RPE changes existing bounded fixture edges; visible dopamine and delta events.
- Live WebSocket and six checked-in deterministic hands (36 events), with exact replay regression and tamper detection.
- Chromium 151.0.7922.34: live/replay, input inspection and weight changes verified; no page errors or horizontal overflow at desktop 1440×1080 or mobile 390×844. Screenshots visually inspected.
- Prominent fixture/development/bio-plastic/teacher-disconnected labels.
- Data-free tests: **15 passed**. Two upstream test-client deprecation warnings are recorded, not correctness failures.
- Short 30-hand fixture benchmark: **26.02 hands/s**, 1.153 seconds, 224 events (no deliberate UI delay). Hardware/platform in `docs/review/fixture-benchmark.json`.

## Artifacts

- `docs/review/fixture-live.png`, `fixture-replay.png`, `fixture-mobile.png`
- `docs/review/browser-check.json`, `fixture-benchmark.json`
- `examples/fixture-demo.jsonl`, `examples/fixture-manifest.json`
- `docs/PROTOCOL.md`, `docs/MODEL_CARD.md`

## Remaining gates

Gate 0 needs the complete golden/symmetry suite at M1. Gates 1–6 and 2A are pending. V0's synthetic Euler solver is not the registered full-graph numerical runtime. Full-state resumable checkpoints, full MaleCNS import, conditioning/controls, both registered training modes, teacher, curricula and confirmatory evidence are later milestones.

## Publication status

Local working branch is `astra/visual-first`. GitHub owner and ADMIN access were verified. Automatic approval review rejected pushing because the existing repository is public; local work proceeds and publication awaits explicit user confirmation. No source/data has been pushed by this build.
