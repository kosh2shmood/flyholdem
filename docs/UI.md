# Viewer

`make demo` starts both the Python backend and its served dashboard at http://127.0.0.1:8766. No frontend build or hosting account is required.

Live events: `/ws`. Graph schematic: `/api/graph`. Example log: `/api/example`. Health: `/api/health`. Live runs bind loopback only. This is a local development server, not a multiuser production deployment.

Use Live / Recorded replay to change sources. Pause freezes the display, not the live worker. Speed changes recorded playback only. Expand “Inspect exact neural input” for every nonzero channel and the canonical visible observation. Legal mask 01 means legal, 00 means masked. Scores are normalized neural rates; masked actions can have spikes but cannot be selected. All-in may be masked when a lower-ID action represents the same wager.

The brain panel uses a 2D schematic of all 126 fixture nodes, with 120 sampled edges out of 1,920. Opacity smoothly interpolates actual reported spike counts. No fabricated background activity or anatomical positions are added. V0 has no full-graph WebGL view yet.

Browser validation: `uv run python -m playwright install chromium`, start `make demo`, then `make browser-check`. Desktop 1440×1080 and mobile 390×844 are checked for overflow; actual live/replay decisions and reinforcement are required. Reports and screenshots are under `docs/review/`.
