# Viewer

`make demo` starts both the Python backend and its served dashboard at http://127.0.0.1:8766. No frontend build or hosting account is required.

Live events: `/ws`. Graph schematic: `/api/graph`. Example log: `/api/example`. Health: `/api/health`. Live runs bind loopback only. This is a local development server, not a multiuser production deployment.

Use Live / Recorded replay to change sources. Pause freezes the display, not the live worker. Speed changes recorded playback only. Expand “Inspect exact neural input” for every nonzero channel and the canonical visible observation. Legal mask 01 means legal, 00 means masked. Scores are normalized neural rates; masked actions can have spikes but cannot be selected. All-in may be masked when a lower-ID action represents the same wager.

The brain panel uses a 2D schematic of all 126 fixture nodes, with 120 sampled edges out of 1,920. Opacity smoothly interpolates actual reported spike counts. No fabricated background activity or anatomical positions are added. V0 has no full-graph WebGL view yet.

Browser validation: `uv run python -m playwright install chromium`, start `make demo`, then `make browser-check`. Desktop 1440×1080 and mobile 390×844 are checked for overflow; actual live/replay decisions and reinforcement are required. Reports and screenshots are under `docs/review/`.

## Animated fly camera

The default 3D table view seats the fully visible fly at the left side of the table and a faceless, abstract calling-station opponent at the right. Each rig moves its articulated hands when that player's committed action arrives. Gestures distinguish check (felt tap), call (chip push), half-pot and pot raises (reach and push), all-in (two-hand stack push), and fold (cards turn face down and slide to the muck). The held cards have curved geometry, a printed front and a distinct patterned back. Their faces point toward their holder and tilt upward toward the spectator, rather than presenting a reversed flat hand. All-in puts the cards on the felt before both hands push. Wing motion is decorative. These are labeled illustrative action-driven gestures, not learned motor dynamics or a whole-body physics simulation. The scene cannot select or submit an action.

The scene and its readable card/stack inset use the same event hash as the decision pane. Opponent cards show an unknown back pattern until the game event reveals them; mucked cards disappear exactly when absent from the public table state. Chip piles are illustrative; exact values are displayed in the inset. Drag/zoom changes only the spectator camera; Reset view restores the player view. Table view retains the overhead layout. Display pause freezes the character's pose as well as event display.

Three.js 0.186.0 is installed from the Node lockfile and served locally. WebGL 2 is required for the 3D scene. If unavailable, the viewer shows an explicit graphics error and the table view remains usable. The fixture/policy process does not depend on WebGL.

Run `uv run python scripts/avatar_check.py` with the server active to verify all six fly gestures and the opponent's real check/call gestures, card/event correspondence, the opponent information boundary, positive card-face orientation toward both holder and default camera, two-hand all-in movement, pause for both rigs, full silhouette bounds, camera reset, desktop/mobile overflow and browser errors. Evidence is in `docs/review/avatar/`; `fly-action-demo.webm` records the actual browser.
