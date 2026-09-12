# Viewer

`make demo` starts both the Python backend and its served dashboard at http://127.0.0.1:8766. No frontend build or hosting account is required.

Live events: `/ws`. Graph schematic: `/api/graph`. Example log: `/api/example`. Health: `/api/health`. Live runs bind loopback only. This is a local development server, not a multiuser production deployment.

Use Live / Recorded replay to change sources. Pause freezes the display, not the live worker. Speed changes recorded playback only. Expand “Inspect exact neural input” for every nonzero channel and the canonical visible observation. Legal mask 01 means legal, 00 means masked. Scores are normalized neural rates; masked actions can have spikes but cannot be selected. All-in may be masked when a lower-ID action represents the same wager.

The brain panel uses a 2D schematic of all 126 fixture nodes, with 120 sampled edges out of 1,920. Opacity smoothly interpolates actual reported spike counts. No fabricated background activity or anatomical positions are added. The V0 fixture view is preserved; native modes use the separate WebGL neuron view below.

Browser validation: `uv run python -m playwright install chromium`, start `make demo`, then `make browser-check`. Desktop 1440×1080 and mobile 390×844 are checked for overflow; actual live/replay decisions and reinforcement are required. Reports and screenshots are under `docs/review/`.

## Animated fly camera

The default 3D table view seats the fully visible fly at the left side of the table and a faceless, abstract calling-station opponent at the right. Each rig moves its articulated hands when that player's committed action arrives. Gestures distinguish check (felt tap), call (chip push), half-pot and pot raises (reach and push), all-in (two-hand stack push), and fold (cards turn face down and slide to the muck). The held cards have curved geometry, a printed front and a distinct patterned back. Their faces point toward their holder and tilt upward toward the spectator, rather than presenting a reversed flat hand. All-in puts the cards on the felt before both hands push. Wing motion is decorative. These are labeled illustrative action-driven gestures, not learned motor dynamics or a whole-body physics simulation. The scene cannot select or submit an action.

The scene and its readable card/stack inset use the same event hash as the decision pane. Opponent cards show an unknown back pattern until the game event reveals them; mucked cards disappear exactly when absent from the public table state. Chip piles are illustrative; exact values are displayed in the inset. Drag/zoom changes only the spectator camera; Reset view restores the player view. Table view retains the overhead layout. Display pause freezes the character's pose as well as event display.

Three.js 0.186.0 is installed from the Node lockfile and served locally. WebGL 2 is required for the 3D scene. If unavailable, the viewer shows an explicit graphics error and the table view remains usable. The fixture/policy process does not depend on WebGL.

Run `uv run python scripts/avatar_check.py` with the server active to verify all six fly gestures and the opponent's real check/call gestures, card/event correspondence, the opponent information boundary, positive card-face orientation toward both holder and default camera, two-hand all-in movement, pause for both rigs, full silhouette bounds, camera reset, desktop/mobile overflow and browser errors. Evidence is in `docs/review/avatar/`; `fly-action-demo.webm` records the actual browser.

## Card and chip readability

A dedicated strip below the 3D scene shows the fly's two cards, all five public community slots and exact balances. Community slots remain empty until the event deals them. The 3D cards are larger and use a larger central rank, while the inset remains readable on a narrow phone screen. Both players have three chip piles positioned inside the rail; their visible count follows the public balance and becomes zero when the stack is empty. The inset contains the exact chip amounts; mesh counts are illustrative.

Browser checks compare the community strip with the canonical events on preflop, flop, turn and river, and verify chip visibility for zero/positive balances for both seats.

## Native connectome spectator

`flyholdem serve --mode full --port 8767` loads the passing quarter-strength registration; `--mode circuit` loads its separate original registration. Both require prepared data and an existing passing registration. Add `--model <frozen-native-model>` to load bounded learned weights. The native loop calls the verified LIF controller for every fly action, commits its exact legal neural argmax, records a hash-chained event, and starts each new hand from fresh neural rest. Native spectator evaluation disables all learning/reward delivery. Model/graph/binary/registration identities accompany events. This is not a poker-training result.

The point cloud contains every retained neuron and follows exact per-neuron spike counts from the last readout window. Full mode renders 166,700 points: 139,662 at uniformly normalized annotated soma coordinates and 27,038 in a separate schematic grid because their soma coordinates are missing. Circuit has 4,505 points, of which 4,490 have soma coordinates. No point is dropped from the retained graph and no missing anatomical coordinate is invented. No full-graph edges are drawn. Color separates the KC input, five readout ensembles, dopamine, other MBONs and other neurons. Filters, orbit/zoom, pause and per-neuron annotation inspection affect display only.

`/api/graph` provides mode/count/checksum metadata. `/api/graph/positions` and `/api/graph/roles` contain compact typed geometry; `/api/graph/neuron/<index>` supplies a selected cell's annotation. Large body IDs are strings. `/api/evidence` serves explicitly maintained written gate findings. Live native logs carry graph and registration identities; native replay rejects different identities. Fixture replay remains separately labeled and switches back to its 126-cell schematic.

Verified with Chromium 151.0.7922.34: all 166,700 points rendered; binary geometry checksums match; displayed activity equals native counts; annotation/filter/pause controls work; fixture replay and return to native live work; desktop 1440×1200 and mobile 390×844 have no overflow or console/page errors. Existing fixture tests again verified all six fly gestures, opponent check/call, clear fly/community cards at 0/3/4/5 public cards, both chip piles and card privacy. Three.js Points/BufferGeometry/ShaderMaterial and OrbitControls use the installed pinned library. See scripts/native_browser_check.py for reproducible checks. Generated screenshots and results remain local.
