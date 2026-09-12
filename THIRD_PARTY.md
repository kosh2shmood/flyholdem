# Third-party provenance

## DOOMFLY reference

- Repository: https://github.com/nftechie/doomfly
- Inspected revision: `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33` (2026-09-12).
- License: MIT, copyright (c) 2026 nftechie and DOOMFLY contributors.
- Inspected: README, license/attribution, dataset registry and SHA-256 source lock, retained-graph importer, LIF engine and training-protocol structure.
- V0 source is original code; no Doom game, artwork, simulator module or dataset has been vendored. DOOMFLY motivates provenance, numerical, refractory and scientific-control requirements. Any later adapted module must carry its own source/revision notice and retain the MIT notice.

## MaleCNS v1.0

Official source: https://male-cns.janelia.org/download/. Attribution: the MaleCNS collaboration, including FlyEM/HHMI Janelia, University of Cambridge, MRC LMB and Google Research, and the release's authors/contributors. Release data uses CC BY 4.0. No MaleCNS bytes are included in V0; the fixture is synthetic. Source hash-lock import is a subsequent milestone. Code license does not relicense data, measurements or third-party marks.

## Runtime dependencies

PokerKit 0.7.5 (MIT): https://github.com/uoftcprg/pokerkit; canonical game rules. Official API inspected at https://pokerkit.readthedocs.io/en/stable/simulation.html.

NumPy (BSD-3-Clause), FastAPI (MIT), Starlette (BSD-3-Clause), Uvicorn (BSD-3-Clause), websockets (BSD-3-Clause), PyYAML (MIT) and their transitive dependencies are installed separately under their own licenses from `uv.lock`. Pytest, HTTPX and Playwright are development tools, not vendored source. The UI uses system fonts and original HTML/CSS/canvas; no remote assets.

PettingZoo's official no-limit documentation was inspected for the five-action reference abstraction; it is not the canonical rules engine or a V0 dependency. No code copied from that implementation.

## Additional inspected references

FlyVis: https://github.com/TuragaLab/flyvis, MIT, official connectome-constrained visual-system PyTorch model. Used as a scope/modeling reference only; no code copied or dependency installed. DOOMFLY live protocol and its independent Brian2 numerical-oracle tests were reviewed at the pinned revision. The linked Prismix/Pong discussion is a secondary failure-analysis reference; biological and numerical claims rely on primary implementation/data sources.

The expected MaleCNS source.lock.json is reproduced from DOOMFLY's pinned provenance record. Its MIT notice is retained at `licenses/DOOMFLY-MIT.txt`. Hash expectations are independently verified on fetch; data is still downloaded from the official release.

## Three-dimensional fly spectator

Three.js 0.186.0 (MIT), https://github.com/mrdoob/three.js, installed separately with npm from the committed lockfile; license retained at licenses/Threejs-MIT.txt. Current official WebGLRenderer/OrbitControls APIs were checked at https://threejs.org/docs/. No remote renderer assets are loaded.

`ui/src/fly-avatar.js` adapts the ellipsoid/rod and veined-wing construction ideas from DOOMFLY `doom-ui/lib/fly-model.ts` at the pinned revision. Its source header and licenses/DOOMFLY-MIT.txt preserve attribution. The upright poker pose, articulated hands, card textures, chips, table and event-driven gestures are original FlyHoldem code. This is an anatomy-inspired illustration; no third-party character mesh, neural-body model or game art is copied.

## Numerical runtime

`src/flyholdem/neural/kernel/doomfly_lif.cpp` retains the all-edge lazy LIF algorithm from DOOMFLY `doom/kernel.cpp` at the pinned revision, with an added provenance header and the MIT notice at licenses/DOOMFLY-MIT.txt. The Python reference, validated ctypes wrapper, checkpoint manager and build record are original FlyHoldem code. The Brian2 oracle test adapts the independent-oracle test design from DOOMFLY. Brian2 2.10.1 uses CeCILL 2.1; it is installed separately in the locked oracle environment, not vendored.

MaleCNS exact-ID and edge-retention helpers in `connectome/import_malecns.py` are adapted from the pinned DOOMFLY `doom/connectome.py`. The transmitter proxy follows the declared logic in `doom/transmitters.py`; FlyHoldem adds whitespace normalization. Attribution and the complete MIT notice are retained. Streaming CSR preparation, audit and resource benchmark orchestration are original. No retinal or motor policy is imported.
