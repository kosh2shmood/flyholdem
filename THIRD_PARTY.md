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
