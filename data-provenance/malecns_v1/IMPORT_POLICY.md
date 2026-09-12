# MaleCNS v1.0 import policy — v1

Authoritative release: https://male-cns.janelia.org/download/. The three exact official files and expected byte counts/SHA-256 digests are in source.lock.json. Expected hashes were obtained from the inspected MIT-licensed DOOMFLY revision 71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33; they must be independently verified after downloading. A registry entry is not proof a file has been downloaded or verified.

Retain every annotation with a nonempty assigned superclass, including uncertain/tbc classes, except explicit `status == Glia`. Do not restrict to Traced status or typed cells. Keep exact integer IDs and original annotation fields, including sides and available neuropil/coordinate fields. Account for excluded objects by reason.

Retain every released directed edge whose endpoints are retained nodes, including weight-one edges and autapses; do not add an extra contact-count threshold or create edges. The upstream release already filters synapse confidence at 0.5. Report retained/excluded rows and synaptic contact counts. Source IDs are never converted through floating point or JavaScript numbers.

Use the per-body consensus neurotransmitter predictions; transmitter sign is a declared model proxy: acetylcholine positive; GABA, glutamate and histamine negative; ambiguous, missing, conflicting or only-modulatory predictions use the preregistered ambiguous-sign parameter (+1 baseline; -1 sensitivity control), without removing edges. This is not receptor physiology. Any distinct separation of modulatory populations from fast currents must be versioned in the prepared-graph/model manifest.

Official data is CC BY 4.0. Credit the MaleCNS collaboration, FlyEM/HHMI Janelia, University of Cambridge, MRC LMB, Google Research and the release authors/contributors. Paper: https://doi.org/10.1016/j.cell.2026.08.015. Changes are the retention/normalization, model-sign and graph-view transformations described here. No creator endorsement is implied.

Raw data and prepared arrays remain in ignored connectome_data/. Download the 1.05 GB aggregate edge table, 14.5 MB annotations and 43.3 MB aggregate neurotransmitters only. Do not fetch the much larger per-synapse tables for the first build. All source files must match the committed source.lock.json before import/training. Never silently rewrite the lock to accept new source bytes.

## Circuit mode and preparation

The circuit subset is fixed by official annotation, before any poker result: classes `Kenyon_Cell`, `MBON`, `DAN`, plus exact types `APL` and `DPM`. It is an induced mushroom-body-focused subgraph, including all annotated DANs, not a full nervous system. Full mode retains every node and edge under the preceding policy. All original node annotation columns survive normalization, including integer soma coordinates when present.

Preparation makes two passes over bounded Arrow record batches, writes stable per-source CSR arrays through memory maps, and preserves contact-one edges and self-edges. It does not perform an all-edge in-memory sort. Signed weights equal contact counts × presynaptic transmitter proxy × 0.275, an explicit experimental dynamics assumption. `configs/runtime.yaml` freezes the sign/gain, circuit criteria, upstream reference counts and non-poker resource stimulus. Prepared files and the exact graph byte sequence are hashed. A loader rejects altered artifacts.
