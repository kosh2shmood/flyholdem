# MaleCNS runtime checkpoint — M2

The official data import, sparse runtime and deterministic regeneration checks passed. This is an engineering result; the current poker dashboard still runs the synthetic fixture, and no conditioning or learned poker result exists yet.

| Mode | Retained neurons | Directed edges | Synaptic contacts | Peak RSS in smoke test |
|---|---:|---:|---:|---:|
| Full retained MaleCNS v1.0 | 166,700 | 25,582,938 | 124,177,617 | 835,305,472 bytes |
| Declared MB circuit | 4,505 | 1,018,725 | 2,794,887 | 150,847,488 bytes |

All three downloaded files independently matched the committed byte counts and SHA-256 digests. The full import retains 10,299,701 contact-one edges and 101 autapses. The 151,856,684 raw edge rows include objects outside the declared retained-neuron set; excluded rows and contacts are explicitly accounted for. No additional edge-strength threshold is used. Circuit mode is an induced annotation-selected subset (Kenyon_Cell, MBON, DAN, APL and DPM), including all annotated DANs.

A second normalization and full/circuit preparation produced identical reports and every output-file SHA-256, including the CSR byte identity. Full graph hash: `70c9daefc6185272d0e001700d75aaa8718648c02d9aa59bbadbe6b4cd9e4c06`. Circuit graph hash: `d85be4e7fe4f78b2747f859b575a5c48ef4d85f8f7e6eab24dd7e80cbb463f94`.

The registered smoke stimulus was 64 annotation-selected Kenyon cells, seed 1729, 50 ms no-input baseline and 100 ms at drive 12. Full mode produced zero baseline spikes and 6,194 stimulated-period spikes across 3,881 cells. It took 0.0712 seconds for the 100 ms stimulus. Circuit mode produced 203 spikes across 96 cells in 0.00118 seconds. Both restored and re-executed their entire mutable neural state exactly. These are short sparse-stimulation measurements, not a bound on memory/time during dense input, learning or long recurrent activity.

The analytic 0.1 ms native LIF solver passed independent scalar Python and Brian2 2.10.1 comparisons, with exact spike counts and 0.002 absolute voltage/conductance tolerance. The oracle uses an isolated Python 3.12 environment; the application uses Python 3.11. Unit/integration tests: 41 passed with the data extra installed; two oracle tests are intentionally skipped there and pass separately. CI passed at execution checkpoint 0dc8111.

Data configuration/code was frozen at 04b86b2; execution manifests record 0dc8111, clean tree, source/config/binary identities, exact stimulus IDs and dependency versions. Machine-readable records are under docs/review/malecns/. Runtime mutable state was 120,835,540 bytes for full and 4,575,043 bytes for circuit. Raw, normalized and prepared data remain ignored under connectome_data/.

Reproduce with `make fetch-malecns`, `make prepare-malecns`, `make audit-malecns`, `make test`, `make test-data` and `make test-oracle`. No concurrent full workers were used. Next is annotation/connectivity-only encoder/readout mapping and the registered controllability gate; poker profit must not influence those choices.

Sources: official MaleCNS v1.0, MaleCNS collaboration / FlyEM / HHMI Janelia / Cambridge / MRC LMB / Google Research, CC BY 4.0. Import and sign policy are documented in data-provenance/malecns_v1/IMPORT_POLICY.md. DOOMFLY-derived generic code retains its MIT attribution in THIRD_PARTY.md and licenses/.
