# Full recorded graph transport

## Current release status

The [complete evidence release](https://github.com/oddeye00/greencert/releases/tag/evidence-451k-20260907)
was published on 2026-09-07 at 22:43:37 UTC after all 14 asset sizes and
SHA256 digests matched. The user approved the original graph, including the
two unchanged hash-bound local-path records listed below. Original source
archives, evidence bytes and older releases were not replaced. The release
receipt and all asset digests are in
`results/full451k_evidence_release_20260907_v1.json` and
`results/full451k_evidence_assets_20260907_v1.json`.
See `451K_COMPLETED_RESULT.md` for the separate outcome observation.

## Reproduce from the evidence release

The evidence tag is `evidence-451k-20260907`. Allow roughly 24 GB of local storage
for the transport and reconstructed graph. The downloader uses public
HTTPS without account credentials and verifies every downloaded asset.

```text
python scripts/download_451k_recorded_graph.py --destination output/full451k_assets
python scripts/materialize_recorded_graph.py --package output/full451k_assets --transport-sha256 b8490b76a42338d7e8bc1c08744e05f537c7a912bbc14c7842bc567f697ae25e --destination output/full451k_evidence --report output/full451k_materialization.json
python scripts/extract_recorded_replay_sources.py --extract output/full451k_portable_replay
python output/full451k_portable_replay/scripts/replay_recorded_window.py --root output/full451k_evidence --manifest output/full451k_evidence/recorded_manifest.json --manifest-sha256 bc6e54b5421ffcf71cd9e905142cadcfbd3dc0856f478fd001c221cd5072b425 --mode replay --report output/full451k_replay.json
```

The separately extracted version-3 driver is intentional: it handles the
recorded Windows paths on both Windows and Linux. Do not overwrite the
original version-1 source archive or overlay new code onto frozen evidence.
These commands check the recorded enclosure and do not rerun the one-shot
future observer. All outcomes are already disclosed in the separate
completed-observation artifact.

## Fresh public-download verification

The commands above completed on DGX from fresh GitHub downloads, not from
the prior SCP cache. All 14 assets authenticated, all 17,713 logical files
were reconstructed and rehashed, and the portable replay returned the
same assembly, Green bound, source hashes and recorded input hashes as the
earlier ARM replay. Reconstruction took 20.2717 seconds and the recorded
numerical replay took 53.7868 seconds, excluding download and original
neural-bound construction. These are replay timings, not verifier-construction
speedups.

Reports: `results/full451k_public_materialization_arm_20260907_v1.json`
and `results/full451k_public_replay_arm_20260907_v1.json`. The latter has
SHA256 `74ef4b9e99cd25b6c57fb3e2d49bb37d37d000c114406d5bf3a6a45f6ff73191`.

The complete dependency graph for the original 451,008-parameter certificate
has been packaged without changing any recorded payload. The transport
contains 17,713 logical files represented by 17,634 distinct SHA256 objects.
Logical size is 11,750,719,606 bytes; the complete transport is
11,193,287,292 bytes across 14 assets: eleven uncompressed object ZIPs, the
original evidence manifest, the tested replay source archive, and a
transport descriptor.

This package reconstructs the inputs for a saved-evidence replay. It does
not itself recompute neural derivatives, issue a certificate, reveal an
outcome, or authorize publication. The original Windows replay is recorded
in `results/recorded_window_exact_replay_windows_20260907.json`.

## Local files

- Package: `output/full_recorded_graph_transport_20260907_v1/`.
- Descriptor: `transport.json` in that directory, SHA256
  `b8490b76a42338d7e8bc1c08744e05f537c7a912bbc14c7842bc567f697ae25e`.
- Original evidence manifest SHA256:
  `bc6e54b5421ffcf71cd9e905142cadcfbd3dc0856f478fd001c221cd5072b425`.
- Builder: `scripts/package_recorded_graph.py`.
  The builder requires the extracted replay toolkit on `PYTHONPATH`; it is
  not needed for the public download/materialization commands above.
- Standalone materializer: `scripts/materialize_recorded_graph.py`, SHA256
  `efd37ecd9dbd5207ef7a290a937f0b77f6035b55df6aed86f2bfab5f7cf29459`.
- Tests: `scripts/test_recorded_graph_transport.py`. Both Windows and ARM
  pass exact-byte reconstruction and 12 integrity/semantic refusal cases.

The materializer verifies segment hashes, object hashes, alias membership,
source compatibility, and every reconstructed logical file. It imports no
archive code. It requires a fresh destination and preserves partial output
on failure. Identical aliases may be hardlinked; no original source file is
modified. The data segments fit within GitHub's documented per-asset limit;
see [GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Publication gate

The text intake scan authenticated and inspected 16,236 files. It found no
credential-shaped strings, but two original records contain local Windows
paths:

- `results/larger_transformer_third_pass/construction_03925/dgx_green_handover/receipt.json`;
- `results/larger_transformer_third_pass/construction_03925/outward_bound_rows/step_001/independent_audit_231.json`.

They are hash-bound provenance records, so redacting them would change the
original evidence graph. The user has approved publishing them unchanged. Scan
report: `results/large_recorded_graph_publication_scan_20260907.json`, SHA256
`a2ac845d0d6182bacfc7b406d898c5f7d679f33061cb2ca375628481ed56e1ad`.
This is a decoded-text intake scan, not a general guarantee that arbitrary
binary data contains no sensitive content.

## Second-host replay

The private SCP transfer completed on 2026-09-07. Full ARM materialization
passed in20.2357798 seconds, checking all17713 logical files/17634 objects,
79 hardlinked aliases and47 original source overlays. Report:
`results/full_recorded_graph_materialization_arm_20260907_v1.json`.

The first original-source ARM replay stopped at imported reference-margin
path parsing: a Windows backslash path was not interpreted by POSIX Path
before its basename check. Original evidence and auditors were retained.
The separate version3 source toolkit adds portable path parsing and a
regression that reproduces three legacy failures on Linux. All source-kit
component and reader tests pass on both hosts.

The version3 ARM full replay then PASSED in55.6175131 seconds. Its event
assembly and Green bound exactly equal the original Windows replay; all
17713 accessed hashes agree after canonicalizing path separators. Report:
`results/recorded_window_exact_replay_arm_20260907_v3.json`, SHA
42727e4d4c1de036c16d4c67c42d3fc29a73420d991d6c5bce2c98a2efb4fa89.
Comparison: `results/recorded_replay_cross_host_comparison_20260907_v1.json`.
These are saved-evidence replay times, not controlled construction speedups.
No future outcome was opened during those replays. The later outcome
observation and publication approval are recorded separately.

The fresh Windows materialization passed: 17,713 logical files, 17,634
distinct objects, 79 hardlinked aliases, and 47 source overlays, all
byte-rechecked in 207.9734278 seconds. Report:
`results/full_recorded_graph_materialization_windows_20260907_v1.json`.
The separately recorded original Windows replay passed in 492.8985672
seconds; its public report is
`results/recorded_window_exact_replay_windows_20260907.json`.
