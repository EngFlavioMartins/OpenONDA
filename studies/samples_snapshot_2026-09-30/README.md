# Complete sample snapshot, 30 September 2026

This lossless snapshot contains all 8,145 regular files found inside the 35
`tutorials/**/samples/` directories at capture, totaling 6,067,553,990 bytes.
Empty directories are recorded in the manifest provenance. Study samples,
superseded restart branches, CSV histories, NumPy arrays and ParaView data are
retained without decimation or exclusions. Build directories and their duplicate
test results are outside the capture scope.

Simulations were allowed to continue. Each source file was copied to disk-backed
scratch storage and accepted only when its device, inode, size, modification time
and change time remained unchanged during the copy. The capture interval and
per-file source sizes/timestamps are recorded in `capture.json`. Files created
after the inventory belong to a subsequent snapshot. This is a stable copy of
each observed file, not a globally synchronized solver checkpoint. It does not
claim that incomplete or stopped simulations have finished or are scientifically
qualified. Original source files were not modified.

`assets/results/manifest.json` records SHA-256 checksums and the original path of
every file. The archive uses the existing `openonda.results` format, with large
payloads split into parts of at most 1 GiB and published as versioned GitHub release
assets. GitHub rejected the attempted LFS upload because this repository exceeded
its LFS budget. Release assets allow publication without changing account billing.
All VTK collection
dependencies are checked during packing and restoration. No data formats or
existing Git commits were rewritten.

## Retrieve after cloning

Clone the `development` branch; Git LFS is not needed. In the OpenONDA Python
environment, run this from the repository root:

```bash
python -m openonda.results restore build/published-samples \
  --bundle studies/samples_snapshot_2026-09-30/assets/results
```

Missing archive parts download automatically from the
[versioned data release](https://github.com/EngFlavioMartins/OpenONDA/releases/tag/tutorial-results-2026-09-30)
and are verified before being cached. All samples appear at
`build/published-samples/tutorials/.../samples/...`, with
their original bytes and relative paths. CSV, NumPy and ParaView readers can open
them there. The target can be another empty directory on any machine. Restoration
verifies archive and member checksums before installing directories and preserves
existing destination sample directories.

Use a separate destination: these newer samples must not be combined with older
numerical solutions from the per-case archives. The normal `./allplot.sh` workflow
continues to restore its matched samples and solution fields. This snapshot
contains samples only and cannot by itself resume a numerical simulation.

## Publication verification

All 18 result assets (9,122,205,342 bytes, including the existing case bundles
and superseded delta-wing archive) were uploaded and checked against GitHub's
SHA-256 digests. This sample snapshot occupies 3,154,626,364 compressed bytes.

A separate network clone retrieved and restored all 8,145 sample files through
anonymous HTTPS downloads. Archive and member checksums and VTK collection
dependencies passed. A second, ordinary shallow clone of `development` succeeded
without LFS skip flags or a recovery bundle; its Taylor–Green archive downloaded
and restored automatically. VTK readers opened both restored VPM sample data
and FVM volume data. The 45 archive tests and 16 cylinder workflow tests passed.
See [the verification record](verification.json).

The 62 initially unpublished commits were published with their original hashes.
Four historical archive commits used GitHub's Git data API, checking both tree
and commit hashes before non-forced fast-forward updates. Normal Git pushes
published the remaining commits. No history rewrite or billing change was
required. The release also retains a verified bundle of those original commits.
Old commits still contain their original LFS pointers and are subject to the
account's existing LFS budget; their result payloads are mirrored in the release.
