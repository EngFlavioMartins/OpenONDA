# Interrupted cube-flow baseline

Preserved on 15 September 2026 before investigating the growing downstream
velocity discrepancy. The coupled FVM ended at time 15.66 and its latest
complete coupled checkpoint is at 15.5. The fine reference completed time 30.
With unit cube diameter and freestream speed, these times equal t U∞ / D.

`samples/` contains byte-preserved native sampler outputs and the existing
matched-time comparison cache. CSV and VTS data are versioned using Git LFS;
NPZ comparison caches remain local under the repository's checkpoint exclusion.
`metadata/` preserves the two coupled configurations, the reference
configuration and the coupled checkpoint descriptor. `telemetry/` contains
losslessly compressed original diagnostic streams. `manifest.json` records
the original paths, local snapshot paths, sizes and SHA-256 checksums.

The approximately 5 GiB `solution/` snapshot contains independent APFS clones
of the coupled states, coupled mesh and fine reference states. It is local
and ignored by Git. Subsequent probes must read these snapshots and write
into separate trial directories. They must not mutate this baseline or the
original tutorial results.

The existing tutorial figures and the current plotting code are included in
the accompanying baseline commit. Runtime metadata does not identify the Git
revision used during integration; the baseline commit records the source
available at capture, rather than asserting an unrecorded run-time revision.

Reproduction of this capture is implemented in
`studies/coupler_accuracy/capture_cube_wake_baseline.py`. It deliberately refuses
to overwrite an existing snapshot. No numerical experiment or solver-source
change was made before this baseline commit.
