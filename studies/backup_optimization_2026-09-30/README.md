# Backup preparation optimization, 30 September 2026

This implements the output-side improvements identified in the
[delta-wing investigation](../delta_wing_runtime_2026-09-29/README.md).
The numerical evolution algorithms and output cadence are unchanged.

## Requirements and implementation

- Keep native HDF5 schema 10.1, compute precision, filenames, directory layout,
  VTU/VTP point/cell arrays, PVD collections, restart identity and atomic commit
  ordering. No checkpoint conversion or tutorial migration is required.
- Reuse accepted transport fields only for a framework-scheduled or final
  backup in the same output event. The preparation is single use and checks
  particle revision/count, clock, background and provider identity. Manual saves
  always refresh, including saves requested inside a sampler callback. No cache
  survives an event, restart, coupling exchange or output failure.
- Accelerate large f32 Gaussian vorticity with the existing source hierarchy.
  Prune only terms beyond 12 pair-mean core radii, where the density already
  underflows to zero. Retain the original kernel, variable radii, self terms
  and particle ordering. Summation order changes float32 rounding. Planar,
  non-Gaussian and f64 calculations retain their existing paths.
- Keep Fourier diagnostics' grid, expansion order, normalization and precision.
  Reduce quadratic forms in blocks of at most 65,536 cells and retain scalars
  instead of copying all three penultimate-order spectra.
- Distinguish evolution, health, diagnostics, sampling, backup preparation and
  file-writing time. Total standalone step time includes scheduled output.

The main changes are in `core/solver.py`, `physics/base.py`, the existing LBVH
implementation, `numerics/fourier_integrals.py`, and profiler/logging labels.
FVM serialization, coupled manifests, restart selection and visualization
writers are unchanged. Coupled coordinators use the optimized VPM preparation
through their existing `_save_backup_to` entry point.

## Measurements

| Controlled workload | Before | After | Observation |
| --- | ---: | ---: | --- |
| Saved delta cloud: vorticity, 106,297 particles, two CPU threads | 135.42 s | 51.64 s warm; 55.66 s cold | 2.62x warmed speedup |
| Fourier diagnostic: 6,000 mixed-core particles, grid 160x72x64, two FFT workers | 10.20 s | 3.68 s | 2.77x speedup |
| Same Fourier process: peak resident memory | 661.0 MiB | 411.8 MiB | 37.7% reduction |

The delta comparison uses the same live device fields and the unchanged direct
sum as its reference. Relative L2 error is 6.92e-7; maximum absolute difference
is 1.07e-4 against a maximum reference magnitude of 89.31. The source checkpoint
SHA-256 matches before and after. No original solution files were rewritten.
The Fourier results, including previous-order estimates and viscous power,
agree within 2e-13 relative / 1e-14 absolute tolerance.

An additional CUDA check with 3,073 mixed-core particles, coincident sources and
1,024-target batches passed against direct summation (relative L2 error 2.87e-8).
It used a native CUDA runtime with fallback disabled.

These are component measurements on a shared machine, not full-simulation
speedups or isolated hardware benchmarks. The Gaussian comparison includes a
fresh hierarchy build on every call. The Fourier input is a controlled synthetic
long wake, not the full historical delta Fourier grid. Removing the duplicate
transport evaluation is separately verified by counting accepted-state RHS
calls: one for health plus a scheduled backup, and no extra call for a final
backup following final diagnostics. FMM evolution and VLM influence costs remain.

The JSON files retain raw results and source hashes. The three Python drivers
are the exact measurement scripts; their paths refer to frozen source trees in
`/tmp/openonda-backup-optimization/frozen` and
`/tmp/openonda-delta-profile/frozen`. They are provenance artifacts, not tutorial
launchers. Copy them into the former temporary directory alongside its frozen
source tree before executing them; `benchmark_fourier.py` accepts `before` or
`after`.

## Validation

Regression coverage checks native restart state, split trajectories, interrupted
writes, history reconciliation, ParaView readers, VLM surfaces, coupled bundles,
field-reuse invalidation and mixed-core Gaussian parity across direct, treecode
and FMM induction. Tutorial checks run setup variants and cleanup only in
temporary copies, and exercise serial and two-rank FVM continuation histories.

Two existing test expectations were corrected: larger regularization storage is
already permitted by restart policy (the old assertion also failed on committed
baseline code), and launchers now include a directory-change preamble. The
multi-stage cylinder campaign also retains its existing `--resume` stage-ledger
flag. No production launcher was changed by this optimization.

All 342 selected pytest checks passed across the main run and targeted reruns;
`validation.json` records the initial failures and their resolution rather than
presenting the first run as clean. The writer-only fixture was also updated to
provide the new preparation helper. Native CUDA parity, numerical benchmark
comparisons, Ruff checks and `git diff --check` passed. Full production-duration
tutorials were not rerun.
