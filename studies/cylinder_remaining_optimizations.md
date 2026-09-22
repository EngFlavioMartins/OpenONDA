# Cylinder optimization follow-up

Preserve Re=150, physical horizons, vorticity cutoff, solver tolerances and the
maximum interface-sweep budget. Keep experimental algorithm changes opt-in
until matched coupled results justify promotion. Existing dirty work is retained.

- [x] Revisit measured phase costs and existing caches before adding new ones.
- [x] Replace competing slab-image atomic writes with target-local reductions;
  check velocity, gradients, stretching, tiling and tail decisions.
- [x] Reuse PETSc normalization vectors and the already-computed matrix/guess
  product; retain the exact current matrix and residual acceptance rules.
- [x] Implement experimental safeguarded Aitken trace acceleration, including
  coherent rollback, flux consistency, MPI decisions and configuration identity.
- [x] Benchmark each accepted optimization on identical inputs, separating
  cold setup, component timings and complete coupled interval timings.
- [x] Run matched short coupled comparisons and restart/configuration checks.
- [x] Update the execution report with measured gains, rejected ideas and limits.

Evidence: [execution report](cylinder_execution_report.md),
[coupled comparison](cylinder_optimization_comparison_2026-09-22.json),
[slab component benchmark](cylinder_slip_slab_accumulation_2026-09-22.json),
and [PETSc component benchmark](petsc_partitioned_workspace_reuse_benchmark.json).
All 80 targeted tests passed, including two-rank rejection rollback and restart
configuration identity; the affected 12 tests passed again after typing fixes.
The optimized installed wheel passed the outside-checkout solver verifier.
Ruff passed; Pyrefly decreased from the existing 123-error baseline to 116.

The existing three-sweep limit means within-interval Aitken, which needs two
ordinary residuals, can improve the third-sweep residual but cannot save a sweep
on that configuration. Do not claim a runtime gain from that alone. The 12-hour
production target and full scientific qualification remain separate open gates.
