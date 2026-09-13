# Real tandem VPM--VLM surface interaction

Run `python setup.py` (or `./allrun.sh`) from this directory.  The case
declares one finite Gaussian vortex ring and two translated delta-wing VLM
surfaces.  The VPM owner advances the ring and the shed wake; the VLM boundary
response is `responsive`, so every particle RK stage receives a temporary
stage-consistent global surface solve.  Only the accepted solve emits wake
particles and load history.

This is an inviscid attached-flow qualification case, not a no-slip or
separation model.  `surface_event_policy="warn"` retains finite-surface
intersections, edge bypasses, and core overlaps as unresolved observer records;
it does not delete, reflect, clip, or transfer particles.

The real run writes inspectable owner outputs under `samples/tandem/`:

- `vlm_surface_forces.csv` and `vlm_forces.csv` for accepted per-surface and
  aggregate loads, including peak and integrated-load inputs;
- `vlm_leakage.csv` and `vlm_surface_leakage.vtp`, with independent on-surface
  probes, two-sided traces, and the finite-target transport operator label;
- `vlm_surface_events.csv` and `.vtp`, with provenance, process, panel,
  accepted interval, crossing position, and core scale;
- `qualification_summary.csv` and `qualification_manifest.json`, generated
  from those actual tables; and
- `solution/tandem/` VPM backups and metadata for continuation checks.

Use `assets/run_qualification_studies.py` to produce separate time-step,
surface-resolution, particle-resolution, two-way tandem, restart, and
precision/backend evidence tables.  Use `./allplot.sh` for the compact final
load figure and `./allclean.sh` to remove generated results.
