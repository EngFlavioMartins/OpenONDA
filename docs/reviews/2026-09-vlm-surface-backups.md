# Native VLM surface backups

9 September 2026. Output repair requested during the sequential flat-plate qualification.

The numerical HDF5 checkpoint already contained the accepted VLM surface geometry,
circulation, velocity, loads and rigid-motion state. Only the VPM particles had a
ParaView companion in `solution/`; surface PolyData was available only through the
independent `VLMSampler` under `samples/`.

Both scheduled backups and the public manual/final backup now write
`solution/<case>/vlm_<step>.vtp` and update `vlm.pvd`. The surface uses the same
accepted physical time as the VPM checkpoint. Repeated writes at the same step do
not duplicate the time index; historical checkpoints can be filled in out of
order. PolyData and its index are published atomically. The existing scientific
sampler cadence is unchanged, and no tutorial setup or shell configuration is
needed for surface backups.

The shared exporter preserves geometry precision and supplies cell fields for
circulation (active color scalar), cumulative circulation, pressure jump,
bound-point and collocation velocity, body velocity, normal, panel area/chord,
panel force, unsteady force, intrinsic force moment, and panel/wing/segment IDs.
Both `time` and VTK's `TimeValue` contain seconds. Forces retain the native
per-density convention; they are not silently converted to newtons. The existing
pressure-jump definition includes the stored unsteady pressure contribution.
This change does not modify any motion, force or wake evolution equation.

## Restoring visualization for existing results

The installed package provides a native one-time conversion command:

```sh
python -m openonda.vlm_backups path/to/solution
```

It also accepts one HDF5 checkpoint. It opens numerical files read-only and uses
their actual accepted geometry and physical fields, without running the solver,
replaying prescribed motion or reading the current tutorial setup. Older
checkpoints omit fields they did not store: in particular, unsteady force/moment
fields and static wing/edge IDs are not invented. Panel IDs and geometric area,
centre and chord can be derived directly from the saved corners.

The first complete backfill restored **252 surface files in 53 time series**:
239 flat-plate checkpoints (including retained qualification histories), twelve
rotor checkpoints and one delta-wing checkpoint. Nineteen completed canonical
flat-plate datasets were included. The active moving10° dataset was excluded;
its frozen v20 runtime predates this output repair. There are no saved quadcopter
checkpoints to convert. Delta and rotor remain interrupted, unqualified runs;
making their saved surfaces visible does not change that status.

Every generated surface was read through VTK and compared exactly with its HDF5
geometry, circulation, normals, bound velocity and stored force/moment fields.
Every indexed time was read through the ParaView-compatible PVD reader.
SHA-256 checks confirmed **576 existing checkpoint, metadata and CSV files were
unchanged**. The surface companions occupy 5,485,094 bytes in total. Detailed
paths, versions, timestamps and hashes are in the external audit workspace's
`vlm-surface-backfill-report.json`; no extra tutorial metadata was written.

## Verification and remaining rollout

- Six focused export/sampler tests passed, including exact double-precision
  moving geometry and independent sample output.
- Thirteen integration/regression tests passed (overlapping five of the focused
  tests), including four actual coupled restart continuations, sparse/manual
  backup dispatch, PVD resume and particle-only backup behavior. No failures or
  skips; runtime 126.391 seconds. Logs and JUnit results are under
  `/tmp/openonda-vlm-surface-backup-*`.
- Ruff and the scoped whitespace check passed.
- Keep the running frozen v20 flat sweep unchanged. Backfill its newly completed
  datasets with the same native exporter and refresh the external verification
  report after the sweep. This is visualization conversion, not a simulation.
- Final normal installation and the overall commit remain part of the parent
  sequential tutorial qualification. Do not alter another task's active runtime.
