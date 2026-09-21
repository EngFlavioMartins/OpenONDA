# Solution layout

Each run keeps its ParaView entry points at the top of `solution/` and stores
the larger immutable frame files by physical representation:

```text
solution/
  fvm.pvd                 # resolved Eulerian field time series
  vpm.pvd                 # particle time series
  vlm.pvd                 # lifting-surface time series, when present
  fvm_metadata.json       # FVM configuration and accepted-clock record
  vpm_metadata.json       # VPM configuration and accepted-clock record
  run_metadata.json       # coupled-run record, when present
  fvm/
    mesh.npz              # native mesh for Python readers
    mesh.vtu              # mesh for ParaView
    fvm_*.vtu|pvtu        # field frames and MPI pieces
  vpm/
    vpm_*.h5              # particle fields and explicit VPM restart targets
    vpm_*.vtu             # ParaView particle frames indexed by vpm.pvd
  vlm/
    vlm_*.vtp             # surface frames
```

Open a root-level `.pvd` file in ParaView. The collection stores relative
paths, so copying or moving the entire `solution/` directory preserves the
series. Python post-processing should read `fvm/mesh.npz` for native FVM mesh
data, `vpm/*.h5` for particle states, and the root metadata records for the
executed configuration and clock. ParaView should open the root `.pvd`
collections; the VPM collection references `.vtu` frames because ParaView's
PVD reader does not accept XDMF collection members.

Standalone VPM restart uses an explicit `.h5` file. FVM backup locations are
set by `BackupConfig`; coupled runs keep a manifest and both solver states in
their configured backup directory (normally `solution/backups/`). Keep those
files together. Visualization files alone cannot restart a coupled run.
