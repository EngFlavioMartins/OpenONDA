# Portable simulation output

Keep a local OpenONDA installation on each computer and put large run directories
on an external SSD. CSV, HDF5, NPZ and VTK output can be read on macOS and Ubuntu.
Moving data is simpler than moving a running calculation: restart also requires
matching code, configuration, mesh partition and a compatible local runtime.
Cross-operating-system continuation has not been qualified for these cases.

## Storage choice

An external SSD is a practical working disk for these repeated field writes.
Use a hard disk for a second archive copy if its lower speed is acceptable.
A small USB flash stick is a poor working-disk choice for sustained simulation
output. Choose capacity from measured frames and allow several times the current
estimate for particle growth, checkpoints, temporary writes and other cases.

For one removable data volume shared by current macOS and Ubuntu, exFAT is a
practical common format. macOS supports it in
[Disk Utility](https://support.apple.com/guide/disk-utility/file-system-formats-dsku19ed921c/mac),
and Linux provides an [exFAT driver](https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/b59e4cae34bfc7f6770047e4dba05faa0780c745/fs/exfat/Kconfig).
Use it for data, not a shared Conda environment or the only copy of results.
The base format does not provide the transaction-safe extension described in
[Microsoft's specification](https://learn.microsoft.com/en-us/windows/win32/fileio/exfat-specification).
Stop writes and eject cleanly before disconnecting. Formatting an occupied disk
erases it; no disk formatting is required to use the already-mounted paths below.

For the most robust arrangement, calculate on APFS on the Mac or ext4 on Ubuntu,
then copy completed output to shared storage. A Linux workstation/NAS serving
the archive over SMB avoids changing disk formats between computers. With limited
internal space, writing directly to a reliably connected external SSD is a
reasonable compromise, provided there is a second copy of important results.

## Run into an external directory

The following commands run from the repository root on macOS. Replace the mount
name with an existing directory on your SSD, and choose a new run directory.
On Ubuntu, use its mounted path, commonly `/media/<user>/<volume>/...`.

```bash
python -m studies.panel_removal.run_cube \
  --output /Volumes/ONDA/runs/cube_panel_free --end-time 30 --snapshot-interval 0.25
python -m studies.panel_removal.run_cylinder \
  --output /Volumes/ONDA/runs/cylinder_planar --end-time 100 --snapshot-interval 0.24
```

These are the panel-free study entry points. The ordinary coupled tutorial
`allrun.sh` files still select their panel-based setups and invoke `allclean.sh`;
do not execute them over results you intend to keep. The cube study requires the
original tutorial `constant/mesh.npz`. Preserve that input when moving the code
or reproducing this particular mesh. The cylinder study requires the recorded
passing cube gate. Neither command copies the currently running trajectory.

Reference `setup.py` supports `--output-root /external/path`. To run an entire
reference campaign externally, replace the literal output-root paths in its
`allrun.sh` with directories on the external volume. Preserve the temporal cube
command's `--mesh` path to that campaign's exact fine native mesh. The launchers
remain direct Python commands; no shell environment or runtime wrapper is needed.

Future default settings are:

| Case | Horizon | Retained volume frames and restart checkpoints |
| --- | ---: | ---: |
| Cube reference fine | 30 s | .25 s |
| Coupled cube | 30 s | .25 s |
| Other cube reference grids/temporal control | 120 s | Existing cadence |
| Cylinder reference fine | 100 s | .25 s |
| Coupled cylinder | 100 s | .24 s |
| Other cylinder reference grids/controls | 100 s | Existing cadence |

The cylinder coupling step remains .04 s. An exact .25 s coupled output interval
would fall between accepted states; .24 s is six steps and does not require
interpolation or a new time discretization. Reference adaptive time stepping
lands on its .25 s output events. Check each file's saved physical time.

These defaults configure new runs. The already-running 20 s coupled cube,
160 s coupled cylinder and original reference campaigns keep their constructed
settings. The existing cylinder completion worker remains attached to its
original 160 s pair. New reference output directories are
`geometric_r15_30s_fine` and `geometric_xy_100s`, separate from that evidence.

## Data volume and retention

Measurements from existing complete frames give approximate planning sizes:

| Output | Measured frame | Requested series estimate |
| --- | ---: | ---: |
| Historical cube fine FVM | 67.57 MB | 7.61 GiB for 121 frames over 30 s |
| Historical coupled cube FVM + VPM at 20 s | 53.32 MB | 6.01 GiB for 121 frames over 30 s |
| Historical cylinder fine FVM at 100 s | 8.13 MB | 3.04 GiB for 401 frames over 100 s |

These extrapolate one frame's size; they exclude mesh, samples, logs, restart
checkpoints and atomic-write headroom. Rank count and field selection can change
compression. The planar cylinder's mature particle count is not yet known, so
its small startup frame is not a reliable 100 s storage estimate.

Retain dense volume histories only for the requested fine and coupled cases.
Keep forces and line probes for every grid: they are small and support convergence
analysis. VTK frames are suitable for visualization; a rolling restart backup
retains state needed to continue, not the full animation history. Coupled backups
also publish retained VPM HDF5/VTU frames, while the FVM schedule retains its volume
frames. Preserve a complete latest checkpoint separately when archiving a run.
Do not remove files from an active checkpoint directory or a VTK collection.

## Moving and resuming

Copy completed, closed runs with an incremental tool such as
[rsync](https://rsync.samba.org/), then verify the destination with checksums before
removing the source. For these ordinary data-file trees, a portable starting
command is `rsync -rt --partial --progress SOURCE/ DESTINATION/`; `rsync -rcn
SOURCE/ DESTINATION/` checks content without changing files. Neither command
requests source deletion. A copy of actively changing output is not a consistent
restart archive unless the writer has completed and closed the chosen checkpoint.

Keep the complete directory structure: PVD collections reference PVTU/VTU pieces
through relative paths. A `.pvd` or one rank's `.vtu` alone is incomplete.
[HDF5 is a portable format](https://www.hdfgroup.org/solutions/hdf5/), but file-format
portability does not guarantee identical numerical results on different devices.

A coupled restart needs all of the following:

- `solution/backups/manifest.json` and every artifact it names, including all FVM
  rank NPZ files, the VPM HDF5 **and VTU**, and boundary-history NPZ. Integrity
  checks require even the VTU listed in the manifest.
- The exact original native mesh and matching cell/face ordering. The existing
  cube checkpoint uses four MPI ranks; the cylinder checkpoint uses two.
- The exact working source snapshot, including uncommitted/new files, configuration
  and geometry assets. A commit ID alone does not capture this working checkout.
- A separately installed compatible environment on the destination. Do not copy
  macOS Conda/native binaries to Ubuntu. Record actual package versions alongside
  the run; the repository's dependency ranges are not an exact environment lock.
- Compatible device, precision and numerical settings. Explicit backend changes
  can fail restart identity checks. `AUTO` may select another backend and is not
  evidence of numerical equivalence. Check a short continuation before committing
  to a long cross-machine run.

For the current study launchers, preserve the cube tutorial's cached mesh; the
cylinder's `--restart` loads its own `solution/fvm/mesh.npz`. Match the saved
horizon and checkpoint interval when resuming an older run: new defaults do not
silently replace its strict restart configuration.
