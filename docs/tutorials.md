# Running tutorials

Install OpenONDA, then list the available cases:

```bash
openonda tutorial list
```

To copy and run the small Taylor–Green FVM example:

```bash
openonda tutorial run fvm/taylor_green --workspace ./first-flow
openonda tutorial plot fvm/taylor_green --workspace ./first-flow
```

To copy a case without running it:

```bash
openonda tutorial create vpm/vortex_ring ./ring-workspace
cd ring-workspace/tutorials/vpm/02_vortex_ring
```

The copied `setup.py` contains the physical parameters, resolution, and run
length. Edit it before launching a long simulation. Existing workspaces are
not overwritten by `create`; `run` reuses them.

## From a case directory

```bash
python setup.py       # Start at zero, or resume the latest backup.
./allrun.sh           # Clean first, then run the listed cases from zero.
./allcontinue.sh      # Run the same cases without cleaning.
./allplot.sh          # Plot saved results as PNG.
./allplot.sh pdf      # Export PDF instead.
./allclean.sh         # Delete generated output for this tutorial.
```

Use the active Python environment; no import-path or interpreter variables
are needed. Shell scripts can also be invoked from another directory.
**Every `allrun.sh` calls `./allclean.sh` before running.** Save results you
need before invoking it. Every tutorial, including reference flows and rotor
studies, supplies `allcontinue.sh`. Individual `setup.py` invocations also
continue automatically; the visible `START_FROM = "latest"` setting controls
this behavior. A completed case performs no additional timesteps. Increasing
the configured final time (or VPM total step count) extends the same case.
See [continuation and backups](continuation.md) for recovery and compatibility.
Running and plotting are separate operations.

Variant arguments select physical or numerical comparisons. For example:

```bash
# tutorials/vpm/01_lamb_oseen_vortex
python setup.py vortex CS

# tutorials/vpm/03_vortex_interactions
python setup.py baseline
python setup.py selective_eddy_viscosity
```

## Reference grid studies

The coupled [cube](../tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/README.md)
and [cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow/README.md)
each have a separate `reference_flow/` directory. Its `allrun.sh` lists the
geometrically spaced grids explicitly. For example, from the cylinder reference:

```bash
python setup.py --name grid_h004 -h 0.04
python postprocess_grid_study.py
```

`--name` selects the output case and `-h` specifies the baseline spacing in
metres. Use `--help` for usage (`-h` means spacing here). One post-processing
file reads the completed grids and compares forces. These reference domains
are larger than the coupled FVM domains; do not substitute one for the other.
See each reference README for its current grid sizes and averaging interval.

## Parallel runs and output

Set `cores` in the FVM case or setup to request MPI workers. The solver factory
launches them, builds the mesh once, and configures numerical thread counts.
Tutorial scripts do not need MPI rank checks or `mpiexec` commands.

The solvers write their configuration to `fvm_metadata.json` and
`vpm_metadata.json`; plotters read those records and the saved samples.
See [solution layout](solution_layout.md) for ParaView and restart files.
Shared plotting functions live in `openonda.plotting`.

The rotor and quadcopter directories marked `PENDING` are unfinished studies,
not validated reference solutions. The other tutorials also need resolution
and time-step checks before their results are used quantitatively.
