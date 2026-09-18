# Two heaving delta wings

Run from this directory using the installed OpenONDA environment:

```bash
./allrun.sh
./allplot.sh
./allplot.sh pdf
```

`allrun.sh` cleans generated output and starts a fresh simulation. The authored
case spans **20 s / 8000 steps**, with a 0.0025 s time step and 1 Hz prescribed
heave. Completion checks read the horizon from native solver metadata, including
for a declared checkpoint continuation.

Two wings heave out of phase and pitch with the changing incident flow. The
downstream wing crosses the upstream wake. Initial geometry pivots precede
translation; prescribed-motion pivots use the world frame. The case uses
CPU/FMM induction, Gaussian particles, wake-core overlap 2.5 and no Pedrizzetti
relaxation. These inputs remain in `setup.py` and native metadata.

The VLM boundary solve, wake induction, forces and plotted field samples remain
active. The model has no particle/wall collision law; assessing boundary
resolution requires a separate convergence study.

## Figures and incomplete runs

`allplot.sh` reads `solution/vpm_metadata.json`. A failed or interrupted run can
produce **clearly labelled partial diagnostics** in `figures/partial/`, without
promoting its lineage to accepted or producing a final animation. It plots the
available force, centroid, power and vortex-strength histories and the latest
common native timestamp of the three wake planes. Fewer than two sampled
heave-velocity peaks cannot establish a period; cycle comparisons are then
skipped. Partial output is not evidence of a completed or converged experiment.

For completed runs, the native output and source lineage are validated before
full-run figures and the GIF are generated. The figure set contains:

- `delta_wing_forces`: vertical force, sampled centroid height and motion input
  power, with a shared time axis.
- `delta_wing_force_cycles`: the last three complete measured heave cycles,
  separated by wing.
- `delta_wing_circulation_history`: the sum of particle vector-strength
  magnitudes, in m³/s. This is not a conserved scalar circulation.
- `delta_wing_wake_streamwise` and `delta_wing_wake_vertical`: separate,
  compact three-plane figures. Completed-run fields are integrated over one
  full measured period using trapezoidal weights, with linear interpolation
  only at the integration endpoints. Insufficient temporal coverage is rejected.

Figures retain **12.5 cm width and 10.95 pt NewPX/Palatino text**. Colours come
from the shared thesis palette: teal/purple distinguish the wings; the wake
uses white-to-teal and teal-white-purple scales. The measured outer y-axis text
has 5.5 pt left clearance, and the plotting area's right margin equals its
left margin. Heights are 10.5 cm for histories, 8.3 cm for cycle comparisons,
7 cm for strength and approximately 8.15 cm for each wake field. Exports are
not cropped or scaled down to fit.

## Native data and validation

Force, loading and motion tables are sampled every accepted step under
`samples/delta_wing/`. Flow integrals and wake planes are sampled every 10 steps.
`solution/` contains coupled H5/XDMF/VTP backups every 10 steps (0.025 s).
Open `solution/vlm.pvd` for the VLM surface series and `solution/vpm.pvd` for
the particle series. The immutable VTP and XDMF/HDF5 frames are below `vlm/`
and `vpm/` respectively; saved owner steps and times match.

For explicit validation after completion:

```bash
python assets/postprocess.py finalize
python assets/postprocess.py validate --pre-plot
```

The accepted lineage in `assets/delta_wing_accepted_lineage.json` supports one
fresh run or a declared fresh prefix and a separate continuation namespace.
Individual plotters accept explicit `--samples` paths for forensic inspection;
the wake plot also accepts matching `--solution` paths. The default individual
plotters retain accepted-lineage checks.

The completed-run animation is `figures/delta_wing_30fps.gif`. Its JSON sidecar
records exact native backups, source namespaces and presentation timestamps.
The renderer selects distinct nearest native states at 30 fps; it does not
interpolate geometry or loads. A fixed oblique projection uses
`s = x + 0.28y` and `h = 0.72y + z`, with one shared thesis circulation colour
scale. The same 12.5 cm width and thesis text size apply.

VLM provides attached-flow circulation and Kutta–Joukowski loading. It does
not model a separated leading-edge vortex or viscous stall; 15° incidence is
not experimental validation of those effects. Solver completion alone does
not establish cycle convergence. Persistent cycle drift requires a longer run
or an explicitly statistical analysis.
