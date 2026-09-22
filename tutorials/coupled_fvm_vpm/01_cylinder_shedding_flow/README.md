# Flow past a cylinder at Re = 150

A body-fitted finite-volume region resolves the cylinder wall and transfers
vorticity to the surrounding vortex-particle domain. The diameter and
freestream speed are 1 m and 1 m/s, giving kinematic viscosity 1/150 m²/s.
The compact FVM box is
`[-1.60, 1.60] × [-1.60, 1.60] × [-0.48, 0.48] m`. Its base setup uses the reference
fine spacing `h = 0.04 m` uniformly in every direction, giving 24 cells across
the resolved span. There are no mesh-refinement or coarsening regions. Force
coefficients use the resolved 0.96 m span.

Coupled runs end at 100 s, retaining volume fields and coupled checkpoints
every 0.24 s (six 0.04 s coupling steps). This preserves the tested time
discretization; an exact 0.25 s event would fall between accepted coupled
states.

`setup.py` shows the coupled configuration. The case is full 3D: it uses
`SlipSlabInduction` with free-slip image planes at `z = ±0.48D`, and the VPM
and FVM domains use the same physical slab. The spanwise lattice closes the
span exactly; it does not stretch or average the z direction.

`allrun.sh` cleans and runs `setup.py`; `allcontinue.sh` resumes its latest
atomic coupled backup. `python setup.py` also resumes automatically.

The separate `python assets/run_pipeline.py` campaign uses the
initial candidate family `h = 0.10, 0.08, 0.064D`; every reference and
coupled case receives its own 12-hour wall limit, log, output directory and
cost summary. Resumed attempts consume the same cumulative per-case allowance.
A complete multi-case campaign can therefore take many days.
All three spacings fit the same outer XY box exactly, and explicit extrusion
preserves the same physical span. Custom spacings may resolve a larger
Cartesian box; the campaign records those bounds so domain changes remain
visible in a grid comparison.
The per-case limit is a budget and measurement boundary; it does not certify
that a finest case completed or that the target accuracy was reached. Use a
bounded pilot first.

The current measurements do **not** certify the twelve-hour target: the
finest pilot takes about 17 seconds per warm startup interval, and
the longer h=.08 screen with the same hp=.08 reaches 26 seconds by t=.8.
These timings precede the final sparse wall-image correction; the execution
report distinguishes successive solver revisions.
The gross twelve-hour allowance at exchange dt=.04 is only 17.28 seconds per
interval before startup. Treat this family as a qualification experiment,
not a proven overnight production configuration.

```bash
python assets/run_pipeline.py --pilot --sensitivity none
```

Pass campaign options to `assets/run_pipeline.py`. The campaign defaults to CPU, selected
from the laptop pilot and available without a GPU driver. Choose
`--compute-device AUTO`, `CUDA`, `VULKAN` or `METAL` for an available GPU
backend. `--coupled-cores 4` controls the coupled MPI/owner thread budget and
is propagated to the sensitivity workers; `--reference-cores 6` controls the
reference MPI ranks. Parallel execution requires a compatible MPI installation
(see [installation](../../../docs/installation.md)). A base pip installation
can run both cases serially with `--coupled-cores 1 --reference-cores 1`.
Backend performance must be measured
on the actual device; software Vulkan rendering is not a GPU benchmark.

For a full run, select `--sensitivity screen` for bounded screens or
`--sensitivity full` for the long paired study. The full study has one baseline
plus two levels for each of the eight supported factors (17 independent
cases), with at most one paired interaction case selected after the OFAT
results. The sensitivity baseline uses `hxy = 0.08` and
`particle_spacing_ratio = 1.0`, giving `hp = 0.08`; span variants keep that
particle spacing so span and particle resolution are not confounded. The
particle-spacing factor then tests ratios `1.25` and `1.5`. The remaining
factors cover core radius, blend width (4 and 7), release width, transfer
amplification cap, exchange clock, span and spanwise spacing. Interface
iteration limits and tolerances remain fixed;
`exchange_dt` changes the exchange clock and is the explicit temporal factor.
Any optional two-factor interaction is checked through the case builder before
launch; invalid body-authority combinations are recorded and skipped.
Screen runs use short bounded coupling segments and are not accuracy
qualifications. Their step budget is defined at the baseline `exchange_dt =
0.04 s` clock, so a 20-step screen uses 40 steps at `0.02 s` and 10 steps at
`0.08 s`; runtime comparisons are normalized per physical time. Full runs
retain the 100 s horizon and common statistics window. An interrupted pipeline
can be continued without cleaning with:

```bash
python assets/run_pipeline.py --run-dir <directory> --resume --sensitivity none
```

The coupled backup manifest is the restart authority. Pipeline provenance is
written to `<directory>/pipeline_manifest.json`; each case has a console log,
`trial.json`, solver timing journals and cost summary below `<directory>/logs`
and its case output directory. Reference grid reports are written under
`<directory>/reference/figures`; `reference_selection.json` records the
`force_grid_qualified` gate. The full production campaign and the 12-hour
runtime certification remain pending measured long-horizon evidence. A
successful launcher return alone is not a production qualification.

The pipeline saves comparison figures automatically. Run `./allplot.sh` to
regenerate PNG figures from the most recently updated campaign, or
`./allplot.sh pdf --run-dir <directory>` to select a saved campaign explicitly.
An incomplete latest campaign is reported as incomplete; plotting does not
silently substitute older results. The individual legacy plotting scripts
remain available for older tutorial-local outputs.

The [fully meshed reference](reference_flow/README.md) describes the explicit
reference-only pipeline and its three conservative candidate grids. Its slip
span boundaries describe a full 3D cylinder flow without endcaps.

Both cases start from the same small, divergence-free velocity perturbation.
It breaks the planar reflection symmetry and includes a span-dependent
component compatible with the slip planes. Re=150 is retained: a resolved
3D mesh is not a claim that a persistent three-dimensional wake instability
must develop. The off-midspan profiles and span/axial-resolution studies test
that response explicitly.
