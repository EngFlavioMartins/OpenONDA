# Vortex interactions — LBM Fig. 5

Run from this directory with the OpenONDA environment active:

```sh
bash allrun.sh
bash allplot.sh
```

`allrun.sh` first calls `allclean.sh`, which deletes this tutorial's previous
`solution/`, `samples/`, `figures/`, and `study_results/` data. The scripts use
plain `python` from the installed OpenONDA environment, with no `PYTHONPATH`
override.

Because OpenONDA is installed in that environment, a single case also works
from another directory with `python /absolute/path/to/setup.py baseline`.

[`setup.py`](setup.py) is the single active case definition. It builds the
unperturbed Re=3000 CS/DNS baseline, then adds either stretching viscosity or
moment-preserving Pedrizzetti relaxation. Earlier ring studies motivated these
two simple candidates; their performance with this Fig. 5 baseline is untested.
The conservative particle transfer is common to every case, including the
baseline; it maintains resolution without explicit damping or projection, but
has finite transfer error.

The physical and numerical baseline is R=1, circulation=π per ring, initial
separation=1, Gaussian physical core=0.1, h=initial σ=0.05, dt=0.00375,
SSPRK3, transposed tree induction, molecular viscosity without an LES closure,
and zero imposed disturbance. Cheng et al. use a 0.05 R0, mode-8 axial
perturbation only for their separate Re=3415 Fig. 3 validation; their Re=3000
Fig. 5 calculation explicitly excludes flow instability. Every case requests
2400 steps (physical t=9, or nondimensional T=t Γ0/R0²≈28.3), subject to
native particle, memory, and numerical-health limits. The shared particle
capacity is 600,000: Gaussian remeshing exceeded the earlier 120,000-particle
cap around steps 1380–1440 while retaining the 0.003 tail budget. A stopped
method does not prevent later commands in `allrun.sh` from running.

Each run starts at t=0 and writes to matching
`solution/<method>/` and `samples/<method>/`
directories. `allclean.sh` removes them before a full rerun. For one method
run directly, clear that method's old output directories first; the solver
appends samples and rejects duplicate initial times.
The old 120,000-particle CS/LES checkpoints cannot be loaded with this CS/DNS
configuration: the solver checks the closure, remeshing limit, and capacity as part of its
restart identity. Start a new run after retaining any failed-run output you
want to inspect. `allrun.sh` deletes the old output directories before it runs.

The baseline is a comparison with Fig. 5, not an exact LBM reproduction: its
treecode solves an unbounded problem, whereas the paper uses periodic boundaries
in a 20 R0 by 7 R0 by 7 R0 box. The particle spacing is 0.05 R0, versus the
paper's LBM grid spacing of 0.005 R0. Spatial/time convergence and the effect
of the transfer-only remesh have not yet been established. The two added
stabilization cases are numerical-method comparisons, not settings used in the
paper.

`group_id` identifies the initial ring only. Gaussian remeshing assigns new
particle labels from the nearest old particle, so grouped particle diagnostics
cannot represent separate physical rings once their vorticity overlaps. The
reference comparison instead tracks maxima of the sampled vorticity field and
stops assigning two identities when the peaks become ambiguous. These tracks
describe separated Eulerian cores, not material ancestry or a measured merger
time. A run without remeshing would retain particle labels but still would not
give diffused vorticity a unique ring of origin; such a run needs its own
resolution and overlap checks before its core trajectories can be trusted.
To compare with the material-particle core shapes in Fig. 4, advect separate
passive markers initialized in each ring with the computed velocity; keep those
markers independent of the vortex-particle remesher. Their labels would track
fluid parcels, not ownership of the merged vorticity field. Before claiming
Fig. 5 core trajectories or merger timing, repeat peak extraction with finer
sampling, varied peak thresholds, and more than one azimuthal section.

`allplot.sh` reads these three cases and writes meridional vorticity sections,
diagnostic histories, and leapfrogging kinematics to
`figures/leapfrogging_study/`. It writes PNG figures by default; use
`./allplot.sh pdf` for PDF figures or `./allplot.sh png` to explicitly select
PNG. All figures use the thesis Matplotlib template at 12.5 cm width,
10.95 pt NewPX text, a complete axes box, and no background grid. The stable
case colors are black (baseline), purple (stretching viscosity), and dark
yellow (moment-preserving relaxation); markers also distinguish the cases.
Solid and dashed curves distinguish the two tracked field cores. The supplied
LBM trajectory is the unperturbed Fig. 5 case; it does not provide a time or a
three-dimensional breakdown field. See [reference provenance](assets/references/README.md).

`assets/` contains the figure scripts, the shared `postprocess.py` plotting
helper, and the digitized LBM reference. The generated `figure_manifest.json`
files link exports to their scripts, sampled fields, run statuses, and hashes.
`allplot.sh` uses whatever sampler output has been saved so far, including
partial runs. Those plots are previews until the runs and comparison have been
validated; only then should an accepted PDF be copied into the Thesis
`thesis_visuals/chapter5/vortex_interactions/` directory and included at its
natural size, with its caption and figure label in LaTeX.
