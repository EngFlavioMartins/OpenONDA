# Vortex interactions — LBM Fig. 5

Run from this directory with the OpenONDA environment active:

```sh
./allrun.sh
./allplot.sh
```

`allrun.sh` first calls `allclean.sh`, which deletes this tutorial's previous
`solution/`, `samples/`, `figures/`, and `study_results/` data. The scripts use
plain `python` from the installed OpenONDA environment, with no `PYTHONPATH`
override.

Because OpenONDA is installed in that environment, a single case also works
from another directory with `python /absolute/path/to/setup.py baseline`.

[`setup.py`](setup.py) is the single active case definition. It builds the
unperturbed Re=3000 CS/DNS baseline, then adds selective eddy viscosity,
moment-preserving Pedrizzetti relaxation, or particle splitting. These are
separate numerical-method comparisons; a method may stop at its numerical
health limit before t=9, and the launcher then proceeds to the next method.
The conservative particle transfer is common to every case, including the
baseline; it maintains resolution without explicit damping or projection, but
has finite transfer error.

The case identifiers are `baseline`, `selective_eddy_viscosity`,
`pedrizzetti_relaxation`, and `particle_splitting`. The launchers list each
command explicitly; tests check them against the supported methods in `setup.py`.
The solver returns after the requested horizon, a wall-time budget, or a
declared numerical health stop. A finite health-limit crossing saves terminal
samples and a restart with status `resolution_lost`. A nonfinite particle state
is recorded as `unstable` without saving the rejected state; previously accepted
samples and backups remain. `allrun.sh` then starts the next method. An unrelated
exception, including a sampler or output failure, stops the launcher immediately.
Stopped methods are partial trajectories, not completed physical results.
Splitting is Winckelmans' local fixed-core bisection: check every five steps,
split at twice the stored reference strength, place children at ±h(t)/4,
halve strength and volume, retain core radius, and reset each child's line
reference. Both children inherit `group_id`, `zone_id`, and viscosity properties.
There is no persistent individual particle ID: array indices change on splitting
and remeshing. Capacity-limited splits report the parents left unsplit. Splitting
preserves the vector-strength sum and impulses, but it changes the field and
does not conserve energy exactly or replace core redistribution.

The viscosity model follows Winckelmans (1995), Eq. (26), positive-production
version, approximating the vorticity direction with particle strength and using
h=V^(1/3). The configured coefficient 0.5 corresponds to C_w=0.5 because
C=2 C_w². It adds diffusion where stretching produces enstrophy. New scripts,
metadata, backups, diagnostics and figures use **Selective eddy viscosity**
and `selective_eddy_viscosity_*` keys. Historical persisted keys are converted
by a read-only migration; the live configuration uses only canonical names.
The optional OpenONDA feedback extension
is disabled here. The Pedrizzetti variant adds an OpenONDA correction that
restores global vector strength and impulses after alignment.

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
The new group-preserving transfer also changes restart identity: old remeshed
checkpoints cannot recover lost ancestry by relabelling. Start the updated
comparison at t=0. The pre-audit dataset is preserved locally in
`artifacts/vpm_ring_audit_20260915/original` at the repository root.

The baseline is a comparison with Fig. 5, not an exact LBM reproduction: its
treecode solves an unbounded problem, whereas the paper uses periodic boundaries
in a 20 R0 by 7 R0 by 7 R0 box. The particle spacing is 0.05 R0, versus the
paper's LBM grid spacing of 0.005 R0. Spatial/time convergence and the effect
of the transfer-only remesh have not yet been established. The three added
stabilization cases are numerical-method comparisons, not settings used in the
paper.

The updated cases remesh each `(group_id, zone_id)` contribution separately
and correct its moments separately. Overlapping groups can therefore have
coincident particles with different labels. This costs additional particles;
the tail budget is respected for each group and capacity applies to their total.
The field-health assessment combines identical blobs so coincident labels do
not artificially improve reported resolution. `RingDiagnosticsSampler` writes
strength-weighted group centroids and radii throughout the run, including its
final state. `group_history` plots this ancestry independently of core detection.
After the rings overlap, the separate group contributions can occupy exactly
the same particle centres. A point rendering coloured by `group_id` can then
hide one label behind the other; view each `group_id` separately to inspect the
two contributions. In ParaView, apply a `Threshold` to the particle XDMF with
point-data `group_id` set to exactly 0 or exactly 1, then show one thresholded
source at a time. The remesher replaces particle indices, so an index does not
identify a continuing physical particle across a transfer. Animate the saved
frames directly; particle-index trails or interpolation across a remesh would
invent trajectories.
These centroids remain defined after merger, but they do not establish two
physical vortex cores. Older nearest-source remeshing labels cannot provide
the same ancestry information retrospectively.

The reference comparison uses positive maxima of sampled `curl(u)_z` in the
meridional plane, not velocity maxima. It follows two dominant cores while any
third peak is below half the weaker core; a strong bridge, comparable competing
peak, or clipped field stops identity assignment. Raw maxima are retained.
This fixes premature termination on a weak satellite lobe without forcing two
identities after merger. Disabling remeshing can preserve labels, but core
spreading then grows smoothing radii and the particle quadrature still distorts.
Splitting alone does not fix those errors; a no-remesh run needs separate
resolution and field-accuracy qualification.
To compare with the material-particle core shapes in Fig. 4, advect separate
passive markers initialized in each ring with the computed velocity; keep those
markers independent of the vortex-particle remesher. Their labels would track
fluid parcels, not ownership of the merged vorticity field. Before claiming
Fig. 5 core trajectories or merger timing, repeat peak extraction with finer
sampling, varied peak thresholds, and more than one azimuthal section.

The saved baseline contains faster successive overtaking. Tracking recovers
passages at t=1.1531, 3.4406 and 5.3192 s; successive intervals are 2.2875 and
1.8786 s (about 18% shorter). The paper describes contraction and acceleration
during passage; these data do not establish a sequence of decreasing full-cycle
periods or quantitative LBM convergence.

In the current comparison, selective viscosity completes t=9 while the baseline
reaches its divergence limit at t=5.98125. On x/R0=[0.55,3.5], the radius
discrepancy decreases from 10.24% to 8.06% of R0. However, the selective run's
two field maxima form the declared bridge at t=4.5, around x/R0=5.5, so a
separated pair cannot be followed into the later LBM oscillations. This is not
only a tracking cutoff: the preserved group histories show the axial separation
and radius exchange remaining small afterward.

The cause is consistent with the configured model. Positive-production
selective viscosity acts in the stretching regions that create the compact,
fast part of a leapfrog. At t=1.5 its mean effective viscosity is 1.66 times the
molecular value and its maximum is 4.59 times the molecular value. At t=4.5
those ratios are 1.41 and 3.20; E/E0 is 0.753 instead of the baseline's 0.810,
Z/Z0 is 0.208 instead of 0.331, and six conservative remeshes have occurred
instead of two. The added diffusion weakens the peaks, damps contraction and
mutual induction, and reaches the remeshing trigger more often. Selective
viscosity is therefore useful here as a stability comparison, but coefficient
0.5 is too dissipative to reproduce the compact late LBM path.

That diagnosis does not make the coefficient the only discrepancy. The LBM
reference uses periodic boundaries and R0/dx=200, whereas this VPM calculation
uses unbounded induction and particle spacing 0.05 R0. A controlled spacing,
remeshing and coefficient study is still required before attributing the full
late-path difference to the stabilization model.
Direct Gaussian summation checks near baseline cores at t=1.5, 3 and 4.5 give
tree-velocity errors of 0.025–0.043%; they do not certify stretching-gradient or
long-time accuracy. The remaining reference discrepancy needs controlled
spatial/time, remeshing and boundary-condition studies.

`allplot.sh` reads these four cases and writes every figure directly to
`figures/`. Core-track CSV tables and JSON manifests go to
`figures/auxiliary/`, leaving only figure files at the top level. It writes PNG
figures by default; use
`./allplot.sh pdf` for PDF figures or `./allplot.sh png` to explicitly select
PNG. All figures use the thesis Matplotlib template at 12.5 cm width,
10.95 pt NewPX text, a complete axes box, and no background grid. The stable
case colors come directly from `openonda.plotting.COLORS`: TUDdark navy
(baseline), VPMpurple (selective eddy viscosity), TUDcyan teal (moment-preserving
relaxation), and AccentGreen (splitting), with RefGray for LBM. Markers also
distinguish the cases. The palette matches the named colours in the thesis's
`thesis.tex`; no case-local hex colours are used.
Compact exports retain the full 10.95 pt text. The 12.5 by 15 cm core-section
figure has one method per column and one requested time per row; unavailable
run/time combinations remain blank. Its shared axes use the local axial
coordinate `(x - x_c) / R0`, with each section's vorticity-weighted absolute
centre retained in the auxiliary manifest. Trajectories are 7 cm high, group
histories are 8.3 cm, and the six-panel diagnostics are 12 cm. Legends use two
columns, and shared time axes omit repeated tick labels between rows.
The outer y-axis text sits 5.5 pt from the left canvas edge. Equal physical
aspect is preserved in every populated core-section panel.
Solid and dashed curves distinguish the two tracked field cores. The supplied
LBM trajectory is the unperturbed Fig. 5 case; it does not provide a time or a
three-dimensional breakdown field. See [reference provenance](assets/references/README.md).

`assets/` contains the figure scripts, the shared `postprocess.py` plotting
helper, and the digitized LBM reference. The generated manifests under
`figures/auxiliary/` link exports to their scripts, sampled fields, run statuses,
and hashes.
Plots read the named current run directly. Recorded source paths and hashes
identify each export; runs never fill one another's missing samples.
`allplot.sh` uses whatever sampler output has been saved so far, including
partial runs. Those plots are previews until the runs and comparison have been
validated; only then should an accepted PDF be copied into the Thesis
`thesis_visuals/chapter5/vortex_interactions/` directory and included at its
natural size, with its caption and figure label in LaTeX.
