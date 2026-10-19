# Where do the coupled-solution errors come from?

Plan agreed with the supervisor on 2026-10-19. The coupled cylinder (case 01)
reproduces the fully meshed FVM reference to about 1% in velocity at equal
shedding phase and within the 2% force target at the baseline. The studies
below give the overall order of the coupled method and attribute the remaining
error to one of four sources, and turn the cylinder sensitivity study into
rules that do not depend on the flow. Every study is a set of runs of case 01
(its sensitivity study), so the convergence and the sensitivity analyses form
one study, plotted by the case's `allplot.sh`.

| Source | What it is | Studied in |
| --- | --- | --- |
| FVM discretization | cell size $h_f$, step $\Delta t_f$, convection and time schemes | S5 (reference levels), S1b |
| VPM discretization | particle spacing $h_p$, core radius, step $\Delta t_p$, diffusion | S1a, S2 |
| Coupling interface | exchange ratio $n=\Delta t_p/\Delta t_f$, interface sweeps, transfer inset and blend, $h_p/h_f$ | S1, S2, S4 |
| Domain and models | FVM box size, outflow fringe, LES | done (box, inset, fringe); S3 checks transfer to other flows |

## Dimensionless parameters

Every parameter is reported as a dimensionless group, so a rule found for the
cylinder can be applied to another flow:

- FVM Courant number $C_f=U_\infty\Delta t_f/h_f$ and particle Courant number
  $C_p=U_\infty\Delta t_p/h_p$;
- particle stability number $\Lambda=\max|S|\,\Delta t_p$, which must stay
  below 1 (the 8-step cylinder run failed at $\Lambda=1.24$);
- exchange ratio $n=\Delta t_p/\Delta t_f$ and resolution ratio $r=h_p/h_f$;
- interface widths (inset, blend, particle-only band) in particle core radii
  $\sigma$;
- box margin relative to the vortex-formation length and the body size.

## Studies

All cylinder runs end at 55 s and are compared with the reference over
$40\le t\le55$ s: mean drag, lift amplitude, Strouhal number, and the
phase-averaged wake-velocity difference at $x/D=2$ and $4$. Every run also
records its wall time per second of flow time.

### S1. Exchange ratio, both directions (case 01)

- **S1a, fixed $\Delta t_f=0.008$ s, larger $\Delta t_p$.** Done for
  $n=2,4,5,6$. Larger ratios are not reachable this way: $\Delta t_p$ is
  capped by the stability limit $\Lambda<1$ at the wall
  ($\Delta t_p\le0.052$ s). The study is complete; it is re-plotted against
  $C_p$ and $\Lambda$.
- **S1b, fixed $\Delta t_p=0.04$ s, smaller $\Delta t_f$.** $n=5$ (baseline),
  10, 20, 40, so $C_f$ falls from 0.2 to 0.025. This gives the larger ratios.
  Because the reference has its own temporal error at $\Delta t_f=0.008$ s,
  the reference is also run at 0.004 and 0.002 s; each coupled run is compared
  with the reference at the same $\Delta t_f$, and with the finest one.
- **Question:** does the coupled error depend on $n$ itself (interface), or
  only on $\Delta t_f$ and $\Delta t_p$ separately (discretization)?

### S2. Particle resolution (case 01)

- $r=h_p/h_f=1$ (baseline) and $1/2$, each at $n=2,4,5,6$ with
  $\Delta t_f=0.008$ s fixed, so $\Delta t_p$ grows with $n$.
- **Pilot first:** the planar cylinder uses one particle layer between slip
  planes one FVM cell apart. A finer particle spacing needs the slip-plane
  spacing to follow $h_p$; one short run checks that the coupler accepts
  $h_p<h_f$ and that the transfer lattice and slip planes stay consistent.
- **Question:** does a finer particle field change the error trends with $n$,
  or only shift them? Does the stability limit move?

### S3. Flow-independent rules

- Express S1, S2, S4 and the earlier box, inset and fringe results in the
  dimensionless groups above, and fit the admissible ranges (for example
  $\Lambda\le0.9$, inset $\ge k\sigma$, pure-FVM core beyond the formation
  length).
- **Check on other flows:** apply the rules, without tuning, to the cylinder
  at $Re=100$ and $250$, each against its own reference. The rules
  generalize if both meet the 2% target without retuning.

### S4. Interface sweeps per exchange (case 01)

- The predictor-corrector repeats the FVM solve over each exchange until the
  boundary trace converges (at most 6 sweeps, tolerance $10^{-5}$). Fixed
  sweep counts $K=1,2,3,4,6$ (tolerance zero, so exactly $K$), plus the
  tolerance-driven baseline.
- **Question:** how fast do forces and wake velocity converge with $K$, and
  is the cost of $K$ sweeps worth it against a smaller $n$?

### S5. Convergence order of the coupled method (case 01)

- **Refinement:** the cells, the steps and the particles are refined together
  by $\sqrt2$ at constant Courant number: $h_f=h_p=0.08$, $0.0566$, $0.04$
  (baseline cells) and $0.0283$ m, with $\Delta t_f\propto h_f$ and
  $n=2$ FVM steps per exchange at every level. The baseline $n=5$ is not used
  here: at constant Courant number the particle step of the two coarse levels
  would exceed the stability limit ($\Lambda=1.10$ and $1.55$); with $n=2$,
  $\Lambda\le0.62$ at every level. The interface widths (inset, blend,
  particle-only band) stay fixed in cells, so they shrink with $h$, as the
  rules of S3 would set them; the box and the particle domain stay fixed.
- **Reference at every level:** the fully meshed FVM reference on the same
  near-wall cells and step (`--cell-size`), refined as a whole.
- **Error-free values:** the mean drag, lift amplitude and Strouhal number of
  the reference extrapolated to zero cell size (Richardson extrapolation from
  its three finest levels, with its observed order). The wake velocity has no
  scalar to extrapolate; it is measured at equal shedding phase against the
  finest reference, which overstates the observed order of the reference
  itself.
- **Split of the coupled error:** coupled error = FVM discretization error
  (reference on the same cells against the extrapolated value) + coupling
  error (coupled against the reference on the same cells). The observed order
  of each is the slope of its error against $h$. The coupling part contains
  the interface, the particle discretization and the coarser outer reference
  mesh (the reference coarsens to $2h$ beyond 1.5 D, the coupled box does
  not). S1a ($\Delta t_p$), S2 ($h_p$) and S4 (sweeps) then attribute the
  coupling part at the baseline cells, and S3 checks that the split, written
  in the dimensionless groups, holds at other Reynolds numbers.
- **Expected:** the reference between first and second order (backward in
  time; limited linear convection, first order at extrema). A coupled order
  equal to the reference's means the FVM discretization dominates; a lower
  one, with a coupling part of lower order, isolates the interface or the
  particles.
- A separate FVM-alone or VPM-alone problem is not needed: the reference is
  the FVM alone on the same flow, and the particles cannot carry the cylinder
  alone; their discretization is varied in S1a and S2 at fixed FVM cells.

## Order of execution and cost

Runs share one laptop with the 3D cube run (about 2.5 more days); at most one
study run alongside it, three once it ends.

| Phase | Work | Runs | Wall time, alone |
| --- | --- | --- | --- |
| 0 | Case flags (`--cell-size`, `--fvm-time-step`, `--particle-spacing`, `--interface-iterations`, `--reynolds-number`), launcher sections, figures (done); S2 pilot (pending) | — | — |
| 1 | S5 convergence: 4 coupled, 3 references (the baseline reference exists) | 7 | about 25 h |
| 2 | S4 sweeps | 5 | about 20 h |
| 3 | S1b ratio, plus 2 references | 3 + 2 | about 45 h |
| 4 | S2 particle resolution | 4 at $r=1/2$ | about 60 h |
| 5 | S3 rules and their check | 4 | about 16 h |

About 170 run-hours in total. Coupled runs use the GPU, which the cube run
needs alone (a second GPU user ran it out of memory), so only the FVM
references run beside the cube; the finest reference ($h=0.0283$ m) needs
more than 4.5 GB to mesh and also waits for the cube.

## Decisions (2026-10-19)

1. Convergence problem: case 01 itself, refined at constant Courant number
   (user decision, 2026-10-19, replacing the convected Lamb–Oseen vortex). The
   vortex ring is not part of this plan.
2. Generality check: the cylinder at $Re=100$ and $250$, each against its own
   reference.
3. The studies run before the NACA wing; one study run at a time alongside the
   cube run, more once it ends.
4. No new tutorial case: the convergence and every cylinder study are runs
   of case 01's sensitivity study (`assets/run_sensitivity_study.sh`
   sections), plotted by its `allplot.sh`.

## Results

To be filled in as the phases complete.
