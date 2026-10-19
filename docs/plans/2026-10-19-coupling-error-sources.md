# Where do the coupled-solution errors come from?

Plan agreed with the supervisor on 2026-10-19. The coupled cylinder (case 01)
reproduces the fully meshed FVM reference to about 1% in velocity at equal
shedding phase and within the 2% force target at the baseline. The studies
below attribute the remaining error, and its convergence, to one of four
sources, and turn the cylinder sensitivity study into rules that do not depend
on the flow.

| Source | What it is | Studied in |
| --- | --- | --- |
| FVM discretization | cell size $h_f$, step $\Delta t_f$, convection and time schemes | S5 |
| VPM discretization | particle spacing $h_p$, core radius, step $\Delta t_p$, diffusion | S2, S5 |
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

### S5. Convergence order: FVM alone, VPM alone, coupled

- **Problem:** a Lamb–Oseen vortex carried by a uniform stream, planar. Its
  viscous solution is exact, so errors are measured against it rather than
  against a finer run. For the coupled case the vortex passes through the FVM
  box, so the interface is crossed. A vortex ring, as discussed, has no exact
  solution and would need Richardson extrapolation from a finer grid; it can
  follow as a three-dimensional check (see decision 1).
- **Runs**, each at four levels $h=h_0/2^k$, $k=0..3$:
  1. FVM alone, VPM alone and coupled at constant Courant number
     ($\Delta t\propto h$), with $h_p=h_f$ and fixed $n$, so each gives one
     combined order curve;
  2. FVM and VPM separately with a tiny time step (spatial order) and on the
     finest mesh with decreasing step (temporal order).
- **Errors:** $L_2$ and $L_\infty$ norms of velocity and vorticity, peak
  vorticity, and vortex-centre position, at fixed times.
- **Expected:** FVM second order in time (backward); in space between first
  and second order, because the limited linear convection scheme reverts to
  first order at extrema such as the vortex peak. VPM second order in time
  (RK2) and second order in the core regularization. The coupled order should
  equal the lower of the two; a lower coupled order, or a larger constant,
  isolates the interface error.

## Order of execution and cost

Runs share one laptop with the 3D cube run (about 2.5 more days); at most one
study run alongside it, three once it ends.

| Phase | Work | Runs | Wall time, alone |
| --- | --- | --- | --- |
| 0 | Case flags (`--fvm-time-step`, `--particle-spacing`, `--interface-sweeps`, `--reynolds-number`), S5 case, S2 pilot | — | 1 day of setup |
| 1 | S5 convergence | about 30 short | about 15 h |
| 2 | S4 sweeps | 5 | about 20 h |
| 3 | S1b ratio, plus 2 references | 3 + 2 | about 45 h |
| 4 | S2 particle resolution | 4 at $r=1/2$ | about 60 h |
| 5 | S3 rules and their check | 4 | about 16 h |

About 160 run-hours in total, so roughly five to six days elapsed.

## Decisions (2026-10-19)

1. Convergence problem: the convected Lamb–Oseen vortex. The vortex ring is not
   part of this plan.
2. Generality check: the cylinder at $Re=100$ and $250$, each against its own
   reference.
3. The studies run before the NACA wing; one study run at a time alongside the
   cube run, more once it ends.
4. The convergence study is a new coupled tutorial case
   (`05_vortex_convergence`); the cylinder studies are further runs of case
   01's sensitivity study, plotted by its `allplot.sh`.

## Results

To be filled in as the phases complete.
