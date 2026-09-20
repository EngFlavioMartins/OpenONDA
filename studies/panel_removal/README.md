# Can the coupled solver dispense with panels?

This document records the initial diagnostic stage. Subsequent implementation,
physical-run qualification and reference campaigns are described in
[EXECUTION.md](EXECUTION.md); the numerical planar model is documented in
[planar_model.md](planar_model.md).

Study performed on 2026-09-20 against the working checkout and existing saved
solutions. The checkout already contained substantial uncommitted solver fixes.
The reconstruction measurements below precede the evolving-trajectory
experiments and reference-campaign changes documented in EXECUTION.md.

## Decision

**Pursue a panel-free coupling formulation, but do not simply disable panels
in the current cylinder tutorial and call it equivalent to the reference.**
The cube results make this direction promising. The cylinder first needs a
consistent spanwise model. Neither a repaired panel mesh nor deleting the panel
solve supplies the missing spanwise wake.

The tests below establish reconstruction feasibility and the size of a frozen
panel correction. They do **not** establish sustained panel-free accuracy,
force agreement, shedding frequency, stability, or simulation speedup.

## What the panel currently does

The FVM imposes the viscous wall condition and produces wall vorticity. Renewal
transfers that vorticity to particles. In `coupling_scope="fvm_vpm"`, the source
panels are refreshed against the particle cloud, and their velocity and gradient
enter the exterior boundary trace and particle Runge--Kutta stages. Removing
panels therefore changes transport and stretching as well as the FVM boundary.

A harmonic body correction is **not mathematically indispensable in every
FVM--VPM formulation**. For a stationary no-slip body, extend the exact fluid
velocity by zero into the solid. The extension is continuous at the wall and
divergence-free, with no velocity-jump vortex sheet. Under the appropriate
unbounded-domain conditions, its complete vorticity and freestream determine
the velocity through Biot--Savart. This argument is dimension-independent.
For moving bodies, incomplete vorticity, different boundary conditions, or a
discrete transfer that fails to preserve that extension, extra treatment is
required. An inviscid slip field with zero volume vorticity is also a different
case and does require its body correction.

Published two-dimensional hybrid work demonstrates a near-wall correction that
avoids a separate VPM wall treatment; this supports the approach, not the
accuracy of this implementation. See Billuart et al.,
[A weak coupling between a near-wall Eulerian solver and a Vortex Particle-Mesh
method for the efficient simulation of 2D external flows](https://www.sciencedirect.com/science/article/pii/S0021999122007896).
The related OpenFOAM hybrid study by Pasolari et al. discusses this distinction:
[Coupling of OpenFOAM with a Lagrangian vortex particle method for external
aerodynamic simulations](https://doi.org/10.1063/5.0165878).

## Cylinder: the span mismatch is measurable

The current FVM spans `z = [-0.5, 0.5]`, with slip end boundaries. Its transfer
box spans approximately `[-0.48125, 0.48125]`. The closed panel cylinder extends
to `z = ±6`, and the VPM domain to `z = ±6.6`. The induction operator does not
periodically replicate the one-span wake. The fully meshed reference is
quasi-two-dimensional. Thus the hybrid and reference do not currently describe
the same exterior vorticity distribution.

For a straight spanwise vorticity element of length L, at midspan and radial
distance r, the finite-span induction relative to infinite-span induction is

`(L/2) / sqrt(r² + (L/2)²)`.

For L=1 and r=1 this is only 0.447. Increasing the panel-body length cannot
change that factor for the particle wake.

The reconstruction uses saved fine-reference cell vorticity and velocity at
t=80, 90 and 100 s. Each native midspan polygon carries its saved constant cell vorticity.
Its induced velocity is integrated using an independent contour formula,
including exact analytical integration along the prescribed span. There is no
panel solve in this reconstruction. The reference velocity uses a 12-neighbour
affine fit at 320 points around the current FVM box at z=0.

All errors below are RMS vector errors divided by U∞=1 m/s. They are boundary
velocity errors, not drag errors, whole-domain errors, or correlation coefficients.

| Vorticity representation, no panels | t=80 s | t=90 s | t=100 s |
|---|---:|---:|---:|
| Full reference x-y wake, length 1D | 15.86% | 15.31% | 14.60% |
| Same wake, length 12D | 0.335% | 0.357% | 0.338% |
| Same wake, infinite span | 1.019% | 1.032% | 0.981% |
| Infinite span, restricted to current VPM x-y bounds | 1.674% | 1.704% | 1.352% |
| Infinite span, only current transfer-box vorticity | 23.34% | 24.54% | 23.76% |

The last row deliberately discards the outer wake; it is an omission control,
not a replay of the production coupler, which retains outer particles.
The 12D result is **not** evidence that 12D is the physically correct model:
finite-span and finite-reference-boundary errors can cancel. The appropriate
comparison to this reference is the span-consistent limit. The remaining
roughly 1% includes finite-domain boundary effects, stored cell-vorticity
error, and reference interpolation error; this study does not separate them.

Quadrature orders 4, 8 and 16 give essentially unchanged infinite-span RMS
error. A preliminary cell-centre point-vortex sum was substantially less
accurate in the wake; the table uses polygon integration instead.

## Cylinder: panel formulation versus panel discretization

The cylinder STL contains 1,280 triangles, with maximum longest/shortest edge
ratio about 122.25. The cube contains 108 triangles with ratio about 1.41.
More specifically, **zero cylinder panel collocation points lie in the
resolved FVM span**. This mesh can represent an almost span-uniform incident
field much better than a wake concentrated around z=0. This is a concrete
spatial-resolution problem, not proof of a broken linear solver.

The clean cylinder panel solution passes a useful analytical check. At r=1.5D,
its difference from infinite-cylinder potential flow is 0.338% U∞ RMS, for
both f64 and f32 panel fields on CPU. The finite 12D cylinder is not exactly
the infinite analytical body, so that difference includes end effects. The
f64 collocation no-penetration RMS is about 2.7e-14 m/s; f32 gives about
7.7e-6 m/s. These checks do not qualify the Metal backend or arbitrary wake
incidence.

With the saved t=100 reference vorticity represented as an actual 1D slab,
an independent 3D Gaussian volume sum gives approximately 14.47% boundary
RMS error without panels and 10.84% with the current cylinder panels. The
panel system solves successfully, but it cannot repair the missing exterior
wake. This is a controlled diagnostic using reference donors, not a saved
coupled-cylinder trajectory.

A panel route remains possible: use a matching physical span and a panel mesh
that resolves the incident field in both circumferential and axial directions;
then refine against off-collocation velocity and wall-normal residuals. Merely
raising `max_n_panels` changes allocation capacity, not the STL resolution.
The implementation accepts audited closed orientable triangulated bodies; this
does not make every arbitrary STL or discretization accurate. A low algebraic
residual is insufficient evidence of physical boundary accuracy.

## Cube: two independent checks support panel-free development

First, integrate the full fine-reference cell vorticity directly in 3D,
without panels. Use 216 probes on the six current FVM-box faces, a
12-neighbour affine reference velocity, and an independent Gaussian
Biot--Savart sum with standard deviation `sigma = 0.5 cbrt(cell_volume)`.

| Time | RMS boundary error without panels |
|---|---:|
| 5 s | 0.151% |
| 10 s | 0.193% |
| 20 s | 0.321% |

Changing sigma/h to 0.25 gives 0.161%, 0.198% and 0.293%; changing it to 1
gives 0.300%, 0.519% and 0.659%. These are quadrature/regularization
sensitivities, not a converged production particle-spacing study.

Second, read the **actual saved coupled particle clouds** at the same times.
Use their strengths and core radii, and recompute the cube-panel correction
against each frozen cloud. This tests whether panels are responsible for the
observed boundary agreement of those clouds.

| Time | Particles | With panels | Without panels | Panel velocity contribution, RMS/U∞ |
|---|---:|---:|---:|---:|
| 5 s | 148,051 | 1.257% | 1.270% | 0.0418% |
| 10 s | 311,827 | 1.320% | 1.343% | 0.0532% |
| 20 s | 496,694 | 5.779% | 5.797% | 0.0451% |

The panel contribution at these probes is small compared with the existing
particle-cloud error. The late-time maximum error is about 36% U∞ in both
replays, so an RMS number should not be mistaken for uniformly good agreement.
The clouds were evolved **with panels**. Removing panels throughout their
history could still alter near-body transport, stretching, interface
convergence and forces. The mixed boundary also uses tangential derivatives;
this study does not qualify their accuracy.

## Concrete implementation and validation sequence

1. Retain the present cube as an A/B baseline. Introduce an explicitly selected
   panel-free experiment, with independent body geometry/masking. Currently
   the cylinder's GBD mask is configured inside panel initialization; setting
   `panel_solver=None` can also lose solid exclusion. Preserve stationary
   no-slip extension and prevent particle regeneration in the solid.
2. For the cylinder reference, implement a consistent quasi-2D or spanwise
   periodic induction/transfer/diffusion path. A 2D path can also avoid the
   current large 3D GBD allocation. Repeating vorticity across a long finite
   span is a sensitivity experiment, not an exact periodic implementation.
3. Check near-wall vorticity transfer, discrete solenoidality in 3D, outer-wake
   retention, startup, and both velocity and gradient boundary traces. Preserve
   circulation and relevant moments, but do not use those budgets as substitutes
   for local velocity accuracy. Avoid retuning the panel-free run to match one
   snapshot.
4. Run clean panel-on/off cube trajectories and a matched cylinder trajectory.
   Compare mean Cd, RMS Cl, shedding frequency, mean/RMS wake profiles, boundary
   error distributions and residuals over common mature time windows, without
   phase fitting or force rescaling. Also inspect transient differences.
5. Suggested engineering gates, to agree before claiming success: changes
   within 2% in mean drag and shedding frequency, 5% in RMS lift, and 2% in
   mean-profile relative L2 error, together with stable accepted steps. These
   are proposed tolerances, not achieved results or published standards.
   Reference mesh/time uncertainty must be included: its own fine/dense lift
   difference is already about 2.4% in the recorded study.
6. Measure end-to-end wall time at matched accuracy. The existing timing
   categories combine panel and non-panel work, so they cannot establish a
   panel-removal speedup. No speedup claim is made here.

The present evidence supports this sequence more strongly than spending time
repairing the cylinder panel mesh alone. It does not justify replacing the
working default with an unvalidated panel-free trajectory.

## Reproduce and inspect

From the repository root, in the OpenONDA environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m studies.panel_removal.diagnose \
  --output studies/panel_removal/results.json
python -m pytest tests/coupler/test_panel_removal_study.py
```

Use `--case cylinder` or `--case cube` for one part. The exact saved reference
frames and cube HDF5 particle files must be available; missing data produces
an error rather than an interpolated or silently substituted time. Output
includes source hashes (including MPI pieces), study-code hashes, raw errors,
and panel diagnostics. Timings are CPU diagnostics with JIT/cache effects,
not production simulation benchmarks. Only the requested JSON file is written.

The four analytical tests check the Rankine-vortex limit, polygon orientation,
finite-filament span factor, 3D vector/sign conventions and Gaussian core
regularity. They pass. The study's operators are intentionally separate from
production induction, FMM, transfer and time integration.

Added files: this report, `diagnose.py`, `operators.py`, `results.json`, and the
four-test diagnostic module. No production solver, tutorial configuration,
reference result, or existing troubleshooting report was edited by this study.
