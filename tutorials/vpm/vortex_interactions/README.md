# Vortex-ring interactions

This tutorial defines two deliberately severe vortex-ring interactions. In the
leapfrogging case, two equal-signed coaxial rings alternately contract, expand,
and pass through one another. In the collision case, two opposite-signed rings
travel towards one another and form strongly distorted vortex tubes. Both
flows amplify azimuthal disturbances and test whether the particle field
remains usable once LES dissipation alone is insufficient.

The reported comparison uses the Smagorinsky LES as its baseline. For
leapfrogging it compares LES with filament refinement by particle splitting
and with Pedrizzetti vortex realignment. The method that performs best in
leapfrogging is then transferred unchanged to the collision. Historical DNS
outputs are retained, but they are not included in the reported figures.

## Physical and numerical setup

Each ring uses the single-ring discretization from `vortex_ring/`:

| Quantity | Value |
|---|---:|
| Ring radius, `R0` | `1.0` m |
| Initial core radius, `a0` | `0.1` m |
| Tube circulation, `Gamma0` | `pi` m2/s |
| Circulation Reynolds number, `Gamma0/nu` | `3000` |
| Particle spacing, `h` | `0.035` m |
| Particle core radius, `sigma_p` | `0.07` m |
| Particles per ring | `8772` |

The two-ring calculations therefore start with `17544` particles. They use a
Gaussian kernel, transposed stretching, coupled RK2 integration, Core
Spreading, and a Barnes--Hut treecode with opening angle `0.30`. The LES cases
use `C_s = 0.20`. The initial centreline disturbance contains modes 1--24 with
a total root-sum-square amplitude of `0.05 R0` and fixed per-ring random seeds.

The nominal time step is

```text
Delta t = 20 h^2 / Gamma0
```

or `Delta t Gamma0 / R0^2 = 0.0245`. The common request is 6000 steps, and
flow-integral and ring diagnostics are sampled every five steps. The solver
stops if the peak particle-strength magnitude exceeds 50 times its initial
value. That guard marks numerical loss of resolution; it is not a physical
transition criterion.

## Kinematic reference

The leapfrogging trajectory plot includes the LBM core-centre paths of Cheng,
Lou, and Lim (2015) at `Re_Gamma = 3000`, `a0/R0 = 0.1`, and `h0/R0 = 1`.
Those values match the present Reynolds number, core size, and initial spacing.
The comparison is kinematic rather than pointwise because the literature case
and the present case do not use the same azimuthal perturbation spectrum. The
digitized vector data and their provenance are stored in
`assets/references/`.

## Running

`allrun.sh` is the single hard-coded campaign driver. It runs the five reported
cases sequentially and then calls the plotting script:

| Interaction | Method | Case directory |
|---|---|---|
| Leapfrogging | LES | `leapfrog_les` |
| Leapfrogging | LES + splitting | `leapfrog_les_splitting` |
| Leapfrogging | LES + realignment | `leapfrog_les_realignment` |
| Collision | LES | `collide_les` |
| Collision | LES + realignment | `collide_les_realignment` |

```sh
./allrun.sh
```

Use `allclean.sh CASE_NAME` to remove one run, or `allclean.sh --all` to remove
all generated results. Existing results are never overwritten silently.

## Stabilization selection

Particle splitting is checked every 25 steps and refines particles whose
strength exceeds five times the initial peak value. This setting extends the
leapfrogging calculation from step 307 to step 335 while adding only 275
particles, or 1.6 percent. Lower thresholds acted too broadly: the most
persistent trial reached step 495 only after particle count grew by 10.5
percent, total particle strength more than doubled, and the impulses drifted
strongly. The retained setting therefore gives the best useful survival gain
without turning refinement into a dominant numerical process.

Realignment applies a Pedrizzetti relaxation factor of 0.005 every 25 steps.
It reaches step 385 with the original 17544 particles, the longest credible
leapfrogging result at fixed particle count. Factors of 0.10 and 0.02 injected
energy and caused substantial impulse drift; 0.01 was cleaner but ended five
steps earlier. The retained weak relaxation reduces the late strength-vector
misalignment relative to LES and avoids the cost and sampling changes of
splitting. It is therefore the selected method for the collision case.

| Case | Last step | Final particles | Interpretation |
|---|---:|---:|---|
| Leapfrogging LES | 307 | 17544 | Baseline loss of filament resolution |
| Leapfrogging LES + splitting | 335 | 17819 | Modest extension with sparse local refinement |
| Leapfrogging LES + realignment | 385 | 17544 | Best leapfrogging quality--cost tradeoff |
| Collision LES | 660 | 17544 | Baseline collision survival |
| Collision LES + realignment | 670 | 17544 | Only a small transfer benefit |

LES dissipation damps unresolved strain, but it cannot add Lagrangian degrees
of freedom or repair a particle-strength direction that has drifted away from
the local vorticity. Splitting addresses the first limitation and realignment
the second. Neither reconstructs the severely distorted and reconnecting
vortex tubes in the collision, which explains why the selected realignment
method adds only ten steps there. Representative selected results are kept in
`solution/` and `samples/`; the useful parameter trials are retained under
`runs/calibration/`. The complete selected leapfrogging backups are
`leapfrog_factor5_interval25/` and
`leapfrog_realign_factor005_interval25/`.

## Figures

Generate PNG figures by default, or request PDF explicitly:

```sh
./allplot.sh
./allplot.sh pdf
```

The figure families are:

- `leapfrogging_trajectory` and `collision_trajectory`: strength-weighted
  material-group centroid and covariance radius;
- `leapfrogging_energy` and `collision_energy`: reconstructed kinetic energy;
- `leapfrogging_circulation` and `collision_circulation`: the circular-tube
  circulation estimate, normalized by the relaxed step-15 sample;
- `leapfrogging_stability` and `collision_stability`: peak particle-strength
  magnitude and the 50-times solver guard.

The trajectory estimator remains meaningful while each initial material group
tracks one coherent tube. After reconnection or complete mixing, group IDs are
material labels rather than unique vortex-core identifiers; the late curve
must then be interpreted together with field visualizations and the other
diagnostics.
