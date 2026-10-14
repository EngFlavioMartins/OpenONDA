# Vortex particles and lifting surfaces

The vortex-particle method (VPM) follows vorticity instead of velocity. Each particle carries a small, smoothed blob of vorticity; the particles move with the flow they induce, their strengths stretch and tilt, and viscosity diffuses them. Lifting surfaces, represented by a vortex-lattice model (VLM), shed their circulation into the particle wake. There is no volume mesh, so a wake can travel far from the body at no extra cost.

| To learn | Start with |
| --- | --- |
| Building particle fields; the four diffusion schemes | [Lamb–Oseen vortices](../tutorials/vpm/01_lamb_oseen_vortex/README.md) |
| Vortex stretching formulations; LES | [Vortex ring](../tutorials/vpm/02_vortex_ring/README.md) |
| Stabilization and particle regularization | [Leapfrogging rings](../tutorials/vpm/03_vortex_interactions/readme.md) |
| Lifting surfaces, startup loads and wakes | [Flat plate](../tutorials/vpm/04_flat_plate/README.md) |
| Prescribed wing motion | [Heaving delta wings](../tutorials/vpm/05_delta_wing/README.md) |
| Rotating blades, wake refinement | [Wind turbine](../tutorials/vpm/06_rotor_flow/README.md), [quadcopter](../tutorials/vpm/07_quadcopter/README.md) |
| Planar flow with one particle layer | [Coupled cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) |

## Physical model and units

For constant viscosity, the vorticity $\boldsymbol{\omega}=\nabla\times\mathbf{u}$ obeys

$$
\frac{D\boldsymbol{\omega}}{Dt}
= (\boldsymbol{\omega}\cdot\nabla)\mathbf{u}
+ \nu\nabla^2\boldsymbol{\omega},
$$

and the velocity follows from the vorticity through the Biot–Savart law. A particle at $\mathbf{x}_p$ carries the vector strength $\boldsymbol{\Gamma}_p=\boldsymbol{\omega}_pV_p$: the vorticity integrated over its volume $V_p$. A radial kernel of core radius $\sigma_p$ spreads it in space, which keeps the induced velocity finite.

| Quantity | Symbol | Unit |
| --- | --- | --- |
| `position` | $\mathbf{x}$ | m |
| `velocity`, `freestream_velocity` | $\mathbf{u}$ | m/s |
| `vortex_strength` | $\boldsymbol{\Gamma}_p=\boldsymbol{\omega}_pV_p$ | m³/s |
| `vorticity` | $\boldsymbol{\omega}$ | 1/s |
| `particle_volume` | $V_p$ | m³ |
| `core_radius` | $\sigma$ | m |
| `spacing` | $h$ | m |
| `kinematic_viscosity`, `eddy_viscosity` | $\nu$, $\nu_t$ | m²/s |
| `time_step_size` | $\Delta t$ | s |

A filament or ring circulation (m²/s) and its physical core radius belong to the flow you model; particle strength and particle core radius belong to its discretisation.

## Particle distributions

`VPMCase.initial_conditions` holds objects whose `build()` returns a `vpm.VortexParticleSet`: positions, volumes, core radii, strengths $\boldsymbol{\omega}V_p$, viscosity, the nominal spacing and optional `group_id` labels that tag each vortex in diagnostics. The tutorials build theirs in `assets/initial_conditions.py`: Gaussian filaments on a triangular lattice for the [Lamb–Oseen vortices](../tutorials/vpm/01_lamb_oseen_vortex/README.md), and Gaussian rings on a toroidal lattice with Widnall disturbances for the [single](../tutorials/vpm/02_vortex_ring/README.md) and [leapfrogging](../tutorials/vpm/03_vortex_interactions/readme.md) rings. Copy one as a template; the [run example](#run-and-inspect) builds a vortex column in a few lines.

Choose $h$ fine enough to resolve the smallest vortex core, and $\sigma/h$ above one so that neighbouring blobs overlap; `core_radius_ratio` in the viscous settings sets it (default 2.5). Large cores smooth small structures; small cores leave holes between particles. Because the blobs themselves smooth the field, a Gaussian vortex of radius $a$ is sampled with the remaining width $\sqrt{a^2-\sigma^2}$, which needs $a>\sigma$. Refine $h$ and $\sigma$ together and compare physical observables.

## Induction and stretching

| Setting | Options |
| --- | --- |
| `particle_kernel` | `GAUSSIAN` (default), `HIGH_ORDER_GAUSSIAN`, `SUPER_GAUSSIAN`, `WINCKELMANS` (algebraic). |
| `induction` | `DirectInduction()`: exact sum over all pairs, for small or reference cases. `TreecodeInduction(theta=0.4, multipole_order=3)`: hierarchical far field for Gaussian and Winckelmans kernels; a smaller `theta` is more accurate. `FMMInduction()`: fast multipole method, every kernel. `SlipSlabInduction(base, z_min, z_max)`: free-slip span planes, see [planar flow](#planar-flow). |
| `stretching_scheme` | `TRANSPOSED` (default) uses $J^T\boldsymbol{\Gamma}$ and conserves total strength; `DIRECT` uses $J\boldsymbol{\Gamma}$; `MIXED` uses the symmetric part $S\boldsymbol{\Gamma}$. The [vortex ring](../tutorials/vpm/02_vortex_ring/README.md) compares them. |
| `integrator` | `RK2()`, `SSPRK3()` or `RK4()`, advancing positions and strengths together. |
| `compute_device`, `precision` | `AUTO`, `CPU`, `CUDA`, `VULKAN` or `METAL`; `f32` or `f64`. |

Choose the induction backend by particle count. On a CUDA GPU, `DirectInduction` is the fastest up to about 50k particles and `FMMInduction()` above that; at 81k particles FMM took 0.6 s per step against 0.9 s for direct summation and 2.8 s for the treecode (core spreading, RTX 3060 Laptop). FMM keeps the velocity within $3\times10^{-4}$ and its gradient within $1.4\times10^{-3}$ of direct summation (relative $L_2$ error on an 81k-particle vortex ring). `TreecodeInduction()` is more accurate, within $3\times10^{-4}$ and $7\times10^{-4}$, about fifty times below the particle discretization error of the 8.8k-particle ring, but slower on GPUs; second-order multipoles are cheaper and more than ten times less accurate. On a CPU the fast methods pay off below 10k particles, and FMM was about twice as fast as the treecode at 81k particles.

Check the fast methods against `DirectInduction` on a small case before trusting forces or instability growth. `freestream_velocity` adds a uniform background flow.

## Diffusion and LES

| `ViscousConfig` | Physical model | Tutorial |
| --- | --- | --- |
| `cs()` | Core spreading: $d\sigma^2/dt=4\nu_\mathrm{eff}$, exact heat spreading of a Gaussian. Cores grow, so long runs need regularization. | [Ring](../tutorials/vpm/02_vortex_ring/README.md), [flat plate](../tutorials/vpm/04_flat_plate/README.md) |
| `rwm()` | Random walk: Brownian steps of variance $2\nu\Delta t$. Uniform viscosity; average an ensemble. | [Lamb–Oseen](../tutorials/vpm/01_lamb_oseen_vortex/README.md) |
| `dvh()` | Diffused vortex hydrodynamics: heat-kernel spreading onto a grid, then regeneration. Uniform viscosity. | [Lamb–Oseen](../tutorials/vpm/01_lamb_oseen_vortex/README.md) |
| `gbd()` | Grid-based diffusion: strengths diffuse on a grid with an explicit Laplacian, then regenerate as particles. Supports variable viscosity and LES. | [Lamb–Oseen](../tutorials/vpm/01_lamb_oseen_vortex/README.md), [coupled cases](coupling.md) |
| `inviscid()` | No diffusion. | — |

The grid schemes (`dvh`, `gbd`) need a `padding` around the particles, in grid cells, and a vorticity `threshold` below which regenerated particles are dropped (`threshold_mode="absolute"` in m³/s, or a `"budget"` fraction of the total).

`TurbulenceConfig.dns()` adds no model. `smagorinsky()` and `equilibrium_smagorinsky()` add an eddy viscosity

$$
\nu_t=(C_s\Delta)^2\sqrt{2S:S},\qquad \Delta=V_p^{1/3},\qquad \nu_\mathrm{eff}=\nu+\nu_t,
$$

with $C_s=0.17$ by default, as in the FVM; `filter_width` fixes $\Delta$ in metres. LES works with core spreading and grid-based diffusion. See the [ring](../tutorials/vpm/02_vortex_ring/README.md) for a DNS/LES comparison and the [wind turbine](../tutorials/vpm/06_rotor_flow/README.md) for an LES wake.

Smagorinsky does not vanish in laminar strain. On resolved vortex rings it dissipates four to eleven times more energy than the molecular viscosity at $Re_\Gamma=3\times10^4$, which mostly lowers the effective Reynolds number. `sigma()` is selective instead. It uses the sigma model (Nicoud et al., 2011),

$$
\nu_t=s\,(C_\sigma\Delta)^2\frac{\sigma_3(\sigma_1-\sigma_2)(\sigma_2-\sigma_3)}{\sigma_1^2},\qquad
s=\frac{\sin^2\theta}{\sin^2\theta+\sin^2\theta_0},
$$

with $\sigma_i$ the singular values of $\nabla\mathbf{u}$ and $C_\sigma=1.35$. The switch $s$ uses the angle $\theta$ between a particle's strength and the resolved vorticity $\nabla\times\mathbf{u}$ at the particle. It stays near zero while the particles represent the vorticity ($\theta\ll\theta_0=3^\circ$) and approaches one where the particle scale is no longer resolved. `subgrid_transport="strength_exchange"`, the default here, applies $\nu_t$ by particle strength exchange after each step. This conserves the total strength and leaves every core to the molecular viscosity: core spreading with a varying $\nu_t$ would grow neighbouring cores unevenly and spoil the overlap where the model acts. Strength exchange is not available between slip planes. The [leapfrogging rings](../tutorials/vpm/03_vortex_interactions/readme.md) compare DNS, this LES and stabilization.

## Stabilization

Particles drift apart and stretch, so a long run gradually loses resolution. `StabilizationConfig` gathers the remedies. Each changes the numerical evolution, so compare with a baseline as the [leapfrogging rings](../tutorials/vpm/03_vortex_interactions/readme.md) do.

| Mechanism | Effect |
| --- | --- |
| `bounded_domain(bounds)` | Removes particles that leave a keep box; `outflow_fringe_depth` fades them out first (below). |
| `selective_eddy_viscosity(coefficient)` | Damps small-scale noise only. |
| `pedrizzetti_relaxation(factor, interval_steps)` | Aligns strengths with the local vorticity. |
| `conservative_filter(...)` | Regenerates the particles on a regular lattice at intervals, conserving total strength and impulse within energy and enstrophy loss limits. |
| `filament_refinement` | Splits over-stretched particles along their vortex lines: `FilamentRefinementConfig.adaptive(interval_steps=...)`. |
| `divergence_relaxation` | Reduces the divergence of the particle vorticity field: `DivergenceRelaxationConfig(...)`. |

Deleting a shed vortex abruptly at the domain face changes the circulation left behind, which induces a spurious alternating velocity at the body. Outflow fringes instead damp strengths smoothly over a depth $L$ inside a face,

$$
\frac{d\boldsymbol{\Gamma}}{dt}=-\lambda(s)\boldsymbol{\Gamma},\qquad
\lambda=\frac{2A\lVert\mathbf{U}_\infty\rVert}{L}s^3(10-15s+6s^2),
$$

with $s\in[0,1]$ the normalized depth and $A$ the number of e-folds (`outflow_fringe_attenuation`, 8 by default). Make a fringe at least two wake wavelengths deep, as in the [coupled cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md).

`ParticleStateLimits` checks every accepted step: finite values, a Lagrangian Courant limit, maximum strength, divergence, strength–vorticity misalignment and growth rates. `RunPlan(state_limit_action="stop")` ends a run cleanly at a crossed limit instead of raising an error.

### Long runs

A particle run usually ends because the particles stop representing a solenoidal vorticity field, not because the physics changes. Noise from the lattice and the fast induction misaligns strengths with the vorticity they induce; stretching then amplifies the divergent part until the divergence limit stops the run. The [leapfrogging rings](../tutorials/vpm/03_vortex_interactions/readme.md) measure each remedy against the DNS while the DNS is still valid:

1. Overlap the cores, $\sigma/h=1.5$. Against $\sigma=h$ it adds about a third to the lifetime without extra dissipation.
2. Remesh only when core spreading has grown the cores (`regularization_core_radius_trigger`). Every regeneration on the lattice seeds lattice-symmetric modes, so remeshing at fixed short intervals shortens the run.
3. If a sub-grid model is needed, use `sigma()`. It stays off laminar vortices that the particles resolve, so it does not lower the Reynolds number the way Smagorinsky does. It does not by itself stop a field from degrading: in the leapfrogging rings the LES and the DNS ended at the same time at both $Re_\Gamma=3000$ and $3\times10^4$.
4. Add Pedrizzetti relaxation with moment restoration: `pedrizzetti_relaxation(factor, preserve_vortex_strength=False, preserve_moments=True)`. Choose the rate $f/\Delta t$ above the physical growth rates (about 1 s⁻¹ here) and well below the core vorticity. At $f/\Delta t\approx5$ s⁻¹ the rings ran to the end of the 12 s run, twice the DNS lifetime, with the same energy, enstrophy and paths. At 1.3 s⁻¹ the run nearly reached the divergence limit during the merger; at 13 s⁻¹ three-dimensional growth was delayed by about three seconds against the DNS.
5. Judge every change by the energy, enstrophy, mode and path histories against the DNS over the time the DNS remains valid. A longer run with a different history is no longer the same flow.

Fast induction adds noise: with `FMMInduction` the rings' DNS ended at 6.0 s, against 7.8 s with direct summation.

## Planar flow

`SlipSlabInduction(base, z_min=..., z_max=...)` bounds the span with two free-slip planes. Each particle then has an infinite row of mirror images: a reflection keeps the spanwise strength and reverses the in-plane components. One particle layer between planes one spacing apart, with all its images, induces the velocity of planar vortices; a Gaussian layer gives planar Gaussian vortices. There is no separate two-dimensional solver: the three-dimensional operators act on the image-complete field, so every kernel, integrator, diffusion scheme and stabilization mechanism carries over.

```python
h = 0.04  # particle spacing, m
span = (-0.5 * h, 0.5 * h)  # slip planes, m
plane = (-5.0, 30.0, -5.0, 5.0)  # x and y bounds, m

numerics = vpm.Numerics(
    induction=vpm.SlipSlabInduction(vpm.FMMInduction(), z_min=span[0], z_max=span[1]),
    viscous=vpm.ViscousConfig.gbd(particle_spacing=h, kinematic_viscosity=nu),
    domain_bounds=(*plane, *span),
    stabilization=vpm.StabilizationConfig.bounded_domain((*plane, *span)),
)
```

Place the particles on $z=0$ with volume $h^3$ and spanwise strength $\Gamma_z=\omega_zh^3$. Then:

- repeat the planes in `domain_bounds` and in any keep box, and keep z-face fringe depths at zero;
- make the slab width a multiple of half the spacing of any grid in use (diffusion, regularization), so the planes fall on nodes or half nodes;
- with `dvh()`, set `padding` to at least `dvh_support_radius_ratio`; LES uses `gbd()`;
- for the algebraic Winckelmans kernel, use $\sigma\ge2h$: a discrete layer of algebraic cores ripples along the span.

`Numerics` rejects inconsistent settings with a message naming both values. The distant images are summed in closed form with a rigorous error bound checked against `tail_tolerance`, so even one layer needs only a few image shells. Thicker slabs model a spanwise-periodic segment between free-slip walls.

## VLM surfaces and wakes

A lifting surface is a lattice of vortex rings whose circulation cancels the normal velocity on the surface; changes of bound circulation are shed into the particle wake.

- **Geometry:** a JSON surface of quadrilateral segments with vertices `a,b` on the leading edge and `d,c` on the trailing edge, and `n_chordwise_panels`, `n_spanwise_panels`. The [flat-plate setup](../tutorials/vpm/04_flat_plate/setup.py) writes one; `VLMMeshConfig.geometric(ratio, region)` clusters panels toward an edge.
- **Placement:** `VLMSurface(surface, translation=..., rotation_degrees=..., rotation_centre=..., kinematics=...)`, collected in `Numerics(vlm=VLMConfig(surfaces=(...), density=..., kinematic_viscosity=...))`.
- **Motion:** `StaticVLM`, `TranslatingVLM`, `SmoothRampVLM`, `RotatingVLM`, `PitchingVLM`, or `ManeuverVLM` driven by your own velocity functions.
- **Loads:** `ForceConfig.kutta_joukowski(unsteady=True)` adds the unsteady pressure load to the bound-vortex forces; `VLMSampler` writes surface geometry, circulation and loads.
- **Wake:** `wake_core_overlap` (default 2.5) sets the core of shed particles relative to the panel and row size; refine $\Delta t$ with the panels.

VLM assumes attached, inviscid flow: it does not model boundary layers, skin friction, stall or leading-edge separation. Progress from the [flat plate](../tutorials/vpm/04_flat_plate/README.md) to [moving wings](../tutorials/vpm/05_delta_wing/README.md), the [wind turbine](../tutorials/vpm/06_rotor_flow/README.md) and the [quadcopter](../tutorials/vpm/07_quadcopter/README.md).

## Run and inspect

This example advances a Gaussian vortex column ($\Gamma=1$ m²/s, core radius 0.35 m, $\nu=0.01$ m²/s) by two steps:

```python
import numpy as np

from openonda import vpm

h, a, nu = 0.2, 0.35, 0.01  # particle spacing [m], core radius [m], viscosity [m²/s]


class GaussianColumn:
    """Vortex column of unit circulation along z, sampled on a cubic lattice."""

    def build(self):
        x = np.arange(-1.0, 1.0 + h / 2, h)
        position = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
        n = len(position)
        radius_squared = position[:, 0] ** 2 + position[:, 1] ** 2
        strength = np.zeros((n, 3))
        strength[:, 2] = np.exp(-radius_squared / a**2) / (np.pi * a**2) * h**3  # omega * V
        return vpm.VortexParticleSet(
            position=position,
            velocity=np.zeros((n, 3)),
            vortex_strength=strength,
            core_radius=np.full(n, 1.2 * h),
            particle_volume=np.full(n, h**3),
            kinematic_viscosity=np.full(n, nu),
            spacing=h,
        )


case = vpm.VPMCase(
    name="gaussian_column",
    directory="first-vpm-case",
    numerics=vpm.Numerics(
        time_step_size=0.01,
        compute_device="cpu",
        max_n_particles=5000,
        integrator=vpm.RK2(),
        induction=vpm.DirectInduction(stretching_scheme="transposed"),
        viscous=vpm.ViscousConfig.cs(kinematic_viscosity=nu, particle_spacing=h),
    ),
    initial_conditions=(GaussianColumn(),),
    run=vpm.RunPlan(end_time=0.02),
    output=vpm.Output(schedule=vpm.EverySteps(10)),
    samplers=vpm.Samplers(samples=(
        vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(10)),
    )),
    backup=vpm.Backup(schedule=vpm.EverySteps(10)),
)
vpm.VPMSolver(case).run(start_from="latest")
```

`RunPlan.end_time` is a whole number of time steps, and a resumed run continues to it; see [continuation](continuation.md). Set `max_n_particles` for the largest cloud the run will reach.

Samplers write to `samples/` on an `EverySteps`, `EveryTime` or `FinalOnly` schedule and record the initial state unless `initial=False`. `FlowIntegralsSampler` records the global invariants per unit density: total strength, linear and angular impulse, kinetic energy and enstrophy. `LineSampler` and `SurfaceSampler` record velocity and vorticity on probes, with the same leading columns as the FVM samplers. Particle frames `solution/vpm/vpm_{step:06d}.vtu`, listed in `solution/vpm.pvd` for ParaView, follow `Output.schedule`; restart backups `solution/vpm/vpm_{step:06d}.h5` follow `Backup.schedule`. `VPMCase.variant` places both trees below a sub-directory. Logs go to `solution/vpm.log` through the `openonda.vpm` logger, and the run metadata to `solution/vpm_metadata.json`.

[Numerical references](../openonda/vpm/REFERENCES.md).
