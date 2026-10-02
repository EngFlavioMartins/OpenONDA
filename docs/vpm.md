# Vortex particles and lifting surfaces

The VPM transports a three-dimensional vorticity field with regularized Biot–Savart velocity, vortex stretching and viscous diffusion. A vortex-lattice model (VLM) adds lifting surfaces and sheds their circulation into the particle wake. Use SI units throughout.

## Variables and units

| Input or field | Symbol | Unit | Meaning |
| --- | --- | --- | --- |
| `position`, distribution bounds | $\boldsymbol{x}$ | m | Particle coordinates in the fixed spatial frame. |
| `velocity`, `freestream_velocity` | $\boldsymbol{u}$ | m/s | Transport velocity, including imposed flow and lifting surfaces. |
| Initializer `circulation` | $\Gamma_0$ | m²/s | Scalar circulation of a filament or ring. |
| `vortex_strength` | $\boldsymbol{\Gamma}_p$ | m³/s | Particle vector strength: $\boldsymbol{\omega}_p V_p$. |
| `vorticity` | $\boldsymbol{\omega}$ | s⁻¹ | Reconstructed vorticity; particle and probe definitions below. |
| `particle_volume` | $V_p$ | m³ | Integration weight of a particle. |
| `core_radius` | $\sigma$ | m | Regularization radius of a particle. |
| Initializer `vortex_core_radius` | $a$ | m | Physical Gaussian vortex-core radius. |
| Distribution `spacing` | $h$ | m | Nominal particle spacing. |
| `kinematic_viscosity` | $\nu$ | m²/s | Molecular viscosity. |
| `eddy_viscosity`, `effective_viscosity` | $\nu_t$, $\nu_{\rm eff}$ | m²/s | LES viscosity and $\nu+\nu_t$. |
| `velocity_gradient`, `strain_rate` | $J$, $S$ | s⁻¹ | $J_{ij}=\partial u_i/\partial x_j$, $S=(J+J^T)/2$. |
| `time_step_size` | $\Delta t$ | s | Physical time increment. |

A particle strength is a vector volume integral; its quadrature vorticity is $\boldsymbol{\Gamma}_p/V_p$. Saved particle `vorticity` reconstructs the regularized blob field at particle locations; line/surface samples use $\nabla\times\boldsymbol{u}$. These differ when the particle field is not divergence-free. Scalar filament circulation and physical vortex-core radius differ from particle strength and particle core radius.

## Particle distributions

VPM uses particle quadrature in place of a volume mesh. A distribution sets positions, volumes and $\sigma/h$; a flow initializer sets vorticity and viscosity.

| Distribution | Geometry and inputs | Example |
| --- | --- | --- |
| `RectangularDistribution` | Cartesian box; `bounds=((xmin,xmax), (ymin,ymax), (zmin,zmax))`. Weights are $h^3$. | Runnable box below. |
| `CylindricalDistribution` | Cartesian points inside a cylinder; `radius`, `length`, `centre`, `axis`. Weights sum to the cylinder volume. | [Constructor](../source/solvers/vpm/initialization/distributions/cylindrical.py). |
| `TriangularPrismDistribution` | Triangular transverse lattice extruded along `axis`; box `bounds`, optional `axial_spacing`. | [Lamb–Oseen vortices](../tutorials/vpm/01_lamb_oseen_vortex/README.md). |
| `ToroidalDistribution` | Ring with a hexagonal cross-section; `ring_radius`, `tube_radius`, `centre`, `axis`. Curved-cell weights include the cylindrical Jacobian. | [Single ring](../tutorials/vpm/02_vortex_ring/README.md), [leapfrogging rings](../tutorials/vpm/03_vortex_interactions/readme.md). |

Choose $h$ small enough to resolve the physical core and retain overlapping particles. `core_radius_ratio` sets $\sigma/h$. Large cores smooth small structures; small cores leave gaps between particles. Refine spacing and core ratio together and compare physical observables.

For Gaussian filaments and rings, `ParticleCoreCompensation()` accounts for particle smoothing when initializing a requested physical core. This requires $a^2>\sigma^2$: the initializer represents the remaining width $\sqrt{a^2-\sigma^2}$. Set the distribution support wide enough to include the Gaussian tail. Ring `tube_radius` is the support radius, not $a$.

`VortexFilament`, `VortexRing`, `VortexDoublet`, `TaylorGreenVortex` and `IsotropicTurbulence` build initial fields. Optional `group_id` labels distinguish vortex contributions; remeshed particle indices do not track fluid parcels.

## Induction and stretching

For constant viscosity, the incompressible vorticity equation is

$$
\frac{D\boldsymbol{\omega}}{Dt}
= (\boldsymbol{\omega}\cdot\nabla)\boldsymbol{u}
+ \nu\nabla^2\boldsymbol{\omega}.
$$

The Gaussian or Winckelmans particle kernel regularizes the velocity near each source. `DirectInduction` sums every pair and is useful for small reference cases. `TreecodeInduction` and `FMMInduction` approximate distant interactions; check their tolerances against direct results. Treecode currently uses f32 with Gaussian or Winckelmans kernels; check the selected backend before changing precision or kernel.

`stretching_scheme` selects a discrete strength update: `DIRECT` uses $J\boldsymbol{\Gamma}$, `TRANSPOSED` uses $J^T\boldsymbol{\Gamma}$, and `MIXED` uses $S\boldsymbol{\Gamma}$. The default is `TRANSPOSED`, which conserves the summed vector strength under the symmetric pair formulation. The [single-ring case](../tutorials/vpm/02_vortex_ring/README.md) compares these choices.

`SlipSlabInduction` adds image vortices between free-slip span planes. See [coupling](coupling.md) for the boundary conditions and compatible diffusion. `domain_bounds` alone does not impose walls or periodic induction.

## Diffusion and LES

| `ViscousConfig` factory | Physical approximation | Example |
| --- | --- | --- |
| `inviscid()` | No viscous diffusion. | Use only when viscosity is intentionally neglected. |
| `cs()` | Core spreading: $d\sigma^2/dt=4\nu_{\rm eff}$. Exact Gaussian heat spreading for uniform viscosity; a second-moment model for Winckelmans cores. | [Ring](../tutorials/vpm/02_vortex_ring/README.md), [flat plate](../tutorials/vpm/04_flat_plate/README.md). |
| `rwm()` | Brownian displacement with coordinate variance $2\nu\Delta t$. Requires uniform viscosity and an ensemble for mean-flow comparisons. | [Lamb–Oseen](../tutorials/vpm/01_lamb_oseen_vortex/README.md). |
| `dvh()` | Heat-kernel diffusion and particle regeneration; may accumulate steps before diffusion. Requires uniform viscosity. | [Lamb–Oseen](../tutorials/vpm/01_lamb_oseen_vortex/README.md). |
| `gbd()` | Grid diffusion followed by particle regeneration; supports spatially varying effective viscosity. | [Lamb–Oseen](../tutorials/vpm/01_lamb_oseen_vortex/README.md), [coupled cases](coupling.md). |

`TurbulenceConfig.dns()` adds no subgrid closure; sufficient resolution still needs checking. `les_smagorinsky()` uses

$$
\nu_t=(C_s\Delta)^2\sqrt{2S:S},\qquad
\Delta=V_p^{1/3},\qquad \nu_{\rm eff}=\nu+\nu_t.
$$

The default $C_s=0.20$; `filter_width` can fix $\Delta$ in metres. LES with CS spreads each core using its local effective viscosity; GBD applies $\nabla\cdot(\nu_{\rm eff}\nabla\boldsymbol{\omega})$. Neither is the complete variable-viscosity stress-curl operator. RWM and DVH reject LES. See the [ring](../tutorials/vpm/02_vortex_ring/README.md) for DNS/LES comparison and [rotor](../tutorials/vpm/06_rotor_flow/README.md) for LES wake inputs.

Core spreading eventually needs redistribution to retain resolution. Selective eddy viscosity, strength alignment and filament splitting change numerical evolution; the [leapfrogging case](../tutorials/vpm/03_vortex_interactions/readme.md) compares them separately. Check time-step, spacing, core overlap and diffusion convergence before interpreting forces or instability growth.

## VLM surfaces and wakes

Surface JSON defines quadrilateral segments with vertices `a,b` on the leading edge and `d,c` on the trailing edge, plus `n_chordwise_panels` and `n_spanwise_panels`. The [flat-plate setup](../tutorials/vpm/04_flat_plate/setup.py) generates this geometry. Change its chord, span and panel counts to build a new wing. `VLMMeshSetup.geometric(ratio=3, region="end")` clusters panels toward the selected edge; `ratio` is largest/smallest panel spacing.

Pass surfaces through `VLMSetup(surfaces=(VLMSurfaceSetup(...),))` in `Numerics.vlm`. Supply matching VPM/VLM viscosity and fluid `density` in kg/m³. `translation` and `rotation_centre` use metres; `rotation_degrees` uses degrees. Motion objects specify translation, rotation or prescribed maneuvers.

`wake_core_overlap=2.5` sets both trailing and transverse emitted cores to 2.5 times the larger local span spacing or convected row length. The older `sigma_factor` affects transverse elements only. Resolve wake rows by refining $\Delta t$ together with the surface panels.

VLM solves bound circulation and sheds its spanwise differences and time changes into the wake. `ForceConfig.kutta_joukowski(unsteady=True)` adds the pressure-time load from changing surface potential jump. The default boundary response holds bound circulation during each particle step; `boundary_response="responsive"` is experimental and needs coupled time-convergence checks.

VLM assumes attached inviscid flow. It does not resolve boundary layers, skin friction, stall, separated delta-wing leading-edge vortices or viscous particle–wall collision. Start with the [flat plate](../tutorials/vpm/04_flat_plate/README.md), then [moving delta wings](../tutorials/vpm/05_delta_wing/README.md), [wind turbine](../tutorials/vpm/06_rotor_flow/README.md) or [quadcopter](../tutorials/vpm/07_quadcopter/README.md).

## Run and inspect

This two-step example advances a finite Gaussian vortex column for 0.02 s, with circulation 1 m²/s and viscosity 0.01 m²/s. Increase `RunPlan.steps` for a longer physical horizon:

```python
from openonda import vpm

h, nu = 0.2, 0.01
case = vpm.VPMCase(
    directory="first-vpm-case",
    numerics=vpm.Numerics(
        time_step_size=0.01,
        compute_device="CPU",
        max_n_particles=5000,
        integrator=vpm.RK2(),
        induction=vpm.DirectInduction(stretching_scheme="TRANSPOSED"),
        viscous=vpm.ViscousConfig.cs(kinematic_viscosity=nu, particle_spacing=h),
    ),
    initial_conditions=(vpm.VortexFilament(
        circulation=1.0,
        vortex_core_radius=0.35,
        kinematic_viscosity=nu,
        distribution=vpm.RectangularDistribution(
            bounds=((-1, 1), (-1, 1), (-1, 1)),
            spacing=h, core_radius_ratio=1.2,
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
    ),),
    backup=vpm.Backup(interval_steps=10),
    samplers=vpm.Samplers(samples=(
        vpm.FlowIntegralsSampler(schedule=vpm.EverySteps(10)),
    )),
    run=vpm.RunPlan(steps=2),
)
vpm.VPMSolver(case).run(start_from="latest")
```

Backups are in `solution/`; open `solution/vpm.pvd` in ParaView. Diagnostics are in `samples/`. Energy, impulse and enstrophy are per unit density: m⁵/s², m⁴/s and m³/s² respectively; VPM enstrophy uses $\int|\boldsymbol{\omega}|^2\,dV$ without a one-half factor. `SurfaceSampler` and `LineSampler` add field probes in metres.

Tutorials use `allrun.sh` for a clean run, `allcontinue.sh` for compatible continuation and `allplot.sh` for figures. Set capacity for the full shed/regenerated cloud. `RunPlan.steps` is the total step target, including restored steps; see [continuation](continuation.md). A numerical health stop produces a partial trajectory. A completed run still needs physical convergence checks.

[Numerical references](../source/solvers/vpm/REFERENCES.md).
