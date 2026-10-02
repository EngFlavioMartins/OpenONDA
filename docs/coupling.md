# FVM–VPM coupling

FVM resolves pressure, viscous walls and the near wake. VPM transports the outer wake as vortex particles. At each exchange, VPM supplies the outer FVM boundary condition; FVM replaces the particle representation in the inner transfer region. The two representations describe the same flow there, so their velocities are not added.

Start with a case matching your physical problem:

| Case | Physics and setup |
| --- | --- |
| [Cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) | Laminar shedding at $Re=150$; no-slip cylinder and free-slip span. |
| [Cube](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) | Separated flow at $Re=1000$; body-fitted wall and equilibrium Smagorinsky LES. |
| [NACA 4412](../tutorials/coupled_fvm_vpm/03_naca4412_flow/README.md) | Finite-span airfoil at $10^\circ$, $Re=1000$; immersed boundary and Smagorinsky LES. |

See the [FVM guide](fvm.md) for meshes and walls, and the [VPM guide](vpm.md) for particle kernels, diffusion and LES.

## Variables and units

| Quantity | Meaning | Unit |
| --- | --- | --- |
| $\mathbf{x}$, $h$, $\sigma$ | Position, particle spacing and kernel core radius | m |
| $\Delta t_\mathrm{FVM}$, $\Delta t_\mathrm{VPM}$ | FVM step and particle/exchange step | s |
| $\mathbf{U}$, $\mathbf{U}_\infty$ | Flow velocity and freestream | m/s |
| $p_k=p/\rho$ | FVM kinematic pressure | m²/s² |
| $\rho$ | Density | kg/m³ |
| $\nu$, $\nu_t$ | Molecular and subgrid kinematic viscosity | m²/s |
| $\boldsymbol{\omega}=\nabla\times\mathbf{U}$ | Vorticity | 1/s |
| $V$ | FVM cell or particle volume | m³ |
| $\boldsymbol{\Gamma}=\boldsymbol{\omega}V$ | Vector particle strength | m³/s |

Match the freestream and molecular viscosity in both solvers. Define Reynolds number as $Re=U_\infty L/\nu$, using the same body length $L$. For LES, match the closure and coefficients; each solver still computes its subgrid viscosity from its own resolved field and filter width. The [cube case](../tutorials/coupled_fvm_vpm/02_cube_flow/README.md) shows matching equilibrium coefficients; [FVM LES](fvm.md#turbulence-and-les) and [VPM diffusion and LES](vpm.md#diffusion-and-les) explain the models.

## Geometry and mesh

1. Define a Cartesian FVM box enclosing the body and near wake. Generate a [body-fitted mesh or immersed body](fvm.md#mesh-setup), with a no-slip solid boundary and an outer patch named `numericalBoundary`.
2. Set `transfer_region_bounds=(xmin, xmax, ymin, ymax, zmin, zmax)` in metres inside the FVM box. The transfer region is rectangular and uses a uniform particle lattice. Resolve the wall-generated vorticity inside this region before releasing it to VPM.
3. Choose particle spacing and core radius together with the FVM near-wall spacing. The cylinder and cube use equal FVM and particle spacing; inspect both resolutions when refining.
4. Give VPM enough domain extent for the desired wake length. Keep the diffusion grid padding required by the selected scheme.

Static, consistently oriented wall triangles and geometrically represented immersed bodies supply solid boundaries for particle motion and diffusion. Marker-only bodies and moving-wall coupling are unsupported. FVM generates no-slip wall vorticity; particle exclusion from the solid does not replace wall resolution.

### Free-slip span

The [cylinder case](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) uses FVM slip planes and `vpm.SlipSlabInduction` at the same two physical $z$ coordinates. Match the FVM slip faces, VPM domain limits and transfer-region span exactly. Free-space induction describes a different boundary condition and is appropriate for the cube and finite-span airfoil cases.

The slab model retains all three velocity and vorticity components. It currently supports laminar GBD diffusion or inviscid `NONE`, without LES, core spreading, random-walk diffusion or DVH. For GBD, use M4-prime remeshing, at least three grid cells of padding, and a spacing placing the slip planes on grid nodes or half nodes. The coupler anchors its slab lattice half a particle spacing above the lower plane.

`tail_tolerance` and `max_shells` control image-sum convergence. Check sensitivity to these settings on a developed wake. Images enforce slip conditions; they are not physical particles or part of the force reference area.

## Boundary conditions

`coupling_patch` selects the outer FVM patch. Keep the [solid and slip boundary conditions](fvm.md#boundary-conditions) separate from this patch.

| `boundary_condition_mode` | Trace supplied by VPM |
| --- | --- |
| `dirichlet` | Velocity; default, used by the NACA case. |
| `vorticity_mixed` | Normal velocity and tangential normal velocity derivative; used by cylinder and cube, with `fixedFluxPressure`. |
| `directional_outflow` | Constrains incoming flow while allowing outgoing flow. |
| `characteristic` | Incoming/outgoing characteristic information. |
| `pressure_gradient` | Velocity and pressure gradient. |
| `vorticity_mixed_pressure_gradient` | Mixed velocity trace and pressure gradient. |

The prescribed-pressure modes depend on VPM acceleration, temporal history and viscous data. Check their pressure-gradient accuracy for the intended flow before replacing the tutorial boundary choice.

## Vorticity transfer

| `transfer_method` | Representation and requirements |
| --- | --- |
| `buffered_m4_renewal` | M4-prime renewal with an advective release buffer; requires GBD. Used by cylinder and cube. |
| `common_lattice` | Maps FVM cell strengths onto a particle lattice and blends with VPM; default experimental path, used by NACA. |
| `projected_renewal` | Experimental sparse Gaussian projection; requires explicit transfer bounds and `eta_blend_width=0`. |

`eta_blend_width` is the inward width, in metres, over which FVM authority increases from zero to one. Zero gives a sharp transition. `vpm_only_width` reserves a band just inside the transfer faces entirely for VPM and must be smaller than the blend width. Cylinder and cube use widths $6h$ and $2h$, respectively.

Buffered renewal provides a release buffer of length

$$
L_\mathrm{buffer}=1.5\lVert\mathbf{U}_\infty\rVert\Delta t_\mathrm{VPM}+2h.
$$

`transfer_vorticity_cutoff` sets the interior pruning threshold in 1/s; it tapers to the configured GBD floor at release. Check wake sensitivity to pruning when choosing spacing and thresholds. Renewal corrects total vector strength and linear impulse $\mathbf{I}=\tfrac12\sum_i\mathbf{x}_i\times\boldsymbol{\Gamma}_i$; this does not guarantee pointwise vorticity accuracy.

## Time stepping and configuration

Coupled runs require fixed steps with an integer ratio $n=\Delta t_\mathrm{VPM}/\Delta t_\mathrm{FVM}$. Each exchange advances VPM, applies its boundary trace during $n$ FVM substeps, then renews the inner particles while retaining the outer wake.

Cylinder and cube set `interface_iterations=3`: repeat the FVM solve and renewal at the same physical endpoint, up to three sweeps. Iteration requires `vorticity_mixed`, `buffered_m4_renewal`, no consistency band, and output times aligned with exchanges. `interface_normal_tolerance` has units m/s; `interface_gradient_tolerance` has units 1/s. Inspect convergence when changing the exchange interval or overlap width.

Edit the physical constants, mesh and solver configurations in a tutorial's `setup.py`. Its `FVM_SETUP`, `VPM_CASE` and `FVM_MESH` supply the flow models and mesh; `CouplerSetup` supplies the exchange choices. For example, the cube configuration uses:

```python
from openonda import coupler

cfg = coupler.CouplerSetup(
    freestream_velocity=[1.0, 0.0, 0.0],
    coupling_patch="numericalBoundary",
    transfer_region_bounds=(-1.45, 1.45, -1.45, 1.45, -1.45, 1.45),
    transfer_method="buffered_m4_renewal",
    boundary_condition_mode="vorticity_mixed",
    eta_blend_width=6 * 0.045,
    vpm_only_width=2 * 0.045,
    interface_iterations=3,
)
with coupler.create_coupler(FVM_SETUP, VPM_CASE, cfg, mesh=FVM_MESH) as solver:
    solver.run(start_from="latest")
```

Run commands and output locations are in each case README. Use [continuation](continuation.md) to resume compatible coupled backups. Read forces and profiles alongside `solution/coupler_diagnostics.jsonl`, then compare mesh, particle-spacing and exchange-step refinements at common physical times. ParaView opens `solution/fvm.pvd` and `solution/vpm.pvd`.
