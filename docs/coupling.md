# FVM–VPM coupling

FVM resolves pressure, viscous walls and the near wake. VPM transports the outer wake as vortex particles. At each exchange, VPM supplies the outer FVM boundary condition; FVM replaces the particle representation in the inner transfer region. The two representations describe the same flow there, so their velocities are not added.

Start with a case matching your physical problem:

| Case | Physics and setup |
| --- | --- |
| [Cylinder](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) | Laminar shedding at $Re=150$; one periodic FVM layer and three-dimensional VPM particles. |
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

For a resolved free-slip span, match the FVM slip faces, `vpm.SlipSlabInduction` planes, VPM domain limits and transfer-region span exactly. Free-space induction describes a different boundary condition and is appropriate for the cube and finite-span airfoil cases.

The slab model retains all three velocity and vorticity components. Coupled runs use laminar GBD diffusion in the slab, without LES. Use M4-prime remeshing, at least three grid cells of padding, and a spacing placing the slip planes on grid nodes or half nodes. The coupler anchors its slab lattice half a particle spacing above the lower plane.

`tail_tolerance` controls image-sum convergence. The Gaussian slab evaluator selects complete reflected shells until both velocity and gradient remainder bounds satisfy that tolerance; `max_shells` is the hard limit. Check sensitivity to these settings on a developed wake. Images enforce slip conditions; they are not physical particles or part of the force reference area.

### Planar flow

The [cylinder case](../tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/README.md) uses one periodic FVM layer across a unit span and three-dimensional cubic VPM particles. `vpm.SlipSlabInduction` applies full-vector reflected sources at the span boundaries. Induction, stretching and diffusion retain the common three-dimensional physics.

## Boundary conditions

`coupling_patch` selects the outer FVM patch. Keep the [solid and slip boundary conditions](fvm.md#boundary-conditions) separate from this patch.

VPM supplies normal velocity and the tangential normal velocity derivative. The FVM pressure condition is `fixedFluxPressure`. This mixed trace is used by every coupled case. Reapplying it retains the previously computed pressure gradient for the next momentum predictor; the pressure correction updates that gradient to enforce the new flux.

The continuous velocity condition follows Billuart, Duponcheel, Winckelmans and Chatelain, [*A weak coupling between a near-wall Eulerian solver and a Vortex Particle-Mesh method for the efficient simulation of 2D external flows*](https://doi.org/10.1016/j.jcp.2022.111726), *Journal of Computational Physics* **473** (2023), 111726, Section 3.1, Eqs. (11)–(14). The original paper treats two-dimensional flow. For a locally planar boundary, define the outward unit normal $\mathbf{n}$, tangential projector $\mathbf{P}=\mathbf{I}-\mathbf{n}\mathbf{n}^T$, and velocity Jacobian $J_{ij}=\partial U_i/\partial x_j$. OpenONDA prescribes

$$
\mathbf{U}_\mathrm{FVM}\cdot\mathbf{n}
=\mathbf{U}_\mathrm{VPM}\cdot\mathbf{n},
\qquad
\mathbf{P}\,\partial_n\mathbf{U}_\mathrm{FVM}
=\mathbf{P}\,\mathbf{J}_\mathrm{VPM}\mathbf{n}.
$$

The planar identity $\mathbf{P}\,\partial_n\mathbf{U}=\nabla_t(\mathbf{U}\cdot\mathbf{n})-\mathbf{n}\times\boldsymbol{\omega}$ connects this derivative to the paper's vorticity condition when $\boldsymbol{\omega}=\nabla\times\mathbf{U}$. OpenONDA evaluates the derivative from the induced VPM velocity; equivalence to a separately reconstructed particle-vorticity trace requires consistency with that curl. Tangential velocity values remain part of the FVM solution, so a tangential velocity difference alone does not indicate a violated boundary condition.

`fixedFluxPressure` is OpenONDA's flux-compatible FVM realization, rather than a pressure value independently supplied by the paper or VPM. It reconstructs the pressure-free momentum field at the boundary face before comparing it with the prescribed normal flux. The same predictor is retained through the pressure corrections belonging to one momentum solve, and refreshed by the next momentum predictor. Comparing a cell-centred momentum field directly with a face-centred velocity would introduce a pressure error even on an orthogonal mesh.

The [face reconstruction](../source/solvers/fvm/fields/mixed_velocity_boundary.py) accounts for tangential owner-to-face displacement on skew faces. Velocity, normal diffusion and pressure face values each receive their corresponding geometric correction; the prescribed VPM derivative is unchanged. The boundary-gradient correction also subtracts the tangential displacement before computing the normal derivative, keeping viscous stress consistent with the face values.

Boundary-only least-squares stencils reuse static mesh geometry and include real processor-neighbour cells, avoiding an additional whole-domain gradient calculation. For a pressure face on a one-cell-thick mesh, real neighbours can leave one derivative undetermined. Only that unresolved direction uses the lagged native boundary data, frozen with the momentum predictor; resolved directions retain their real-cell reconstruction. This selective lagged closure does not establish higher-order accuracy in a direction without enough real-cell support. Unresolved directions without usable boundary support raise an error.

## Vorticity transfer

Every coupled case uses M4-prime particle renewal with an advective release buffer and GBD diffusion. Set the particle and GBD grid spacings equal, use M4-prime remeshing and choose an absolute vorticity pruning threshold. FVM replaces the inner particle representation while VPM retains the released wake.

`eta_blend_width` is the inward width, in metres, over which the FVM blending weight increases from zero to one. Zero gives a sharp transition. `vpm_only_width` reserves a band just inside the transfer faces entirely for VPM and must be smaller than the blend width. The tutorials use widths $6h$ and $2h$, respectively.

Renewal compares FVM vorticity with the Gaussian field represented by the particles, then corrects the existing particle strengths using that difference. An already matching pair of fields is unchanged by this representation correction. Directly blending FVM vorticity with particle coefficients would apply an extra smoothing at each exchange. `transfer_amplification_cap` limits the local correction to avoid accumulating large coefficients when a near-wall target cannot be resolved by the particle cores. Remeshing and pruning have their own errors, so this consistency property does not replace resolution checks.

Buffered renewal provides a release buffer of length

$$
L_\mathrm{buffer}=1.5\lVert\mathbf{U}_\infty\rVert\Delta t_\mathrm{VPM}+2h.
$$

`transfer_vorticity_cutoff` sets the interior pruning threshold in 1/s; it tapers to the configured GBD floor at release. Check wake sensitivity to pruning when choosing spacing and thresholds. Renewal corrects total vector strength and linear impulse $\mathbf{I}=\tfrac12\sum_i\mathbf{x}_i\times\boldsymbol{\Gamma}_i$; this does not guarantee pointwise vorticity accuracy.

GBD pruning also preserves angular impulse and keeps moment recovery within connected, wall-visible fluid components. A weak component can have adequate matrix rank yet require excessive strength correction after pruning. In that case GBD retains its original post-diffusion donor nodes within the declared particle capacity and repeats the unchanged moment checks. The stronger components and successful recovery path retain their existing treatment. `support_augmented_node_count` records the added support; insufficient capacity raises before the diffusion grid is changed.

## Time stepping and configuration

An optional `CouplerSetup.freestream` declares a `coupler.VelocityRamp` in m/s
and seconds. The native driver applies the background velocity at each
accepted exchange endpoint and restores it on continuation. The declared
history belongs to the checkpoint configuration. To initialize a spatial
velocity disturbance, pass a physical cell-centre function as
`solver.run(initial_velocity=...)`; native continuation restores its saved
field without repeating the disturbance.

Coupled runs require fixed steps with an integer ratio $n=\Delta t_\mathrm{VPM}/\Delta t_\mathrm{FVM}$. Each exchange advances VPM, applies its boundary trace during $n$ FVM substeps, then renews the inner particles while retaining the outer wake.

`interface_iterations` limits repeated FVM solves and renewal at the same physical endpoint. Cylinder allows six sweeps and cube allows three. The initial interface estimate uses accepted trace history; a rejected estimate is retried from the unpredicted trace. Output times must align with exchanges. `interface_normal_tolerance` has units m/s; `interface_gradient_tolerance` has units 1/s. Inspect convergence when changing the exchange interval or overlap width.

Edit the physical constants, mesh and solver configurations in a tutorial's `setup.py`. `FVMSetup`, `VPMCase` and the mesh define the flow problem; `CouplerSetup` supplies the overlap, convergence tolerances and output schedule. The cylinder constructs these objects in `build_case()`. For example, the cube configuration uses:

```python
from openonda import coupler

cfg = coupler.CouplerSetup(
    freestream_velocity=[1.0, 0.0, 0.0],
    coupling_patch="numericalBoundary",
    transfer_region_bounds=(-1.45, 1.45, -1.45, 1.45, -1.45, 1.45),
    eta_blend_width=6 * 0.045,
    vpm_only_width=2 * 0.045,
    interface_iterations=3,
)
with coupler.create_coupler(FVM_SETUP, VPM_CASE, cfg, mesh=FVM_MESH) as solver:
    solver.run(start_from="latest")
```

Run commands and output locations are in each case README. Use [continuation](continuation.md) to resume compatible coupled backups. Read forces and profiles alongside `solution/coupler_diagnostics.jsonl`, then compare mesh, particle-spacing and exchange-step refinements at common physical times. ParaView opens `solution/fvm.pvd` and `solution/vpm.pvd`.
