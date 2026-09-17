# FVM--VPM coupling guide

`FVMVPMCoupler` exchanges a resolved inner FVM flow with an outer VPM particle
representation. It is a time-synchronized state-transfer driver, not an additive force
model. The public entry points are `CouplerSetup`, `FVMVPMCoupler`, and
`create_coupler` from [`openonda.coupler`](../source/coupler/__init__.py).

## Ownership and construction

The coupler receives already constructed solvers:

```python
from openonda import coupler

cfg = coupler.CouplerSetup(
    coupling_patch="numericalBoundary",
    transfer_method="common_lattice",
    freestream_velocity=[1.0, 0.0, 0.0],
)
driver = coupler.FVMVPMCoupler(fvm_solver, vpm_solver, cfg)
driver.run(max_coupling_steps=2, backup_at_stop=True)
```

`fvm_solver` exists on every MPI rank. `vpm_solver` is owned by rank zero; inactive
ranks pass `None` and participate in the FVM collectives. The coupler adopts the
injected objects and does not clone their fields. Call `initialize()` explicitly when
inspection of derived coupling components is needed; `run()` initializes idempotently.

The FVM configuration owns its mesh, density, boundaries, and FVM step. The VPM
configuration owns its particle discretization, induction, viscosity, and VPM step.
Initialization derives an integer subcycling ratio

\[
  n_\mathrm{FVM} = \operatorname{round}(\Delta t_\mathrm{VPM}/\Delta t_\mathrm{FVM}),
\]

and rejects incompatible time-step ratios. A coupled run currently requires a fixed
FVM step; maximum-Courant adaptive stepping is for standalone FVM/reference runs.

## Coupled time sequence

For every accepted coupling step `k`, the driver performs the following sequence:

```text
VPM advance (Δt_vpm, deferred output)
        ↓ synchronize VPM device
sample VPM velocity / boundary trace at FVM boundary
        ↓
n_fvm substeps: solve_pimple(candidate) → advance_time(commit)
        ↓
collect FVM velocity and ∇U at all donor cells
        ↓
replace/blend FVM-authoritative particle state in VPM
        ↓
VPM scheduled samples, transfer diagnostics, optional backup
```

The VPM particle evolution is complete before the boundary trace is sampled. The FVM
pressure solve is repeated as needed inside its candidate/commit contract, but the
accepted FVM clock advances only in `advance_time()`. The transfer then replaces the
inner VPM representation while retaining the outer wake. The next coupling interval
starts from the synchronized accepted state.

`run(start_step=..., restart_from=...)` supports a complete run or a bounded segment.
`max_coupling_steps` is an execution limit, not a physical configuration change.
`save_backup` writes both solver states and coupling history; `load_backup` restores
both clocks, fields, and boundary-history arrays. A bounded stop can write a restart
with `backup_at_stop=True`.

## Shared units and layouts

All coordinates and lengths are m, times are s, velocities are m/s, kinematic pressure
is m²/s², density is kg/m³, vorticity/velocity gradients are 1/s, and vortex strength
is m³/s. The transfer uses the following logical arrays:

| Array | Shape | Source | Meaning |
| --- | --- | --- | --- |
| FVM cell centres | `(M, 3)` | FVM | Donor locations. |
| FVM cell volumes | `(M,)` | FVM | Quadrature weights for `ωV`. |
| FVM velocity | `(M, 3)` | FVM | Accepted cell-centred velocity. |
| FVM velocity gradient | `(M, 3, 3)` | FVM | Jacobian used to derive vorticity and diagnostics. |
| FVM vorticity | `(M, 3)` | FVM | Curl of the FVM gradient in the coupler's declared layout. |
| VPM position | `(N, 3)` | VPM | Active particle positions. |
| VPM vortex strength | `(N, 3)` | VPM | Particle-strength vector retained/replaced by transfer. |

The FVM donor count `M` must match cell centres, cell volumes, velocity, and gradient.
The VPM active count `N` can change during renewal; all particle fields remain aligned.
The transfer records before/after populations, L1/net strength, first moments, closure,
divergence, amplification, and pruning diagnostics in `TransferResult`.

## Boundary trace

`coupling_patch` identifies the FVM boundary where VPM supplies the outer trace. The
driver samples face centres, outward face normals, and face areas. The selected
`boundary_condition_mode` determines which part of the trace is imposed:

* `dirichlet` imposes the sampled velocity;
* `characteristic` uses incoming/outgoing characteristic information;
* `directional_outflow` preserves outgoing flow while constraining incoming content;
* `pressure_gradient` prescribes velocity and the pressure-gradient trace;
* `vorticity_mixed` prescribes normal velocity and tangential `du/dn`, with native
  `fixedFluxPressure`; and
* `vorticity_mixed_pressure_gradient` is an opt-in combination of the mixed
  velocity trace and a prescribed pressure gradient. Its pressure evaluation
  includes the viscous term as well as convection and the available temporal
  history. It requires separate qualification of VPM pressure-gradient accuracy.

Both mixed modes retain normal-velocity and tangential-gradient histories.
Both prescribed-pressure modes retain the pressure-gradient and Eulerian velocity
histories used during subcycling and restart. The combined mode keeps all of
these histories, including refreshing the velocity snapshot after particle
replacement at fixed physical time.

The mixed face velocity is `U_b = (I - nnᵀ) U_owner + n U_n + d g_t`.
Its tangential value therefore depends on the adjacent FVM cell. Momentum
convection retains the diagonal part of that dependence implicitly, just as
the directional diffusion condition does; cross-component terms converge
through the outer iterations. The reconstructed face value is used to evaluate
the flux, while the cell dependence belongs in the momentum matrix.

The pressure datum has a constant nullspace. Coupling does not shift the numerical FVM
pressure field merely to improve presentation; a pressure offset can be applied to an
output copy when required. Closed-body pressure forces are invariant to that datum.

The optional `fvm_consistency_width` creates a resolved-scale band outside the transfer
region. It is a diagnostic/consistency projection and must fit between every transfer
face and the outer FVM boundary.

## Transfer methods

`transfer_method` selects how the absolute FVM state is represented on the VPM lattice:

### `common_lattice`

Builds a common lattice aligned with FVM donor cells, scatters FVM `ωV` to lattice
nodes, and blends the FVM-authoritative state with the retained VPM state in the
transfer box. `eta_blend_width=0` is a hard partition; a positive width uses a C1
smoothstep authority ramp. `vpm_only_width` reserves an inner band where FVM authority
is exactly zero and must be smaller than `eta_blend_width`.

### `projected_renewal`

Solves a sparse Gaussian projection for a replacement particle basis and verifies the
projected vorticity and boundary velocity errors. It requires explicit transfer bounds
and currently requires `eta_blend_width=0`. The sparse solve tolerance, Gaussian tail,
vorticity error limit, and velocity error limit are independent controls.

### `buffered_m4_renewal`

Uses the stable buffered M4-prime renewal path: a release/retention buffer accounts for
advection, a stable lattice represents the FVM-authoritative belt, and invariant/error
checks constrain pruning and amplification. The current implementation requires the
VPM GBD diffusion scheme. The buffer length includes an advection safety factor and
complete M4-prime support:

\[
  L_\mathrm{buffer} = s\,||U_\infty||\,\Delta t_\mathrm{couple} + 2h.
\]

The configured pruning threshold applies in the FVM-owned interior. It decreases with
FVM authority through the overlap and reaches the VPM GBD scheme's absolute vorticity
floor at the release surface. The VPM floor is read from its solver configuration, so
the transfer cannot silently impose a second, stronger cutoff after FVM authority has
decayed. The `transfer_amplification_cap` limits represented-state corrections. A
transfer failure rolls back the VPM particle fields through the atomic replacement API
before surfacing the error.

## Conservation and diagnostics

The coupler treats `Γ` and the first moments as explicit budgets. Diagnostics distinguish
the donor budget, the mapped target budget, the blended/replaced budget, the retained
outer population, and pruned strength. They are not interchangeable with pointwise
vorticity error. In particular:

* `net_vortex_strength` measures the vector sum of `Γ`;
* L1 strength measures `Σ ||Γ_i||` and detects cancellation-insensitive population change;
* first moments measure the spatial distribution of the transferred strength;
* divergence/closure diagnostics test whether the particle representation remains
  compatible with a solenoidal field;
* velocity/pressure boundary mismatches measure interface consistency, not global
  conservation.

Use the recorded `TransferResult` and `coupling_diagnostics` together with mesh/particle
resolution studies. A run completing without an exception is not evidence that transfer
errors are below a physical accuracy target.

## Output and restart artifacts

The solution root is the user-facing ParaView launch point. Open `solution/fvm.pvd`,
`solution/vpm.pvd`, or `solution/vlm.pvd` for the resolved fields, particles, or lifting
surfaces respectively. Each collection uses relative paths to immutable files below its
matching `fvm/`, `vpm/`, or `vlm/` directory, so the complete solution directory can be
moved without editing the collection. Solver metadata and coupled diagnostics remain at
the solution root for Python post-processing; VPM scientific samples remain under the
VPM solver's `samples/` path.

Restart state is not a visualization product, and this layout does not prescribe a
restart location. It remains independent of the component directories above.

## Failure modes and limitations

Construction/initialization rejects missing injected solvers, mismatched viscosity,
invalid donor bounds, incompatible step ratios, missing VPM particle spacing, invalid
patch geometry, and unsupported transfer-method combinations. Runtime transfer rejects
non-finite or mismatched arrays, failed projection/closure limits, excessive
amplification, and failed rollback prerequisites.

MPI ranks must enter collective solver, field-gather, output, and backup calls in the
same order. The VPM owner is rank zero. Coupled adaptive FVM stepping, arbitrary
unqualified combinations of transfer/diffusion kernels, and convergence claims without
the local validation reports are outside the current guaranteed contract.

## Small-domain accuracy qualification

The [coupler accuracy study](../studies/coupler_accuracy/README.md) records
independent manufactured-field checks, an exact unsteady FVM boundary test,
and induction experiments using a saved cylinder reference snapshot. It also
distinguishes the finite 3D coupled cylinder from the quasi-2D reference.
These component results do not establish agreement of complete cylinder runs.

Buffered renewal uses fluid-domain membership and native wall geometry
independently of distance to the nearest FVM cell centre, so anisotropic donor
cells retain their authority. Its represented Gaussian uses the physical VPM
kernel without discrete normalization. Generic body-fitted walls provide
oriented native surface triangles through a collective FVM getter; this
geometry is assumed static.

The cylinder tutorial's `allplot.sh` now adds common-time reference overlays
and a numerical comparison report. Set `REFERENCE_GRID=medium` to choose the
reference, or run `assets/compare_reference.py --reference medium` directly
for short histories. The comparison does not fit a phase shift or force scale.
