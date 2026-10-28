# Coupling overhaul — 5-minute brief

**Headline:** the FVM–VPM coupling is now one physically consistent algorithm:
a mixed boundary condition from the particles on the outer FVM boundary, and a
vorticity transfer by M4′ remeshing onto a fixed transfer lattice. The work of
the last week removed competing experimental paths and fixed the two errors
that could otherwise accumulate over repeated coupling intervals.

## What changed

| Area | Before | Now / benefit |
|---|---|---|
| Boundary data | Several selectable velocity/pressure paths | One mixed boundary condition: the VPM supplies the normal velocity and the normal gradient of the tangential velocity; the FVM uses `fixedFluxPressure`. The outer FVM boundary is treated the same way in every case. |
| Vorticity transfer | Multiple paths (`common_lattice`, projected transfer, extended-lattice transfer) | One supported route: M4′ remeshing onto a fixed transfer lattice extended by the remeshing region, GBD diffusion, and free particles in the inset band. Fewer settings and one validation target. |
| Particle-represented vorticity | FVM vorticity could be blended directly with the particle strengths | The update compares the FVM target strengths with the vorticity the particles represent. A matching state is a fixed point: no artificial smoothing in every coupling interval. |
| Wall | Cells just inside a solid could feed the strength correction | Lattice nodes inside the solid are excluded before the correction, so targets inside the wall cannot leak into nearby fluid nodes. |
| Stability and diagnostics | Strength growth and the cost of the boundary data were less explicit | The strength-ratio limit `κ_max` keeps unresolved structures from accumulating; the time of the first and repeated boundary-data evaluations is recorded. |
| Restart | The coupled cylinder restart had inconsistent boundary data | Starting from the latest backup now restores the boundary data and the transfer lattice of the accepted coupling step. |

## The algorithm now

```text
advance the VPM over one coupling interval
      ↓
evaluate the boundary data of the particles on the outer FVM boundary
      ↓
advance the FVM over N substeps with the normal velocity and the
normal gradient of the tangential velocity
      ↓
transfer the FVM vorticity to the transfer lattice; particles in the inset band evolve freely
      ↓
update the boundary data (repeat as coupling iterations if configured)
```

The mixed boundary condition on the outer FVM boundary is

$$
U_{FVM}\cdot n=U_{VPM}\cdot n, \qquad
P\,\partial_n U_{FVM}=P\,J_{VPM}n.
$$

## Key implementation change: correction against the particle-represented vorticity

The strength correction is now (notation: `G` = Gaussian representation,
`eta` = blending weight):

```python
represented = G(vpm_strength)
mismatch = eta[:, None] * (fvm_strength - represented)
residual = mismatch - G(mismatch)
correction = mismatch + beta * residual
corrected_strength = baseline + correction
```

This replaces direct blending of strengths and fields. Therefore, if
`fvm_strength == G(vpm_strength)`, then `mismatch == 0` and the vorticity
transfer does not change the particle field. `beta` is bounded by
`max_strength_ratio`; the implementation also limits the update along
`correction` so that no node grows beyond the allowed strength.

The wall fix is applied *before* this correction:

```python
output_weight = lattice.fluid_weight * ~lattice.solid_interior
mismatch[~physical] = 0.0
baseline[~physical] = 0.0
correction[~physical] = 0.0
```

## Boundary data, in code

```python
velocity, normal_gradient = vpm.compute_velocity_and_tangential_normal_gradient_at_points(
    face_centre, face_normal, particle_spacing=h
)
normal_velocity = einsum("ij,ij->i", velocity, face_normal)
# FVM: fixedFluxPressure + normal_velocity + normal_gradient
```

The net-flux check remains: a physically significant net VPM flux through the
outer FVM boundary stops the coupling step instead of being hidden by a projection.

## Take-away / slide close

**Several partially overlapping coupling modes became one verifiable
algorithm.** It keeps the particle wake outside the transfer surface, replaces
the near-field vorticity with the FVM vorticity without repeated Gaussian damping, respects the solid
wall, and applies one mixed boundary condition in every coupling interval and
after every restart.

**Relevant changes:** `24a2be62` (29 Sep), `b1ff9eb6` (3 Oct), `379697e4` (5
Oct). Main implementation: `openonda/coupler/transfer_lattice.py`,
`openonda/coupler/boundary.py`, and `openonda/coupler/vorticity_transfer.py`.
