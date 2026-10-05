# Coupling overhaul — 5-minute brief

**Headline:** the FVM–VPM exchange is now one canonical, physically consistent
algorithm: a mixed VPM-to-FVM boundary trace plus fixed-lattice, buffered M4′
renewal.  The last-week work removed competing experimental paths and fixed the
two errors that could otherwise accumulate at repeated exchanges.

## What changed

| Area | Before | Now / benefit |
|---|---|---|
| Boundary hand-off | Several selectable velocity/pressure paths | One mixed trace: VPM supplies normal velocity and tangential normal derivative; FVM uses `fixedFluxPressure`. This makes the patch treatment consistent across cases. |
| FVM → VPM renewal | Multiple paths (`common_lattice`, projected renewal, buffered renewal) | One supported route: fixed-lattice M4′ renewal, GBD diffusion, an advective VPM-owned release buffer. Less configuration surface and one validation target. |
| Field representation | FVM *physical vorticity* could be blended directly with VPM *Gaussian coefficients* | The new update compares the FVM field with the field **represented by** particles. A matching state is a fixed point—no artificial smoothing at every exchange. |
| Wall handling | Tapered cells just inside a solid could feed the inverse correction | Solid-interior nodes are excluded before correction; impossible wall targets cannot leak into nearby fluid coefficients. |
| Stability / observability | Coefficient growth and boundary-query cost were less explicit | Directional amplitude cap prevents unresolved structures from accumulating; initial/refresh boundary-induction timing is recorded. |
| Continuation | Coupled cylinder continuation had boundary-state/workflow inconsistencies | Start/latest continuation now preserves the canonical boundary/renewal lifecycle. |

## The algorithm now

```text
advance VPM over one exchange interval
      ↓
evaluate VPM trace on the FVM coupling patch
      ↓
FVM subcycle with normal velocity + tangential normal-gradient trace
      ↓
renew the FVM-owned inner VPM lattice; preserve released outer wake
      ↓
refresh boundary trace (repeat fixed-point sweep if configured)
```

The physical interface condition is

$$
U_{FVM}\cdot n=U_{VPM}\cdot n, \qquad
P\,\partial_n U_{FVM}=P\,J_{VPM}n.
$$

## Key implementation change: represented-state correction

The essential correction is now (notation: `G` = Gaussian representation,
`eta` = FVM authority ramp):

```python
represented = G(vpm_strength)
mismatch = eta[:, None] * (fvm_strength - represented)
residual = mismatch - G(mismatch)
correction = mismatch + beta * residual
corrected_strength = baseline + correction
```

This replaces direct coefficient/field blending. Therefore, if
`fvm_strength == G(vpm_strength)`, then `mismatch == 0` and the exchange does
not change the particle field. `beta` is bounded by
`transfer_amplification_cap`; the actual implementation additionally clips the
step along `correction` so that a node cannot grow beyond the allowed magnitude.

The wall fix is deliberately applied *before* this inverse-like correction:

```python
output_weight = lattice.fluid_weight * ~lattice.solid_interior
mismatch[~physical] = 0.0
baseline[~physical] = 0.0
correction[~physical] = 0.0
```

## Boundary hand-off, in code

```python
velocity, tangential_normal_gradient = (
    vpm.compute_velocity_and_tangential_normal_gradient_at_points(
        face_centre, face_normal, particle_spacing=h
    )
)
normal_velocity = einsum("ij,ij->i", velocity, face_normal)
# FVM: fixedFluxPressure + normal_velocity + tangential_normal_gradient
```

The flux check remains a guardrail: a physically significant net VPM flux
rejects the exchange rather than being silently hidden by a projection.

## Take-away / slide close

**We traded several partially overlapping coupling modes for one auditable
pipeline.** It preserves the VPM wake at the release boundary, regenerates the
FVM-owned near field without repeated Gaussian damping, respects solid
geometry, and applies one consistent mixed boundary condition at every
exchange and restart.

**Relevant changes:** `24a2be62` (29 Sep), `b1ff9eb6` (3 Oct), `379697e4` (5
Oct).  Main implementation: `source/coupler/stable_renewal.py`,
`source/coupler/boundary.py`, and `source/coupler/vorticity_transfer.py`.
