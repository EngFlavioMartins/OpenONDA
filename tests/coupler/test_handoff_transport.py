"""Repeated hand-off transport: does error accumulate cycle after cycle?

A single hand-off is measured in ``test_handoff_convergence``.  What decides
whether a coupled run drifts is what happens when the same operators are applied
every coupling step for hundreds of steps.  Two regimes matter and they behave
completely differently:

* **FVM-authority zone** (``eta = 1``).  Particle strengths are overwritten from
  the FVM every step, so the transfer error is *re-imposed*, not accumulated.
  It must stay flat.
* **Lagrangian zone** (``eta = 0``, inside the remesh buffer).  Particles are
  remeshed every step but never reset, so each remesh is a fresh application of
  a smoothing operator.  This is the numerical diffusion of the coupling, and it
  is what turns a resolved wake into an over-damped one as it crosses the
  overlap region.

The second test measures that diffusion instead of assuming it, and pins it
against the analytic expectation so a regression cannot creep back in.
"""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.core.helpers.continuous_overlap import continuous_handoff

H = 0.05
BOX = np.array([-0.6, 0.6, -0.6, 0.6, -0.6, 0.6])
U_INF = np.array([1.0, 0.0, 0.0])


def gaussian_ring(points: np.ndarray, centre: np.ndarray, radius: float, core: float, gamma: float):
    """Circulation per cell of a Gaussian-cored vortex ring about the x axis.

    Closed vortex lines, so the field is one a particle method can represent
    exactly -- unlike a spanwise-uniform ("quasi-2D") wake, whose lines leave
    the domain.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.asarray(centre)
    rho = np.hypot(p[:, 1], p[:, 2])
    safe = np.maximum(rho, 1e-300)
    distance_sq = (rho - radius) ** 2 + p[:, 0] ** 2
    strength = gamma * np.exp(-distance_sq / core**2) / (np.pi * core**2)
    # Azimuthal direction e_phi = (0, -z, y) / rho
    out = np.zeros_like(p)
    out[:, 1] = -strength * p[:, 2] / safe
    out[:, 2] = strength * p[:, 1] / safe
    return out * H**3


def _invariants(pos, circ):
    return np.sum(circ, axis=0), 0.5 * np.sum(np.cross(pos, circ), axis=0)


def _cycle(pos, circ, target_fn, dt, mesh_weight=None):
    """Advect by a uniform stream, then hand off once.

    ``mesh_weight = 0`` means "no FVM data here", which drives eta to zero and
    leaves the particles purely Lagrangian -- the regime where remesh error
    accumulates instead of being overwritten.
    """
    pos = pos + U_INF * dt
    result = continuous_handoff(
        pos,
        circ,
        BOX,
        H,
        circulation_at_node=target_fn,
        mesh_weight_at_node=mesh_weight,
        u_inf=U_INF,
        ramp_width=4 * H,
        buffer_length=4 * H,
        threshold_abs=0.0,
        radius_ratio=1.0,
        amplification_cap=2.0,
        u_max=1.0,
        dt=dt,
        lattice_anchor=np.array([-0.625, -0.625, -0.625]),
    )
    return result.pos, result.circ, result


def _seed_from_field(field_fn):
    """Build an on-lattice particle set from an analytic circulation field."""
    n = 41
    axis = (np.arange(n) - (n - 1) / 2.0) * H
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    circ = field_fn(grid)
    keep = np.linalg.norm(circ, axis=1) > 0.0
    return grid[keep], circ[keep]


@pytest.mark.verification
def test_fvm_authority_zone_does_not_accumulate_error():
    """With eta = 1 and an exact target, the error must be flat, not growing."""
    radius, core, gamma = 0.25, 6.0 * H, 1.0
    dt = 0.02

    def target(points):
        return gaussian_ring(points, np.array([0.0, 0.0, 0.0]), radius, core, gamma)

    pos, circ = _seed_from_field(lambda q: target(q))
    errors = []
    for _ in range(25):
        # Hold the ring fixed so the analytic target stays valid; the advection
        # is what forces a fresh off-lattice remesh every cycle.
        pos, circ, result = _cycle(pos, circ, target, dt)
        pos = pos - U_INF * dt
        inner = np.all(np.abs(pos) < 0.4, axis=1)
        exact = target(pos[inner])
        errors.append(
            float(np.linalg.norm(circ[inner] - exact) / (np.linalg.norm(exact) + 1e-30))
        )

    errors = np.asarray(errors)
    assert np.all(np.isfinite(errors))
    growth = errors[-5:].mean() / (errors[:5].mean() + 1e-30)
    assert growth < 1.15, f"error grew by {growth:.2f}x over 25 cycles: {errors[::5]}"
    assert errors.max() < 0.10, f"resolved ring should transfer to <10%, got {errors.max():.3%}"


@pytest.mark.verification
def test_lagrangian_zone_conserves_circulation_and_impulse_exactly():
    """Remesh-only transport must not leak circulation or linear impulse."""
    radius, core, gamma = 0.2, 5.0 * H, 1.0
    dt = 0.02
    pos, circ = _seed_from_field(
        lambda q: gaussian_ring(q, np.array([0.0, 0.0, 0.0]), radius, core, gamma)
    )
    gamma0, impulse0 = _invariants(pos, circ)

    zero = lambda points: np.zeros((len(np.atleast_2d(points)), 3))  # noqa: E731
    no_fvm = lambda points: np.zeros(len(np.atleast_2d(points)))  # noqa: E731
    for _ in range(20):
        pos, circ, _ = _cycle(pos, circ, zero, dt, mesh_weight=no_fvm)

    gamma1, impulse1 = _invariants(pos, circ)
    np.testing.assert_allclose(gamma1, gamma0, atol=1e-12, rtol=0)
    # Impulse translates with the ring; compare in the co-moving frame.
    shift = U_INF * dt * 20
    impulse1_comoving = impulse1 - 0.5 * np.cross(shift, gamma1)
    np.testing.assert_allclose(impulse1_comoving, impulse0, atol=1e-12, rtol=0)


@pytest.mark.verification
def test_lagrangian_zone_diffusion_is_bounded_and_measured(capsys):
    """Quantify the per-cycle amplitude loss of remesh-only transport."""
    radius, gamma = 0.2, 1.0
    dt = 0.02
    rows = []
    for cells_per_core in (2.0, 4.0, 6.0):
        core = cells_per_core * H
        pos, circ = _seed_from_field(
            lambda q, c=core: gaussian_ring(q, np.array([0.0, 0.0, 0.0]), radius, c, gamma)
        )
        peak0 = np.linalg.norm(circ, axis=1).max()
        zero = lambda points: np.zeros((len(np.atleast_2d(points)), 3))  # noqa: E731
        no_fvm = lambda points: np.zeros(len(np.atleast_2d(points)))  # noqa: E731
        for _ in range(20):
            pos, circ, _ = _cycle(pos, circ, zero, dt, mesh_weight=no_fvm)
        peak = np.linalg.norm(circ, axis=1).max() / peak0
        rows.append((cells_per_core, peak, peak ** (1.0 / 20.0)))

    with capsys.disabled():
        print("\n  r_c/h   peak after 20 remeshes   per-cycle retention")
        for cells, peak, per_cycle in rows:
            print(f"  {cells:5.1f} {peak:22.4f} {per_cycle:22.5f}")

    # A core resolved by 4+ cells must survive 20 remeshes essentially intact;
    # this is the bound that decides whether a wake crossing the overlap region
    # arrives at the right strength.
    resolved = {cells: peak for cells, peak, _ in rows}
    assert resolved[4.0] > 0.90, f"4 cells/core lost {(1 - resolved[4.0]):.1%} over 20 remeshes"
    assert resolved[6.0] > 0.96
    # And the trend must be monotone: coarser cores must lose more.
    assert resolved[2.0] < resolved[4.0] < resolved[6.0]
