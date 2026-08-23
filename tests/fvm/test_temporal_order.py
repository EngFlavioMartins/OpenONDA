"""Temporal order-of-accuracy verification for the transient momentum operator.

Uses *self-convergence* (Richardson on the time axis) rather than comparison to
an analytic solution, so the measured order isolates the **time-integration
error** from spatial discretisation and boundary-condition limitations — the
right tool while the public BC API still only supports uniform Dirichlet values.

Problem: a linear transient momentum equation with a frozen, constant advecting
velocity ``a`` and a manufactured source so that ``u(x,t) = e^{-t} φ(x)`` is the
exact solution, where ``φ`` is the divergence-free 2D Taylor–Green field::

    ∂u/∂t + (a·∇)u − ν∇²u = S(x, t)

Integrating to a fixed time ``T`` with ``N``, ``2N``, ``4N`` implicit steps and
comparing the final fields gives ``order = log2(‖u_N−u_2N‖ / ‖u_2N−u_4N‖)``.

Today the assembler hard-codes implicit (backward) Euler, so the expected order
is ≈ 1.  This same harness is the acceptance gate for the Phase-2 BDF2 /
Crank–Nicolson schemes, which must lift it to ≈ 2.
"""

import numpy as np

from source.solvers.fvm.assemble.convection import compute_volumetric_face_flux
from source.solvers.fvm.assemble.momentum import assemble_momentum_equation
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.solve.linear_interface import solve_linear_system

from ._structured_mesh import structured_box

PI = np.pi


def _scalar_field(x, y):
    """Divergence-free 2D Taylor–Green spatial profile (z-uniform)."""
    return np.column_stack(
        [np.sin(PI * x) * np.cos(PI * y), -np.cos(PI * x) * np.sin(PI * y), np.zeros_like(x)]
    )


def _advection_direction_dot_scalar_field_gradient(x, y, a):
    """(a·∇)φ for the field above, a constant."""
    sx, sy = np.sin(PI * x), np.sin(PI * y)
    cx, cy = np.cos(PI * x), np.cos(PI * y)
    gx = a[0] * PI * cx * cy - a[1] * PI * sx * sy
    gy = a[0] * PI * sx * sy - a[1] * PI * cx * cy
    return np.column_stack([gx, gy, np.zeros_like(x)])


def _source(x, y, t, a, kinematic_viscosity):
    """S = g'φ + g(a·∇)φ − ν g ∇²φ with g=e^{-t}, ∇²φ = −2π²φ."""
    g = np.exp(-t)
    scalar_field = _scalar_field(x, y)
    adv = _advection_direction_dot_scalar_field_gradient(x, y, a)
    return g * (-scalar_field + adv + 2.0 * kinematic_viscosity * PI**2 * scalar_field)


def _set_ghosts(velocity, mesh, geo, t):
    """Write u_exact(t) into the boundary ghost cells (Dirichlet)."""
    n_elem = mesh["n_cells"]
    n_int = mesh["n_interior_faces"]
    fc = geo["face_centre"]
    g = np.exp(-t)
    for b in mesh["boundary"]:
        b["boundary_condition_type"] = "fixedValue"
        b["velocity_type"] = "fixedValue"
        b["velocity_value"] = [0.0, 0.0, 0.0]
        for j in range(b["n_faces"]):
            fi = b["start_face"] + j
            gi = n_elem + (fi - n_int)
            velocity[gi] = g * _scalar_field(np.array([fc[fi, 0]]), np.array([fc[fi, 1]])).ravel()


def _integrate(mesh, geo, n_steps, T, a, kinematic_viscosity, ddt_scheme="euler"):
    """March the linear momentum operator in time; return interior U(T).

    ``ddt_scheme`` selects BDF1 (``"euler"``) or BDF2 (``"backward"``).  BDF2 is
    self-starting (first step uses BDF1), so ``U_old_old`` is None on step 0.
    """
    n_elem = mesh["n_cells"]
    n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
    cc = geo["cell_centre"]
    time_step_size = T / n_steps

    # Frozen advecting flux from the constant velocity a.
    a_field = np.tile(a, (n_elem + n_bnd, 1)).astype(np.float64)
    phi_flux = compute_volumetric_face_flux(a_field, mesh, geo)

    velocity = np.zeros((n_elem + n_bnd, 3))
    velocity[:n_elem] = _scalar_field(cc[:, 0], cc[:, 1])  # u(0) = φ
    _set_ghosts(velocity, mesh, geo, 0.0)
    p = np.zeros(n_elem + n_bnd)

    velocity_older = None
    t = 0.0
    for _ in range(n_steps):
        t_new = t + time_step_size
        velocity_old = velocity.copy()
        _set_ghosts(velocity, mesh, geo, t_new)  # Dirichlet at the new time level
        S = _source(cc[:, 0], cc[:, 1], t_new, a, kinematic_viscosity)
        # Use the fully-implicit central scheme (cell-Péclet < 2 here, so it is
        # bounded) to isolate the *time*-integration order: the deferred
        # scheme's explicit central correction is lagged by one solve and would
        # cap the observed order at ~1 without outer correctors.
        # The deviatoric transpose-stress contribution is explicit in
        # each momentum assembly and is refreshed by the PIMPLE outer loop.
        # Converge that fixed point here too; a single lagged evaluation would
        # inject a first-order splitting error into a test intended to isolate
        # the BDF time operator.
        for _picard in range(8):
            previous_iterate = velocity[:n_elem].copy()
            mom = assemble_momentum_equation(
                velocity,
                p,
                phi_flux,
                1.0,
                kinematic_viscosity,
                mesh,
                geo,
                mesh["boundary"],
                convection_scheme="central",
                time_step_size=time_step_size,
                velocity_old=velocity_old,
                velocity_older=velocity_older,
                ddt_scheme=ddt_scheme,
                source_explicit=S,
            )
            for i, comp in enumerate(["x", "y", "z"]):
                velocity[:n_elem, i] = solve_linear_system(
                    mom[comp]["A"],
                    mom[comp]["b"],
                    method="spsolve",
                    equation_type="momentum",
                )
            change = np.linalg.norm(velocity[:n_elem] - previous_iterate)
            if change <= 1e-12 * max(np.linalg.norm(velocity[:n_elem]), 1.0):
                break
        velocity_older = velocity_old
        t = t_new
    return velocity[:n_elem].copy()


def _l2(a, b, vol):
    d = a - b
    return np.sqrt(np.sum(vol[:, None] * d**2) / np.sum(vol))


class TestTemporalOrder:
    A = (1.0, 0.5, 0.0)
    NU = 0.05
    T = 0.4

    def _order(self, mesh, geo, ddt_scheme):
        vol = geo["cell_volume"]
        u_n = _integrate(mesh, geo, 8, self.T, self.A, self.NU, ddt_scheme)
        u_2n = _integrate(mesh, geo, 16, self.T, self.A, self.NU, ddt_scheme)
        u_4n = _integrate(mesh, geo, 32, self.T, self.A, self.NU, ddt_scheme)
        e1 = _l2(u_n, u_2n, vol)
        e2 = _l2(u_2n, u_4n, vol)
        return float(np.log2(e1 / e2))

    def test_backward_euler_first_order(self):
        mesh = structured_box(12, 12, 1)
        geo = compute_mesh_geometry(mesh)
        order = self._order(mesh, geo, "euler")
        assert 0.8 < order < 1.3, f"observed temporal order {order:.2f} not ≈ 1 (Euler)"

    def test_bdf2_second_order(self):
        """BDF2 must lift the observed temporal order to ≈ 2."""
        mesh = structured_box(16, 16, 1)
        geo = compute_mesh_geometry(mesh)
        order = self._order(mesh, geo, "backward")
        assert order > 1.7, f"observed BDF2 temporal order {order:.2f} not ≈ 2"
