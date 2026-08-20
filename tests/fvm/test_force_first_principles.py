"""First-principles force certification on the carved-cube mesh (AGENT_PLAN M3).

Complements ``test_surface_forces.py`` (building blocks on a plain box) and
``test_wall_force_certification.py`` (drag sign, wall ghost, boundary
diffusion flux) with the remaining spec Phase-10 checks:

1. uniform pressure on the closed body → zero net force AND zero moment about
   an arbitrary centre;
2. an arbitrary constant pressure-gauge shift does not change the closed-body
   force or moment;
3. a manufactured LINEAR velocity field gives the analytic viscous traction
   ``μ·dev(∇U + ∇Uᵀ)·n`` exactly (gradient reconstruction and the snGrad
   correction are exact for linear fields);
4. the discrete pressure force converges at second order to the analytic
   surface integral for a smooth pressure field (midpoint rule per flat face);
5. a symmetric external-flow solve produces symmetric forces (|Cy|, |Cz| and
   the transverse moments bounded by the linear-solver tolerance, not by an
   observed value);
6. the diagnosed wall force agrees with a control-volume momentum balance on
   the outer boundary, with the imbalance decreasing under mesh refinement
   (the diagnostic uses the deviatoric snGrad-corrected traction, the CV uses
   the discrete boundary fluxes: their difference is discretisation error);
7. auto-selection of force patches uses the mesh patch TYPE, never the name.

Sign conventions certified here (n̂ = face normal out of the fluid, i.e. into
the body):  F_body = ∮ (ρ p_kin n̂ − τ·n̂) dS,  τ = μ dev(∇U + ∇Uᵀ).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from source.solvers.FVM import (
    BoundaryConfig,
    DiscretizationConfig,
    ForceSampler,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.fields import gradients
from source.solvers.FVM.fields.diagnostics import compute_surface_forces
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.mesh.rectilinear import box_mesh_3d, coupling_box_mesh
from source.solvers.FVM.sampling.base import SamplingSchedule

BOX = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
HOLE = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)


def _carved(spacing: float):
    mesh = coupling_box_mesh(BOX, spacing, hole_box=HOLE, wall_patch_name="cube")
    for b in mesh["boundary"]:  # gradient reconstruction needs BC types
        b["bc_type"] = "zeroGradient"
        b["velocity_type"] = "zeroGradient"
    return mesh, compute_mesh_geometry(mesh)


@pytest.fixture(scope="module")
def cube():
    return _carved(0.25)


def _fields(mesh):
    n_total = mesh["n_cells"] + (mesh["n_faces"] - mesh["n_interior_faces"])
    return np.zeros((n_total, 3)), np.zeros(n_total)


def _wall_faces(mesh):
    (patch,) = [b for b in mesh["boundary"] if b["name"] == "cube"]
    faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    ghost = mesh["n_cells"] + (faces - mesh["n_interior_faces"])
    return faces, ghost


# ────────────────────────────────────────────────────────────────────────────
# 1. + 2.  Closed-body invariants
# ────────────────────────────────────────────────────────────────────────────


def test_uniform_pressure_zero_force_and_moment(cube):
    mesh, geo = cube
    velocity, p = _fields(mesh)
    p[:] = 2.19
    res = compute_surface_forces(
        velocity,
        p,
        0.0,
        1.0,
        mesh,
        geo,
        mesh["boundary"],
        patch_names=["cube"],
        moment_centre=[0.3, -0.7, 1.1],
    )["cube"]
    assert np.allclose(res["Fp"], 0.0, atol=1e-12)
    assert np.allclose(res["Mtot"], 0.0, atol=1e-12), f"M={res['Mtot']}"


def test_gauge_shift_does_not_change_closed_body_force(cube):
    mesh, geo = cube
    rng = np.random.default_rng(7)
    velocity, p = _fields(mesh)
    _, ghost = _wall_faces(mesh)
    p[ghost] = rng.standard_normal(ghost.size)

    def forces(shift):
        return compute_surface_forces(
            velocity,
            p + shift,
            0.0,
            1.3,
            mesh,
            geo,
            mesh["boundary"],
            patch_names=["cube"],
            moment_centre=[0.2, 0.1, -0.4],
        )["cube"]

    base, shifted = forces(0.0), forces(37.5)
    np.testing.assert_allclose(shifted["Fp"], base["Fp"], atol=1e-10)
    np.testing.assert_allclose(shifted["Mtot"], base["Mtot"], atol=1e-10)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Manufactured linear velocity → analytic full-tensor viscous traction
# ────────────────────────────────────────────────────────────────────────────


def test_linear_velocity_field_gives_analytic_traction(cube):
    """U = A·x is reproduced exactly by the gradient reconstruction AND by the
    snGrad correction, so the computed traction must equal
    μ·dev(A + Aᵀ)·n̂ per face to round-off.  This exercises the full stress
    tensor (tangential derivatives and the deviatoric part), which the
    boundary-diffusion-flux test cannot see."""
    mesh, geo = cube
    A = np.array([[0.3, -1.1, 0.4], [0.9, 0.2, -0.5], [-0.7, 0.6, -0.5]])  # tr(A)=0 not required
    mu = 0.037
    n_elem = mesh["n_cells"]
    n_int = mesh["n_interior_faces"]
    centroids = geo["cell_centroids"][:n_elem]
    velocity, p = _fields(mesh)
    velocity[:n_elem] = centroids @ A.T
    # Exact values on every boundary face (ghost slots follow the interior).
    face_centroids = geo["face_centroids"][n_int:]
    velocity[n_elem:] = face_centroids @ A.T

    res = compute_surface_forces(
        velocity, p, mu, 1.0, mesh, geo, mesh["boundary"], patch_names=["cube"]
    )["cube"]

    faces, _ = _wall_faces(mesh)
    sf = geo["face_sf"][faces]
    two_symm = A + A.T
    dev = two_symm - (2.0 / 3.0) * np.trace(A) * np.eye(3)
    # F_v = −Σ μ dev(2·symm A)·Sf  (Sf out of the fluid)
    expected = -mu * (sf @ dev.T).sum(axis=0)
    np.testing.assert_allclose(res["Fv"], expected, rtol=1e-9, atol=1e-12)


# ────────────────────────────────────────────────────────────────────────────
# 4.  Mesh convergence of the pressure force (2nd-order midpoint rule)
# ────────────────────────────────────────────────────────────────────────────


def _smooth_p(x, y, z):
    return np.sin(1.3 * x + 0.2) * np.cos(0.9 * y - 0.1) + 0.5 * z * z


def _exact_cube_pressure_force(p_fn=_smooth_p, n_quad: int = 1500) -> np.ndarray:
    """∮ p n̂ dS over the unit cube via dense midpoint quadrature (per flat
    face; error ~n_quad⁻² ≈ 4e-7, far below the coarsest mesh error)."""
    t = -0.5 + (np.arange(n_quad) + 0.5) / n_quad
    u, v = np.meshgrid(t, t, indexing="ij")
    w = 1.0 / n_quad**2
    force = np.zeros(3)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            coords = [None, None, None]
            coords[axis] = np.full_like(u, 0.5 * sign)
            tang = [a for a in range(3) if a != axis]
            coords[tang[0]], coords[tang[1]] = u, v
            p = p_fn(*coords)
            # n̂ out of the fluid points INTO the body: n̂ = −sign·ê_axis on
            # the face at coordinate 0.5·sign (certified by
            # test_exact_reference_orientation below).
            force[axis] -= sign * w * p.sum()
    return force


def test_smooth_pressure_force_converges_second_order():
    """Discrete Fp = Σ p(x_f)·Sf is the midpoint rule on each flat wall face
    → error O(h²) against the analytic surface integral."""
    exact = _exact_cube_pressure_force()
    errors = []
    spacings = [0.25, 0.125, 0.0625]
    for h in spacings:
        mesh, geo = _carved(h)
        velocity, p = _fields(mesh)
        faces, ghost = _wall_faces(mesh)
        fc = geo["face_centroids"][faces]
        p[ghost] = _smooth_p(fc[:, 0], fc[:, 1], fc[:, 2])
        res = compute_surface_forces(
            velocity, p, 0.0, 1.0, mesh, geo, mesh["boundary"], patch_names=["cube"]
        )["cube"]
        errors.append(np.linalg.norm(res["Fp"] - exact))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(orders) > 1.8, f"errors={errors}, observed orders={orders}"


def test_exact_reference_orientation():
    """The quadrature reference must use n̂ out of the fluid (into the body).
    For p = a·x the closed-body force is F = ∮ p n̂ dS = −∇p·V = (−a·V, 0, 0):
    higher pressure downstream pushes the body upstream.  Both the discrete
    sum and the quadrature reference must give exactly that."""
    a = 2.0
    mesh, geo = _carved(0.25)
    velocity, p = _fields(mesh)
    faces, ghost = _wall_faces(mesh)
    p[ghost] = a * geo["face_centroids"][faces][:, 0]
    res = compute_surface_forces(
        velocity, p, 0.0, 1.0, mesh, geo, mesh["boundary"], patch_names=["cube"]
    )["cube"]
    assert res["Fp"][0] == pytest.approx(-a * 1.0, abs=1e-9)  # −a·V on the body
    # The quadrature reference must agree on the same linear field.
    ref = _exact_cube_pressure_force(p_fn=lambda x, y, z: a * x, n_quad=200)
    np.testing.assert_allclose(ref, [-a, 0.0, 0.0], atol=1e-12)


# ────────────────────────────────────────────────────────────────────────────
# 5. + 6.  Solve-based checks: symmetry and control-volume momentum balance
# ────────────────────────────────────────────────────────────────────────────

_EXT_BOX = (-3.0, 5.0, -3.0, 3.0, -3.0, 3.0)
_MOMENTUM_TOL = 1e-6
_PRESSURE_TOL = 1e-10


def _external_flow_solver(tmp_path, spacing: float, n_steps: int, time_step_size: float = 0.05):
    mesh = box_mesh_3d(
        *(np.arange(_EXT_BOX[2 * a], _EXT_BOX[2 * a + 1] + spacing / 2, spacing) for a in range(3)),
        hole_box=HOLE,
        wall_patch_name="cube",
    )
    config = FVMSetup(
        case_name="cv-balance",
        time=TimeConfig.transient(
            time_step_size=time_step_size,
            duration=n_steps * time_step_size,
            output_interval_steps=10**9,
        ),
        schemes=DiscretizationConfig(
            convection_scheme="central", gradient_scheme="gauss", time_scheme="euler_implicit"
        ),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="amg",
            momentum_tolerance=_MOMENTUM_TOL,
            pressure_tolerance=_PRESSURE_TOL,
        ),
        pimple=PimpleControl(n_correctors=2, n_outer_correctors=2),
        samplers=[
            ForceSampler(
                patch_names=["cube"],
                ref_velocity=1.0,
                ref_area=1.0,
                ref_length=1.0,
                schedule=SamplingSchedule(every_n_steps=1),
            )
        ],
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.05),
        boundaries=[
            BoundaryConfig.inlet("inlet", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("outlet", 0.0),
            BoundaryConfig.slip("ymin"),
            BoundaryConfig.slip("ymax"),
            BoundaryConfig.slip("zmin"),
            BoundaryConfig.slip("zmax"),
            BoundaryConfig.wall("cube"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],  # symmetric start: no perturbation
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(config, case_dir=str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
    return solver


def test_symmetric_flow_gives_symmetric_forces(tmp_path):
    """Geometry, BCs, and the initial field are all symmetric in y and z, so
    the transverse force and moment coefficients must vanish.  In exact
    arithmetic they are identically zero; the only asymmetry source is the
    linear-solver residual (momentum_tol=1e-6, pressure_tol=1e-10), so the
    bound is set at 1e-4 — two orders above the residual floor, ten times
    below any physical asymmetry."""
    solver = _external_flow_solver(tmp_path, spacing=0.5, n_steps=10)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(10):
            solver.advance()
    forces = solver.last_forces["cube"]
    fx = abs(forces["Ftot"][0])
    assert fx > 1e-3, "drag should be nonzero for the impulsively started flow"
    assert abs(forces["Ftot"][1]) < 1e-4 * max(fx, 1.0), f"Fy={forces['Ftot'][1]}"
    assert abs(forces["Ftot"][2]) < 1e-4 * max(fx, 1.0), f"Fz={forces['Ftot'][2]}"
    # Transverse moments about the body centre likewise vanish by symmetry.
    assert abs(forces["Mtot"][1]) < 1e-4, f"My={forces['Mtot'][1]}"
    assert abs(forces["Mtot"][2]) < 1e-4, f"Mz={forces['Mtot'][2]}"


def _cv_forces(tmp_path, spacing: float, n_steps: int = 8) -> dict[str, float]:
    """Return the x-components of three independent wall-force evaluations.

    * ``cv``   — outer control-volume momentum balance
        F_body = −d/dt ∫ρu dV − ∮_outer ρ u (u·n̂) dS + ∮_outer (−ρ p n̂ + τ·n̂) dS
      evaluated with the solver's own fields (φ for the mass flux, patch ghost
        values for u and p, and the complete deviatoric stress for the
        outer traction; implicit-Euler d/dt from the last two committed states,
        consistent with the solver's time scheme).
    * ``disc`` — wall force from the DISCRETE boundary fluxes the momentum
      equation actually applies: pressure, implicit Laplacian, and explicit
      transpose/deviatoric stress.
    * ``diag`` — the production diagnostic (deviatoric snGrad-corrected
      traction, ``compute_surface_forces``).
    """
    solver = _external_flow_solver(tmp_path, spacing, n_steps)
    mesh, geo = solver.mesh_data, solver.geo_data
    n_elem = mesh["n_cells"]
    n_int = mesh["n_interior_faces"]
    rho = solver.setup.transport.density
    mu = rho * solver.setup.transport.kinematic_viscosity
    volumes = geo["cell_volumes"][:n_elem]

    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n_steps - 1):
            solver.advance()
        momentum_before = rho * (volumes[:, None] * solver.velocity[:n_elem]).sum(axis=0)
        solver.advance()
    momentum_after = rho * (volumes[:, None] * solver.velocity[:n_elem]).sum(axis=0)
    dmdt = (momentum_after - momentum_before) / solver.setup.time.time_step_size

    velocity, p, face_flux = (
        np.asarray(solver.velocity),
        np.asarray(solver.kinematic_pressure),
        np.asarray(solver.face_flux),
    )
    grad_u = gradients._resolve_gradient_fn(geo)(velocity, mesh, geo)

    def _transpose_stress_flux(faces):
        """Dynamic ``mu*dev2(T(grad(U))) . Sf`` used by momentum assembly."""
        ghost = n_elem + (faces - n_int)
        grad_face = grad_u[ghost]
        sf = geo["face_sf"][faces]
        trace = np.einsum("fii->f", grad_face)
        transposed = np.einsum("fji,fi->fj", grad_face, sf)
        return mu * (transposed - (2.0 / 3.0) * trace[:, None] * sf)

    f_cv = -dmdt
    for b in mesh["boundary"]:
        if b["name"] == "cube":
            continue
        faces = np.arange(b["start_face"], b["start_face"] + b["n_faces"])
        ghost = n_elem + (faces - n_int)
        sf = geo["face_sf"][faces]
        dist = geo["wall_dist"][faces][:, None]
        areas = np.linalg.norm(sf, axis=1)[:, None]
        # Convective outflow of momentum through the patch (φ = u·Sf).
        f_cv -= (rho * face_flux[faces][:, None] * velocity[ghost]).sum(axis=0)
        # Pressure on the outer boundary (p is kinematic).
        f_cv -= (rho * p[ghost][:, None] * sf).sum(axis=0)
        # Viscous traction on the outer boundary from the normal gradient.
        f_cv += (mu * (velocity[ghost] - velocity[mesh["owners"][faces]]) / dist * areas).sum(
            axis=0
        )
        f_cv += _transpose_stress_flux(faces).sum(axis=0)

    (wall,) = [b for b in mesh["boundary"] if b["name"] == "cube"]
    faces = np.arange(wall["start_face"], wall["start_face"] + wall["n_faces"])
    ghost = n_elem + (faces - n_int)
    sf = geo["face_sf"][faces]
    dist = geo["wall_dist"][faces][:, None]
    areas = np.linalg.norm(sf, axis=1)[:, None]
    f_disc = (rho * p[ghost][:, None] * sf).sum(axis=0) - (
        mu * (velocity[ghost] - velocity[mesh["owners"][faces]]) / dist * areas
    ).sum(axis=0)
    f_disc -= _transpose_stress_flux(faces).sum(axis=0)

    f_diag = np.asarray(solver.last_forces["cube"]["Ftot"])
    return {"cv": float(f_cv[0]), "disc": float(f_disc[0]), "diag": float(f_diag[0])}


@pytest.mark.parametrize("spacing", [0.5, 0.25])
def test_wall_force_control_volume_balance(tmp_path, spacing):
    """Two-level certification of the wall force against the momentum balance.

    (a) EXACT closure: the wall force assembled from the discrete boundary
        fluxes must close the outer-CV discrete momentum balance.  In exact
        arithmetic the interior fluxes telescope and the closure is identity;
        the residual comes from linear-solve/corrector convergence (measured
        0.008–0.08% of drag across these meshes).  The 1% bound is chosen an
        order below the smallest defect this test exists to catch (a missing
        or double-counted flux term, a ρ mix-up, a sign error — all ≥ 100%),
        and an order above the convergence floor.

    (b) PHYSICAL consistency: the production diagnostic (deviatoric
        snGrad-corrected traction) differs from the discrete-flux force by
        tangential-gradient/deviatoric terms — pure discretisation content,
        measured ~5% on these deliberately coarse meshes (the boundary layer
        δ≈√(νt)≈0.14 is unresolved at h=0.25; asymptotic convergence of this
        gap is exercised on the wall-refined production mesh in the benchmark
        validation phase, not here).  The 15% bound distinguishes convention
        errors (wrong sign, missing μ, full vs deviatoric stress → ≥ 33%)
        from legitimate discretisation content."""
    f = _cv_forces(tmp_path, spacing=spacing)
    drag = abs(f["disc"])
    assert drag > 1e-3
    assert abs(f["disc"] - f["cv"]) < 0.01 * drag, (
        f"discrete wall flux does not close the CV balance: disc={f['disc']:.5f} cv={f['cv']:.5f}"
    )
    assert abs(f["diag"] - f["disc"]) < 0.15 * drag, (
        f"diagnostic traction inconsistent with discrete flux: "
        f"diag={f['diag']:.5f} disc={f['disc']:.5f}"
    )


# ────────────────────────────────────────────────────────────────────────────
# 7.  Patch auto-selection
# ────────────────────────────────────────────────────────────────────────────


def test_patch_autoselection_uses_mesh_type_only(cube):
    """With patch_names=None only patches of TYPE 'wall' are selected; a
    non-wall patch whose NAME contains 'wall' must not be."""
    mesh, geo = cube
    boundaries = [dict(b) for b in mesh["boundary"]]
    outer = next(b for b in boundaries if b["name"] != "cube")
    outer["name"] = "seawallFarfield"  # name contains 'wall'; type stays 'patch'
    velocity, p = _fields(mesh)
    res = compute_surface_forces(velocity, p, 0.0, 1.0, mesh, geo, boundaries, patch_names=None)
    assert set(res) == {"cube"}, f"auto-selected {set(res)}"
