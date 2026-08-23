"""Certification of the wall-force algorithm on the production carved-cube mesh.

`tests/fvm/test_surface_forces.py` certifies the force building blocks on a
simple all-outer-faces box.  The production hybrid/reference topology is
different: the cube is *carved out* of the mesh and its exposed faces form a
``wall`` patch (``coupling_box_mesh(.., hole_box=..)``).  The subtle,
previously-wrong parts live exactly there — the wall-face normal orientation
(which sets the drag sign) and the wall-pressure ghost that feeds the pressure
force. These tests pin both analytically using only the native FVM solver.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.fields.diagnostics import compute_surface_forces, compute_y_plus
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh
from source.solvers.fvm.sampling.base import SamplingSchedule
from source.solvers.fvm.sampling.forces import ForceSampler

BOX = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
HOLE = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
SPACING = 0.25  # small mesh: fast, and the cube faces land on grid planes


@pytest.fixture(scope="module")
def cube_mesh():
    mesh = coupling_box_mesh(BOX, SPACING, hole_box=HOLE, wall_patch_name="cube")
    geo = compute_mesh_geometry(mesh)
    return mesh, geo


def _cube_patch(mesh):
    return next(b for b in mesh["boundary"] if b["name"] == "cube")


def _blank_fields(mesh):
    n_total = mesh["n_cells"] + (mesh["n_faces"] - mesh["n_interior_faces"])
    return np.zeros((n_total, 3)), np.zeros(n_total)  # U, p


def _wall_face_slice(mesh):
    """(global face index array, ghost-slab index array) for the cube patch."""
    cube = _cube_patch(mesh)
    n_elem, n_int = mesh["n_cells"], mesh["n_interior_faces"]
    faces = np.arange(cube["start_face"], cube["start_face"] + cube["n_faces"])
    ghost = n_elem + (faces - n_int)
    return faces, ghost


def test_cube_patch_is_closed_surface(cube_mesh):
    """The carved cube's outward area vectors sum to zero (closed surface)."""
    mesh, geo = cube_mesh
    faces, _ = _wall_face_slice(mesh)
    net_sf = geo["face_area_vector"][faces].sum(axis=0)
    assert np.allclose(net_sf, 0.0, atol=1e-12), f"cube not closed: ΣSf={net_sf}"
    # And it is the physical cube surface area: 6 faces × (1×1).
    area = np.linalg.norm(geo["face_area_vector"][faces], axis=1).sum()
    assert area == pytest.approx(6.0, abs=1e-9)


def test_uniform_pressure_gives_zero_net_force(cube_mesh):
    """Uniform p on the closed cube surface → net pressure force = 0."""
    mesh, geo = cube_mesh
    velocity, p = _blank_fields(mesh)
    p[:] = 3.7
    res = compute_surface_forces(
        velocity, p, 0.0, 1.0, mesh, geo, mesh["boundary"], patch_names=["cube"]
    )["cube"]
    assert np.allclose(res["pressure_force"], 0.0, atol=1e-12), (
        f"pressure_force={res['pressure_force']}"
    )


def test_drag_sign_is_positive_for_front_high_back_low(cube_mesh):
    """High pressure on the upstream (x=-0.5) face and low on the downstream
    (x=+0.5) face must give POSITIVE drag (Fx>0).  This certifies the wall-face
    normals point out of the fluid — the orientation that makes Cd's sign right.
    """
    mesh, geo = cube_mesh
    velocity, p = _blank_fields(mesh)
    faces, ghost = _wall_face_slice(mesh)
    fc = geo["face_centre"][faces]
    # Select the upstream (x=-0.5) and downstream (x=+0.5) cube faces purely by
    # geometry; the test then certifies the force ORIENTATION is physical
    # (no assumption about the stored normal sign).
    front = np.abs(fc[:, 0] + 0.5) < 1e-9  # upstream face
    back = np.abs(fc[:, 0] - 0.5) < 1e-9  # downstream face
    assert front.any() and back.any(), "front/back cube faces not identified"
    p[ghost[front]] = 1.0  # stagnation over-pressure upstream
    p[ghost[back]] = -0.5  # base suction downstream
    res = compute_surface_forces(
        velocity,
        p,
        0.0,
        1.0,
        mesh,
        geo,
        mesh["boundary"],
        patch_names=["cube"],
        reference_velocity=1.0,
        reference_area=1.0,
    )["cube"]
    assert res["pressure_force"][0] > 0.0, (
        f"expected positive drag, got Fx={res['pressure_force'][0]}"
    )
    assert res["coeffs"]["drag_coefficient"] > 0.0, (
        f"expected drag_coefficient>0, got {res['coeffs']['drag_coefficient']}"
    )
    # front area = back area = 1 → Fx = -(p_front*(-1) + p_back*(+1)) = 1.0 - (-0.5) = 1.5
    assert res["pressure_force"][0] == pytest.approx(1.5, abs=1e-9)
    assert np.allclose(res["pressure_force"][1:], 0.0, atol=1e-12)


def test_linear_pressure_field_matches_analytic_force(cube_mesh):
    """p = a·x on the cube → analytic net force.  For F = ∮ -p n̂_body dA over a
    unit cube, p=a·x gives F = (-a·V_cube, 0, 0) = (-a, 0, 0) (divergence theorem,
    V=1).  Here Fp = ρ·p·Sf with Sf out of the fluid (= -n̂_body dA), so
    Fp_x = +a·∮ x·(Sf_x) ... verified against a direct face sum."""
    mesh, geo = cube_mesh
    velocity, p = _blank_fields(mesh)
    faces, ghost = _wall_face_slice(mesh)
    a, density = 2.0, 1.5
    fc = geo["face_centre"][faces]
    p[ghost] = a * fc[:, 0]
    res = compute_surface_forces(
        velocity, p, 0.0, density, mesh, geo, mesh["boundary"], patch_names=["cube"]
    )["cube"]
    expected = density * (geo["face_area_vector"][faces] * p[ghost][:, None]).sum(axis=0)
    assert np.allclose(res["pressure_force"], expected, atol=1e-12)
    # divergence-theorem value on the unit cube: |Fp_x| = ρ·a·V = ρ·a·1
    assert abs(res["pressure_force"][0]) == pytest.approx(density * a * 1.0, abs=1e-9)


def test_viscous_force_matches_boundary_diffusion_flux(cube_mesh):
    """On the downstream (x=+0.5, normal ‖x) cube face, a tangential ghost
    velocity U_y=1 gives viscous force Fv_y = -μ·Σ(A/d) — the boundary diffusion
    flux — mirroring the certified box-mesh test on the carved-cube topology."""
    mesh, geo = cube_mesh
    for b in mesh["boundary"]:  # gradient reconstruction needs BC types
        b["boundary_condition_type"] = "zeroGradient"
        b["velocity_type"] = "zeroGradient"
    velocity, p = _blank_fields(mesh)
    faces, ghost = _wall_face_slice(mesh)
    fc = geo["face_centre"][faces]
    back = np.abs(fc[:, 0] - 0.5) < 1e-9  # x=+0.5 faces (wall normal ‖ x)
    assert back.any()
    mu = 0.01
    velocity[ghost[back], 1] = 1.0  # tangential (y) slip on the x-normal face only
    res = compute_surface_forces(
        velocity, p, mu, 1.0, mesh, geo, mesh["boundary"], patch_names=["cube"]
    )["cube"]
    areas = np.linalg.norm(geo["face_area_vector"][faces[back]], axis=1)
    expected_y = -mu * np.sum(areas / geo["wall_distance"][faces[back]])
    assert res["viscous_force"][1] == pytest.approx(expected_y, rel=1e-9)
    assert abs(res["viscous_force"][0]) < 1e-9 and abs(res["viscous_force"][2]) < 1e-9


def test_yplus_on_couette_field_matches_analytic(cube_mesh):
    """y+ on the x-normal cube faces, from a known tangential owner-cell
    velocity, equals the closed form u_tau·d/ν with u_tau = sqrt(ν·Ut/d)."""
    mesh, geo = cube_mesh
    for b in mesh["boundary"]:
        b.setdefault("velocity_type", "noSlip" if b["name"] == "cube" else "zeroGradient")
    velocity, _ = _blank_fields(mesh)
    faces, _ = _wall_face_slice(mesh)
    fc = geo["face_centre"][faces]
    back = np.abs(fc[:, 0] - 0.5) < 1e-9  # x-normal faces: U_y is tangential
    owners = mesh["owners"][faces[back]]
    kinematic_viscosity, ut = 1e-3, 0.5
    velocity[owners, 1] = ut
    stats = compute_y_plus(
        velocity, kinematic_viscosity, mesh, geo, mesh["boundary"], patch_names=["cube"]
    )
    assert "cube" in stats
    s = stats["cube"]
    assert s["max"] > 0.0 and s["avg"] > 0.0
    d0 = float(geo["wall_distance"][faces[back][0]])
    yplus_ref = np.sqrt(kinematic_viscosity * ut / d0) * d0 / kinematic_viscosity
    assert s["max"] == pytest.approx(yplus_ref, rel=1e-6)


def test_yplus_uses_local_cell_viscosity():
    """A kinematic_viscosity field is sampled at each wall-face owner, not averaged."""
    velocity = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    kinematic_viscosity = np.array([1.0e-6, 4.0e-6])
    mesh = {
        "n_cells": 2,
        "owners": np.array([0, 1], dtype=np.int32),
    }
    geometry = {
        "wall_distance": np.array([1.0e-3, 1.0e-3]),
        "face_area_vector": np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    }
    boundaries = [{"name": "wall", "type": "wall", "start_face": 0, "n_faces": 2}]

    stats = compute_y_plus(
        velocity, kinematic_viscosity, mesh, geometry, boundaries, patch_names=["wall"]
    )["wall"]
    expected = np.sqrt(kinematic_viscosity / 1.0e-3) * 1.0e-3 / kinematic_viscosity

    assert stats["min"] == pytest.approx(float(np.min(expected)))
    assert stats["max"] == pytest.approx(float(np.max(expected)))
    assert stats["avg"] == pytest.approx(float(np.mean(expected)))


def _split_outer_patches(mesh):
    """Split the single merged coupling patch of coupling_box_mesh into the six
    named outer patches (inlet/outlet/ymin/ymax/zmin/zmax) + cube wall, so a
    standalone external-flow solver can be built on the carved cube."""
    nx = int(round((5.0 - -3.0) / 0.5))
    ny = nz = int(round((3.0 - -3.0) / 0.5))
    counts = (ny * nz, ny * nz, nx * nz, nx * nz, nx * ny, nx * ny)
    names = ("inlet", "outlet", "ymin", "ymax", "zmin", "zmax")
    start = mesh["n_interior_faces"]
    patches = []
    for name, count in zip(names, counts, strict=True):
        patches.append({"name": name, "start_face": start, "n_faces": count, "type": "patch"})
        start += count
    cube = mesh["boundary"][-1]
    patches.append({**cube, "start_face": start, "type": "wall"})
    mesh["boundary"] = patches
    return mesh


def test_wall_pressure_ghost_is_physical_after_solve(tmp_path):
    """Certify the value that FEEDS the pressure force: after a few solver steps
    on the carved cube, the cube-patch pressure ghost must carry the real wall
    pressure (front-face stagnation > back-face base suction) — not a stale zero.
    A zero/stale ghost is the failure mode that silently nulls the pressure drag
    regardless of the summation being correct."""
    mesh = _split_outer_patches(
        coupling_box_mesh(
            (-3.0, 5.0, -3.0, 3.0, -3.0, 3.0),
            0.5,
            hole_box=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
            wall_patch_name="cube",
        )
    )
    params_schemes = DiscretizationConfig(
        convection_scheme="central", gradient_scheme="gauss", time_scheme="euler_implicit"
    )
    params_linear = LinearSolverConfig(momentum_solver="bicgstab", pressure_solver="amg")
    params_pimple = PimpleControl(n_correctors=2, n_outer_correctors=2)
    params_forces = [
        ForceSampler(
            patch_names=["cube"],
            reference_velocity=1.0,
            reference_area=1.0,
            reference_length=1.0,
            schedule=SamplingSchedule(every_n_steps=1),
        )
    ]
    config = FVMSetup(
        case_name="ghost-cert",
        time=TimeConfig.transient(time_step_size=0.05, duration=0.5, output_interval_steps=10**9),
        schemes=params_schemes,
        linear=params_linear,
        pimple=params_pimple,
        samplers=params_forces,
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
        initial_velocity=[1.0, 0.02, 0.0],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(config, case_dir=str(tmp_path), mesh_data=mesh)
        solver.auto_write = False
        for _ in range(10):
            solver.advance()

    geo = solver.geo_data
    n_elem, n_int = mesh["n_cells"], mesh["n_interior_faces"]
    cube = next(b for b in mesh["boundary"] if b["name"] == "cube")
    faces = np.arange(cube["start_face"], cube["start_face"] + cube["n_faces"])
    ghost = n_elem + (faces - n_int)
    p_ghost = np.asarray(solver.kinematic_pressure)[ghost]
    fc = geo["face_centre"][faces]

    # (1) the ghost is a real, populated field — not stale zeros
    assert np.all(np.isfinite(p_ghost))
    assert np.std(p_ghost) > 1e-6, "cube pressure ghost is ~uniform/stale"
    # (2) physical signature: upstream face pressure > downstream face pressure
    front = p_ghost[np.abs(fc[:, 0] + 0.5) < 1e-9].mean()
    back = p_ghost[np.abs(fc[:, 0] - 0.5) < 1e-9].mean()
    assert front > back, f"no stagnation/base signature: front={front:.4f} back={back:.4f}"
    # (3) the force built from this ghost is a physical, positive drag
    cd = solver.last_forces["cube"]["coeffs"]["drag_coefficient"]
    assert cd > 0.0, f"non-physical drag from wall ghost: Cd={cd}"
