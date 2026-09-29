"""Native non-box meshes advance with GBD and reproduce a restarted step."""

import json

import numpy as np
import pytest

from source.coupler import CouplerSetup, FVMVPMCoupler
from source.solvers.fvm import (
    BoundaryConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.immersed_boundary.body import ImmersedBody
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh
from source.solvers.fvm.mesh.validation import extract_cell_subset_mesh
from source.solvers.vpm import DirectInduction, Numerics, ViscousConfig, VPMCase, VPMSolver


def body_mesh(kind):
    bounds = (-1.0, 1.0) * 3
    if kind == "immersed":
        return coupling_box_mesh(bounds, 0.25)
    if kind in {"rotated", "thin"}:
        nodes = None
        hole = (-0.25, 0.25) * 3
        if kind == "thin":
            nodes = (
                np.r_[np.linspace(-1, -0.01, 5), np.linspace(0.01, 1, 5)],
                np.linspace(-1, 1, 9),
                np.linspace(-1, 1, 9),
            )
            hole = (-0.01, 0.01, -0.5, 0.5, -0.5, 0.5)
        mesh = coupling_box_mesh(bounds, 0.25, hole_box=hole, wall_patch_name="body", nodes=nodes)
        if kind == "rotated":
            points = mesh["vertex_position"]
            angle = 0.32 * np.clip((1 - np.max(np.abs(points), axis=1)) / 0.5, 0, 1)
            x, y = points[:, 0].copy(), points[:, 1].copy()
            points[:, 0] = np.cos(angle) * x - np.sin(angle) * y
            points[:, 1] = np.sin(angle) * x + np.cos(angle) * y
        return mesh
    full = coupling_box_mesh(bounds, 0.25)
    centres = compute_mesh_geometry(full, compute_lsq=False)["cell_centre"]
    x, y, z = centres.T
    if kind == "concave":
        solid = (np.abs(x) < 0.5) & (np.abs(y) < 0.5) & (np.abs(z) < 0.25) & ((x < 0) | (y < 0))
    else:
        solid = (np.abs(x) > 0.25) & (np.abs(x) < 0.75) & (np.abs(y) < 0.25) & (np.abs(z) < 0.25)
    mesh = extract_cell_subset_mesh(full, np.flatnonzero(~solid))
    outer_count = full["n_faces"] - full["n_interior_faces"]
    inner = mesh["n_interior_faces"]
    walls = mesh["n_faces"] - inner - outer_count
    mesh["boundary"] = [
        {"name": "body", "type": "wall", "start_face": inner, "n_faces": walls},
        {
            "name": "numericalBoundary",
            "type": "patch",
            "start_face": inner + walls,
            "n_faces": outer_count,
        },
    ]
    return mesh


def make_coupler(directory, kind, transfer_method):
    velocity = [0.2, 0.03, 0]
    fvm = FVMSolver(
        FVMSetup(
            case_name=kind,
            linear=LinearSolverConfig(momentum_tolerance=1e-10, pressure_tolerance=1e-10),
            time=TimeConfig(time_step_size=0.005, end_time=0.02),
            transport=TransportConfig(kinematic_viscosity=0.01),
            initial_velocity=velocity,
            boundaries=[
                *([] if kind == "immersed" else [BoundaryConfig.wall("body")]),
                BoundaryConfig(
                    name="numericalBoundary",
                    velocity_type="fixedValue",
                    velocity_value=velocity,
                    pressure_type="fixedFluxPressure",
                ),
            ],
        ),
        case_dir=directory,
        mesh_data=body_mesh(kind),
    )
    if kind == "immersed":
        body = ImmersedBody.extruded_polygon_z(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0], [0, 0], [0, 0.5], [-0.5, 0.5]],
            z_bounds=[-0.25, 0.25],
            grid_spacing=0.25,
            caps=True,
        )
        fvm.set_immersed_bodies([body], grid_spacing=0.25)
        # A resolved solenoidal perturbation exercises nonzero initial
        # transfer and avoids a roundoff-sized linear residual denominator.
        x, y, _ = fvm.get_cell_centre_coordinates().T
        initial = np.broadcast_to(velocity, (len(x), 3)).copy()
        initial[:, 0] += 0.01 * np.sin(np.pi * x) * np.cos(np.pi * y)
        initial[:, 1] -= 0.01 * np.cos(np.pi * x) * np.sin(np.pi * y)
        fvm.set_initial_velocity(initial)
    vpm = VPMSolver(
        VPMCase(
            directory=directory,
            numerics=Numerics(
                time_step_size=0.01,
                compute_device="CPU",
                max_n_particles=20000,
                domain_bounds=(-2.0, 2.0) * 3,
                freestream_velocity=velocity,
                induction=DirectInduction(),
                viscous=ViscousConfig.gbd(
                    kinematic_viscosity=0.01,
                    particle_spacing=0.25,
                    padding=4,
                    threshold=1e-5,
                    threshold_mode=(
                        "absolute" if transfer_method == "buffered_m4_renewal" else "budget"
                    ),
                    max_nodes=20000,
                ),
            ),
        )
    )
    return FVMVPMCoupler(
        fvm,
        vpm,
        CouplerSetup(
            freestream_velocity=velocity,
            transfer_method=transfer_method,
            eta_blend_width=0,
            backup_interval_steps=100,
            transfer_discretization_error_limit=1.0,
        ),
    )


def particle_state(coupler):
    particles = coupler.vpm_solver.particles
    count = particles.n_particles_total
    physics = coupler.vpm_solver.physics
    positions = physics._download_vector_field(particles.position, count)
    strengths = physics._download_vector_field(particles.vortex_strength, count)
    assert count > 0
    assert not coupler.vorticity_transfer.solid_boundary.contains(positions).any()
    order = np.lexsort(positions.T[::-1])
    return positions[order], strengths[order]


@pytest.mark.parametrize("transfer_method", ["common_lattice", "buffered_m4_renewal"])
@pytest.mark.parametrize("kind", ["rotated", "concave", "multiple", "thin", "immersed"])
def test_arbitrary_walls_advance_and_restore_latest(tmp_path, kind, transfer_method):
    with make_coupler(tmp_path, kind, transfer_method) as first:
        assert first.run(start_from="latest", max_coupling_steps=1) == 1
        assert first.vorticity_transfer.solid_boundary is not None
        first.solve(start_step=1)
        velocity = first.fvm_solver.get_velocity_field().copy()
        pressure = first.fvm_solver.get_pressure_field().copy()
        positions, strengths = particle_state(first)
    manifest = tmp_path / "solution/backups/manifest.json"
    assert json.loads(manifest.read_text())["coupling_step"] == 1
    with make_coupler(tmp_path, kind, transfer_method) as resumed:
        assert resumed.run(start_from="latest") == 2
        np.testing.assert_allclose(resumed.fvm_solver.get_velocity_field(), velocity, atol=1e-7)
        # VPM fields and GBD atomic accumulation use float32; accept its
        # roundoff in the pressure trace while catching lost boundary history.
        np.testing.assert_allclose(resumed.fvm_solver.get_pressure_field(), pressure, atol=2e-7)
        restored_positions, restored_strengths = particle_state(resumed)
        np.testing.assert_allclose(restored_positions, positions, atol=2e-7)
        np.testing.assert_allclose(restored_strengths, strengths, atol=2e-7)
    diagnostics = tmp_path / "solution/coupler_diagnostics.jsonl"
    assert [json.loads(row)["step"] for row in diagnostics.read_text().splitlines()] == [1, 2]
