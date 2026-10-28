"""Native non-box meshes advance with GBD and reproduce a restarted step."""

import json

import numpy as np
import pytest

from openonda import coupler, fvm, vpm
from openonda.fvm.mesh.geometry import compute_mesh_geometry
from openonda.fvm.mesh.rectilinear import coupling_box_mesh
from openonda.fvm.mesh.validation import extract_cell_subset_mesh


def body_mesh(kind):
    bounds = (-1.0, 1.0) * 3
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


def make_coupler(directory, kind, *, les=False, error_limit=1.0):
    velocity = (0.2, 0.03, 0.0)
    flow = fvm.FVMCase(
        name=kind,
        directory=directory,
        numerics=fvm.Numerics(
            time_step_size=0.005,
            transport=fvm.TransportConfig(kinematic_viscosity=0.01),
            linear=fvm.LinearSolverConfig(momentum_tolerance=1e-10, pressure_tolerance=1e-10),
            turbulence=fvm.TurbulenceConfig.smagorinsky() if les else fvm.TurbulenceConfig.dns(),
        ),
        initial_conditions=fvm.InitialFields(velocity=velocity),
        run=fvm.RunPlan(end_time=0.02),
        mesh=body_mesh(kind),
        boundaries=(
            fvm.BoundaryConfig.wall("body"),
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_value=velocity,
                pressure_type="fixed_flux_pressure",
            ),
        ),
    )
    particles = vpm.VPMCase(
        name=kind,
        directory=directory,
        numerics=vpm.Numerics(
            time_step_size=0.01,
            compute_device="cpu",
            max_n_particles=20000,
            domain_bounds=(-2.0, 2.0) * 3,
            freestream_velocity=velocity,
            induction=vpm.DirectInduction(),
            turbulence=vpm.TurbulenceConfig.smagorinsky() if les else vpm.TurbulenceConfig.dns(),
            viscous=vpm.ViscousConfig.gbd(
                kinematic_viscosity=0.01,
                particle_spacing=0.25,
                padding=4,
                threshold=1e-5,
                threshold_mode="absolute",
            ),
        ),
        run=vpm.RunPlan(end_time=0.02),
    )
    config = coupler.CouplerConfig(
        # One coupling iteration, so the restart must reproduce the run exactly
        max_coupling_iterations=1,
        transfer_inset_spacings=0.0,
        blend_end_depth=0.0,
        freestream_velocity=velocity,
        max_moment_correction_fraction=error_limit,
    )
    return coupler.CoupledSolver(flow, particles, config)


def particle_state(solver):
    positions = solver.vpm_solver.particle_position
    strengths = solver.vpm_solver.particle_vortex_strength
    assert len(positions) > 0
    assert not solver.vorticity_transfer.solid_boundary.contains(positions).any()
    order = np.lexsort(positions.T[::-1])
    return positions[order], strengths[order]


@pytest.mark.parametrize("kind", ["rotated", "concave", "multiple", "thin"])
def test_arbitrary_walls_advance_and_restore_latest(tmp_path, kind):
    with make_coupler(tmp_path, kind) as first:
        first.start_from("latest")
        first.advance()
        first.save_backup()
        assert first.vorticity_transfer.solid_boundary is not None
        first.advance()
        velocity = first.fvm_solver.gather_velocity_field().copy()
        pressure = first.fvm_solver.gather_pressure_field().copy()
        positions, strengths = particle_state(first)
    with make_coupler(tmp_path, kind) as resumed:
        assert resumed.start_from("latest")
        assert resumed.step == 1
        resumed.advance()
        np.testing.assert_allclose(resumed.fvm_solver.gather_velocity_field(), velocity, atol=1e-7)
        # VPM fields and GBD atomic accumulation use float32; accept its
        # roundoff in the pressure while catching lost boundary data.
        np.testing.assert_allclose(resumed.fvm_solver.gather_pressure_field(), pressure, atol=1e-6)
        restored_positions, restored_strengths = particle_state(resumed)
        np.testing.assert_allclose(restored_positions, positions, atol=2e-7)
        np.testing.assert_allclose(restored_strengths, strengths, atol=2e-7)
    diagnostics = tmp_path / "solution/coupler_diagnostics.jsonl"
    assert [json.loads(row)["step"] for row in diagnostics.read_text().splitlines()] == [1, 2]


def test_finite_wall_body_with_les_uses_three_dimensional_coupling(tmp_path):
    with make_coupler(tmp_path, "concave", les=True, error_limit=0.08) as driver:
        driver.advance()
        assert driver.fvm_case.numerics.turbulence.model == "smagorinsky"
        assert driver.vpm_case.numerics.turbulence.model == "smagorinsky"
        position, strength = particle_state(driver)
        assert np.ptp(position[:, 2]) > 0.5
        assert np.all(np.isfinite(strength))
        diagnostics = driver._last_transfer_result
        assert (
            diagnostics.transfer_conservation_error
            <= diagnostics.transfer_vortex_strength_tolerance
        )
        assert (
            diagnostics.transfer_linear_impulse_error
            <= diagnostics.transfer_linear_impulse_tolerance
        )
