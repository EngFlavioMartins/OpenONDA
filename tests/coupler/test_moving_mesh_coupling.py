"""FVM-VPM coupling on a translating FVM mesh: frame invariance and restart."""

from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from openonda import coupler, fvm, vpm
from openonda.fvm.diagnostics.fields import compute_surface_forces
from openonda.fvm.mesh.rectilinear import coupling_box_mesh

SPACING = 0.125  # m, FVM cell and particle spacing
STREAM = np.array([0.4, 0.1, 0.0])  # m/s
VISCOSITY = 0.01  # m^2/s
FVM_STEP = 0.01  # s
EXCHANGE_STEP = 0.02  # s


def _mesh():
    return coupling_box_mesh(
        (-1.0, 1.0) * 3, SPACING, hole_box=(-0.25, 0.25) * 3, wall_patch_name="body"
    )


def _fvm_case(directory, stream, motion, *, time_step_size=FVM_STEP, end_time=4 * EXCHANGE_STEP):
    return fvm.FVMCase(
        name="moving_body",
        directory=directory,
        numerics=fvm.Numerics(
            time_step_size=time_step_size,
            transport=fvm.TransportConfig(kinematic_viscosity=VISCOSITY),
            schemes=fvm.DiscretizationConfig(time_scheme="backward"),
            linear=fvm.LinearSolverConfig(momentum_tolerance=1e-12, pressure_tolerance=1e-12),
        ),
        initial_conditions=fvm.InitialFields(velocity=tuple(stream)),
        run=fvm.RunPlan(end_time=end_time),
        mesh=_mesh,
        boundaries=(
            fvm.BoundaryConfig.wall("body"),
            fvm.BoundaryConfig(
                name="numericalBoundary",
                velocity_value=tuple(stream),
                pressure_type="fixed_flux_pressure",
            ),
        ),
        mesh_motion=motion,
    )


def _coupled_solver(directory, stream, motion, *, end_time=4 * EXCHANGE_STEP, backup=None):
    particles = vpm.VPMCase(
        name="moving_body",
        directory=directory,
        numerics=vpm.Numerics(
            time_step_size=EXCHANGE_STEP,
            compute_device="cpu",
            max_n_particles=50_000,
            domain_bounds=(-3.0, 3.0) * 3,
            freestream_velocity=tuple(stream),
            induction=vpm.DirectInduction(),
            viscous=vpm.ViscousConfig.gbd(
                kinematic_viscosity=VISCOSITY,
                particle_spacing=SPACING,
                padding=4,
                threshold=1e-5,
                threshold_mode="absolute",
            ),
        ),
        run=vpm.RunPlan(end_time=end_time),
    )
    config = coupler.CouplerConfig(
        freestream_velocity=tuple(stream),
        eta_blend_width=0.0,
        backup=backup or coupler.Backup(),
    )
    return coupler.CoupledSolver(
        _fvm_case(directory, stream, motion, end_time=end_time), particles, config
    )


def test_body_translating_through_still_fluid_couples_like_a_fixed_body(tmp_path):
    """Claim: the coupled flow round a body moving at -U through still fluid
    equals the flow past the fixed body in a stream U.

    The VPM runs in the frame of the translating mesh, where both cases are
    the same particle problem. Both start from one divergence-free FVM state,
    since an impulsive start violates discrete continuity and then depends on
    the frame through the conservative convection term.
    """
    with fvm.FVMSolver(
        _fvm_case(tmp_path / "start", STREAM, None, time_step_size=1e-4, end_time=1.0)
    ) as start:
        start.advance()
        start.advance()

    results = {}
    for moving in (False, True):
        frame_velocity = -STREAM if moving else np.zeros(3)
        with _coupled_solver(
            tmp_path / ("moving" if moving else "fixed"),
            STREAM + frame_velocity,
            fvm.MeshMotion(fvm.TranslatingMotion(tuple(frame_velocity))) if moving else None,
        ) as solver:
            flow, particles = solver.fvm_solver, solver.vpm_solver
            mesh_flux = flow.mesh_motion.mesh_face_flux if moving else 0.0
            for name in ("velocity", "velocity_old", "velocity_older"):
                getattr(flow, name)[:] = getattr(start, name) + frame_velocity
            for name in (
                "volumetric_face_flux",
                "volumetric_face_flux_old",
                "volumetric_face_flux_older",
            ):
                getattr(flow, name)[:] = getattr(start, name) + mesh_flux
            flow.kinematic_pressure[:] = start.kinematic_pressure
            if moving:
                flow.mesh_motion.update_boundaries(flow)
            solver.start_from("initial")
            while solver.step < solver.n_steps:
                solver.advance()
            count = particles.particles.n_particles_total
            position = np.asarray(particles.particle_position)[:count]
            strength = np.asarray(particles.particle_vortex_strength)[:count]
            order = np.lexsort(np.round(position, 6).T)
            force = compute_surface_forces(
                flow.velocity,
                flow.kinematic_pressure,
                VISCOSITY,
                1.0,
                flow.mesh_data,
                flow.geo_data,
                flow.boundaries,
                patch_names=["body"],
            )["body"]["total_force"]
            results[moving] = (
                flow.velocity - frame_velocity,
                position[order],
                strength[order],
                force,
                flow.geo_data["cell_centre"].mean(axis=0),
            )

    fixed, moving = results[False], results[True]
    np.testing.assert_allclose(moving[4] - fixed[4], -STREAM * 4 * EXCHANGE_STEP, atol=1e-12)
    np.testing.assert_allclose(moving[0], fixed[0], rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(moving[1], fixed[1])
    np.testing.assert_allclose(moving[2], fixed[2], rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(moving[3], fixed[3], rtol=0.0, atol=1e-6 * np.abs(fixed[3]).max())
    assert len(fixed[1]) > 1000


def _coupled_state(solver):
    """Copy the FVM and particle state before the solver closes its device fields."""
    flow, particles = solver.fvm_solver, solver.vpm_solver
    count = particles.particles.n_particles_total
    return {
        "step": flow.step,
        "cell_centre": flow.geo_data["cell_centre"].copy(),
        "velocity": flow.velocity.copy(),
        "freestream_velocity": np.array(particles.freestream_velocity),
        "position": np.asarray(particles.particle_position)[:count].copy(),
        "vortex_strength": np.asarray(particles.particle_vortex_strength)[:count].copy(),
    }


def test_heaving_mesh_coupled_restart_continues_the_same_trajectory(tmp_path):
    """A coupled restart restores the mesh, the frame and the particle cloud."""
    heave = fvm.MeshMotion(fvm.HeavingMotion(0.1, 2.0, direction=(0.0, 1.0, 0.0)))
    end_time = 3 * EXCHANGE_STEP
    backup = coupler.Backup(schedule=coupler.EverySteps(1))
    with _coupled_solver(
        tmp_path / "continuous", STREAM, heave, end_time=end_time, backup=backup
    ) as continuous:
        continuous.start_from("initial")
        while continuous.step < continuous.n_steps:
            continuous.advance()
        expected = _coupled_state(continuous)

    with _coupled_solver(
        tmp_path / "restarted", STREAM, heave, end_time=end_time, backup=backup
    ) as first:
        first.start_from("initial")
        first.advance()
        first.advance()
    with _coupled_solver(
        tmp_path / "restarted", STREAM, heave, end_time=end_time, backup=backup
    ) as restarted:
        assert restarted.start_from("latest")
        assert restarted.step == 2
        while restarted.step < restarted.n_steps:
            restarted.advance()
        actual = _coupled_state(restarted)

    assert actual["step"] == expected["step"]
    np.testing.assert_array_equal(actual["cell_centre"], expected["cell_centre"])
    np.testing.assert_allclose(actual["velocity"], expected["velocity"], atol=1e-6)
    np.testing.assert_allclose(actual["freestream_velocity"], expected["freestream_velocity"])
    np.testing.assert_array_equal(actual["position"], expected["position"])
    np.testing.assert_allclose(actual["vortex_strength"], expected["vortex_strength"], atol=1e-7)


def test_rotating_mesh_is_rejected_by_the_coupling(tmp_path):
    """The VPM has no rotating frame, so a rotating FVM mesh cannot be coupled."""
    pitch = fvm.MeshMotion(fvm.PitchingMotion(5.0, 1.0, rotation_axis=(0, 0, 1)))
    with pytest.raises(NotImplementedError, match="translating meshes"):
        _coupled_solver(tmp_path, STREAM, pitch)


_PARTITIONED_CASE = """
import sys
from pathlib import Path
from openonda import coupler, fvm, vpm
from openonda.fvm.mesh.rectilinear import coupling_box_mesh

cores = int(sys.argv[1])
stream = (0.4, 0.1, 0.0)
flow = fvm.FVMCase(
    name="moving", directory=Path.cwd(), cores=cores,
    numerics=fvm.Numerics(
        time_step_size=0.01,
        transport=fvm.TransportConfig(kinematic_viscosity=0.01),
        linear=fvm.LinearSolverConfig(momentum_tolerance=1e-12, pressure_tolerance=1e-12),
    ),
    initial_conditions=fvm.InitialFields(velocity=stream),
    run=fvm.RunPlan(end_time=0.06),
    samplers=fvm.Samplers((fvm.ForceSampler(["body"], schedule=fvm.EverySteps(2)),)),
    mesh=lambda: coupling_box_mesh((-1.0, 1.0) * 3, 0.125, hole_box=(-0.25, 0.25) * 3,
        wall_patch_name="body"),
    boundaries=(
        fvm.BoundaryConfig.wall("body"),
        fvm.BoundaryConfig(name="numericalBoundary", velocity_value=stream,
            pressure_type="fixed_flux_pressure"),
    ),
    mesh_motion=fvm.MeshMotion(fvm.HeavingMotion(0.1, 2.0, direction=(0.0, 1.0, 0.0))),
)
particles = vpm.VPMCase(
    name="moving", directory=Path.cwd(),
    numerics=vpm.Numerics(
        time_step_size=0.02, compute_device="cpu", max_n_particles=50_000,
        domain_bounds=(-3.0, 3.0) * 3, freestream_velocity=stream,
        induction=vpm.DirectInduction(),
        viscous=vpm.ViscousConfig.gbd(kinematic_viscosity=0.01, particle_spacing=0.125,
            padding=4, threshold=1e-5, threshold_mode="absolute"),
    ),
    run=vpm.RunPlan(end_time=0.06),
)
config = coupler.CouplerConfig(
    freestream_velocity=stream, eta_blend_width=0.0, backup=coupler.Backup(write_at_end=False)
)
with coupler.CoupledSolver(flow, particles, config) as solver:
    solver.run(start_from="initial")
"""


def test_partitioned_heaving_mesh_couples_like_the_serial_run(tmp_path):
    """Each rank moves its mesh partition; the body forces equal the serial run."""
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc are required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec is required")
    script = tmp_path / "case.py"
    script.write_text(_PARTITIONED_CASE)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path / "taichi-cache")
    forces = []
    for cores in (1, 2):
        directory = tmp_path / f"cores_{cores}"
        directory.mkdir()
        result = subprocess.run(
            [sys.executable, str(script), str(cores)],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        forces.append(
            np.genfromtxt(
                directory / "samples/forces_history.csv",
                delimiter=",",
                names=True,
                usecols=("total_force_x", "total_force_y"),
            )
        )
    for name in ("total_force_x", "total_force_y"):
        np.testing.assert_allclose(forces[1][name], forces[0][name], rtol=1e-6, atol=1e-9)
