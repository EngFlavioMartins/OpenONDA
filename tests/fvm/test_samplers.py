from __future__ import annotations

import contextlib
import csv
import io

import numpy as np
import pytest

from source.solvers.FVM import (
    BoundaryConfig,
    ForceSampler,
    FVMSetup,
    LinearSolverConfig,
    LineSampler,
    PimpleControl,
    SchemesConfig,
    Solver,
    SurfaceSampler,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.sampling.base import SamplingSchedule

from ._structured_mesh import structured_box

LINE_HEADER = [
    "flow_time",
    "time_step",
    "x",
    "y",
    "z",
    "Ux",
    "Uy",
    "Uz",
    "omega_x",
    "omega_y",
    "omega_z",
    "p",
]

FORCES_HEADER = [
    "time",
    "step",
    "dt",
    "patch",
    "Fpx",
    "Fpy",
    "Fpz",
    "Fvx",
    "Fvy",
    "Fvz",
    "Ftx",
    "Fty",
    "Ftz",
    "Cd",
    "Cl",
    "Cz",
    "Cm",
]


def _config(samplers=()):
    return FVMSetup(
        case_name="samplers",
        time=TimeConfig.transient(dt=0.01, duration=0.1, write_interval=1),
        schemes=SchemesConfig(convection_scheme="upwind", time_scheme="euler_implicit"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, nu=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [0.5, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax"),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        samplers=samplers,
        initial_U=[0.2, 0.0, 0.0],
    )


def _solver(config, path):
    with contextlib.redirect_stdout(io.StringIO()):
        result = Solver(config, str(path), mesh_data=structured_box(3, 3, 3))
    result.auto_write = False
    return result


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.reader(stream))


def test_force_history_lands_in_samples_with_unchanged_schema(tmp_path):
    solver = _solver(_config(samplers=(ForceSampler(),)), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.evolve()

    csv_path = tmp_path / "samples" / "forces_history.csv"
    assert csv_path.exists()
    assert not (tmp_path / "solution" / "forces_history.csv").exists()

    rows = _rows(csv_path)
    assert rows[0] == FORCES_HEADER
    assert len(rows) > 1


def test_line_sampler_appends_a_time_aware_row_per_point(tmp_path):
    sampler = LineSampler(
        start=[0.1, 0.5, 0.5], end=[0.9, 0.5, 0.5], n_points=4, file_name="centerline"
    )
    solver = _solver(_config(samplers=(sampler,)), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(2):
            solver.evolve()

    csv_path = tmp_path / "samples" / "centerline.csv"
    rows = _rows(csv_path)
    assert rows[0] == LINE_HEADER
    # One header + n_points rows per sampling event, two events.
    assert len(rows) == 1 + 2 * 4
    # Successive events carry distinct, increasing flow times.
    assert float(rows[1][0]) < float(rows[5][0])


def test_line_sampler_interpolates_a_uniform_field_exactly(tmp_path):
    sampler = LineSampler(start=[0.1, 0.5, 0.5], end=[0.9, 0.5, 0.5], n_points=5)
    solver = _solver(_config(), tmp_path)
    solver.U[:] = 0.0
    solver.U[:, 0] = 3.0
    solver.p[:] = 7.0

    data = sampler.sample(solver)

    np.testing.assert_allclose(data["Ux"], 3.0)
    np.testing.assert_allclose(data["Uy"], 0.0)
    np.testing.assert_allclose(data["Uz"], 0.0)
    np.testing.assert_allclose(data["p"], 7.0)


def test_line_sampler_reports_vorticity_of_a_linear_shear(tmp_path):
    sampler = LineSampler(start=[0.3, 0.5, 0.5], end=[0.7, 0.5, 0.5], n_points=4)
    solver = _solver(_config(), tmp_path)
    mesh = solver.mesh_data
    n = mesh["n_elements"]
    n_interior = mesh["n_interior_faces"]

    # u_x = 2*y  ->  omega_z = dv/dx - du/dy = -2. Boundary-face values must be
    # set consistently too, or the gradient reconstruction skews boundary cells.
    y_cells = solver.geo_data["element_centroids"][:n, 1]
    y_faces = solver.geo_data["face_centroids"][n_interior:, 1]
    solver.U[:] = 0.0
    solver.U[:n, 0] = 2.0 * y_cells
    solver.U[n:, 0] = 2.0 * y_faces
    solver._invalidate_derived_fields()

    data = sampler.sample(solver)

    np.testing.assert_allclose(data["omega_z"], -2.0, atol=1e-6)


def test_surface_sampler_writes_a_vts_with_vpm_compatible_arrays(tmp_path):
    pv = pytest.importorskip("pyvista")
    sampler = SurfaceSampler(
        point=[0, 0, 0.5],
        normal=[0, 0, 1],
        bounds=[0.2, 0.8, 0.2, 0.6],
        spacing=0.2,
        file_name="slice_z0",
    )
    solver = _solver(_config(samplers=(sampler,)), tmp_path)
    n = solver.mesh_data["n_elements"]
    n_interior = solver.mesh_data["n_interior_faces"]
    cells = solver.geo_data["element_centroids"][:n]
    faces = solver.geo_data["face_centroids"][n_interior:]
    solver.U[:n, 0] = 2.0 * cells[:, 0] + 3.0 * cells[:, 1]
    solver.U[n:, 0] = 2.0 * faces[:, 0] + 3.0 * faces[:, 1]
    solver._invalidate_derived_fields()

    expected = sampler.sample(solver)
    samples = tmp_path / "samples"
    samples.mkdir()
    output = samples / "slice_z0_000001.vts"
    sampler.save_vts(solver, str(output))

    written = sorted(samples.glob("slice_z0_*.vts"))
    assert len(written) == 1

    grid = pv.read(written[0])
    assert grid.dimensions == (4, 3, 1)
    assert set(grid.point_data) == {
        "Velocity",
        "VelocityMagnitude",
        "Vorticity",
        "VorticityMagnitude",
        "Pressure",
    }
    assert grid.point_data["Velocity"].shape == (12, 3)
    assert "OpenONDASurfaceOrdering" in grid.field_data
    np.testing.assert_allclose(
        grid.point_data["Velocity"][:, 0],
        expected["Ux"].reshape(sampler.grid_shape).ravel(order="F"),
    )


def test_surface_sampler_cadence_is_owned_by_the_schedule(tmp_path):
    # Cadence lives in the schedule, not a mutable stride counter, so live and
    # offline runs select the same physical states.  The executor runs after
    # accepted steps (there is no t=0 event), so every_n_steps=2 fires at
    # steps 2 and 4.
    sampler = SurfaceSampler(
        point=[0, 0, 0.5],
        normal=[0, 0, 1],
        bounds=[0.2, 0.8, 0.2, 0.8],
        spacing=0.3,
        file_name="slice_z0",
        schedule=SamplingSchedule(every_n_steps=2),
    )
    solver = _solver(_config(samplers=(sampler,)), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(4):
            solver.evolve()

    written = sorted((tmp_path / "samples").glob("slice_z0_*.vts"))
    assert len(written) == 2


def test_explicit_force_sampler_writes_its_own_named_file(tmp_path):
    extra = ForceSampler(patch_names=["ymin"], ref_velocity=0.5, file_name="wall_loads")
    solver = _solver(_config(samplers=(extra,)), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.evolve()

    rows = _rows(tmp_path / "samples" / "wall_loads.csv")
    assert rows[0] == FORCES_HEADER
    assert {row[3] for row in rows[1:]} == {"ymin"}
    # No legacy auto-ForceSampler runs behind explicit samplers, so there is
    # exactly one force history file: the explicit sampler's own output.
    assert not (tmp_path / "samples" / "forces_history.csv").exists()


def test_sampler_failure_aborts_the_step(tmp_path):
    class Exploding:
        file_name = "boom"

        def write_csv(self, solver, samples_dir):
            raise RuntimeError("sampler blew up")

    solver = _solver(_config(samplers=(Exploding(),)), tmp_path)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        pytest.raises(RuntimeError, match="Sampler 'boom' failed"),
    ):
        solver.evolve()

    assert solver.time_step == 1
    assert not (tmp_path / "samples" / "boom.csv").exists()
