"""The tutorial plots actual SurfaceSampler fields and their recorded times."""

from types import SimpleNamespace

import numpy as np
import pytest

import openonda.vpm as vpm
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager
from tests._tutorial_helpers import load_tutorial_module

setup = load_tutorial_module("vpm/vortex_interactions")
postprocess = load_tutorial_module("vpm/vortex_interactions", "assets.postprocess")
plot_core_sections = load_tutorial_module("vpm/vortex_interactions", "assets.plot_core_sections")
discover = postprocess.discover_core_sections
arrange_records = plot_core_sections.arrange_records
read_plane = postprocess.read_core_section


def test_plane_sampler_round_trip_preserves_curl_orientation_and_physical_time(tmp_path):
    # u_y=x^2 gives curl(u)_z=2x. A varying field catches VTK transpose errors.
    def evaluate(points, *, particle_spacing):
        velocity = np.zeros_like(points)
        velocity[:, 1] = points[:, 0] ** 2
        gradient = np.zeros((len(points), 3, 3))
        gradient[:, 1, 0] = 2 * points[:, 0]
        return velocity, gradient

    sampler = vpm.SurfaceSampler(
        point=[0, 0, 0],
        normal=[0, 0, 1],
        bounds=[-1, 1, 0, 1],
        spacing=0.25,
        file_name="core_section",
        include_derivatives=False,
        schedule=vpm.EveryTime(1.5),
        initial=True,
    )
    solver = SimpleNamespace(
        case=SimpleNamespace(
            samplers=vpm.Samplers(samples=(sampler,), directory="baseline"),
            backup=vpm.Backup(interval_steps=0),
        ),
        case_dir=tmp_path,
        step=200,
        time=1.5,
        time_step_size=0.0075,
        particles=SimpleNamespace(n_particles_total=1),
        compute_velocity_and_gradient_at_points=evaluate,
    )
    manager = OutputManager(solver)
    solver.step, solver.time = 0, 0.0
    manager.dispatch(OutputEvent.INITIAL)
    solver.step, solver.time = 200, 1.5
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    records = discover(tmp_path / "samples")
    assert [record["time"] for record in records] == [0.0, 1.5]
    x, r, omega = read_plane(records[-1]["path"])
    np.testing.assert_allclose(omega, np.broadcast_to(2 * x[:, None] / 100, (len(x), len(r))))
    assert np.min(omega) < 0 < np.max(omega)


def test_setup_records_initial_periodic_and_final_planes():
    case = setup.build_case("baseline")
    planes = [s for s in case.samplers.samples if getattr(s, "file_name", None) == "core_section"]
    assert len(planes) == 2
    assert planes[0].initial
    assert planes[0].schedule.interval == pytest.approx(0.15)
    assert planes[1].schedule.is_final_only
    np.testing.assert_allclose(planes[0].normal, [0, 0, 1])
    assert planes[0].spacing == pytest.approx(0.02)


def test_discovery_uses_only_the_requested_current_run(tmp_path):
    for name in ("baseline", "selective_eddy_viscosity"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "core_section_000000.vts").touch()
        (directory / "core_section.pvd").write_text(
            '<VTKFile><Collection><DataSet timestep="0" file="core_section_000000.vts"/>'
            "</Collection></VTKFile>"
        )
    records = discover(tmp_path, ["selective_eddy_viscosity"])
    assert len(records) == 1
    assert records[0]["run"] == "selective_eddy_viscosity"
    assert records[0]["path"].parent == tmp_path / "selective_eddy_viscosity"
    (tmp_path / "selective_eddy_viscosity" / "core_section_000000.vts").unlink()
    assert discover(tmp_path, ["selective_eddy_viscosity"]) == []


def test_grid_preserves_missing_run_time_combinations():
    records = [
        {"run": "baseline", "time": 1.5},
        {"run": "selective_eddy_viscosity", "time": 1.5},
        {"run": "selective_eddy_viscosity", "time": 3.0},
    ]
    grid = arrange_records(
        records,
        ["baseline", "selective_eddy_viscosity"],
        [1.5, 3.0],
    )
    assert [[record is not None for record in row] for row in grid] == [
        [True, True],
        [False, True],
    ]
