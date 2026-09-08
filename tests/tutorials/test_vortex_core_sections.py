"""The tutorial plots actual SurfaceSampler fields and their recorded times."""

from types import SimpleNamespace

import numpy as np
import pytest

import openonda.vpm as vpm
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager
from tutorials.vpm.vortex_interactions import setup
from tutorials.vpm.vortex_interactions.assets.plot_core_sections import discover, read_plane


def test_plane_sampler_round_trip_preserves_curl_orientation_and_physical_time(tmp_path):
    # u_y=x^2 gives curl(u)_z=2x. A varying field catches VTK transpose errors.
    def evaluate(points, *, particle_spacing):
        velocity = np.zeros_like(points)
        velocity[:, 1] = points[:, 0] ** 2
        gradient = np.zeros((len(points), 3, 3))
        gradient[:, 1, 0] = 2 * points[:, 0]
        return velocity, gradient

    sampler = setup.CoreSectionSampler(
        point=[0, 0, 0],
        normal=[0, 0, 1],
        bounds=[-1, 1, 0, 1],
        spacing=0.25,
        file_name="core_section",
        include_derivatives=False,
        schedule=vpm.EveryTime(1.5),
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
    OutputManager(solver).dispatch(OutputEvent.ACCEPTED_STEP)
    records = discover(tmp_path / "samples", tmp_path / "absent")
    assert len(records) == 1
    assert records[0]["time"] == 1.5
    x, r, omega = read_plane(records[0]["path"])
    np.testing.assert_allclose(omega, np.broadcast_to(2 * x[:, None] / 100, (len(x), len(r))))
    assert np.min(omega) < 0 < np.max(omega)


def test_setup_and_study_share_initial_periodic_and_final_plane_sampling(tmp_path):
    from tutorials.vpm.vortex_interactions.study import build_experiment, parser

    cases = [setup.build_case("baseline"), build_experiment(parser().parse_args([]), tmp_path)]
    for case, interval in zip(cases, (1.5, 0.15), strict=True):
        planes = [
            s for s in case.samplers.samples if getattr(s, "file_name", None) == "core_section"
        ]
        assert len(planes) == 2
        assert planes[0].initial
        assert planes[0].schedule.interval == pytest.approx(interval)
        assert planes[1].schedule.is_final_only
        np.testing.assert_allclose(planes[0].normal, [0, 0, 1])
        assert planes[0].spacing == pytest.approx(0.02)


def test_study_uses_only_vpm_samplers_without_particle_archives(tmp_path):
    from tutorials.vpm.vortex_interactions.study import build_experiment, parser

    case = build_experiment(parser().parse_args([]), tmp_path)
    assert all(
        isinstance(s, vpm.FlowIntegralsSampler | vpm.RingDiagnosticsSampler | vpm.SurfaceSampler)
        for s in case.samplers.samples
    )
