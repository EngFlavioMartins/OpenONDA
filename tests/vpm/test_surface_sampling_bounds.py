"""Declared sampling domains remain exact when spacing does not divide them."""

import numpy as np
import pytest

from source.solvers.vpm.io.manifest import _sampler_identity
from source.solvers.vpm.io.sampling.field_samplers import SurfaceSampler


@pytest.mark.parametrize("normal_axis", [0, 1, 2])
@pytest.mark.parametrize(
    "bounds,spacing",
    [([-7.8, 7.8, -7.8, 7.8], 1 / 6), ([-1.7, 2.3, -2.6, 0.7], 0.31), ([-1, 1, -1, 1], 0.5)],
)
def test_exact_domain_and_uniform_spacing(normal_axis, bounds, spacing):
    normal = np.eye(3)[normal_axis]
    sampler = SurfaceSampler([0, 0, 0], normal, bounds, spacing)
    axes = [i for i in range(3) if i != normal_axis]
    np.testing.assert_array_equal(sampler.grid_points[:, normal_axis], 0)
    for i, axis in enumerate(axes):
        coordinates = np.unique(sampler.grid_points[:, axis])
        lower, upper = sampler.bounds[2 * i : 2 * i + 2]
        assert coordinates[0] == lower
        assert coordinates[-1] == upper
        assert np.all(np.diff(coordinates) > 0)
        assert np.max(np.diff(coordinates)) <= spacing + 2e-6
        np.testing.assert_allclose(np.diff(coordinates), np.diff(coordinates).mean(), atol=2e-6)
        np.testing.assert_allclose(coordinates + coordinates[::-1], lower + upper, atol=1e-6)
    assert _sampler_identity(sampler)["grid_layout"] == "bounded_uniform_v1"


@pytest.mark.parametrize(
    "bounds,spacing",
    [([1, -1, 0, 1], 0.1), ([0, 0, 0, 1], 0.1), ([0, 1, 0, np.nan], 0.1), ([0, 1, 0, 1], np.nan)],
)
def test_invalid_domain_cannot_produce_a_silent_empty_or_nonfinite_grid(bounds, spacing):
    with pytest.raises(ValueError):
        SurfaceSampler([0, 0, 0], [0, 0, 1], bounds, spacing)


def test_resume_rejects_changed_grid_before_evaluating_or_appending(tmp_path):
    from types import SimpleNamespace

    from source.solvers.vpm.config.artifacts import Samplers
    from source.solvers.vpm.io.sampler import OutputEvent, OutputManager, SamplingContext
    from tests.vpm.test_surface_sampling_resume import write_frame

    sampler = SurfaceSampler([0, 0, 0], [1, 0, 0], [-1, 1, -1, 1], 0.31, file_name="plane")
    previous = SurfaceSampler([0, 0, 0], [1, 0, 0], [-1, 1, -1, 1], 0.29)
    write_frame(previous, tmp_path / "plane_000001.vts")
    OutputManager._write_pvd(tmp_path, "plane", [(0.1, "plane_000001.vts")])
    before = (tmp_path / "plane.pvd").read_bytes()
    solver = SimpleNamespace(case=SimpleNamespace(samplers=Samplers(samples=(sampler,))))
    manager = OutputManager(solver)
    context = SamplingContext(
        solver=solver,
        output_directory=tmp_path,
        step=2,
        time=0.2,
        event=OutputEvent.ACCEPTED_STEP,
        continuing_output=True,
    )
    with pytest.raises(ValueError, match="sampling grid differs"):
        manager._write(sampler, context)
    assert (tmp_path / "plane.pvd").read_bytes() == before
    assert not (tmp_path / "plane_000002.vts").exists()


def test_unchanged_grid_can_continue(tmp_path):
    from tests.vpm.test_surface_sampling_resume import write_frame

    sampler = SurfaceSampler([0, 0, 0], [1, 0, 0], [-1, 1, -1, 1], 0.31)
    write_frame(sampler, tmp_path / "prior.vts")
    sampler.validate_existing_vtk(tmp_path / "prior.vts")
