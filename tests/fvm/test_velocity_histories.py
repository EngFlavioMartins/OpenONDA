"""Native physical velocity histories survive current-format continuation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from openonda import coupler, fvm
from source.coupler.backup import config_digest
from tests.fvm.test_restart_and_diagnostics import _setup
from tests.support.fvm_mesh import structured_box


def setup(end):
    ramp = fvm.VelocityRamp((0.5, 0.1, 0.0), (0.5, 0.0, 0.0), 0.01, 0.03)
    original = _setup()
    return replace(
        original,
        time=replace(original.time, end_time=end),
        boundaries=[
            fvm.BoundaryConfig.inlet("xmin", list(ramp.initial)),
            fvm.BoundaryConfig.outlet("xmax"),
            *(fvm.BoundaryConfig.slip(name) for name in ("ymin", "ymax", "zmin", "zmax")),
        ],
        velocity_boundaries=(
            fvm.VelocityBoundary(("xmin",), ramp),
            fvm.VelocityBoundary(("ymin", "ymax"), ramp, normal_only=True),
        ),
        initial_velocity=list(ramp.initial),
    )


def solver(directory, end):
    result = fvm.FVMSolver(setup(end), str(directory), mesh_data=structured_box(2, 2, 2))
    result.auto_write = False
    return result


def test_velocity_history_uses_implicit_endpoint_and_restores_native_slip(tmp_path):
    with solver(tmp_path, 0.04) as flow:
        for expected in (0.1, 0.05, 0.0, 0.0):
            flow.advance()
            inlet = next(item for item in flow.boundaries if item["name"] == "xmin")
            np.testing.assert_allclose(inlet["velocity_value_field"][:, 1], expected, atol=1e-15)
            for boundary in flow.boundaries:
                if boundary["name"] in ("ymin", "ymax"):
                    assert boundary["velocity_type"] == (
                        "slip" if expected == 0 else "normalValueTangentialGradient"
                    )
        assert flow.step == 4 and flow.time == pytest.approx(0.04)


def test_native_continuation_restores_history_without_reseeding_or_sidecars(tmp_path):
    seeded = []

    def initial(points):
        seeded.append(points.shape)
        return np.tile([0.5, 0.1, 0.0], (len(points), 1))

    with solver(tmp_path, 0.02) as flow:
        flow.run(start_from="initial", initial_velocity=initial)
    assert seeded == [(8, 3)]
    with solver(tmp_path, 0.04) as flow:
        flow.run(start_from="latest", initial_velocity=initial)
        assert flow.step == 4
        assert flow.run_status == "complete"
        assert all(
            item["velocity_type"] == "slip"
            for item in flow.boundaries
            if item["name"] in ("ymin", "ymax")
        )
    assert seeded == [(8, 3)]
    assert not list((tmp_path / "solution").glob("*startup*"))


def test_native_restart_rejects_a_changed_velocity_history(tmp_path):
    with solver(tmp_path, 0.02) as flow:
        flow.run(start_from="initial")
    changed = setup(0.04)
    changed.velocity_boundaries = tuple(
        replace(item, velocity=replace(item.velocity, end_time=0.04))
        for item in changed.velocity_boundaries
    )
    with (
        fvm.FVMSolver(changed, str(tmp_path), mesh_data=structured_box(2, 2, 2)) as flow,
        pytest.raises(RuntimeError, match="configuration hash"),
    ):
        flow.start_from("latest")


def test_velocity_history_serializes_without_callables_or_external_records(tmp_path):
    original = setup(0.04)
    path = tmp_path / "setup.json"
    original.save(str(path))
    restored = fvm.FVMSetup.load(str(path))
    assert restored.velocity_boundaries == original.velocity_boundaries


def test_coupled_runtime_background_preserves_authored_restart_configuration():
    ramp = coupler.VelocityRamp((1.0, 0.2, 0.0), (1.0, 0.0, 0.0), 0.0, 2.0)
    settings = coupler.CouplerSetup(freestream_velocity=list(ramp.initial), freestream=ramp)
    configuration = config_digest(settings)
    values = []
    flow = coupler.FVMVPMCoupler.__new__(coupler.FVMVPMCoupler)
    flow.setup = settings
    flow._is_master = True
    flow.vorticity_transfer = SimpleNamespace(config=settings)
    flow.vpm_solver = SimpleNamespace(_set_freestream_velocity=lambda value: values.append(value))
    for time in (0.0, 1.0, 2.0):
        flow._update_freestream(time)
        assert config_digest(flow.setup) == configuration
        np.testing.assert_array_equal(
            flow.vorticity_transfer.config.freestream_velocity, ramp.at(time)
        )
    np.testing.assert_allclose(values, [[1, 0.2, 0], [1, 0.1, 0], [1, 0, 0]])


def test_quintic_velocity_has_zero_endpoint_acceleration():
    ramp = fvm.VelocityRamp((0, 0.2, 0), (0, 0, 0), 1, 2)
    h = 1e-5
    for end, direction in ((1.0, 1), (2.0, -1)):
        difference = ramp.at(end + direction * h) - ramp.at(end)
        assert np.linalg.norm(difference) < 3e-15
