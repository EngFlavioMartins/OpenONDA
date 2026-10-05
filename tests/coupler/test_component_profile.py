"""Qualification instrumentation must not persist on solver objects."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_profile():
    asset = (
        Path(__file__).resolve().parents[2] / "tests/support/cylinder/profile_solver_components.py"
    )
    spec = importlib.util.spec_from_file_location("component_profile_test_asset", asset)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.profile_components


def test_nested_methods_restore_after_error(monkeypatch):
    import taichi as ti

    monkeypatch.setattr(ti, "sync", lambda: None)

    class Operator:
        def evaluate(self):
            return 42

    def noop(*_args):
        return 12

    stage = Operator()
    stage.position_guard = None
    stepper = SimpleNamespace(
        **dict.fromkeys(
            (
                "_advance_particles",
                "_apply_viscous_diffusion",
                "_update_velocity_and_gradients",
                "_apply_grid_diffusion",
            ),
            noop,
        )
    )
    induction = SimpleNamespace(evaluate_stage=noop, _images=noop)
    output = SimpleNamespace(dispatch=noop)
    coupler = SimpleNamespace(
        _is_master=True,
        vpm_solver=SimpleNamespace(
            stepper=stepper, stage_rhs=stage, induction=induction, output_manager=output
        ),
        _transfer_vorticity_to_vpm=noop,
    )
    measurements = {}
    with pytest.raises(ValueError, match="injected"), load_profile()(coupler, measurements):
        assert stage.evaluate() == 42
        assert induction._images() == 12
        assert stepper._advance_particles(0.04) == 12
        raise ValueError("injected")
    assert "evaluate" not in vars(stage)
    assert induction._images is noop
    assert all(method is noop for method in vars(stepper).values())
    assert coupler._transfer_vorticity_to_vpm is noop
    assert len(measurements) == 3
    assert all(row["calls"] == 1 and row["seconds"] >= 0 for row in measurements.values())


def test_nonmaster_does_not_access_particle_or_device_state():
    measurements = {}
    with load_profile()(SimpleNamespace(_is_master=False), measurements):
        pass
    assert measurements == {}


def test_reuse_dispatch_is_timed_without_modifying_existing_backend(monkeypatch):
    import taichi as ti

    monkeypatch.setattr(ti, "sync", lambda: None)

    class Backend:
        def evaluate_stage(self):
            return 17

        def _images(self):
            return 23

    class Stage:
        position_guard = None

        def evaluate_induction(self):
            return induction.evaluate_stage()

        def evaluate(self):
            return self.evaluate_induction()

    induction, stage = Backend(), Stage()
    coupler = SimpleNamespace(
        _is_master=True,
        vpm_solver=SimpleNamespace(
            stepper=SimpleNamespace(),
            stage_rhs=stage,
            induction=induction,
            output_manager=SimpleNamespace(),
        ),
    )
    measurements = {}
    with load_profile()(coupler, measurements):
        assert stage.evaluate() == 17
        assert "evaluate_stage" not in vars(induction)
        assert "_images" not in vars(induction)
    assert "evaluate_induction" not in vars(stage)
    assert set(measurements) == {"vpm.complete_induction", "vpm.complete_stage_rhs"}
    assert all(row["calls"] == 1 for row in measurements.values())
