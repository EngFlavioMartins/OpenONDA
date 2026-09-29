"""Cylinder coefficients use the represented frontal area and freestream."""

from types import SimpleNamespace

import pytest

from openonda.fvm import IBMForceSampler
from tests._tutorial_helpers import load_tutorial_module


@pytest.mark.parametrize("depth", [0.0625, 0.125])
@pytest.mark.parametrize("speed", [1.0, 2.0])
def test_cylinder_drag_coefficient_is_independent_of_extrusion_depth(depth, speed, monkeypatch):
    case = load_tutorial_module("fvm/cylinder_ibm")
    monkeypatch.setattr(case, "FREESTREAM_VELOCITY", speed)
    setup = case.create_fvm_setup(30.0, 1.0, depth, 0.01, 0.01)
    (sampler,) = (sample for sample in setup.samplers if isinstance(sample, IBMForceSampler))
    expected_drag = 1.8
    force = 0.5 * case.DENSITY * speed**2 * case.DIAMETER * depth * expected_drag
    context = SimpleNamespace(setup=setup)
    drag, lift = sampler.summary(context, {"forces": {"cylinder": (force, 0.0, 0.0)}})["cylinder"]
    assert drag == pytest.approx(expected_drag)
    assert lift == 0.0
