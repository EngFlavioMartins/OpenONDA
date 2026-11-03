"""An error while recording a diagnostic or logging loads reaches the caller."""

from types import SimpleNamespace

import numpy as np
import pytest

from openonda.vpm.io.logging import Logging
from openonda.vpm.physics.evaluation import ParticleFieldEvaluation

VLM_LOADS = {
    "force_x": 1.0,
    "force_y": 0.0,
    "force_z": 2.0,
    "lift_coefficient": 0.5,
    "drag_coefficient": 0.02,
    "dynamic_pressure": 0.6125,
    "reference_area": 10.0,
}


def test_vortex_centroid_weights_positions_by_vortex_strength_magnitude():
    history = {"vortex_centroid": []}
    position = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vortex_strength = np.array([[0.0, 0.0, 1.0], [0.0, 3.0, 0.0]])
    ParticleFieldEvaluation.record_vortex_centroid_history(history, position, vortex_strength)
    assert history["vortex_centroid"] == [(0.75, 0.0, 0.0)]


def test_vortex_centroid_of_mismatched_particle_arrays_raises():
    history = {"vortex_centroid": []}
    with pytest.raises(ValueError):
        ParticleFieldEvaluation.record_vortex_centroid_history(
            history, np.zeros((3, 3)), np.ones((2, 3))
        )


def _system_with_vlm_loads(loads):
    return SimpleNamespace(
        vlm_solver=SimpleNamespace(_solved=True, _last_forces=loads),
        particles=SimpleNamespace(n_particles_total=0),
    )


def test_vlm_loads_are_logged():
    Logging.vlm_forces(_system_with_vlm_loads(VLM_LOADS))


def test_logging_vlm_loads_without_a_coefficient_raises():
    loads = {name: value for name, value in VLM_LOADS.items() if name != "lift_coefficient"}
    with pytest.raises(KeyError, match="lift_coefficient"):
        Logging.vlm_forces(_system_with_vlm_loads(loads))
