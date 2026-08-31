"""Mathematical initialization checks for the vortex-interactions tutorial."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np

SETUP_PATH = (
    Path(__file__).resolve().parents[2] / "tutorials" / "vpm" / "vortex_interactions" / "setup.py"
)


def _load_setup():
    spec = importlib.util.spec_from_file_location("vortex_interactions_tutorial", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_leapfrogging_ring_pair_is_a_translated_symmetric_initial_condition():
    setup = _load_setup()
    left = setup.create_ring(-0.5 * setup.RING_SEPARATION, group_id=0).build()
    right = setup.create_ring(0.5 * setup.RING_SEPARATION, group_id=1).build()
    translation = np.array([setup.RING_SEPARATION, 0.0, 0.0])

    np.testing.assert_allclose(right.position, left.position + translation, atol=2.0e-15)
    np.testing.assert_allclose(right.vortex_strength, left.vortex_strength, atol=2.0e-15)
    np.testing.assert_array_equal(left.particle_volume, right.particle_volume)
    np.testing.assert_array_equal(left.core_radius, right.core_radius)
    np.testing.assert_array_equal(left.group_id, np.zeros(len(left), dtype=np.int32))
    np.testing.assert_array_equal(right.group_id, np.ones(len(right), dtype=np.int32))


def test_case_restores_global_weak_particle_pruning():
    setup = _load_setup()
    case = setup.build_case("leapfrog_les")

    assert case.initial_weak_particle_percent == setup.WEAK_PARTICLE_PERCENT == 5.0


def test_case_keeps_transposed_rk3_without_an_uncalibrated_macro_step_cfl_gate():
    setup = _load_setup()
    case = setup.build_case("leapfrog_les")

    assert case.numerics.time_integration == "COUPLED"
    assert case.numerics.advection.scheme == "RK3"
    assert case.numerics.stretching.scheme == "RK3"
    assert case.numerics.stretching.mode == "TRANSPOSED"
    assert case.numerics.health_limits.lagrangian_cfl.maximum is None


def test_cases_compare_only_supported_particle_stabilization_mechanisms():
    setup = _load_setup()
    expected = {
        "leapfrog_les": (False, False),
        "leapfrog_les_splitting": (True, False),
        "leapfrog_les_remeshing": (False, True),
        "leapfrog_les_splitting_remeshing": (True, True),
    }

    assert expected == setup.CASES
    for name, (has_splitting, has_remeshing) in expected.items():
        case = setup.build_case(name)
        stabilization = case.numerics.stabilization
        assert stabilization.filament_refinement.enabled is has_splitting
        assert (stabilization.regularization_interval_steps > 0) is has_remeshing
        assert case.numerics.turbulence.model == "LES_SMAGORINSKY"
        assert case.numerics.turbulence.smagorinsky_coefficient == setup.SMAGORINSKY_COEFFICIENT


def test_widnall_mode_changes_vorticity_without_moving_quadrature_nodes():
    setup = _load_setup()
    disturbed = setup.create_ring(-0.5).build()
    undisturbed_model = replace(setup.create_ring(-0.5), disturbance=None)
    undisturbed = undisturbed_model.build()

    np.testing.assert_array_equal(undisturbed.position, disturbed.position)
    np.testing.assert_array_equal(undisturbed.particle_volume, disturbed.particle_volume)
    np.testing.assert_array_equal(undisturbed.core_radius, disturbed.core_radius)
    assert not np.allclose(undisturbed.vortex_strength, disturbed.vortex_strength)
