"""Checks for the isolated reference-driven cylinder boundary controls."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.fvm.io.backup import RestartState, config_hash
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from tests.support.cylinder.run_boundary_condition_study import (
    Reconstruction,
    balanced_trace,
    quiet_setup,
    reference_setup,
    restart_state_comparison,
    restrict_initial_state,
    trace_at_step,
    validate_saved_settings,
)
from tests.support.fvm_mesh import structured_box


def test_reference_output_controls_do_not_change_native_restart_configuration():
    setup = reference_setup(112.0)
    assert config_hash(quiet_setup(setup)) == config_hash(setup)


def test_affine_reconstruction_preserves_full_jacobian_and_planar_velocity():
    x, y = np.meshgrid(np.linspace(-1, 1, 13), np.linspace(-1, 1, 13))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    targets = np.array([[0.12, -0.24, 0], [0.71, 0.31, 0], [-0.45, 0.82, 0]])
    jacobian = np.array([[0.3, 0.4, 0], [-0.2, -0.3, 0], [0, 0, 0]])
    field = points @ jacobian.T + [1.0, 0.1, 0]
    reconstruction = Reconstruction(points, targets)
    np.testing.assert_allclose(
        reconstruction.apply(field), targets @ jacobian.T + [1, 0.1, 0], atol=1e-13
    )
    gradient = np.broadcast_to(jacobian.T, (len(points), 3, 3))
    np.testing.assert_allclose(
        reconstruction.apply(gradient).swapaxes(1, 2),
        np.broadcast_to(jacobian, (len(targets), 3, 3)),
        atol=1e-13,
    )


def test_reference_flux_balancing_retains_tangential_velocity_and_gradient():
    normals = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])
    areas = np.ones(4)
    jacobian = np.broadcast_to(np.diag([0.2, -0.2, 0]), (4, 3, 3))
    velocity = np.tile([1.0, 0.3, 0], (4, 1)) + 1e-5 * normals
    balanced, un, tangent, diagnostic = balanced_trace(velocity, jacobian, normals, areas)
    np.testing.assert_allclose(np.dot(un, areas), 0, atol=1e-15)
    np.testing.assert_allclose(balanced, np.tile([1.0, 0.3, 0], (4, 1)), atol=1e-15)
    np.testing.assert_allclose(tangent, 0, atol=1e-15)
    assert diagnostic["normal_velocity_correction"] == pytest.approx(1e-5)
    with pytest.raises(ValueError, match="excessive flux correction"):
        balanced_trace(velocity + 0.01 * normals, jacobian, normals, areas)


def test_restriction_preserves_each_bdf_level_and_matching_native_flux(tmp_path):
    mesh = structured_box(8, 8, 1, 2, 2, 1)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False, logger=None)
    count = mesh["n_cells"]
    total = count + mesh["n_faces"] - mesh["n_interior_faces"]
    field = np.zeros((total, 3))
    field[:count] = geometry["cell_centre"]
    field[:, 2] = 0.0
    pressure = np.zeros(total)
    pressure[:count] = geometry["cell_centre"][:, 0]
    flux = np.arange(mesh["n_faces"], dtype=np.float64) * 0.002
    old_flux = flux + 0.1
    older_flux = flux + 0.2
    for boundary in mesh["boundary"]:
        if boundary["name"] in ("zmin", "zmax"):
            faces = slice(boundary["start_face"], boundary["start_face"] + boundary["n_faces"])
            for values in (flux, old_flux, older_flux):
                values[faces] = 0.0
    reference = SimpleNamespace(
        mesh_data=mesh,
        geo_data=geometry,
        velocity=field,
        velocity_old=field + [0.1, 0, 0],
        velocity_older=field + [0.2, 0, 0],
        kinematic_pressure=pressure,
        volumetric_face_flux=flux,
        volumetric_face_flux_old=old_flux,
        volumetric_face_flux_older=older_flux,
        time=100.0,
        step=12500,
        _n_committed_time_steps=12500,
        time_step_size=0.008,
        _accepted_time_step_size=0.008,
        _previous_time_step_size=0.008,
        max_courant_number=0.2,
        _n_consecutive_accepted_steps={},
    )
    path = tmp_path / "initial.npz"
    report = restrict_initial_state(reference, mesh, geometry, path)
    with np.load(path, allow_pickle=False) as stored:
        for name in ("velocity", "velocity_old", "velocity_older", "kinematic_pressure"):
            np.testing.assert_allclose(
                stored[name][:count], getattr(reference, name)[:count], atol=1e-14
            )
        for name in (
            "volumetric_face_flux",
            "volumetric_face_flux_old",
            "volumetric_face_flux_older",
        ):
            np.testing.assert_allclose(stored[name], getattr(reference, name), atol=1e-14)
        assert int(stored["step"]) == 12500
        assert float(stored["time"]) == 100
    assert report["matching_oriented_native_faces"] == mesh["n_faces"]


def test_subcycling_uses_only_exchange_endpoints_and_preserves_their_values():
    times = np.arange(11) * 0.008
    trace = {
        "time": times,
        "velocity": np.broadcast_to((times**2)[:, None, None], (11, 4, 3)).copy(),
        "normal_velocity": np.broadcast_to((times**2)[:, None], (11, 4)).copy(),
        "tangential_gradient": np.broadcast_to((times**3)[:, None, None], (11, 4, 3)).copy(),
    }
    original = {name: values.copy() for name, values in trace.items()}
    for index in range(11):
        values = trace_at_step(trace, index, coupling_substeps=5)
        lower, upper = (0, 5) if index <= 5 else (5, 10)
        fraction = (index - lower) / 5
        for name in ("velocity", "normal_velocity", "tangential_gradient"):
            expected = (1 - fraction) * trace[name][lower] + fraction * trace[name][upper]
            np.testing.assert_array_equal(values[name], expected)
            if index % 5 == 0:
                np.testing.assert_array_equal(values[name], trace[name][index])
            np.testing.assert_array_equal(trace[name], original[name])
    assert not np.array_equal(
        trace_at_step(trace, 2, coupling_substeps=5)["velocity"], trace["velocity"][2]
    )
    with pytest.raises(ValueError, match="endpoint"):
        trace_at_step(trace, 9, coupling_substeps=6)
    with pytest.raises(ValueError, match="positive integer"):
        trace_at_step(trace, 1, coupling_substeps=1.5)


def test_saved_settings_allow_only_output_horizon_and_operator_implementation_changes():
    frozen = {
        "configuration": {
            "time": {"time_step_size": 0.008, "end_time": 100, "output_schedule": {}},
            "execution": {"operator_backend": "numba", "linear_backend": "scipy"},
            "pimple": {"n_correctors": 2, "type": "PimpleControl"},
            "logging": {"console": True},
            "samplers": [],
        }
    }
    actual = deepcopy(frozen)
    actual["configuration"]["time"]["end_time"] = 112
    actual["configuration"]["execution"]["operator_backend"] = "numpy"
    actual["configuration"]["logging"]["console"] = False
    assert validate_saved_settings(actual, frozen)["matched"]
    actual["configuration"]["pimple"]["n_correctors"] = 3
    with pytest.raises(ValueError, match="pimple"):
        validate_saved_settings(actual, frozen)


def test_rollback_state_comparison_preserves_gauge_and_all_temporal_levels():
    reference = RestartState(
        fields={
            "velocity": np.zeros((4, 3)),
            "velocity_old": np.ones((4, 3)),
            "velocity_older": np.full((4, 3), 2.0),
            "kinematic_pressure": np.array([1.0, 2.0, 1.0, 2.0]),
            "volumetric_face_flux": np.zeros(3),
            "volumetric_face_flux_old": np.ones(3),
            "volumetric_face_flux_older": np.full(3, 2.0),
        },
        eddy_viscosity=np.empty(0),
        time=100.04,
        step=12505,
        n_committed_time_steps=12505,
        time_step_size=0.008,
        accepted_time_step_size=0.008,
        previous_time_step_size=0.008,
        kinematic_viscosity=0.01,
        max_courant_number=0.2,
        n_consecutive_accepted_steps={"velocity": 3},
    )
    assert restart_state_comparison(reference, deepcopy(reference), np.array([1, 2]))[
        "exact_state_equal"
    ]
    shifted = deepcopy(reference)
    shifted.fields["kinematic_pressure"] += 3.0
    result = restart_state_comparison(reference, shifted, np.array([1, 2]))
    assert not result["exact_state_equal"]
    assert result["pressure_gauge_difference"] == 3.0
    assert result["gauge_aligned_pressure_max_abs_difference"] == 0
    assert result["clock_and_history_controls_equal"]
    shifted.fields["velocity_older"][0, 0] += 0.25
    shifted.fields["volumetric_face_flux_old"][1] -= 0.5
    result = restart_state_comparison(reference, shifted, np.array([1, 2]))
    assert result["fields_max_abs_difference"]["velocity_older"] == 0.25
    assert result["fields_max_abs_difference"]["volumetric_face_flux_old"] == 0.5
