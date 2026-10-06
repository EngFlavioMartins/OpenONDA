"""A streamed oracle must retain the native donor ordering and accepted clock."""

import json

import h5py
import numpy as np
import pytest

from tests.support.cylinder.capture_reference_volume_history import (
    GRADIENT_CONVENTION,
    SCHEMA,
    authored_input_provenance,
    bounded_difference,
    endpoint_steps,
    validate_volume_history,
)


def _fixture(path, *, historical_authored_hashes=True):
    information = {
        "start_time": 100., "end_time": 100.08, "start_step": 12500,
        "step_size": .008, "steps": 10, "reference_config_hash": "native-equations",
        "source_sha256": {"source/physical_equation.py": "unchanged"},
        "authored_inputs_sha256": {"reference_flow/setup.py": "physical-inputs"},
        "files": {"reference_mesh.npz": {"sha256": "ordered-mesh"},
                  "reference_backup.npz": {"sha256": "native-BDF-state"}},
    }
    if not historical_authored_hashes:
        information.pop("authored_inputs_sha256")
    points = np.array([[.2, -.1, 0], [.6, .3, 0]])
    jacobian = np.array([[.2, -.3, 0], [.1, -.2, 0], [0, 0, 0]])
    velocity = points @ jacobian.T + [1, 0, 0]
    with h5py.File(path, "x") as stored:
        stored.attrs.update(
            schema=SCHEMA, complete=True, cell_count=2,
            published_frame_count=3, gradient_convention=GRADIENT_CONVENTION,
            exchange_step_size=.04, configuration_hash="native-equations",
            source_sha256=json.dumps(information["source_sha256"]),
            authored_input_provenance=json.dumps(authored_input_provenance(
                information, {"reference_flow/setup.py": "new-current-hash"})),
            reference_mesh_archive_sha256="ordered-mesh",
            native_reference_checkpoint_sha256="native-BDF-state", native_mesh_hash="native-topology",
        )
        stored.create_dataset("time", data=[100., 100.04, 100.08])
        stored.create_dataset("step", data=np.array([12500, 12505, 12510], dtype=np.int64))
        stored.create_dataset("accepted_time_step_size", data=np.full(3, .008))
        stored.create_dataset("cell_centre", data=points)
        stored.create_dataset("velocity", data=np.broadcast_to(velocity, (3, 2, 3)))
        stored.create_dataset("velocity_gradient", data=np.broadcast_to(jacobian.T, (3, 2, 3, 3)))
    return information, jacobian


def test_reference_consumer_admits_complete_native_ordered_affine_fields_and_rejects_corruption(tmp_path):
    path = tmp_path / "reference-volume.h5"
    information, jacobian = _fixture(path)
    with h5py.File(path, "r+") as stored:
        result = validate_volume_history(stored, information, inspect_fields=True)
        assert result["frame_count"] == 3
        np.testing.assert_array_equal(stored["velocity_gradient"][1].swapaxes(1, 2),
                                      np.broadcast_to(jacobian, (2, 3, 3)))
        # A changing publication may not be consumed even when fields are finite.
        stored.attrs["complete"] = False
        with pytest.raises(ValueError, match="incomplete"):
            validate_volume_history(stored, information)
        stored.attrs["complete"] = True
        stored["step"][1] = 12504
        with pytest.raises(ValueError, match="skipped or repeated"):
            validate_volume_history(stored, information)
        stored["step"][1] = 12505
        stored["velocity_gradient"][1, 0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="nonfinite velocity_gradient"):
            validate_volume_history(stored, information, inspect_fields=True)
        stored["velocity_gradient"][1, 0, 0, 0] = jacobian[0, 0]
        stored.attrs["configuration_hash"] = "changed-pressure-tolerance"
        with pytest.raises(ValueError, match="configuration"):
            validate_volume_history(stored, information)
        stored.attrs["configuration_hash"] = "native-equations"
        stored.attrs["gradient_convention"] = "J[i,j]=d(u_i)/d(x_j)"
        with pytest.raises(ValueError, match="gradient convention"):
            validate_volume_history(stored, information)


def test_reference_clock_cannot_truncate_an_exchange_or_hide_changed_deterministic_fields():
    information = {"start_time": 100., "end_time": 112., "start_step": 12500,
                   "step_size": .008, "steps": 1500}
    steps = endpoint_steps(information, .04)
    assert len(steps) == 301 and steps[-1] == 1500
    with pytest.raises(ValueError, match="complete native coupling"):
        endpoint_steps({**information, "steps": 1499}, .04)
    with pytest.raises(ValueError, match="complete native coupling"):
        endpoint_steps(information, .041)
    with pytest.raises(ValueError, match="determinism bound exceeded"):
        bounded_difference(np.array([1.2, .08]), np.array([1.2, .01]), "drag force")
    with pytest.raises(ValueError, match="finite matching"):
        bounded_difference(np.array([1.2, np.nan]), np.array([1.2, .01]), "drag force")


def test_original_frozen_input_schema_without_authored_hashes_cannot_invent_historical_evidence(tmp_path):
    # The original immutable cylinder inputs.json has source_sha256 and files,
    # but no authored_inputs_sha256. Current file hashes cannot fill that gap.
    path = tmp_path / "original-schema-volume.h5"
    information, _jacobian = _fixture(path, historical_authored_hashes=False)
    with h5py.File(path, "r+") as stored:
        validate_volume_history(stored, information, inspect_fields=True)
        provenance = json.loads(stored.attrs["authored_input_provenance"])
        assert provenance["historical_hashes_recorded"] is False
        assert provenance["historical_sha256"] == {}
        assert provenance["newly_collected_sha256"] == {"reference_flow/setup.py": "new-current-hash"}
        assert "do not establish historical byte equivalence" in provenance["scope"]
        assert "deterministic original outer traces" in provenance["equation_equivalence_basis"]
        provenance["historical_hashes_recorded"] = True
        provenance["historical_sha256"] = provenance["newly_collected_sha256"]
        stored.attrs["authored_input_provenance"] = json.dumps(provenance)
        with pytest.raises(ValueError, match="misstates historical"):
            validate_volume_history(stored, information)
