"""A completed FVM input cannot hide an unstable or changed image oracle."""

from copy import deepcopy
import json

import h5py
import numpy as np
import pytest

from tests.support.cylinder.capture_reference_volume_history import (
    GRADIENT_CONVENTION,
)
from tests.support.cylinder.capture_reference_volume_history import (
    SCHEMA as VOLUME_SCHEMA,
)
from tests.support.cylinder.merge_reference_volume_images import merge
from tests.support.cylinder.replay_boundary_inputs import admit_cold_particle_image, digest

pytest_plugins = ("tests.coupler.test_reference_image_ranges",)


@pytest.fixture
def image_input(cold_ranges, tmp_path):
    """Manufactured complete cold ranges with an unchanged native trace channel."""
    directories, volume_directory = cold_ranges
    volume = volume_directory / "reference_volume_history.h5"
    reference = tmp_path / "reference_capture"
    (reference / "inputs").mkdir(parents=True)
    (reference / "reference/samples").mkdir(parents=True)
    forces = reference / "reference/samples/forces_history.csv"
    forces.write_text("Manufactured source identity only; no native flow claim.\n")
    worker = json.loads((directories[0] / "report.json").read_text())
    information = {
        "start_time": 100.0,
        "end_time": 112.0,
        "start_step": 12500,
        "steps": 1500,
        "step_size": 0.008,
        "reference_config_hash": "manufactured-native-configuration",
        "source_sha256": worker["source_sha256"],
        "files": {
            "reference_mesh.npz": {"sha256": "manufactured-ordered-mesh"},
            "reference_backup.npz": {"sha256": "manufactured-original-BDF"},
        },
    }
    inputs = reference / "inputs/inputs.json"
    inputs.write_text(json.dumps(information))
    original = {"initial_restriction": {"initial_state_sha256": "manufactured-initial-state"}}
    with h5py.File(volume, "r+") as stored:
        stored.attrs.update(
            schema=VOLUME_SCHEMA,
            cell_count=2,
            published_frame_count=301,
            gradient_convention=GRADIENT_CONVENTION,
            exchange_step_size=0.04,
            configuration_hash=information["reference_config_hash"],
            source_sha256=json.dumps(information["source_sha256"]),
            authored_input_provenance=json.dumps(
                {
                    "historical_hashes_recorded": False,
                    "historical_sha256": {},
                    "newly_collected_sha256": {"manufactured_geometry": "new-test-hash"},
                }
            ),
            reference_mesh_archive_sha256="manufactured-ordered-mesh",
            native_reference_checkpoint_sha256="manufactured-original-BDF",
            native_mesh_hash="manufactured-native-mesh",
        )
        stored.create_dataset("step", data=np.arange(12500, 14001, 5, dtype=np.int64))
        stored.create_dataset("accepted_time_step_size", data=np.full(301, 0.008))
        stored.create_dataset("cell_centre", data=np.array([[2.0, 0, 0], [3.0, 0, 0]]))
        stored.create_dataset("velocity", data=np.zeros((301, 2, 3)))
        stored.create_dataset("velocity_gradient", data=np.zeros((301, 2, 3, 3)))
        with h5py.File(reference / "reference_traces.h5", "x") as exact:
            exact.create_dataset("time", data=100 + 0.008 * np.arange(1501))
            for name in ("face_centre", "face_normal", "face_area"):
                exact.create_dataset(name, data=stored[name][:])
            for name in (
                "raw_velocity",
                "velocity",
                "jacobian",
                "normal_velocity",
                "tangential_gradient",
                "flux_correction",
            ):
                source = "outer_velocity" if name == "velocity" else name
                value = stored[source][0]
                exact.create_dataset(name, data=np.broadcast_to(value, (1501, *np.shape(value))))
    volume_hash = digest(volume)
    final_fields = (
        "velocity",
        "velocity_old",
        "velocity_older",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
    )
    (volume_directory / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "history_sha256": volume_hash,
                "equation_equivalence_verified": True,
                "mapped_initial_state_sha256": "manufactured-initial-state",
                "native_reference_config_hash": information["reference_config_hash"],
                "reference_mesh_sha256": "manufactured-ordered-mesh",
                "original_capture_inputs_sha256": digest(inputs),
                "accepted_native_steps": 1500,
                "endpoint_count": 301,
                "accepted_time": 112.0,
                "maximum_outer_trace_difference": dict.fromkeys(
                    (
                        "raw_velocity",
                        "velocity",
                        "jacobian",
                        "normal_velocity",
                        "tangential_gradient",
                        "flux_correction",
                    ),
                    0.0,
                ),
                "final_primary_and_BDF_difference": dict.fromkeys(final_fields, 0.0),
                "force_determinism": {
                    "record_count": 300,
                    "original_sha256": digest(forces),
                    "candidate_sha256": digest(forces),
                    "maximum_absolute_field_difference": {
                        "drag_coefficient": 0.0,
                        "lift_coefficient": 0.0,
                    },
                },
            }
        )
    )
    for directory in directories:
        path = directory / "report.json"
        report = json.loads(path.read_text())
        report["volume_history_sha256"] = volume_hash
        report["outer_repetitions"] = 10
        comparison = {
            "particle_count": 4,
            "positions_bitwise_equal": True,
            "strengths_bitwise_equal": True,
        }
        report["cold_repeat_qualification"]["coefficient_comparison"] = {
            name: deepcopy(comparison) for name in ("native_masked", "full_reference_native_masked")
        }
        with h5py.File(directory / "particle_image_traces.h5", "r+") as stored:
            stored.attrs["volume_history_sha256"] = volume_hash
        report["output_sha256"] = digest(directory / "particle_image_traces.h5")
        path.write_text(json.dumps(report))
    report = merge(directories, volume_directory, tmp_path / "merged_admission")
    return {
        "source": tmp_path / "merged_admission/particle_image_traces.h5",
        "report": report,
        "information": information,
        "original_report": original,
        "reference": reference,
        "workers": directories,
        "volumes": volume_directory,
    }


def admit(case):
    with h5py.File(case["source"], "r") as stored:
        return admit_cold_particle_image(
            stored,
            case["report"],
            source=case["source"],
            information=case["information"],
            original_report=case["original_report"],
            reference=case["reference"],
        )


def rehash_worker_report(case, index, changed):
    path = case["workers"][index] / "report.json"
    path.write_text(json.dumps(changed))
    case["report"]["range_provenance"][index]["report_sha256"] = digest(path)
    with h5py.File(case["source"], "r+") as stored:
        stored.attrs["range_provenance"] = json.dumps(
            case["report"]["range_provenance"], sort_keys=True
        )


def test_complete_cold_image_admission_retains_native_budget_and_original_exact_trace(image_input):
    result = admit(image_input)
    assert result["frame_count"] == 301 and result["independent_range_count"] == 3
    assert result["both_supports_independently_repeated"]
    assert result["native_moment_closure_correction_limit"] == 0.08
    assert result["unchanged_native_volume_validation"]["frame_count"] == 301


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("schema", "openonda-cylinder-reference-particle-image/1"),
        ("projection_history", "warm"),
        ("auxiliary_state_policy", "reuse_previous_frame"),
        ("outer_repetitions", 25),
        ("stop_frame", 300),
    ],
)
def test_new_replay_admission_rejects_warm_legacy_incomplete_and_changed_inverse_policy(
    image_input, attribute, value
):
    with h5py.File(image_input["source"], "r+") as stored:
        stored.attrs[attribute] = value
    with pytest.raises(ValueError, match="schema-2 independent cold horizon"):
        admit(image_input)


def test_new_replay_admission_rejects_missing_second_support_repeat_even_after_rehash(image_input):
    worker = json.loads((image_input["workers"][0] / "report.json").read_text())
    worker["cold_repeat_qualification"]["coefficient_comparison"].pop(
        "full_reference_native_masked"
    )
    rehash_worker_report(image_input, 0, worker)
    with pytest.raises(ValueError, match="both supports"):
        admit(image_input)


def test_new_replay_admission_rejects_native_moment_budget_failure_even_after_rehash(image_input):
    worker = json.loads((image_input["workers"][1] / "report.json").read_text())
    worker["frame_reports"][0]["image"]["auxiliary_renewals"][-1]["native_replacement_budget"][
        "closure_correction_fraction"
    ] = 0.08001
    rehash_worker_report(image_input, 1, worker)
    with pytest.raises(ValueError, match="native storage/moment"):
        admit(image_input)


def test_new_replay_admission_rejects_changed_merged_field_with_valid_worker_sources(image_input):
    with h5py.File(image_input["source"], "r+") as stored:
        stored["production/raw_velocity"][0, 0, 0] += 0.02
    image_input["report"]["output_sha256"] = digest(image_input["source"])
    with pytest.raises(ValueError, match="differs from its admitted cold range"):
        admit(image_input)


def test_new_replay_admission_rejects_changed_volume_BDF_determinism(image_input):
    path = image_input["volumes"] / "report.json"
    report = json.loads(path.read_text())
    report["final_primary_and_BDF_difference"]["velocity_old"] = 0.01
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact original force, outer-trace and primary/BDF"):
        admit(image_input)


def test_new_replay_admission_rejects_loss_of_absolute_integer_frame_precision(image_input):
    with h5py.File(image_input["source"], "r+") as stored:
        values = stored["frame_index"][:].astype(np.float64)
        del stored["frame_index"]
        stored.create_dataset("frame_index", data=values)
    with pytest.raises(ValueError, match="schema-2 independent cold horizon"):
        admit(image_input)
