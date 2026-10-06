"""Independent image-range ownership, admission and complete merge checks."""

from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.support.cylinder.audit_reference_particle_image import admit_final_native_image
from tests.support.cylinder.merge_reference_volume_images import merge, validate_projection_subsets
from tests.support.cylinder.project_reference_volume_images import (
    SCHEMA,
    SUPPORTS,
    TRACE_FIELDS,
    auxiliary_image_admission,
    projection_frame_range,
    repeat_image_comparison,
)
from tests.support.cylinder.run_boundary_condition_study import balanced_trace, digest


def image_row():
    return {
        "population_pruned_count": 0,
        "stored_particle_count": 4,
        "particle_capacity": 20,
        "stored_inside_solid_count": 0,
        "maximum_stored_axial_strength": 0.2,
        "stored_strength_l1": 0.8,
        "native_replacement_budget": {
            "closure_correction_fraction": 0.03,
            "closure_correction_limit": 0.08,
            "vortex_strength_error": 0.0,
            "vortex_strength_tolerance": 1e-5,
            "linear_impulse_error": 0.0,
            "linear_impulse_tolerance": 1e-5,
        },
    }


def image_report():
    row = image_row()
    return {
        "auxiliary_target": {"maximum_absolute_axial_strength": 0.1, "axial_strength_l1": 0.5},
        "auxiliary_renewals": [deepcopy(row) for _ in range(10)],
        "auxiliary_history_input_particle_count": 0,
        "final_auxiliary_native_admission": admit_final_native_image(row),
    }


def test_independent_half_open_ranges_cover_history_and_reject_invalid_bounds():
    ranges = [
        projection_frame_range(301, start_frame=a, stop_frame=b)
        for a, b in ((0, 101), (101, 201), (201, 301))
    ]
    np.testing.assert_array_equal(np.concatenate(ranges), np.arange(301))
    np.testing.assert_array_equal(
        projection_frame_range(301, start_frame=101, maximum_frames=2), [101, 102]
    )
    for options in (
        {"start_frame": -1},
        {"start_frame": 301},
        {"stop_frame": 302},
        {"start_frame": True},
        {"maximum_frames": 0},
    ):
        with pytest.raises(ValueError):
            projection_frame_range(301, **options)


def test_image_oracle_rejects_coefficient_growth_history_and_native_guard_failure():
    normal = image_report()
    assert auxiliary_image_admission(normal)["passed"]
    for key, value in (("maximum_stored_axial_strength", 30.0), ("stored_strength_l1", 340.0)):
        bad = deepcopy(normal)
        bad["auxiliary_renewals"][-1][key] = value
        with pytest.raises(ValueError, match="target-relative"):
            auxiliary_image_admission(bad)
    bad = deepcopy(normal)
    bad["auxiliary_history_input_particle_count"] = 4
    with pytest.raises(ValueError, match="zero coefficients"):
        auxiliary_image_admission(bad)
    bad = image_row()
    bad["native_replacement_budget"]["closure_correction_fraction"] = 0.08001
    with pytest.raises(ValueError, match="native storage/moment"):
        admit_final_native_image(bad)


def test_cold_repeat_checks_both_support_coefficient_arrays():
    pair = (np.zeros((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32), [])
    first = {name: deepcopy(pair) for name in ("native_masked", "full_reference_native_masked")}
    assert repeat_image_comparison(first, deepcopy(first))["native_masked"][
        "strengths_bitwise_equal"
    ]
    changed = deepcopy(first)
    changed["full_reference_native_masked"][1][0, 2] += 1
    with pytest.raises(ValueError, match="changed when repeated"):
        repeat_image_comparison(first, changed)


@pytest.fixture
def cold_ranges(tmp_path):
    volume_directory = tmp_path / "volumes"
    volume_directory.mkdir()
    volume = volume_directory / "reference_volume_history.h5"
    configuration_path = tmp_path / "configuration.json"
    configuration = {
        "coupler": {"transfer_discretization_error_limit": 0.08},
        "vpm": {"precision": "f32", "max_n_particles": 20},
    }
    configuration_path.write_text(json.dumps({"config": configuration}))
    source = tmp_path / "frozen_test_source.txt"
    source.write_text("Manufactured range-admission fixture; no native flow claim.\n")
    sources = {str(source): digest(source)}
    geometry = {
        "face_centre": np.array([[1.0, 0, 0], [-1.0, 0, 0], [0.0, 1, 0], [0.0, -1, 0]]),
        "face_normal": np.array([[1.0, 0, 0], [-1.0, 0, 0], [0.0, 1, 0], [0.0, -1, 0]]),
        "face_area": np.ones(4),
    }
    raw = np.tile([1.0, 0.02, 0], (4, 1))
    jacobian = np.zeros((4, 3, 3))
    velocity, normal, tangent, balance = balanced_trace(
        raw, jacobian, geometry["face_normal"], geometry["face_area"]
    )
    fields = {
        "raw_velocity": raw,
        "velocity": velocity,
        "jacobian": jacobian,
        "normal_velocity": normal,
        "tangential_gradient": tangent,
        "flux_correction": balance["normal_velocity_correction"],
    }
    times = 100 + 0.04 * np.arange(301)
    with h5py.File(volume, "x") as h:
        h.attrs.update(complete=True, mapped_initial_state_sha256="manufactured-initial-state")
        for name, values in geometry.items():
            h.create_dataset(name, data=values)
        h.create_dataset("time", data=times)
        for name, values in fields.items():
            h.create_dataset(
                "outer_velocity" if name == "velocity" else name,
                data=np.broadcast_to(values, (301, *np.shape(values))),
            )
    volume_hash = digest(volume)
    (volume_directory / "report.json").write_text(
        json.dumps({"status": "complete", "history_sha256": volume_hash})
    )
    directories = []
    for start, stop in ((0, 101), (101, 201), (201, 301)):
        directory = tmp_path / ("worker_" + str(start))
        directory.mkdir()
        directories.append(directory)
        output = directory / "particle_image_traces.h5"
        repeat = {
            "frame_index": start,
            "passed": True,
            "repeated_admission": {"passed": True},
            "coefficient_comparison": {
                "manufactured": {"positions_bitwise_equal": True, "strengths_bitwise_equal": True}
            },
        }
        with h5py.File(output, "x") as h:
            h.attrs.update(
                schema=SCHEMA,
                complete=True,
                full_reference_horizon_complete=False,
                projection_history="cold",
                kernel_backend="cpu",
                auxiliary_state_policy="zero_coefficients_each_frame",
                auxiliary_search_policy="fixed internal nonphysical coefficient iterations; final atomic replacement guards enforced",
                qualified_reference_frame_count=301,
                start_frame=start,
                stop_frame=stop,
                configuration_sha256=digest(configuration_path),
                native_configuration=json.dumps(configuration, sort_keys=True),
                source_sha256=json.dumps(sources, sort_keys=True),
                volume_history_sha256=volume_hash,
                mapped_initial_state_sha256="manufactured-initial-state",
                gradient_convention="J[i,j]=d(velocity_i)/d(x_j)",
                coefficient_storage_dtype="float32",
                exchange_step_size=0.04,
                fvm_step_size=0.008,
                outer_repetitions=10,
            )
            for name, values in geometry.items():
                h.create_dataset(name, data=values)
            h.create_dataset("time", data=times[start:stop])
            h.create_dataset("frame_index", data=np.arange(start, stop, dtype=np.int64))
            for support in SUPPORTS:
                group = h.create_group(support)
                for name, values in fields.items():
                    group.create_dataset(
                        name, data=np.broadcast_to(values, (stop - start, *np.shape(values)))
                    )
        frames = []
        for index in range(start, stop):
            image = image_report()
            frames.append(
                {
                    "index": index,
                    "time": times[index],
                    "image": image,
                    "auxiliary_image_admission": auxiliary_image_admission(image),
                    "production_renewals": [image_row()],
                    "reference_support_renewals": [image_row()],
                }
            )
        report = {
            "status": "complete",
            "output": str(output),
            "output_sha256": digest(output),
            "projection_history": "cold",
            "kernel_backend": "cpu",
            "configuration_path": str(configuration_path),
            "configuration_sha256": digest(configuration_path),
            "native_configuration": configuration,
            "source_sha256": sources,
            "volume_history_sha256": volume_hash,
            "volume_history_path": str(volume),
            "cold_repeat_qualification": repeat,
            "start_frame": start,
            "stop_frame": stop,
            "frames": stop - start,
            "qualified_reference_frame_count": 301,
            "frame_reports": frames,
        }
        (directory / "report.json").write_text(json.dumps(report))
    return directories, volume_directory


def test_merge_complete_disjoint_cold_ranges_preserves_order_and_hash_provenance(
    cold_ranges, tmp_path
):
    directories, volumes = cold_ranges
    report = merge(list(reversed(directories)), volumes, tmp_path / "merged")
    assert report["complete_qualified_horizon"] and report["frames"] == 301
    assert len(report["range_provenance"]) == 3
    with h5py.File(report["output"], "r") as h:
        assert h.attrs["complete"] and h.attrs["full_reference_horizon_complete"]
        np.testing.assert_array_equal(h["frame_index"][:], np.arange(301))
        for support in SUPPORTS:
            assert all(np.isfinite(h[support][name][:]).all() for name in TRACE_FIELDS)


def test_merge_rejects_missing_range_warm_artifact_and_changed_hash(cold_ranges):
    directories, volumes = cold_ranges
    with pytest.raises(ValueError, match="gaps"):
        validate_projection_subsets([directories[0], directories[2]], volumes)
    path = directories[1] / "report.json"
    original = json.loads(path.read_text())
    changed = deepcopy(original)
    changed["projection_history"] = "warm"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Warm"):
        validate_projection_subsets(directories, volumes)
    path.write_text(json.dumps(original))
    with h5py.File(directories[1] / "particle_image_traces.h5", "r+") as h:
        h["production/velocity"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="changed"):
        validate_projection_subsets(directories, volumes)


def test_merge_rejects_rehashed_nonfinite_geometry_and_configuration_changes(cold_ranges):
    directories, volumes = cold_ranges
    output = directories[1] / "particle_image_traces.h5"
    with h5py.File(output, "r+") as h:
        h["production/velocity"][0, 0, 0] = np.nan
    path = directories[1] / "report.json"
    report = json.loads(path.read_text())
    report["output_sha256"] = digest(output)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="nonfinite"):
        validate_projection_subsets(directories, volumes)
    with h5py.File(output, "r+") as h:
        h["production/velocity"][0, 0, 0] = 1.0
        h["face_centre"][0, 0] = 2.0
    report["output_sha256"] = digest(output)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="geometry"):
        validate_projection_subsets(directories, volumes)
    configuration = Path(report["configuration_path"])
    configuration.write_text("{}")
    with pytest.raises(ValueError, match="configuration receipt changed"):
        validate_projection_subsets(directories, volumes)
