"""Merge disjoint qualified cold-image ranges without shared HDF5 writers.

Every worker owns its private output directory. This read-only admission first
checks complete reports, frozen sources/configuration/volume hashes, ordered
finite fields, identical geometry, native trace projections and cold-image
budgets. Publication requires exactly the original complete 301 endpoints.
These are reference-image diagnostics, not autonomous particle simulations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from tests.support.cylinder.audit_reference_particle_image import admit_final_native_image
from tests.support.cylinder.project_reference_volume_images import (
    SCHEMA,
    SUPPORTS,
    TRACE_FIELDS,
    auxiliary_image_admission,
)
from tests.support.cylinder.run_boundary_condition_study import balanced_trace, digest

SAME_ATTRIBUTES = (
    "schema",
    "exchange_step_size",
    "fvm_step_size",
    "outer_repetitions",
    "volume_history_sha256",
    "mapped_initial_state_sha256",
    "source_sha256",
    "gradient_convention",
    "coefficient_storage_dtype",
    "native_configuration",
    "configuration_sha256",
    "qualified_reference_frame_count",
    "projection_history",
    "auxiliary_state_policy",
    "auxiliary_search_policy",
    "kernel_backend",
)
EXACT_FIELDS = {
    "raw_velocity": "raw_velocity",
    "velocity": "outer_velocity",
    "jacobian": "jacobian",
    "normal_velocity": "normal_velocity",
    "tangential_gradient": "tangential_gradient",
    "flux_correction": "flux_correction",
}


def admit_image_budget_configuration(row, configuration):
    """Keep final test-image budgets tied to the immutable native settings."""
    if (
        row["particle_capacity"] != configuration["vpm"]["max_n_particles"]
        or row["native_replacement_budget"]["closure_correction_limit"]
        != configuration["coupler"]["transfer_discretization_error_limit"]
    ):
        raise ValueError("Final image budget changed the native capacity or closure limit")
    return admit_final_native_image(row)


def validate_projection_subsets(subset_directories, volume_directory):
    """Admit complete independent ranges and return their ordered provenance."""
    volume_directory = Path(volume_directory).resolve()
    volume_path = volume_directory / "reference_volume_history.h5"
    volume_report_path = volume_directory / "report.json"
    volume_report = json.loads(volume_report_path.read_text())
    if (
        volume_report.get("status") != "complete"
        or digest(volume_path) != volume_report["history_sha256"]
    ):
        raise ValueError("Qualified native reference volume changed or is incomplete")
    entries = []
    for directory in subset_directories:
        directory = Path(directory).resolve()
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text())
        output = directory / "particle_image_traces.h5"
        if (
            report.get("status") != "complete"
            or Path(report["output"]).resolve() != output
            or digest(output) != report["output_sha256"]
        ):
            raise ValueError("Cold worker output changed or did not complete")
        if report.get("projection_history") != "cold":
            raise ValueError("Warm auxiliary outputs cannot enter an independent cold merge")
        configuration_path = Path(report["configuration_path"])
        if digest(configuration_path) != report["configuration_sha256"]:
            raise ValueError("Worker native configuration receipt changed")
        configuration = json.loads(configuration_path.read_text())["config"]
        if configuration != report["native_configuration"]:
            raise ValueError("Worker native configuration differs from its recorded receipt")
        for name, expected in report["source_sha256"].items():
            if digest(Path(name)) != expected:
                raise ValueError("Worker numerical or verification source changed: " + name)
        if (
            report["volume_history_sha256"] != volume_report["history_sha256"]
            or Path(report["volume_history_path"]).resolve() != volume_path
        ):
            raise ValueError("Workers did not consume the same qualified native volume history")
        repeated = report["cold_repeat_qualification"]
        if not repeated.get("passed") or not repeated["repeated_admission"].get("passed"):
            raise ValueError("Worker first image was not independently repeated")
        for comparison in repeated["coefficient_comparison"].values():
            if (
                not comparison["positions_bitwise_equal"]
                or not comparison["strengths_bitwise_equal"]
            ):
                raise ValueError("Worker first image failed independent coefficient repeat")
        entries.append(
            {
                "directory": str(directory),
                "report_path": str(report_path),
                "report_sha256": digest(report_path),
                "output": str(output),
                "output_sha256": report["output_sha256"],
                "report": report,
            }
        )
    if not entries:
        raise ValueError("At least one complete cold range is required")
    entries.sort(key=lambda entry: entry["report"]["start_frame"])
    first_attributes = None
    cursor = 0
    with h5py.File(volume_path, "r") as volumes:
        qualified_count = len(volumes["time"])
        if qualified_count != 301 or not volumes.attrs.get("complete"):
            raise ValueError("Merge requires the complete qualified 100-112 s native history")
        for entry in entries:
            report = entry["report"]
            start, stop = report["start_frame"], report["stop_frame"]
            if start != cursor or not start < stop <= qualified_count:
                raise ValueError("Cold worker ranges overlap, have gaps or extend past the horizon")
            cursor = stop
            indices = np.arange(start, stop, dtype=np.int64)
            if (
                report["frames"] != len(indices)
                or report["qualified_reference_frame_count"] != qualified_count
                or len(report["frame_reports"]) != len(indices)
            ):
                raise ValueError("Worker report has an inconsistent frame count")
            if report["cold_repeat_qualification"]["frame_index"] != start:
                raise ValueError("Worker repeat did not qualify its own first frame")
            with h5py.File(entry["output"], "r") as stored:
                attributes = {name: stored.attrs[name] for name in SAME_ATTRIBUTES}
                if (
                    attributes["schema"] != SCHEMA
                    or not stored.attrs.get("complete")
                    or attributes["projection_history"] != "cold"
                    or attributes["auxiliary_state_policy"] != "zero_coefficients_each_frame"
                ):
                    raise ValueError("Worker is not a complete independent cold schema-2 image")
                if first_attributes is None:
                    first_attributes = attributes
                elif attributes != first_attributes:
                    raise ValueError("Worker source, native configuration or image policy differs")
                if (
                    attributes["source_sha256"]
                    != json.dumps(report["source_sha256"], sort_keys=True)
                    or attributes["native_configuration"]
                    != json.dumps(report["native_configuration"], sort_keys=True)
                    or attributes["configuration_sha256"] != report["configuration_sha256"]
                    or attributes["volume_history_sha256"] != volume_report["history_sha256"]
                    or attributes["mapped_initial_state_sha256"]
                    != volumes.attrs["mapped_initial_state_sha256"]
                ):
                    raise ValueError(
                        "Worker HDF provenance disagrees with native admission receipts"
                    )
                if (
                    stored.attrs["start_frame"] != start
                    or stored.attrs["stop_frame"] != stop
                    or not np.array_equal(stored["frame_index"][:], indices)
                    or not np.array_equal(stored["time"][:], volumes["time"][indices])
                ):
                    raise ValueError("Worker absolute frame indices or clocks changed")
                geometry = {
                    name: stored[name][:] for name in ("face_centre", "face_normal", "face_area")
                }
                for name, values in geometry.items():
                    if not np.array_equal(values, volumes[name][:]):
                        raise ValueError(
                            "Worker face geometry/order differs from the native reference"
                        )
                normals, areas = geometry["face_normal"], geometry["face_area"]
                shapes = {
                    "raw_velocity": (len(normals), 3),
                    "velocity": (len(normals), 3),
                    "jacobian": (len(normals), 3, 3),
                    "normal_velocity": (len(normals),),
                    "tangential_gradient": (len(normals), 3),
                    "flux_correction": (),
                }
                for support in SUPPORTS:
                    for name, shape in shapes.items():
                        if stored[support][name].shape != (len(indices), *shape) or stored[support][
                            name
                        ].dtype != np.dtype(np.float64):
                            raise ValueError("Worker trace shape or storage precision differs")
                for row, index in enumerate(indices):
                    frame = report["frame_reports"][row]
                    if frame["index"] != index or frame["time"] != volumes["time"][index]:
                        raise ValueError("Worker frame diagnostic order or time differs")
                    admission = auxiliary_image_admission(frame["image"])
                    if admission != frame["auxiliary_image_admission"]:
                        raise ValueError("Worker coefficient admission budget differs")
                    if len(frame["image"]["auxiliary_renewals"]) != attributes["outer_repetitions"]:
                        raise ValueError(
                            "Worker changed its declared fixed inverse iteration count"
                        )
                    native_admission = admit_image_budget_configuration(
                        frame["image"]["auxiliary_renewals"][-1], report["native_configuration"]
                    )
                    if native_admission != frame["image"]["final_auxiliary_native_admission"]:
                        raise ValueError("Final auxiliary native guard receipt differs")
                    for name in ("production_renewals", "reference_support_renewals"):
                        admit_image_budget_configuration(
                            frame[name][-1], report["native_configuration"]
                        )
                    if report["kernel_backend"] == "cuda" and row in (0, len(indices) - 1):
                        for support in ("production", "reference_support"):
                            comparison = frame["kernel_cpu_comparison"][support]
                            if (
                                comparison is None
                                or comparison["maximum_velocity_absolute_error"] > 2e-6
                                or comparison["maximum_jacobian_absolute_error"] > 2e-6
                            ):
                                raise ValueError(
                                    "Worker first/final CUDA trace lacks independent CPU proof"
                                )
                    for support in SUPPORTS:
                        fields = {name: stored[support][name][row] for name in TRACE_FIELDS}
                        if not all(np.isfinite(value).all() for value in fields.values()):
                            raise ValueError("Worker has a nonfinite projected trace")
                        balanced, normal, tangent, balance = balanced_trace(
                            fields["raw_velocity"], fields["jacobian"], normals, areas
                        )
                        for name, expected in (
                            ("velocity", balanced),
                            ("normal_velocity", normal),
                            ("tangential_gradient", tangent),
                            ("flux_correction", balance["normal_velocity_correction"]),
                        ):
                            if not np.allclose(fields[name], expected, rtol=0, atol=2e-12):
                                raise ValueError("Worker trace projection is inconsistent: " + name)
                        if support == "exact_reference":
                            for field, source in EXACT_FIELDS.items():
                                if not np.array_equal(fields[field], volumes[source][index]):
                                    raise ValueError("Exact native reference channel changed")
                if report["kernel_backend"] == "cuda":
                    comparison = report["native_cuda_cpu_comparison"]
                    if (
                        comparison is None
                        or comparison["maximum_velocity_absolute_error"] > 2e-6
                        or comparison["maximum_jacobian_absolute_error"] > 2e-6
                    ):
                        raise ValueError("Worker CUDA fields lack independent CPU qualification")
    if cursor != qualified_count:
        raise ValueError("Cold worker ranges do not cover all 301 qualified endpoints")
    return entries, first_attributes, volume_report


def merge(subset_directories, volume_directory, directory):
    """Publish one ordered complete artifact only after every subset is admitted."""
    directory = Path(directory).resolve()
    if not directory.is_relative_to(Path("/tmp")):
        raise ValueError("Merged diagnostic traces require a private /tmp directory")
    started = perf_counter()
    entries, attributes, volume_report = validate_projection_subsets(
        subset_directories, volume_directory
    )
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "particle_image_traces.h5"
    merge_source = digest(Path(__file__))
    with h5py.File(entries[0]["output"], "r") as first, h5py.File(output, "x") as stored:
        stored.attrs.update(attributes)
        stored.attrs.update(
            complete=False,
            full_reference_horizon_complete=False,
            start_frame=0,
            stop_frame=301,
            merged_independent_ranges=True,
            merge_source_sha256=merge_source,
        )
        for name in ("face_centre", "face_normal", "face_area"):
            stored.create_dataset(name, data=first[name][:])
        stored.create_dataset("frame_index", data=np.arange(301, dtype=np.int64))
        stored.create_dataset("time", shape=(301,), dtype=np.float64)
        for support in SUPPORTS:
            group = stored.create_group(support)
            for name in TRACE_FIELDS:
                source = first[support][name]
                group.create_dataset(name, shape=(301, *source.shape[1:]), dtype=source.dtype)
        frame_reports = []
        for entry in entries:
            report = entry["report"]
            start, stop = report["start_frame"], report["stop_frame"]
            with h5py.File(entry["output"], "r") as subset:
                stored["time"][start:stop] = subset["time"][:]
                for support in SUPPORTS:
                    for name in TRACE_FIELDS:
                        stored[support][name][start:stop] = subset[support][name][:]
            frame_reports.extend(report["frame_reports"])
        for entry in entries:
            if (
                digest(Path(entry["output"])) != entry["output_sha256"]
                or digest(Path(entry["report_path"])) != entry["report_sha256"]
            ):
                raise ValueError("Worker artifact changed during merge")
        if digest(Path(__file__)) != merge_source:
            raise ValueError("Merge helper changed during publication")
        provenance = [
            {key: value for key, value in entry.items() if key != "report"} for entry in entries
        ]
        stored.attrs["range_provenance"] = json.dumps(provenance, sort_keys=True)
        stored.attrs["complete"] = True
        stored.attrs["full_reference_horizon_complete"] = True
    report = {
        "status": "complete",
        "output": str(output),
        "output_sha256": digest(output),
        "frames": 301,
        "start_frame": 0,
        "stop_frame": 301,
        "qualified_reference_frame_count": 301,
        "complete_qualified_horizon": True,
        "projection_history": "cold",
        "outer_repetitions": int(attributes["outer_repetitions"]),
        "source_sha256": json.loads(attributes["source_sha256"]),
        "configuration_sha256": attributes["configuration_sha256"],
        "native_configuration": json.loads(attributes["native_configuration"]),
        "volume_history_sha256": volume_report["history_sha256"],
        "mapped_initial_state_sha256": attributes["mapped_initial_state_sha256"],
        "range_provenance": provenance,
        "merge_source_sha256": merge_source,
        "frame_reports": frame_reports,
        "wall_seconds": perf_counter() - started,
        "scope": "301 independent cold reference images from identical qualified native volumes, configuration and ordered face geometry; paired supports share each complete coefficient image. This is not an autonomous evolving VPM solution or force-recovery proof.",
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-directory", type=Path, action="append", required=True)
    parser.add_argument("--volume-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    arguments = parser.parse_args()
    report = merge(arguments.subset_directory, arguments.volume_directory, arguments.directory)
    print(json.dumps({name: report[name] for name in ("frames", "output_sha256", "wall_seconds")}))


if __name__ == "__main__":
    main()
