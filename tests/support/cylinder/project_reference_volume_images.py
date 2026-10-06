"""Stream simultaneous native reference volumes into paired planar-image traces.

Consumes an already qualified private reference volume capture. One complete
reference coefficient image supplies both supports: the production trace removes
only particles outside the prescribed VPM bounds, and the reference-support
trace retains those same coefficients. No simulation is launched here.
Every endpoint starts its auxiliary Gaussian coefficients from zero. A previous
warm diagnostic accumulated unstable unobserved outer-belt coefficients and is
retired. The auxiliary Gaussian recovery count is an explicit diagnostic parameter;
these traces do not reproduce autonomous VPM evolution or prove force recovery.
Native FVM replay may consume the generated .04 s endpoints with its .008 s
linear endpoint interpolation, using the unchanged restricted initial histories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
from time import perf_counter

import h5py
import numpy as np

from tests.support.cylinder.audit_reference_particle_image import ReferenceParticleImage
from tests.support.cylinder.audit_saved_wall_circulation import gaussian_velocity_and_gradient
from tests.support.cylinder.capture_reference_volume_history import validate_volume_history
from tests.support.cylinder.run_boundary_condition_study import (
    balanced_trace,
    check_numerical_sources,
    digest,
)
from tests.support.cylinder.stream_planar_image_induction import PlanarImageInduction

SCHEMA = "openonda-cylinder-reference-particle-image/2"
SUPPORTS = ("production", "reference_support", "exact_reference")
TRACE_FIELDS = (
    "raw_velocity",
    "velocity",
    "jacobian",
    "normal_velocity",
    "tangential_gradient",
    "flux_correction",
)
MAXIMUM_AUXILIARY_TARGET_RATIO = 4.0
MAXIMUM_AUXILIARY_L1_RATIO = 10.0


def projection_frame_range(frame_count, *, start_frame=0, stop_frame=None, maximum_frames=None):
    """Select a disjoint half-open range in one qualified complete volume history."""
    stop_frame = frame_count if stop_frame is None else stop_frame
    if (
        isinstance(start_frame, bool)
        or not isinstance(start_frame, int)
        or isinstance(stop_frame, bool)
        or not isinstance(stop_frame, int)
        or not 0 <= start_frame < stop_frame <= frame_count
    ):
        raise ValueError("Projection range must satisfy 0 <= start < stop <= frame count")
    if maximum_frames is not None:
        if (
            isinstance(maximum_frames, bool)
            or not isinstance(maximum_frames, int)
            or not 1 <= maximum_frames <= stop_frame - start_frame
        ):
            raise ValueError("Requested frame bound is outside the qualified range")
        stop_frame = start_frame + maximum_frames
    return np.arange(start_frame, stop_frame, dtype=np.int64)


def auxiliary_image_admission(image_report):
    """Bound the diagnostic oracle independently of its residual denominator.

    These additional test-image admission limits are never solver settings or
    replacements for native capacity, moment, support or strength guards.
    """
    target = image_report["auxiliary_target"]
    rows = image_report["auxiliary_renewals"]
    target_maximum = float(target["maximum_absolute_axial_strength"])
    target_l1 = float(target["axial_strength_l1"])
    maxima = np.asarray([row["maximum_stored_axial_strength"] for row in rows])
    l1 = np.asarray([row["stored_strength_l1"] for row in rows])
    if (
        not np.isfinite([target_maximum, target_l1]).all()
        or not np.isfinite(maxima).all()
        or not np.isfinite(l1).all()
        or target_maximum <= 0
        or target_l1 <= 0
        or not len(rows)
    ):
        raise ValueError(
            "Cold image requires a finite nonzero simultaneous target and coefficients"
        )
    maximum_ratio = float(maxima.max() / target_maximum)
    l1_ratio = float(l1.max() / target_l1)
    if maximum_ratio > MAXIMUM_AUXILIARY_TARGET_RATIO or l1_ratio > MAXIMUM_AUXILIARY_L1_RATIO:
        raise ValueError(
            "Auxiliary image exceeds its target-relative admission limits: "
            f"maximum ratio={maximum_ratio:.6g}, L1 ratio={l1_ratio:.6g}"
        )
    if image_report["auxiliary_history_input_particle_count"] != 0:
        raise ValueError("Independent cold projection must begin with zero coefficients")
    return {
        "maximum_coefficient_to_target_ratio": maximum_ratio,
        "maximum_coefficient_l1_to_target_l1_ratio": l1_ratio,
        "maximum_coefficient_ratio_limit": MAXIMUM_AUXILIARY_TARGET_RATIO,
        "coefficient_l1_ratio_limit": MAXIMUM_AUXILIARY_L1_RATIO,
        "zero_initial_coefficients": True,
        "passed": True,
    }


def repeat_image_comparison(first, second):
    """Require deterministic coefficients after a second independent cold image."""
    result = {}
    for name in ("native_masked", "full_reference_native_masked"):
        position, strength, _rows = first[name]
        repeated_position, repeated_strength, _repeated_rows = second[name]
        equal = np.array_equal(position, repeated_position) and np.array_equal(
            strength, repeated_strength
        )
        if not equal:
            raise ValueError("Independent cold image changed when repeated: " + name)
        result[name] = {
            "particle_count": len(position),
            "positions_bitwise_equal": True,
            "strengths_bitwise_equal": True,
        }
    return result


def project(
    capture_directory,
    volume_directory,
    checkpoint_directory,
    directory,
    *,
    outer_repetitions=10,
    maximum_frames=None,
    start_frame=0,
    stop_frame=None,
    kernel_backend="cpu",
    projection_history="cold",
):
    capture_directory, volume_directory = Path(capture_directory), Path(volume_directory)
    checkpoint_directory, directory = Path(checkpoint_directory), Path(directory)
    if not directory.resolve().is_relative_to(Path("/tmp")):
        raise ValueError("Reference-image traces must use a private /tmp directory")
    if projection_history != "cold":
        raise ValueError("Warm auxiliary images are inadmissible; use independent cold projection")
    if kernel_backend not in ("cpu", "cuda"):
        raise ValueError("Image kernel backend must be cpu or cuda")
    if not isinstance(outer_repetitions, int) or not 1 <= outer_repetitions <= 25:
        raise ValueError("Frozen auxiliary recovery count must be between 1 and 25")
    information = json.loads((capture_directory / "inputs/inputs.json").read_text())
    volume_report = json.loads((volume_directory / "report.json").read_text())
    volume_path = volume_directory / "reference_volume_history.h5"
    if (
        volume_report.get("status") != "complete"
        or digest(volume_path) != volume_report["history_sha256"]
    ):
        raise ValueError("Reference volume history is incomplete or changed after capture")
    check_numerical_sources(information)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("audit_reference_particle_image.py"),
        Path(__file__).with_name("audit_saved_wall_circulation.py"),
        Path(__file__).with_name("capture_reference_volume_history.py"),
        Path(__file__).with_name("reference_cut_cell_deposition.py"),
        Path("source/coupler/vorticity_transfer.py"),
        Path("source/coupler/stable_renewal.py"),
        Path("source/coupler/interpolation.py"),
        Path("source/solvers/vpm/physics/induction/planar.py"),
        Path(__file__).with_name("stream_planar_image_induction.py"),
    ]
    source_hashes = {str(path): digest(path) for path in source_paths}
    configuration_path = checkpoint_directory / "checkpoint/checkpoint_info.json"
    configuration = json.loads(configuration_path.read_text())["config"]
    started = perf_counter()
    model = ReferenceParticleImage(
        capture_directory / "inputs", configuration, support="production"
    )
    construction_seconds = perf_counter() - started
    directory.mkdir(parents=True, exist_ok=False)
    output_path = directory / "particle_image_traces.h5"
    frame_reports = []
    with h5py.File(volume_path, "r") as volumes, h5py.File(output_path, "x") as stored:
        validation = validate_volume_history(volumes, information)
        if not np.array_equal(volumes["cell_centre"][:], model.geometry["cell_centre"]):
            raise ValueError(
                "Reference image donor ordering/geometry differs from qualified volume capture"
            )
        if volumes.attrs["reference_mesh_archive_sha256"] != digest(
            capture_directory / "inputs/reference_mesh.npz"
        ):
            raise ValueError("Reference image uses a different native mesh archive")
        indices = projection_frame_range(
            validation["frame_count"],
            start_frame=start_frame,
            stop_frame=stop_frame,
            maximum_frames=maximum_frames,
        )
        frames = len(indices)
        full_horizon = np.array_equal(indices, np.arange(validation["frame_count"]))
        points, normals, areas = (
            volumes[key][:] for key in ("face_centre", "face_normal", "face_area")
        )
        evaluator = (
            PlanarImageInduction(
                points,
                float(model.dtype(model.h * model.viscous["core_radius_ratio"])),
                model.span,
                model.dtype,
            )
            if kernel_backend == "cuda"
            else None
        )
        for key, values in (
            ("face_centre", points),
            ("face_normal", normals),
            ("face_area", areas),
        ):
            stored.create_dataset(key, data=values)
        stored.create_dataset("time", data=volumes["time"][indices])
        stored.create_dataset("frame_index", data=indices)
        stored.attrs.update(
            schema=SCHEMA,
            complete=False,
            projection_history="cold",
            kernel_backend=kernel_backend,
            auxiliary_state_policy="zero_coefficients_each_frame",
            auxiliary_search_policy="fixed internal nonphysical coefficient iterations; final atomic replacement guards enforced",
            qualified_reference_frame_count=validation["frame_count"],
            start_frame=int(indices[0]),
            stop_frame=int(indices[-1] + 1),
            configuration_sha256=digest(configuration_path),
            native_configuration=json.dumps(configuration, sort_keys=True),
            exchange_step_size=volumes.attrs["exchange_step_size"],
            fvm_step_size=volumes.attrs["fvm_step_size"],
            outer_repetitions=outer_repetitions,
            volume_history_sha256=volume_report["history_sha256"],
            mapped_initial_state_sha256=volumes.attrs["mapped_initial_state_sha256"],
            source_sha256=json.dumps(source_hashes, sort_keys=True),
            gradient_convention="J[i,j]=d(velocity_i)/d(x_j)",
            coefficient_storage_dtype=str(np.dtype(model.dtype)),
        )
        groups = {}
        for support in SUPPORTS:
            group = stored.create_group(support)
            for field, shape in {
                "raw_velocity": (len(points), 3),
                "velocity": (len(points), 3),
                "jacobian": (len(points), 3, 3),
                "normal_velocity": (len(points),),
                "tangential_gradient": (len(points), 3),
                "flux_correction": (),
            }.items():
                group.create_dataset(field, shape=(frames, *shape), dtype=np.float64)
            groups[support] = group
        cold_repeat_qualification = None
        for row, index in enumerate(indices):
            index = int(index)
            velocity, gradient = volumes["velocity"][index], volumes["velocity_gradient"][index]
            if not np.isfinite(velocity).all() or not np.isfinite(gradient).all():
                raise ValueError(f"Reference volume is nonfinite at frame {index}")
            variants, image_report, _target = model.sample(
                velocity,
                gradient,
                repetitions=1,
                outer_repetitions=outer_repetitions,
                include_cut_cells=False,
            )
            frame_report = {
                "index": index,
                "time": float(volumes["time"][index]),
                "image": image_report,
                "kernel_wall_seconds": {},
                "kernel_cpu_comparison": {},
                "flux_correction": {},
                "auxiliary_image_admission": auxiliary_image_admission(image_report),
            }
            if row == 0:
                repeated, repeated_report, _ = model.sample(
                    velocity,
                    gradient,
                    repetitions=1,
                    outer_repetitions=outer_repetitions,
                    include_cut_cells=False,
                )
                cold_repeat_qualification = {
                    "frame_index": index,
                    "time": float(volumes["time"][index]),
                    "coefficient_comparison": repeat_image_comparison(variants, repeated),
                    "repeated_admission": auxiliary_image_admission(repeated_report),
                    "repeat_wall_seconds": repeated_report["sample_wall_seconds"],
                    "passed": True,
                }
                del repeated, repeated_report
            for support, name in (
                ("production", "native_masked"),
                ("reference_support", "full_reference_native_masked"),
            ):
                positions, strengths, renewal_rows = variants[name]
                kernel_started = perf_counter()
                if evaluator is None:
                    induced, jacobian = gaussian_velocity_and_gradient(
                        points,
                        positions.astype(float),
                        strengths[:, 2].astype(float) / model.span,
                        float(model.dtype(model.h * model.viscous["core_radius_ratio"])),
                    )
                else:
                    induced, jacobian = evaluator.evaluate(
                        positions, strengths, check_cpu=(row == 0 or row == frames - 1)
                    )
                frame_report["kernel_cpu_comparison"][support] = (
                    dict(evaluator.validation)
                    if evaluator is not None and (row == 0 or row == frames - 1)
                    else None
                )
                raw_velocity = induced + model.freestream
                balanced, normal, tangent, balance = balanced_trace(
                    raw_velocity, jacobian, normals, areas
                )
                frame_report["kernel_wall_seconds"][support] = perf_counter() - kernel_started
                frame_report["flux_correction"][support] = balance
                frame_report[support + "_renewals"] = renewal_rows
                for field, values in {
                    "raw_velocity": raw_velocity,
                    "velocity": balanced,
                    "jacobian": jacobian,
                    "normal_velocity": normal,
                    "tangential_gradient": tangent,
                    "flux_correction": balance["normal_velocity_correction"],
                }.items():
                    groups[support][field][row] = values
            for field, source in (
                ("raw_velocity", "raw_velocity"),
                ("velocity", "outer_velocity"),
                ("jacobian", "jacobian"),
                ("normal_velocity", "normal_velocity"),
                ("tangential_gradient", "tangential_gradient"),
                ("flux_correction", "flux_correction"),
            ):
                groups["exact_reference"][field][row] = volumes[source][index]
            frame_reports.append(frame_report)
            stored.flush()
            print(
                f"Reference-image endpoint saved: {float(volumes['time'][index]):.6f} s", flush=True
            )
        if {str(path): digest(path) for path in source_paths} != source_hashes:
            raise ValueError(
                "Reference-image numerical/verification sources changed during projection"
            )
        check_numerical_sources(information)
        if digest(volume_path) != volume_report["history_sha256"]:
            raise ValueError("Qualified reference volume history changed during projection")
        stored.attrs["complete"] = True
        stored.attrs["full_reference_horizon_complete"] = full_horizon
        stored.attrs["cold_repeat_qualification"] = json.dumps(
            cold_repeat_qualification, sort_keys=True
        )
    report = {
        "status": "complete",
        "output": str(output_path),
        "output_sha256": digest(output_path),
        "frames": frames,
        "complete_qualified_horizon": full_horizon,
        "qualified_reference_frame_count": validation["frame_count"],
        "start_frame": int(indices[0]),
        "stop_frame": int(indices[-1] + 1),
        "cold_repeat_qualification": cold_repeat_qualification,
        "construction_wall_seconds": construction_seconds,
        "wall_seconds": perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "source_sha256": source_hashes,
        "volume_history_sha256": volume_report["history_sha256"],
        "volume_history_path": str(volume_path.resolve()),
        "configuration_sha256": digest(configuration_path),
        "configuration_path": str(configuration_path.resolve()),
        "native_configuration": configuration,
        "outer_repetitions": outer_repetitions,
        "kernel_backend": kernel_backend,
        "projection_history": projection_history,
        "native_cuda_cpu_comparison": None if evaluator is None else evaluator.validation,
        "frame_reports": frame_reports,
        "scope": "Simultaneous complete native reference field imaged at each accepted coupling endpoint; identical coefficients for both support traces. No VPM advection/diffusion or autonomous coupled force claim.",
        "limitations": [
            "Auxiliary full-reference renewal is a guarded finite Gaussian image, not exact inverse or evolved particle state.",
            "The finite reference outer blending belt excludes straddling exit curl and cannot reconstruct unknown wake beyond the reference domain.",
            "Small FVM replay must preserve the original restricted initial primary/BDF/face-flux fields and use the native linear .04→.008 endpoint schedule.",
        ],
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-directory", "volume-directory", "checkpoint-directory", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--outer-repetitions", type=int, default=10)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--kernel-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--projection-history", choices=("cold",), default="cold")
    args = parser.parse_args()
    report = project(
        args.capture_directory,
        args.volume_directory,
        args.checkpoint_directory,
        args.directory,
        outer_repetitions=args.outer_repetitions,
        maximum_frames=args.maximum_frames,
        start_frame=args.start_frame,
        stop_frame=args.stop_frame,
        kernel_backend=args.kernel_backend,
        projection_history=args.projection_history,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("frames", "wall_seconds", "process_peak_rss_bytes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
