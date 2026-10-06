"""Compare bounded native CUDA planar image sums with saved independent CPU data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from tests.support.cylinder.run_boundary_condition_study import digest
from tests.support.cylinder.stream_planar_image_induction import PlanarImageInduction


def qualify(image_path, output_path):
    image_path, output_path = Path(image_path), Path(output_path)
    report_path = image_path.with_suffix(".json")
    report = json.loads(report_path.read_text())
    if not report.get("sources_unchanged"):
        raise ValueError("Frozen CPU image does not have unchanged-source proof")
    image_hash = digest(image_path)
    started = perf_counter()
    sources = [
        Path(__file__),
        Path(__file__).with_name("stream_planar_image_induction.py"),
        Path("source/solvers/vpm/physics/induction/planar.py"),
    ]
    hashes = {str(path): digest(path) for path in sources}
    with np.load(image_path) as stored:
        points = stored["face_centre"]
        evaluator = PlanarImageInduction(
            points, report["core_radius"], 1.0, np.dtype(report["output_dtype"])
        )
        measurements = {}
        for name in ("native_masked", "full_reference_native_masked"):
            kernel_started = perf_counter()
            velocity, jacobian = evaluator.evaluate(
                stored[name + "_position"], stored[name + "_strength"], check_cpu=True
            )
            total_velocity = velocity + np.array([1.0, 0.0, 0.0])
            saved_error = {
                "maximum_saved_velocity_absolute_error": float(
                    np.max(np.abs(total_velocity - stored[name + "_velocity"]))
                ),
                "maximum_saved_jacobian_absolute_error": float(
                    np.max(np.abs(jacobian - stored[name + "_jacobian"]))
                ),
            }
            if max(saved_error.values()) > 2e-6:
                raise ValueError("Native CUDA field disagrees with the exact saved CPU image")
            measurements[name] = {
                "cpu_comparison": evaluator.validation,
                "saved_cpu_comparison": saved_error,
                "wall_seconds": perf_counter() - kernel_started,
            }
    if digest(image_path) != image_hash or {str(path): digest(path) for path in sources} != hashes:
        raise ValueError("Frozen input or numerical/projection sources changed during CUDA audit")
    receipt = {
        "status": "passed",
        "image_sha256": image_hash,
        "image_report_sha256": digest(report_path),
        "source_sha256": hashes,
        "measurements": measurements,
        "wall_seconds": perf_counter() - started,
        "scope": "356 face targets, actual frozen FP32 source arrays/core, bounded native PlanarInduction CUDA sum in FP64 compared against independent streamed CPU and saved CPU fields. No solver or evolved-force claim.",
    }
    output_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.image, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
