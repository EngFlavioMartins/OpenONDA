"""UNWIRED native three-family stencil preparation cost/identity comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._cupy_cardinal_stencil import cardinal_stencil_gpu
from tests.vpm._cupy_slab_field_mesh import _weights
from tests.vpm._finite_slab_field_mesh_reference import slab_coordinates
from tests.vpm._finite_slab_native_qualification import authenticate_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    solution = root / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output = args.output.resolve()
    if output.parent != solution or output.suffix != ".json" or output.exists():
        raise ValueError("new normal-solution JSON evidence required")
    names = ("_profile_cupy_cardinal_stencil.py", "_cupy_cardinal_stencil.py", "_cupy_slab_field_mesh.py",
             "_finite_slab_field_mesh_reference.py", "_finite_slab_native_qualification.py")
    hashes = {str(Path(__file__).with_name(name)): hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in names}
    report = {"status": "running", "production_admissible": False,
              "scope": "Three native source-even/source-odd/target stencils, no field solve or solver advancement",
              "source_hashes": hashes, "measurements": []}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        import cupy as cp

        data = authenticate_inputs(args.checkpoint, args.oracle)
        physical = data["position"]
        slab = data["identity"]["configuration"]["induction"]
        start = time.perf_counter()
        x, steps, cells = slab_coordinates(physical, slab["z_min"], slab["z_max"], .035)
        q, _, _ = slab_coordinates(physical, slab["z_min"], slab["z_max"], .035)
        reflected = x.copy()
        reflected[:, 2] *= -1
        points = np.concatenate((x, reflected, q))
        origin = np.floor(points.min(0))-10
        shape = tuple((np.ceil(points.max(0)-origin)+11).astype(np.int64))
        geometry_seconds = time.perf_counter()-start
        arrays = {"source_even": x, "source_odd": reflected, "target": q}
        references, host_seconds = {}, {}
        for name, points in arrays.items():
            start = time.perf_counter()
            references[name] = _weights(points, origin, 10, shape)
            host_seconds[name] = time.perf_counter()-start
        report.update(identity=data["identity"], slab_cells=cells, spacing=steps.tolist(),
                      shape=list(map(int, shape)), origin=origin.tolist(),
                      common_host_geometry_seconds=geometry_seconds,
                      reference_cpu_seconds=host_seconds, reference_cpu_total_seconds=sum(host_seconds.values()),
                      max_scope_bytes=256*1024**2)
        for dtype in ("float32", "float64"):
            pool = cp.cuda.MemoryPool()
            pool.set_limit(size=256*1024**2)
            for repeat in range(3):
                control_cpu, control_upload, candidate_prepare, validate_seconds = 0., 0., 0., 0.
                families = []
                for name, points in arrays.items():
                    first, reference = references[name]
                    expected = reference.astype(dtype)
                    start = time.perf_counter()
                    current_first, current_weights = _weights(points, origin, 10, shape)
                    control_cpu += time.perf_counter()-start
                    cp.cuda.get_current_stream().synchronize()
                    start = time.perf_counter()
                    with cp.cuda.using_allocator(pool.malloc):
                        first_control, weights_control = cp.asarray(current_first), cp.asarray(current_weights, dtype=dtype)
                    cp.cuda.get_current_stream().synchronize()
                    control_upload += time.perf_counter()-start
                    del first_control, weights_control, current_first, current_weights
                    first_d, weights_d, diagnostics = cardinal_stencil_gpu(points, origin, 10, shape, dtype=dtype, pool=pool)
                    candidate_prepare += diagnostics["wall_seconds"]
                    start = time.perf_counter()
                    actual_first, actual = cp.asnumpy(first_d), cp.asnumpy(weights_d)
                    delta = np.asarray(actual, dtype=np.float64)-np.asarray(expected, dtype=np.float64)
                    first_exact = bool(np.array_equal(actual_first, first))
                    max_error = float(np.max(np.abs(delta), initial=0))
                    epsilon = np.finfo(dtype).eps
                    if not first_exact or not np.allclose(actual, expected, rtol=8*epsilon, atol=2*epsilon):
                        raise AssertionError("native GPU cardinal product differs beyond preregistered roundoff allowance")
                    families.append({"family": name, "first_exact": first_exact,
                                     "weights_exact": bool(np.array_equal(actual, expected)),
                                     "max_absolute_weight_difference": max_error,
                                     "weight_difference_l2": float(np.linalg.norm(delta)),
                                     "max_partition_sum_error": float(np.abs(actual.sum(2)-1).max()),
                                     "operator": diagnostics})
                    validate_seconds += time.perf_counter()-start
                    del first_d, weights_d, actual, actual_first, delta, expected
                item = {"dtype": dtype, "repeat": repeat, "control_cpu_and_upload_seconds": control_cpu+control_upload,
                        "control_cpu_seconds": control_cpu, "control_upload_seconds": control_upload,
                        "candidate_three_family_seconds": candidate_prepare,
                        "validation_output_copy_seconds": validate_seconds, "families": families,
                        "pool_high_water_bytes": pool.total_bytes()}
                report["measurements"].append(item)
                print(json.dumps({key: value for key, value in item.items() if key != "families"}), flush=True)
                publish()
            pool.free_all_blocks()
        report.update(status="complete", checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==data["identity"]["checkpoint_sha256"])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["sources_changed"] = [name for name, digest in hashes.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
        if report["sources_changed"]:
            report["status"]="failed"
        publish()
    if report["status"] != "complete":
        raise RuntimeError("qualification input/source changed")


if __name__ == "__main__":
    main()
