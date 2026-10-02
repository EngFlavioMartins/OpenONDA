"""UNWIRED native finite-image compact-query reuse screen; no advancement.

Uses every authenticated source and initial particle target, then three
disjoint native subsets and projected lower/upper-plane queries. Each wall
set is evaluated both at the exact f64 configured plane and at the f32
roundtrip coordinates needed for a later identical-target FMM comparison.
These are finite image-only fields, with local correction already included.
No primary field, wall cancellation assertion, or tail approval is inferred.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._cupy_slab_compact_fields import CompactSlabFieldMeshGPU
from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(current_u, current_j, expected_u, expected_j):
    result = {}
    for name, actual, expected in (("velocity", current_u, expected_u),
                                    ("gradient", current_j, expected_j)):
        actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("invalid comparison fields")
        delta = actual-expected
        normal = float(np.linalg.norm(expected))
        result[name] = {"absolute_l2": float(np.linalg.norm(delta)),
                        "relative_l2": float(np.linalg.norm(delta)/normal) if normal else None,
                        "max_absolute_component": float(np.max(np.abs(delta), initial=0)),
                        "max_point_norm": float(np.linalg.norm(delta.reshape(len(delta), -1), axis=1).max(initial=0))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True, help="Existing finite513 f32 coalesced JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("bounded1..3 query repeats required")
    root = Path(__file__).resolve().parents[2]
    solution = root / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output, archive = args.output.resolve(), args.output.resolve().with_suffix(".npz")
    if output.parent != solution or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("new JSON/NPZ evidence required in ordinary cylinder solution")
    names = ("_profile_cupy_slab_compact.py", "_cupy_slab_compact_fields.py",
             "_cupy_slab_field_mesh.py", "_cupy_slab_field_mesh_fused.py",
             "_cupy_cardinal_stencil.py", "_gaussian_core_correction_gpu.py",
             "_finite_slab_native_qualification.py", "_finite_slab_field_mesh_reference.py",
             "_finite_image_mesh_reference.py", "_slip_periodic_gaussian_oracle.py")
    hashes = {str(Path(__file__).with_name(name)): _digest(Path(__file__).with_name(name)) for name in names}
    report = {"status": "running", "production_admissible": False, "tail_certified": False,
              "scope": "immutable-source finite513 images plus local correction; no primary/solver/tail approval",
              "source_hashes": hashes, "measurements": []}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    engine, fields = None, {}
    started = time.perf_counter()
    try:
        import cupy as cp

        data = authenticate_inputs(args.checkpoint, args.oracle)
        x, gamma, sigma = data["position"], data["strength"], data["core"]
        slab = data["identity"]["configuration"]["induction"]
        images = [item for block in data["blocks"] for item in block["images"]]
        report.update(identity=data["identity"], oracle_provenance=data["oracle_provenance"],
                      admission_helper_sha256=data["admission_helper_sha256"])
        control_path = args.control.resolve()
        control = json.loads(control_path.read_text())
        control_npz = control_path.with_suffix(".npz")
        params = control["parameters"]
        if (control["status"] != "complete" or control["identity"] != data["identity"]
                or not params["coalesce_finite_blocks"] or params["dtype"] != "float32"
                or params["correction_dtype"] != "float32" or params["blocks"] != "all"
                or params["spacing"] != .035 or _digest(control_npz) != control["archive_sha256"]):
            raise ValueError("control identity, finite operator, parameters or digest mismatch")
        with np.load(control_npz, allow_pickle=False) as saved:
            for label, actual in (("source_position", x), ("source_strength", gamma), ("source_core", sigma)):
                if not np.array_equal(saved[label], actual):
                    raise ValueError("control source arrays differ")
            expected_u, expected_j = saved["velocity"], saved["gradient"]
        report["control"] = {"json": str(control_path), "json_sha256": _digest(control_path),
                             "npz_sha256": _digest(control_npz)}
        if len(x) < 3*8192:
            raise ValueError("native query census requires three disjoint8192-source subsets")
        indices = data["oracle"]["indices"].astype(np.int64)
        exact = data["oracle"]["direct"]
        queries = [("initial_full", np.asarray(x, np.float64), np.arange(len(x), dtype=np.int64))]
        for number, first in enumerate((0, len(x)//2, len(x)-8192)):
            subset = np.arange(first, first+8192, dtype=np.int64)
            queries.append((f"subset_{number}", np.asarray(x[subset], np.float64), subset))
        wall_indices = indices[:24]
        for label, plane in (("lower", slab["z_min"]), ("upper", slab["z_max"])):
            intended = np.array(x[wall_indices], dtype=np.float64, copy=True)
            intended[:, 2] = plane
            rounded = intended.astype(np.float32).astype(np.float64)
            fields[f"wall_{label}_source_indices"] = wall_indices
            fields[f"wall_{label}_intended_position"] = intended
            queries.extend(((f"wall_{label}_f64", intended, None),
                            (f"wall_{label}_f32", rounded, None)))
        free, total = cp.cuda.runtime.memGetInfo()
        if free < 3*1024**3:
            raise MemoryError("at least3GiB free device memory required")
        report["initial_device_memory"] = {"free": free, "total": total}
        report["runtime"] = {"cupy": cp.__version__, "cupy_path": cp.__file__}
        clock = time.perf_counter()
        engine = CompactSlabFieldMeshGPU(x, gamma, sigma, x, zmin=slab["z_min"], zmax=slab["z_max"],
                                        tau=.12, spacing=.035, order=10, cutoff=.6,
                                        dtype="float32", correction_dtype="float32", stencil_backend="gpu")
        report["scope_setup_seconds"] = time.perf_counter()-clock
        report["prepare"] = engine.prepare(images)
        for label, positions, source_indices in queries:
            repeats = 1 if label == "initial_full" else args.repeats
            for repeat in range(repeats):
                if time.perf_counter()-started > 550:
                    raise TimeoutError("bounded native compact screen allowance exhausted")
                u, j, diagnostics = engine.evaluate_prepared(positions)
                copy_start = time.perf_counter()
                host_u, host_j = cp.asnumpy(u), cp.asnumpy(j)
                copy_seconds = time.perf_counter()-copy_start
                del u, j
                item = {"query": label, "repeat": repeat, "operator": diagnostics,
                        "output_transfer_seconds": copy_seconds}
                if source_indices is not None:
                    item["same_finite_control_error"] = _summary(host_u, host_j,
                        expected_u[source_indices], expected_j[source_indices])
                    shared, query_rows, oracle_rows = np.intersect1d(source_indices, indices, return_indices=True)
                    if len(shared):
                        item["direct_oracle_indices"] = shared.tolist()
                        item["selected_direct_error"] = compare_fields(
                            host_u[query_rows], host_j[query_rows],
                            exact[oracle_rows, :3], exact[oracle_rows, 3:12].reshape(-1, 3, 3))
                if repeat == repeats-1:
                    fields[f"{label}_position"] = positions
                    fields[f"{label}_velocity"] = host_u
                    fields[f"{label}_gradient"] = host_j
                    if source_indices is not None:
                        fields[f"{label}_source_indices"] = source_indices
                report["measurements"].append(item)
                print(json.dumps({"query": label, "repeat": repeat, "count": len(positions),
                                  "seconds": diagnostics["query_seconds"],
                                  "gather_seconds": diagnostics["gather_seconds"],
                                  "correction_seconds": diagnostics["correction"]["query_seconds"]}), flush=True)
                publish()
        clock = time.perf_counter()
        engine.close()
        report["scope_teardown_seconds"] = time.perf_counter()-clock
        engine = None
        fields["oracle_indices"] = indices
        with archive.open("xb") as stream:
            np.savez(stream, **fields)
        report.update(status="complete", archive_sha256=_digest(archive),
                      checkpoint_unchanged=_digest(args.checkpoint)==data["identity"]["checkpoint_sha256"],
                      elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if engine is not None:
            try:
                engine.close()
            except BaseException as error:
                report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        report["sources_changed"] = [name for name, digest in hashes.items() if _digest(name) != digest]
        if report["sources_changed"]:
            report["status"] = "failed"
        publish()


if __name__ == "__main__":
    main()
