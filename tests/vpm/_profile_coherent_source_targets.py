"""UNWIRED authenticated native whole SOURCE-ONLY target accuracy/cost screen.

No particle advancement or particle-stage substitution. The common finite
operator contains primary once plus513images. Existing complete direct fields
at24 native points and exact wall-normal pairing serve as independent oracles.
No finite/infinite tail or runtime tolerance certificate is inferred.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._cupy_coherent_source_targets import CoherentSourceTargetsGPU
from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_evidence(prefix, identity):
    prefix = Path(prefix).resolve().with_suffix("")
    report = json.loads(prefix.with_suffix(".json").read_text())
    archive = prefix.with_suffix(".npz")
    if (report.get("status") != "complete" or report.get("identity") != identity
            or not report.get("checkpoint_unchanged") or report.get("sources_changed")
            or _digest(archive) != report.get("archive_sha256")):
        raise ValueError("prior evidence identity, immutability or digest mismatch")
    with np.load(archive, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "oracle", "compact", "wall-audit", "primary-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    solution = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output = args.output.resolve()
    archive = output.with_suffix(".npz")
    if output.parent != solution or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("new evidence paths required in ordinary solution directory")
    names = ("_profile_coherent_source_targets.py", "_cupy_coherent_source_targets.py",
             "_cupy_slab_compact_fields.py", "_cupy_slab_field_mesh_fused.py",
             "_cupy_slab_field_mesh.py", "_cupy_cardinal_stencil.py",
             "_gaussian_core_correction_gpu.py", "_finite_slab_native_qualification.py",
             "_finite_image_mesh_reference.py", "_finite_slab_field_mesh_reference.py",
             "_slip_periodic_gaussian_oracle.py")
    hashes = {str(Path(__file__).with_name(name)): _digest(Path(__file__).with_name(name)) for name in names}
    report = {"status": "running", "production_admissible": False, "tail_certified": False,
              "scope": "source-only arbitrary targets; primary once +513images; no particle-stage substitution",
              "source_hashes": hashes, "measurements": []}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    engine = None
    arrays = {}
    try:
        import cupy as cp

        data = authenticate_inputs(args.checkpoint, args.oracle)
        x, gamma, sigma = data["position"], data["strength"], data["core"]
        slab = data["identity"]["configuration"]["induction"]
        report["identity"] = data["identity"]
        report["oracle_provenance"] = data["oracle_provenance"]
        compact_report, compact = _load_evidence(args.compact, data["identity"])
        wall_report, wall = _load_evidence(args.wall_audit, data["identity"])
        _, primary = _load_evidence(args.primary_audit, data["identity"])
        indices = data["oracle"]["indices"].astype(np.int64)
        if (not np.array_equal(primary["indices"], indices)
                or not np.array_equal(primary["position"], x[indices])
                or not np.array_equal(primary["core"], sigma[indices])
                or not np.all(sigma == sigma[0])):
            raise ValueError("saved pair-mean direct primary requires identical actual source cores here")
        # This is an oracle-equivalence check for this fixed input, not a
        # particle runtime dispatch rule: every actual source core is identical.
        report["oracle_primary_equivalence"] = "all actual source cores bitwise equal; saved pair mean equals source-only"
        if (compact_report["prepare"]["finite_image_count"] != 513
                or not wall_report.get("source_fields_unchanged")
                or not wall_report.get("imported_sources_unchanged")):
            raise ValueError("saved compact/wall contract differs")
        images = [item for block in data["blocks"] for item in block["images"]]
        queries = [("oracle24", np.asarray(x[indices], np.float64), indices)]
        for number in range(3):
            label = f"subset_{number}"
            q, selected = compact[label+"_position"], compact[label+"_source_indices"]
            if len(q) != 8192 or not np.array_equal(q, x[selected]):
                raise ValueError("saved8192 query positions differ")
            queries.append((label, q, selected))
        for side in ("lower", "upper"):
            for precision in ("f64", "f32"):
                label = f"wall_{side}_{precision}"
                queries.append((label, compact[label+"_position"], None))
        free, total = cp.cuda.runtime.memGetInfo()
        if free < 3*1024**3:
            raise MemoryError("at least3GiB free GPU memory required")
        report["initial_device_memory"] = {"free": free, "total": total}
        started = time.perf_counter()
        engine = CoherentSourceTargetsGPU(
            x, gamma, sigma, x, include_physical_primary=True,
            zmin=slab["z_min"], zmax=slab["z_max"], tau=.12, spacing=.035,
            order=10, cutoff=.6, dtype="float32", correction_dtype="float32", stencil_backend="gpu")
        report["construction_seconds"] = time.perf_counter()-started
        report["prepare"] = engine.prepare(images)
        for label, q, selected in queries:
            for repeat in range(2):
                started = time.perf_counter()
                u, j, diagnostics = engine.evaluate_prepared(q)
                host_u, host_j = cp.asnumpy(u), cp.asnumpy(j)
                seconds = time.perf_counter()-started
                del u, j
                item = {"query": label, "repeat": repeat, "query_and_copy_seconds": seconds,
                        "operator": diagnostics}
                if selected is not None:
                    shared, rows, oracle_rows = np.intersect1d(selected, indices, return_indices=True)
                    if len(shared):
                        item["direct_oracle_indices"] = shared.tolist()
                        item["independent_direct_error"] = compare_fields(host_u[rows], host_j[rows],
                            primary["exact_velocity"][oracle_rows], primary["exact_gradient"][oracle_rows])
                if label.startswith("wall_"):
                    _, side, precision = label.split("_")
                    side_index = 0 if side == "lower" else 1
                    if precision == "f64":
                        true_normal = np.zeros(len(q)) if side == "lower" else wall["true_finite_upper_normal"]
                        item["finite_geometric_normal_error_max"] = float(np.max(np.abs(host_u[:, 2]-true_normal)))
                    else:
                        # Rounded queries differ from the exact plane by~1e−8m.
                        # Keep this observed comparator separate from exact truth.
                        old_normal = wall["old_slab_velocity"][side_index*24:(side_index+1)*24, 2]
                        item["old_slab_normal_max"] = float(np.max(np.abs(old_normal)))
                        item["difference_from_old_slab_normal_max"] = float(np.max(np.abs(host_u[:, 2]-old_normal)))
                    item["normal_max_absolute"] = float(np.max(np.abs(host_u[:, 2])))
                if repeat == 1:
                    arrays[label+"_position"] = q
                    arrays[label+"_velocity"] = host_u
                    arrays[label+"_gradient"] = host_j
                report["measurements"].append(item)
                print(json.dumps({"query": label, "repeat": repeat, "seconds": seconds,
                                  "normal": item.get("normal_max_absolute"),
                                  "direct": item.get("independent_direct_error")}), flush=True)
                publish()
        started = time.perf_counter()
        engine.close()
        engine = None
        report["teardown_seconds"] = time.perf_counter()-started
        with archive.open("xb") as stream:
            np.savez(stream, **arrays)
        report.update(status="complete", archive_sha256=_digest(archive),
                      checkpoint_unchanged=_digest(args.checkpoint)==data["identity"]["checkpoint_sha256"])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if engine is not None:
            engine.close()
        report["sources_changed"] = [name for name, digest in hashes.items() if _digest(name) != digest]
        if report["sources_changed"]:
            report["status"] = "failed"
        publish()


if __name__ == "__main__":
    main()
