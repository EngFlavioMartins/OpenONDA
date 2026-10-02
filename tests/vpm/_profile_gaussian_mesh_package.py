"""Read-only native extraction parity/ownership screen, never a solver run."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields
from tests.vpm._profile_cupy_slab_compact import _summary


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(path, identity):
    path = Path(path).resolve().with_suffix(".json")
    record = json.loads(path.read_text())
    archive = path.with_suffix(".npz")
    if (record.get("status") != "complete" or record.get("identity") != identity
            or not record.get("checkpoint_unchanged") or record.get("sources_changed")
            or digest(archive) != record.get("archive_sha256")):
        raise ValueError("prior evidence identity/digest/immutability mismatch")
    with np.load(archive, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    return arrays, {"json": str(path), "json_sha256": digest(path), "npz_sha256": digest(archive)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "oracle", "compact", "coherent", "primary-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    solution = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output, archive = args.output.resolve(), args.output.resolve().with_suffix(".npz")
    if output.parent != solution or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("new package evidence required in ordinary solution directory")
    package = root/"source/solvers/vpm/physics/induction/gaussian_mesh"
    files = [package/name for name in ("__init__.py", "coordinates.py", "correction.py",
                                      "fields.py", "runtime.py", "stencil.py")]
    files += [Path(__file__), Path(__file__).with_name("_finite_slab_native_qualification.py"),
              Path(__file__).with_name("_profile_cupy_slab_compact.py")]
    hashes = {str(path): digest(path) for path in files}
    report = {"status": "running", "runtime_admissible": False, "tail_certified": False,
              "source_hashes": hashes, "scopes": [],
              "scope": "finite image/coherent source-only fields; no primary particle replacement or advancement"}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    owner, arrays = None, {}
    try:
        import cupy as cp

        data = authenticate_inputs(args.checkpoint, args.oracle)
        x, gamma, sigma = data["position"], data["strength"], data["core"]
        identity, indices = data["identity"], data["oracle"]["indices"].astype(np.int64)
        slab = identity["configuration"]["induction"]
        report.update(identity=identity, oracle_provenance=data["oracle_provenance"])
        compact, compact_evidence = evidence(args.compact, identity)
        coherent, coherent_evidence = evidence(args.coherent, identity)
        primary, primary_evidence = evidence(args.primary_audit, identity)
        report["controls"] = {"compact": compact_evidence, "coherent": coherent_evidence,
                              "primary": primary_evidence}
        if (not np.array_equal(primary["indices"], indices)
                or not np.array_equal(primary["position"], x[indices])
                or not np.all(sigma == sigma[0])):
            raise ValueError("saved primary oracle requires identical actual native cores")
        images = [image for block in data["blocks"] for image in block["images"]]
        # This unrelated client stays live throughout both package owners.
        # Our explicit plans must neither borrow nor clear its cache entry.
        foreign = cp.arange(8*9*10, dtype=cp.float32).reshape(8, 9, 10)
        foreign_fft = cp.fft.rfftn(foreign)
        foreign_saved = cp.asnumpy(foreign_fft)
        cache = cp.fft.config.get_plan_cache()
        cache_before = (cache.get_curr_size(), cache.get_curr_memsize(), cache.get_size(), cache.get_memsize())
        allocator_before = cp.cuda.get_allocator()
        report["foreign_fft_cache_before"] = cache_before
        for source_only in (False, True):
            label = "source_only" if source_only else "images"
            reference = coherent if source_only else compact
            descriptors = [(0, False), *images] if source_only else images
            queries = []
            if not source_only:
                queries.append(("initial_full", np.asarray(x, np.float64), np.arange(len(x))))
            queries.append(("oracle24", np.asarray(x[indices], np.float64), indices))
            for number in range(3):
                name = f"subset_{number}"
                selected = compact[name+"_source_indices"]
                q = compact[name+"_position"]
                if len(q) != 8192 or not np.array_equal(q, x[selected]):
                    raise ValueError("saved subset positions differ")
                queries.append((name, q, selected))
            for side in ("lower", "upper"):
                for precision in ("f64", "f32"):
                    name = f"wall_{side}_{precision}"
                    queries.append((name, compact[name+"_position"], None))
            started = time.perf_counter()
            owner = GaussianImageFields(
                x, gamma, sigma, x, zmin=slab["z_min"], zmax=slab["z_max"],
                tau=.12, spacing=.035, cutoff=.6, order=10, dtype="float32",
                correction_dtype="float32", source_only_primary=source_only,
                max_images=len(descriptors))
            scope = {"label": label, "setup_seconds": time.perf_counter()-started,
                     "prepare": owner.prepare(descriptors), "queries": []}
            report["scopes"].append(scope)
            for name, q, selected in queries:
                for repeat in range(1 if name == "initial_full" else 2):
                    u, j, diagnostics = owner.evaluate_prepared(q)
                    copied = time.perf_counter()
                    host_u, host_j = cp.asnumpy(u), cp.asnumpy(j)
                    copy_seconds = time.perf_counter()-copied
                    del u, j
                    item = {"query": name, "repeat": repeat, "operator": diagnostics,
                            "output_transfer_seconds": copy_seconds}
                    if name+"_velocity" in reference:
                        item["extraction_error"] = _summary(host_u, host_j,
                            reference[name+"_velocity"], reference[name+"_gradient"])
                    elif name == "oracle24" and not source_only:
                        item["extraction_error"] = _summary(host_u, host_j,
                            compact["initial_full_velocity"][indices], compact["initial_full_gradient"][indices])
                    if selected is not None:
                        shared, rows, oracle_rows = np.intersect1d(selected, indices, return_indices=True)
                        if len(shared):
                            exact_u = primary["exact_velocity"] if source_only else data["oracle"]["direct"][:, :3]
                            exact_j = primary["exact_gradient"] if source_only else data["oracle"]["direct"][:, 3:12].reshape(-1, 3, 3)
                            item["direct_error"] = compare_fields(host_u[rows], host_j[rows],
                                                                   exact_u[oracle_rows], exact_j[oracle_rows])
                    if name.startswith("wall_"):
                        item["normal_max_absolute"] = float(np.max(np.abs(host_u[:, 2])))
                    arrays[f"{label}_{name}_position"] = q
                    arrays[f"{label}_{name}_velocity"] = host_u
                    arrays[f"{label}_{name}_gradient"] = host_j
                    scope["queries"].append(item)
                    print(json.dumps({"scope": label, "query": name, "repeat": repeat,
                                      "seconds": diagnostics["query_seconds"],
                                      "normal": item.get("normal_max_absolute")}), flush=True)
                    publish()
            started = time.perf_counter()
            owner.close()
            owner = None
            scope["teardown_seconds"] = time.perf_counter()-started
            if (cache.get_curr_size(), cache.get_curr_memsize(), cache.get_size(), cache.get_memsize()) != cache_before:
                raise AssertionError("foreign FFT cache changed")
            if cp.cuda.get_allocator() is not allocator_before:
                raise AssertionError("foreign allocator changed")
            np.testing.assert_array_equal(cp.asnumpy(cp.fft.rfftn(foreign)), foreign_saved)
        report["foreign_fft_cache_unchanged"] = True
        report["foreign_allocator_and_live_fields_unchanged"] = True
        with archive.open("xb") as stream:
            np.savez(stream, **arrays, oracle_indices=indices)
        report.update(status="complete", archive_sha256=digest(archive),
                      checkpoint_unchanged=digest(args.checkpoint)==identity["checkpoint_sha256"])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if owner is not None:
            try:
                owner.close()
            except BaseException as error:
                report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        report["sources_changed"] = [name for name, before in hashes.items() if digest(name) != before]
        if report["sources_changed"]:
            report["status"] = "failed"
        publish()


if __name__ == "__main__":
    main()
