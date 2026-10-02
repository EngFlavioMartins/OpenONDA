"""Bounded saved-cloud GPU local-correction qualification, not a solver run."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields
from tests.vpm._gaussian_core_correction_gpu import GaussianCoreCorrectionGPU, _possibly_near
from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


def independent_selected(x, gamma, sigma, q, images, tau=.12, cutoff=.6):
    """Different f64 incomplete-gamma/series oracle; scalar cutoff, no cells."""
    u, j = np.zeros((len(q), 3)), np.zeros((len(q), 3, 3))
    candidate_count = 0
    for shift, odd in images:
        if not _possibly_near(x.min(0), x.max(0), q.min(0), q.max(0), shift, odd, cutoff):
            continue
        for start in range(0, len(x), 16_384):
            xx = np.asarray(x[start:start+16_384], dtype=np.float64).copy()
            gg = np.asarray(gamma[start:start+16_384], dtype=np.float64).copy()
            ss = np.asarray(sigma[start:start+16_384], dtype=np.float64)
            if odd:
                xx[:, 2] *= -1
                gg[:, :2] *= -1
            xx[:, 2] += shift
            for t, point in enumerate(q):
                displacement = point-xx
                keep = np.sum(displacement*displacement, axis=1) < cutoff*cutoff
                if not keep.any():
                    continue
                candidate_count += int(keep.sum())
                narrow_u, narrow_j = gaussian_pairs(displacement[keep], gg[keep], ss[keep])
                broad_u, broad_j = gaussian_pairs(displacement[keep], gg[keep], tau)
                u[t] += (narrow_u-broad_u).sum(0)
                j[t] += (narrow_j-broad_j).sum(0)
    return u, j, candidate_count


def _summary(u, j, expected_u, expected_j):
    result = {}
    for name, actual, expected in (("velocity", u, expected_u), ("gradient", j, expected_j)):
        delta = np.asarray(actual, dtype=np.float64)-np.asarray(expected, dtype=np.float64)
        result[name] = {"absolute_l2": float(np.linalg.norm(delta)),
                        "relative_l2": float(np.linalg.norm(delta)/np.linalg.norm(expected)),
                        "max_absolute_component": float(np.abs(delta).max(initial=0))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=850)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3 or not 0 < args.max_seconds <= 850:
        parser.error("bounded1..3 repeats and <=850s internal allowance required")
    root = Path(__file__).resolve().parents[2]
    solution = root / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output = args.output.resolve()
    archive = output.with_suffix(".npz")
    if output.parent != solution or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("new JSON/NPZ evidence required in normal cylinder solution directory")
    names = ("_profile_gaussian_core_correction.py", "_gaussian_core_correction_gpu.py",
             "_finite_slab_native_qualification.py", "_slip_periodic_gaussian_oracle.py")
    hashes = {str(Path(__file__).with_name(name)): hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in names}
    report = {"status": "running", "production_admissible": False, "source_hashes": hashes,
              "parameters": {"tau": .12, "cutoff": .6, "strict_cutoff": True, "repeats": args.repeats},
              "measurements": [], "scope": "finite image-only LOCAL correction; excludes smooth field and solver"}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    started = time.perf_counter()
    fields = {}
    try:
        import cupy as cp

        data = authenticate_inputs(args.checkpoint, args.oracle)
        x, gamma, sigma = data["position"], data["strength"], data["core"]
        slab = data["identity"]["configuration"]["induction"]
        zmin, width = slab["z_min"], slab["z_max"]-slab["z_min"]
        images = [(2*k*width+(2*zmin if odd else 0), odd) for block in data["blocks"] for k, odd in block["images"]]
        indices = data["oracle"]["indices"]
        query = np.asarray(x, dtype=np.float64)
        report.update(identity=data["identity"], oracle_provenance=data["oracle_provenance"],
                      selected_indices=indices.tolist(), finite_images=len(images), checkpoint=str(args.checkpoint.resolve()),
                      direct_oracle="Independent incomplete-gamma narrow minus broad kernels, strict cutoff, no cell index")
        reference_start = time.perf_counter()
        expected = {}
        for label, current_images in (("first", images[:1]), ("all", images)):
            expected[label] = independent_selected(x, gamma, sigma, query[indices], current_images)
        report["selected_independent_seconds"] = time.perf_counter()-reference_start
        free, total = cp.cuda.runtime.memGetInfo()
        if free < 1024**3:
            raise MemoryError("at least1GiB free device memory required")
        report["initial_device_memory"] = {"free": free, "total": total}
        for dtype in ("float32", "float64"):
            build_started = time.perf_counter()
            with GaussianCoreCorrectionGPU(x, gamma, sigma, tau=.12, cutoff=.6,
                                           accumulation_dtype=dtype, max_scratch_bytes=256*1024**2) as owner:
                build_seconds = time.perf_counter()-build_started
                for label, current_images in (("first", images[:1]), ("all", images)):
                    for repeat in range(args.repeats):
                        if time.perf_counter()-started > args.max_seconds:
                            raise TimeoutError("bounded correction allowance exhausted between kernels")
                        u, j, diagnostics = owner.evaluate(query, current_images)
                        transfer_start = time.perf_counter()
                        host_u, host_j = cp.asnumpy(u), cp.asnumpy(j)
                        transfer_seconds = time.perf_counter()-transfer_start
                        del u, j
                        eu, ej, selected_count = expected[label]
                        item = {"dtype": dtype, "block": label, "repeat": repeat,
                                "scope_build_seconds": build_seconds, "output_transfer_seconds": transfer_seconds,
                                "operator": diagnostics,
                                "selected_direct_error": compare_fields(host_u[indices], host_j[indices], eu, ej),
                                "selected_direct_accepted_pairs": selected_count}
                        report["measurements"].append(item)
                        print(json.dumps({"dtype": dtype, "block": label, "repeat": repeat,
                                          "seconds": diagnostics["query_seconds"],
                                          "candidate_pairs": diagnostics["candidate_pairs"],
                                          "accepted_pairs": diagnostics["accepted_pairs"],
                                          "memory": diagnostics["pool_high_water_bytes"]}), flush=True)
                        if label == "all" and repeat == args.repeats-1:
                            fields[f"velocity_{dtype}"] = host_u
                            fields[f"gradient_{dtype}"] = host_j
                        publish()
        report["all_target_float32_vs_float64"] = _summary(fields["velocity_float32"], fields["gradient_float32"],
                                                           fields["velocity_float64"], fields["gradient_float64"])
        with archive.open("xb") as stream:
            np.savez(stream, **fields, indices=indices,
                     selected_direct_velocity=expected["all"][0], selected_direct_gradient=expected["all"][1])
        report.update(status="complete", archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==data["identity"]["checkpoint_sha256"],
                      elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        report["sources_changed"] = [name for name, digest in hashes.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
        if report["sources_changed"]:
            report["status"]="failed"
        publish()


if __name__ == "__main__":
    main()
