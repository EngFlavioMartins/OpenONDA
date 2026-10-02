"""Read-only native finite-field qualification under controlled memory pressure.

The injected admission value is explicitly synthetic, never claimed as a
measurement of a historical failure. All transforms/corrections execute on the
real GPU. Each process imports one preserved or current source tree, with the
same native sources, grid, precision, finite images and unchanged resource caps.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--free-mib", type=int)
    parser.add_argument("--expect-memory-rejection", action="store_true")
    args = parser.parse_args()
    output, archive = args.output.resolve(), args.output.resolve().with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError("device-budget qualification evidence already exists")
    root, checkpoint = args.source_root.resolve(strict=True), args.checkpoint.resolve(strict=True)
    before = digest(checkpoint)
    with h5py.File(checkpoint, "r") as saved:
        config = json.loads(saved["solver"].attrs["numerical_configuration"])
        x, gamma, sigma = (saved["particles/"+name][:] for name in
                           ("position", "vortex_strength", "core_radius"))
        clock = float(saved["solver"].attrs["time"])
    sys.path.insert(0, str(root))
    import cupy as cp

    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
    from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabPolicy

    policy = replace(GaussianSlabPolicy(), backend=config["induction"]["gaussian_mesh_policy"]["backend"])
    if asdict(policy) != config["induction"]["gaussian_mesh_policy"]:
        raise ValueError("native numerical policy differs from this qualification")
    resolved, slab = policy.mesh.resolve(sigma), config["induction"]
    shell = int(slab["max_shells"])-1
    images = tuple((k, odd) for k in range(-shell, shell+1) for odd in (False, True)
                   if (k, odd) != (0, False))
    package = root/"source/solvers/vpm/physics/induction/gaussian_mesh"
    hashes = {str(path): digest(path) for path in package.glob("*.py")}
    actual_free, actual_total = cp.cuda.runtime.memGetInfo()
    record = {"scope": "unchanged native finite-image fields; no particle advance or tail-policy change",
              "status": "running", "source_root": str(root), "source_hashes": hashes,
              "checkpoint_sha256": before, "time": clock, "source_count": len(x),
              "query_count": len(x), "images": len(images), "policy": asdict(policy),
              "actual_device_free_bytes_before": int(actual_free),
              "actual_device_total_bytes": int(actual_total),
              "synthetic_admission_free_bytes": None if args.free_mib is None else args.free_mib*1024**2}
    original_meminfo, owner = cp.cuda.runtime.memGetInfo, None
    started = time.perf_counter()
    try:
        if args.free_mib is not None:
            cp.cuda.runtime.memGetInfo = lambda: (args.free_mib*1024**2, actual_total)
        try:
            owner = GaussianImageFields(x, gamma, sigma, x,
                zmin=slab["z_min"], zmax=slab["z_max"], tau=resolved.tau,
                spacing=resolved.spacing, cutoff=resolved.correction_cutoff,
                order=resolved.order, dtype="float32", correction_dtype="float32",
                max_images=len(images), max_query_points=policy.max_query_points,
                max_scratch_bytes=policy.max_scratch_bytes, max_plan_bytes=policy.max_plan_bytes,
                max_correction_bytes=policy.max_correction_bytes, max_total_bytes=policy.max_total_bytes)
        finally:
            cp.cuda.runtime.memGetInfo = original_meminfo
        if args.expect_memory_rejection:
            raise AssertionError("preserved admission unexpectedly accepted controlled pressure")
        prepare = owner.prepare(images)
        u, j, query = owner.evaluate_prepared(x)
        host_u, host_j = cp.asnumpy(u), cp.asnumpy(j)
        del u, j
        if not np.isfinite(host_u).all() or not np.isfinite(host_j).all():
            raise AssertionError("nonfinite finite-field output")
        with archive.open("xb") as stream:
            np.savez(stream, position=x, velocity=host_u, gradient=host_j)
        record.update(status="complete", prepare=prepare, query=query,
                      shape=owner.shape, fft_shape=owner.fft_shape,
                      execution_plan=asdict(owner.execution_plan), archive_sha256=digest(archive))
    except MemoryError as error:
        if not args.expect_memory_rejection:
            raise
        record.update(status="expected-memory-rejection", error=str(error))
    finally:
        cp.cuda.runtime.memGetInfo = original_meminfo
        if owner is not None:
            owner.close()
        record["wall_seconds"] = time.perf_counter()-started
        record["checkpoint_unchanged"] = digest(checkpoint) == before
        record["changed_sources"] = [path for path, value in hashes.items() if digest(Path(path)) != value]
        with output.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
    if not record["checkpoint_unchanged"] or record["changed_sources"]:
        raise AssertionError("qualification inputs changed")
    print(json.dumps({key: record[key] for key in ("status", "source_count", "wall_seconds")}))


if __name__ == "__main__":
    main()
