"""Host-only actual-wall qualification of unwired exact grid evidence reuse.

Accepted checkpoint grids are not the post-RK production GBD grids. Timings
exclude GPU allocation/upload, diffusion, and particle processing. No solver
is initialized and no saved solution fields or native observations are changed.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from tests.vpm._gbd_body_host_oracle import legacy_host_type
from tests.vpm._gbd_geometry_cache_prototype import ExactBodyGridCache
from tests.vpm._profile_gbd_geometry_census import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, case = args.source_root.resolve(), args.case.resolve()
    output = args.output.resolve()
    if output.parent != case / "solution" or output.exists():
        raise ValueError("Require a new ordinary solution report")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/support/cylinder"))
    from profile_diffusion_checkpoint import (
        array_digest,
        load_inputs,
        runtime_environment,
        source_identity,
    )

    source_hashes = source_identity(root)
    census = json.loads(args.census.read_text())
    if digest(root / "source/solvers/vpm/physics/diffusion/grid.py") != census["grid_source_sha256"]:
        raise ValueError("Census numerical geometry source differs")
    for row in census["records"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Census particle input changed")
    started = time.perf_counter()
    _, boundary, _, admitted = load_inputs(root, args.checkpoint.resolve(),
                                          case / "solution/fvm/mesh.npz", case / "setup.py")
    preflight_seconds = time.perf_counter() - started
    cfg = admitted["configuration"]
    spacing = cfg["viscous"]["gbd_grid_spacing"]
    if spacing != census["spacing"]:
        raise ValueError("Native and census spacing differ")
    slab = cfg["induction"]["z_min"], cfg["induction"]["z_max"]
    host_type = legacy_host_type(root)
    calls = {"membership_calls": 0, "membership_points": 0, "segment_calls": 0, "segments": 0}

    def contains(points):
        calls["membership_calls"] += 1
        calls["membership_points"] += len(points)
        return boundary.contains(points, include_boundary=False)

    def blocks(starts, ends):
        calls["segment_calls"] += 1
        calls["segments"] += len(starts)
        return boundary.blocks_segments(starts, ends)

    owner = host_type((1, 1, 1), contains, blocks, slab)
    member_callback, segment_callback = owner._body_interior_at_particles, owner._body_blocked_segments
    cache = ExactBodyGridCache(qualified_bindings=((member_callback, segment_callback),))
    measurements, controls = [], []

    def measure(name, row, operation):
        for body in boundary.bodies:
            body._cache.clear()
            body._cache_bytes = 0
        for key in calls:
            calls[key] = 0
        started = time.perf_counter()
        mask, links, stats = operation()
        elapsed = time.perf_counter() - started
        measurements.append({"name": name, "step": row["step"], "shape": row["shape"],
                             "origin": row["origin"], "seconds": elapsed,
                             "callback_counts": calls.copy(), "reuse": stats,
                             "mask_sha256": array_digest(mask), "links_sha256": array_digest(links)})
        return mask, links

    for row in census["records"]:
        def fresh(row=row):
            host = host_type(tuple(row["shape"]), contains, blocks, slab)
            host._prepare_body_mask_current_grid(np.asarray(row["origin"]), spacing, *row["shape"])
            return host._body_mask_host, host._body_links_host, None

        controls.append(measure("original_fresh_grid", row, fresh))
    for row, reference in zip(census["records"], controls, strict=True):
        def reuse(row=row):
            return cache.prepare(row["origin"], spacing, row["shape"], contains=member_callback,
                                 blocks=segment_callback, revision=boundary.revision, slab=slab)

        actual = measure("prototype_geometry_history", row, reuse)
        for field, expected in zip(actual, reference, strict=True):
            np.testing.assert_array_equal(field, expected)
        measurements[-1]["bitwise_matches_original"] = True
    last = census["records"][-1]
    repeated = measure("prototype_identical_grid_repeat", last,
                       lambda: cache.prepare(last["origin"], spacing, last["shape"],
                                             contains=member_callback, blocks=segment_callback,
                                             revision=boundary.revision, slab=slab))
    for field, expected in zip(repeated, controls[-1], strict=True):
        np.testing.assert_array_equal(field, expected)
    measurements[-1]["bitwise_matches_original"] = True
    if source_identity(root) != source_hashes:
        raise ValueError("Imported geometry source changed during qualification")
    import taichi as ti
    if ti.lang.impl.get_runtime().prog is not None:
        raise RuntimeError("Host-only qualification unexpectedly initialized Taichi")
    report = {"scope": __doc__, "status": "complete", "input": admitted,
              "source_hashes": source_hashes, "census_sha256": digest(args.census),
              "host_preflight_seconds": preflight_seconds, "runtime": runtime_environment(),
              "measurements": measurements, "gpu_initialized": False,
              "production_state_modified": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "complete", "measurements": measurements}, indent=2))


if __name__ == "__main__":
    main()
