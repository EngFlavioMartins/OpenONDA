"""Authenticated native fields plus EXTENDED-QUERY capacity, not future flow.

Only query-envelope corners extend to x15 / configured FVM bounds. Physical
source positions, strengths and cores are the unmodified saved checkpoint.
No solver advancement, no particle-core replacement and no tail approval.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields
from tests.vpm._profile_gaussian_mesh_package import evidence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "oracle", "coherent", "primary-audit", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    solution = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output, archive = args.output.resolve(), args.output.resolve().with_suffix(".npz")
    if output.parent != solution or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("fresh evidence required in ordinary solution directory")
    package = root/"source/solvers/vpm/physics/induction/gaussian_mesh"
    files = [package/name for name in ("__init__.py", "coordinates.py", "correction.py",
                                      "fields.py", "runtime.py", "stencil.py", "planning.py")]
    files += [Path(__file__), Path(__file__).with_name("_finite_slab_native_qualification.py"),
              Path(__file__).with_name("_profile_gaussian_mesh_package.py")]
    hashes = {str(path): digest(path) for path in files}
    report = {"status": "running", "runtime_admissible": False, "tail_certified": False,
              "source_hashes": hashes, "scopes": [], "extended_geometry_is_queries_only": True}
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
        coherent, coherent_provenance = evidence(args.coherent, identity)
        primary, primary_provenance = evidence(args.primary_audit, identity)
        report["controls"] = {"coherent": coherent_provenance, "primary": primary_provenance}
        images = [image for block in data["blocks"] for image in block["images"]]
        corners = np.array(list(itertools.product((-1.6, 15.), (-1.6, 1.6),
                                                   (slab["z_min"], slab["z_max"]))), dtype=np.float64)
        report["extended_query_corners"] = corners.tolist()
        walls = np.concatenate([coherent[f"wall_{side}_{dtype}_position"]
                                for side in ("lower", "upper") for dtype in ("f64", "f32")])
        for spacing, extended, source_only in ((.035, False, False), (.03, False, False),
                                               (.035, True, False), (.03, True, False),
                                               (.03, True, True)):
            label = f"h{spacing:g}_{'extended' if extended else 'current'}_{'whole' if source_only else 'images'}"
            envelope = np.concatenate((x, corners)) if extended else x
            descriptors = [(0, False), *images] if source_only else images
            started = time.perf_counter()
            owner = GaussianImageFields(x, gamma, sigma, envelope,
                zmin=slab["z_min"], zmax=slab["z_max"], tau=.12, spacing=spacing, cutoff=.6,
                dtype="float32", correction_dtype="float32", source_only_primary=source_only,
                max_images=len(descriptors), profile=False)
            scope = {"label": label, "spacing": spacing, "extended_query_only": extended,
                     "source_only_primary": source_only, "setup_seconds": time.perf_counter()-started,
                     "source_bounds": [x.min(axis=0).tolist(), x.max(axis=0).tolist()],
                     "source_count": len(x), "initial_query_count": len(envelope), "runs": []}
            report["scopes"].append(scope)
            query = np.concatenate((x[indices], walls)) if source_only else x
            for repeat in range(2):
                prepare = owner.prepare(descriptors)
                u, j, diagnostics = owner.evaluate_prepared(query)
                host_u, host_j = cp.asnumpy(u), cp.asnumpy(j)
                del u, j
                if source_only:
                    direct = compare_fields(host_u[:len(indices)], host_j[:len(indices)],
                        primary["exact_velocity"], primary["exact_gradient"])
                else:
                    direct = compare_fields(host_u[indices], host_j[indices],
                        data["oracle"]["direct"][:, :3], data["oracle"]["direct"][:, 3:12].reshape(-1, 3, 3))
                item = {"repeat": repeat, "prepare": prepare, "query": diagnostics, "direct_error": direct}
                if source_only:
                    item["wall_normal_max"] = float(np.max(np.abs(host_u[len(indices):, 2])))
                scope["runs"].append(item)
                arrays[label+"_position"], arrays[label+"_velocity"], arrays[label+"_gradient"] = query, host_u, host_j
                print(json.dumps({"case": label, "repeat": repeat,
                    "plan": prepare["execution_plan"], "prepare_seconds": prepare["seconds"],
                    "query_seconds": diagnostics["query_seconds"],
                    "pool_reserved_bytes": diagnostics["smooth_pool_reserved_bytes"],
                    "plan_bytes": prepare["plan_bytes"], "plan_build_seconds": prepare["plan_build_seconds"]}), flush=True)
                publish()
            started = time.perf_counter()
            owner.close()
            owner = None
            scope["teardown_seconds"] = time.perf_counter()-started
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
