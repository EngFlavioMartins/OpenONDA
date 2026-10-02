"""Read-only native Gaussian session ownership, certificates and field check."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "oracle", "coherent", "primary-audit", "fenv-extension", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    solution = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output = args.output.resolve()
    archive = output.with_suffix(".npz")
    if output.parent != solution or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("new session evidence required directly in ordinary solution directory")
    name = "source.solvers.vpm.numerics._fenv"
    extension = args.fenv_extension.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, extension)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    sys.modules[name] = bridge

    from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabFieldSession
    from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields
    from tests.vpm._profile_gaussian_mesh_package import evidence

    # Include lazy numerical imports as well as helpers and the compiled guard.
    files = [path for folder in ("gaussian_mesh", "gaussian_tail")
             for path in (root/"source/solvers/vpm/physics/induction"/folder).glob("*.py")]
    files += list((root/"source/solvers/vpm/numerics").glob("*.py"))
    files += [extension, root/"source/solvers/vpm/numerics/_fenv.c", Path(__file__),
              Path(__file__).with_name("_finite_slab_native_qualification.py"),
              Path(__file__).with_name("_profile_gaussian_mesh_package.py")]
    hashes = {str(path): digest(path) for path in files}
    report = {"status": "running", "source_hashes": hashes, "evaluations": [],
              "scope": "real session/tail/cutoff admission and explicit field transfers; no evolution, self-FMM or coupled step",
              "runtime_dispatch_qualified": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    engine, arrays = None, {}
    try:
        data = authenticate_inputs(args.checkpoint, args.oracle)
        x, gamma, sigma = data["position"], data["strength"], data["core"]
        identity, indices = data["identity"], data["oracle"]["indices"].astype(np.int64)
        coherent, coherent_evidence = evidence(args.coherent, identity)
        primary, primary_evidence = evidence(args.primary_audit, identity)
        if not np.array_equal(primary["indices"], indices):
            raise ValueError("primary oracle indices disagree")
        report.update(identity=identity, oracle_provenance=data["oracle_provenance"],
                      controls={"coherent": coherent_evidence, "primary": primary_evidence})
        # Deliberately exercise the actual Taichi host FTZ environment without
        # constructing any physical solver or allocating its particle fields.
        import taichi as ti
        ti.init(arch=ti.cuda, offline_cache=True, default_fp=ti.f32)

        def host_probe():
            tiny = np.array([2.**-1022], np.float64)
            return float((tiny*.5)[0]).hex()

        before = host_probe()
        slab = identity["configuration"]["induction"]
        engine = GaussianSlabFieldSession(z_min=slab["z_min"], z_max=slab["z_max"],
            tail_tolerance=slab["tail_tolerance"], max_shells=slab["max_shells"],
            velocity_scale=slab["velocity_scale"], gradient_scale=slab["gradient_scale"], dtype="float32")
        report["resolved_mesh"] = vars(engine.policy.mesh.resolve(sigma))
        for source_only in (False, True):
            label = "source_only" if source_only else "images"
            parts = [("oracle24", x[indices])]
            if source_only:
                for side in ("lower", "upper"):
                    for precision in ("f64", "f32"):
                        key = f"wall_{side}_{precision}"
                        parts.append((key, coherent[key+"_position"]))
                query = np.concatenate([points for _, points in parts])
            else:
                query = x
            for repeat in range(2):
                started = perf_counter()
                u, j, diagnostics = engine.evaluate(x, gamma, sigma, query, source_only=source_only)
                elapsed = perf_counter()-started
                record = {"role": label, "repeat": repeat, "complete_session_seconds": elapsed,
                          "session": diagnostics}
                if source_only:
                    exact_u, exact_j = primary["exact_velocity"], primary["exact_gradient"]
                    tested_u, tested_j = u[:len(indices)], j[:len(indices)]
                    offset = len(indices)
                    walls = {}
                    for key, points in parts[1:]:
                        walls[key] = float(np.max(np.abs(u[offset:offset+len(points), 2])))
                        offset += len(points)
                    record["wall_normal_max_absolute"] = walls
                    if max(walls.values()) > 1e-8:
                        raise AssertionError("coherent source-only wall cancellation regressed")
                else:
                    exact_u = data["oracle"]["direct"][:, :3]
                    exact_j = data["oracle"]["direct"][:, 3:12].reshape(-1, 3, 3)
                    tested_u, tested_j = u[indices], j[indices]
                record["direct_error"] = compare_fields(tested_u, tested_j, exact_u, exact_j)
                if (record["direct_error"]["velocity"]["relative_l2"] > 1e-5
                        or record["direct_error"]["gradient"]["relative_l2"] > 1e-5):
                    raise AssertionError("finite-field direct accuracy exceeds qualification envelope")
                if host_probe() != before:
                    raise AssertionError("session did not restore Taichi host floating environment")
                if repeat and not diagnostics["field_owner_hit"]:
                    raise AssertionError("identical second query did not reuse its exact field owner")
                report["evaluations"].append(record)
                arrays[label+"_position"], arrays[label+"_velocity"], arrays[label+"_gradient"] = query, u, j
                print(json.dumps({"role": label, "repeat": repeat, "seconds": elapsed,
                                  "direct_relative_l2": {key: value["relative_l2"]
                                      for key, value in record["direct_error"].items()}}), flush=True)
                publish()
        engine.close()
        engine = None
        with archive.open("xb") as stream:
            np.savez(stream, **arrays, oracle_indices=indices)
        report.update(status="complete", archive_sha256=digest(archive),
                      host_environment_restored=True,
                      checkpoint_unchanged=digest(args.checkpoint)==identity["checkpoint_sha256"])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if engine is not None:
            try:
                engine.close()
            except BaseException as error:
                report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        report["sources_changed"] = [path for path, before in hashes.items() if digest(path) != before]
        if report["sources_changed"]:
            report["status"] = "failed"
        publish()


if __name__ == "__main__":
    main()
