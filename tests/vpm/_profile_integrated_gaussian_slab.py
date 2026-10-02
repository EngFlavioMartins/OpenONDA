"""Read-only native public-SlipSlab qualification; no solver advancement.

Exercises unchanged pair-mean primary FMM, the actual optional session/tail
admission, Taichi accumulation and coherent source-only query publication.
Cold, cached-image and deliberately fresh-image timings are separate.
"""

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


def difference(current, reference):
    delta = current.astype(np.float64)-reference.astype(np.float64)
    denominator = float(np.linalg.norm(reference.astype(np.float64)))
    return {"relative_l2": float(np.linalg.norm(delta))/denominator if denominator else None,
            "max_component": float(np.abs(delta).max(initial=0)),
            "max_point_norm": float(np.linalg.norm(delta.reshape(len(delta), -1), axis=1).max(initial=0))}


def authenticate_baseline_provenance(baseline_evidence, primary_evidence):
    """Bind the old-error gate to the baseline bytes used by the direct audit.

    Merely hashing today's baseline is not authentication. The independently
    authenticated primary audit already records the baseline report/archive
    hashes. Copies at different paths are allowed only when both bytes match.
    """
    audit_path = Path(primary_evidence["json"])
    raw = audit_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != primary_evidence["json_sha256"]:
        raise ValueError("authenticated primary audit changed before baseline admission")
    recorded = json.loads(raw).get("baseline")
    if not isinstance(recorded, dict) or any(
        recorded.get(key) != baseline_evidence.get(key)
        for key in ("report_sha256", "archive_sha256")
    ):
        raise ValueError("baseline differs from the authenticated primary-direct audit provenance")
    return {"primary_audit_json_sha256": primary_evidence["json_sha256"],
            "baseline_report_sha256": recorded["report_sha256"],
            "baseline_archive_sha256": recorded["archive_sha256"],
            "admission": "both baseline byte digests match authenticated direct-audit provenance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "oracle", "baseline", "coherent", "primary-audit", "fenv-extension", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    solution = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output, archive = args.output.resolve(), args.output.resolve().with_suffix(".npz")
    if output.parent != solution or output.suffix != ".json" or output.exists() or archive.exists():
        raise ValueError("new integrated evidence required in ordinary solution directory")
    extension = args.fenv_extension.resolve(strict=True)
    name = "source.solvers.vpm.numerics._fenv"
    spec = importlib.util.spec_from_file_location(name, extension)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    sys.modules[name] = bridge

    from tests.vpm._audit_composite_primary_direct import _load_fields
    from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields
    from tests.vpm._profile_gaussian_mesh_package import evidence

    data = authenticate_inputs(args.checkpoint, args.oracle)
    x, gamma, sigma = data["position"], data["strength"], data["core"]
    identity, indices = data["identity"], data["oracle"]["indices"].astype(np.int64)
    baseline, baseline_evidence = _load_fields(args.baseline, identity, composite=False)
    coherent, coherent_evidence = evidence(args.coherent, identity)
    primary, primary_evidence = evidence(args.primary_audit, identity)
    baseline_provenance = authenticate_baseline_provenance(baseline_evidence, primary_evidence)
    if (not np.array_equal(primary["indices"], indices)
            or not np.array_equal(primary["position"], x[indices])
            or not np.all(sigma == sigma[0])):
        raise ValueError("saved selected primary oracle requires these identical uniform native cores")
    paths = set((root/"source").rglob("*.py"))
    paths.update(Path(module.__file__).resolve() for module_name, module in sys.modules.items()
                 if module_name.startswith("tests.vpm.") and getattr(module, "__file__", None))
    paths.update((Path(__file__).resolve(), extension, root/"source/solvers/vpm/numerics/_fenv.c",
                  solution.parent/"assets/verify_image_operator_checkpoint.py"))
    hashes = {str(path): digest(path) for path in sorted(paths)}
    report = {"status": "running", "identity": identity, "source_hashes": hashes,
              "oracle_provenance": data["oracle_provenance"],
              "controls": {"baseline": baseline_evidence, "coherent": coherent_evidence,
                           "primary": primary_evidence}, "stages": [], "queries": [],
              "baseline_provenance_admission": baseline_provenance,
              "scope": "public Slab stage/target dispatch on unchanged checkpoint275; no physical advance",
              "coupled_evolution_qualified": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    slab = None
    arrays = {}
    try:
        import cupy as cp
        import taichi as ti

        from source.solvers.vpm.kernels.base import make_vortex_kernel
        from source.solvers.vpm.physics.base import PhysicsBase
        from source.solvers.vpm.physics.induction.fmm import FMMInduction
        from source.solvers.vpm.physics.induction.gaussian_mesh.session import GaussianSlabPolicy
        from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction

        config, count = identity["configuration"], len(x)
        controls = config["induction"]
        if controls["stretching_scheme"] != "TRANSPOSED":
            raise ValueError("authenticated saved rate oracle uses transposed stretching")
        ti.init(arch=ti.cuda, default_fp=ti.f32, offline_cache=True, cpu_max_num_threads=2)
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError("CUDA required; no silent fallback")
        physics = PhysicsBase("GAUSSIAN", config["max_n_particles"], ti.f32,
                              max_evaluation_points=128)
        slab = SlipSlabInduction(FMMInduction(stretching_scheme=controls["stretching_scheme"]),
            **{key: controls[key] for key in ("z_min", "z_max", "tail_tolerance", "max_shells",
                                              "velocity_scale", "gradient_scale")},
            gaussian_mesh_policy=GaussianSlabPolicy()).bind(physics, kernel=make_vortex_kernel("GAUSSIAN"))
        report["mesh_parameters"] = vars(slab.gaussian_mesh_policy.mesh.resolve(sigma))
        xx, gg, u, rate = (ti.Vector.field(3, ti.f32, shape=count) for _ in range(4))
        rr = ti.field(ti.f32, shape=count)
        j = ti.Matrix.field(3, 3, ti.f32, shape=count)
        for field, value in ((xx, x), (gg, gamma), (rr, sigma)):
            field.from_numpy(value)
        direct_u, direct_j = primary["exact_velocity"], primary["exact_gradient"]
        direct_rate = primary["exact_rate"]
        old_errors = compare_fields(baseline["velocity"][indices], baseline["gradient"][indices],
                                    direct_u, direct_j)
        old_rate_error = difference(baseline["rate"][indices], direct_rate)
        report["preserved_control_direct_error"] = {**old_errors, "rate": old_rate_error}
        for repeat in range(3):
            teardown = 0.0
            if repeat == 2:
                started = perf_counter()
                slab.close_mesh_session()
                teardown = perf_counter()-started
            free, total = cp.cuda.runtime.memGetInfo()
            ti.sync()
            started = perf_counter()
            slab.evaluate_stage(position=xx, vortex_strength=gg, core_radius=rr, count=count,
                                velocity_out=u, velocity_gradient_out=j, vortex_strength_rate_out=rate)
            ti.sync()
            elapsed = perf_counter()-started
            values = {"velocity": u.to_numpy(), "gradient": j.to_numpy(), "rate": rate.to_numpy()}
            if not all(np.isfinite(value).all() for value in values.values()):
                raise FloatingPointError("nonfinite public stage output")
            selected = compare_fields(values["velocity"][indices], values["gradient"][indices], direct_u, direct_j)
            selected["rate"] = difference(values["rate"][indices], direct_rate)
            for key in ("velocity", "gradient", "rate"):
                if selected[key]["relative_l2"] > report["preserved_control_direct_error"][key]["relative_l2"]:
                    raise AssertionError(f"selected whole physical {key} norm error exceeds preserved control")
            session = slab.last_tail["mesh"]
            if bool(session["field_owner_hit"]) != (repeat == 1):
                raise AssertionError("cold/cached/fresh source-owner sequence disagrees")
            record = {"repeat": repeat, "owner_mode": ("cold", "cached", "fresh_after_close")[repeat],
                      "complete_public_stage_seconds": elapsed, "preceding_owner_close_seconds": teardown,
                      "gpu_free_before_bytes": free, "gpu_total_bytes": total,
                      "gpu_free_after_bytes": cp.cuda.runtime.memGetInfo()[0], "tail": slab.last_tail,
                      "direct_error": selected,
                      "all_field_difference_from_preserved_control": {
                          key: difference(value, baseline[key]) for key, value in values.items()}}
            report["stages"].append(record)
            arrays.update(values)
            print(json.dumps({"stage": repeat, "seconds": elapsed,
                              "direct_relative_l2": {key: item["relative_l2"] for key, item in selected.items()}}), flush=True)
            publish()

        # Real disabled-rate/no-J publication, not merely a fake-session branch.
        j.fill(37)
        slab.evaluate_stage(position=xx, vortex_strength=gg, core_radius=rr, count=count,
                            velocity_out=u, velocity_gradient_out=None, vortex_strength_rate_out=rate,
                            strength_rate_enabled=False)
        ti.sync()
        if not np.all(j.to_numpy() == 37) or np.any(rate.to_numpy()):
            raise AssertionError("omitted gradient/disabled rate publication regressed")
        optional_u = u.to_numpy()
        if not np.isfinite(optional_u).all():
            raise FloatingPointError("nonfinite no-J/disabled-rate velocity")
        optional_error = difference(optional_u[indices], direct_u)
        if optional_error["relative_l2"] > old_errors["velocity"]["relative_l2"]:
            raise AssertionError("no-J/disabled-rate direct24 velocity error exceeds preserved control")
        report["stage_optional_outputs"] = {
            "all_field_difference_from_full_stage": difference(optional_u, values["velocity"]),
            "direct_velocity_error": optional_error,
            "criterion": "same direct24 relative-L2 noninferiority to authenticated old control as full stage",
            "omitted_gradient_unchanged": True, "disabled_rate_exact_zero": True,
        }
        parts = [("oracle24", x[indices].astype(np.float64))]
        for side in ("lower", "upper"):
            for precision in ("f64", "f32"):
                key = f"wall_{side}_{precision}"
                parts.append((key, coherent[key+"_position"]))
        query = np.concatenate([points for _, points in parts])
        q = ti.Vector.field(3, ti.f64, shape=len(query))
        qu = ti.Vector.field(3, ti.f32, shape=len(query))
        qj = ti.Matrix.field(3, 3, ti.f32, shape=len(query))
        background = ti.Vector.field(3, ti.f32, shape=())
        background[None] = [0.25, -0.5, 0.125]
        q.from_numpy(query)
        for repeat in range(2):
            started = perf_counter()
            slab.evaluate_targets(target_position=q, source_position=xx, source_vortex_strength=gg,
                source_core_radius=rr, target_velocity=qu, target_velocity_gradient=qj,
                target_count=len(query), source_count=count, include_freestream=False,
                background_velocity=background)
            ti.sync()
            elapsed = perf_counter()-started
            vu, vj = qu.to_numpy(), qj.to_numpy()
            selected = compare_fields(vu[:len(indices)], vj[:len(indices)], direct_u, direct_j)
            if max(value["relative_l2"] for value in selected.values()) > 1e-5:
                raise AssertionError("coherent whole source-only direct error exceeds its existing envelope")
            offset, walls = len(indices), {}
            for key, points in parts[1:]:
                walls[key] = float(np.abs(vu[offset:offset+len(points), 2]).max())
                arrays[key+"_position"] = points
                arrays[key+"_velocity"], arrays[key+"_gradient"] = vu[offset:offset+len(points)], vj[offset:offset+len(points)]
                offset += len(points)
            if max(walls.values()) > 1e-8:
                raise AssertionError("coherent wall normal cancellation regressed")
            report["queries"].append({"repeat": repeat, "seconds": elapsed, "tail": slab.last_tail,
                                      "direct_error": selected, "wall_normal_max": walls})
            arrays.update(query_position=query, query_velocity=vu, query_gradient=vj)
            publish()
        qj.fill(43)
        slab.evaluate_targets(target_position=q, source_position=xx, source_vortex_strength=gg,
            source_core_radius=rr, target_velocity=qu, target_velocity_gradient=None,
            target_count=len(query), source_count=count, include_freestream=True, background_velocity=background)
        np.testing.assert_allclose(qu.to_numpy(), vu+np.array([.25, -.5, .125], np.float32), rtol=0, atol=2e-6)
        if not np.all(qj.to_numpy() == 43):
            raise AssertionError("omitted target Jacobian was overwritten")
        qu.fill(47)
        slab.evaluate_targets(target_position=q, source_position=xx, source_vortex_strength=gg,
            source_core_radius=rr, target_velocity=None, target_velocity_gradient=qj,
            target_count=len(query), source_count=count, include_freestream=True, background_velocity=background)
        np.testing.assert_allclose(qj.to_numpy(), vj, rtol=0, atol=2e-6)
        if not np.all(qu.to_numpy() == 47):
            raise AssertionError("omitted target velocity was overwritten")
        report["target_optional_outputs_and_freestream_once"] = True

        # Empty active ranges use actual CUDA-backed public fields/session, not
        # fake backends. Inactive allocated entries must remain untouched.
        empty_records = []
        for source_count in (count, 0):
            qu.fill(53)
            qj.fill(59)
            slab.evaluate_targets(target_position=q, source_position=xx, source_vortex_strength=gg,
                source_core_radius=rr, target_velocity=qu, target_velocity_gradient=qj,
                target_count=0, source_count=source_count, include_freestream=True,
                background_velocity=background)
            ti.sync()
            if not np.all(qu.to_numpy() == 53) or not np.all(qj.to_numpy() == 59):
                raise AssertionError("zero-target call overwrote inactive allocated entries")
            empty_records.append({"target_count": 0, "source_count": source_count,
                                  "allocated_outputs_unchanged": True})
        for want_u, want_j, free in ((True, True, False), (True, True, True),
                                     (True, False, True), (False, True, True)):
            qu.fill(61)
            qj.fill(67)
            slab.evaluate_targets(target_position=q, source_position=xx, source_vortex_strength=gg,
                source_core_radius=rr, target_velocity=qu if want_u else None,
                target_velocity_gradient=qj if want_j else None,
                target_count=len(query), source_count=0, include_freestream=free,
                background_velocity=background)
            ti.sync()
            expected_u = np.broadcast_to(
                np.array([.25, -.5, .125], np.float32) if free else np.zeros(3, np.float32),
                (len(query), 3),
            ) if want_u else np.full((len(query), 3), 61, np.float32)
            expected_j = np.full((len(query), 3, 3), 0 if want_j else 67, np.float32)
            np.testing.assert_array_equal(qu.to_numpy(), expected_u)
            np.testing.assert_array_equal(qj.to_numpy(), expected_j)
            if slab.last_tail["mesh"].get("empty_field") is not True:
                raise AssertionError("zero-source session did not report its empty-field path")
            empty_records.append({"target_count": len(query), "source_count": 0,
                                  "velocity_requested": want_u, "gradient_requested": want_j,
                                  "include_freestream": free, "exact_empty_publication": True})
        u.fill(71)
        j.fill(73)
        rate.fill(79)
        slab.evaluate_stage(position=xx, vortex_strength=gg, core_radius=rr, count=0,
                            velocity_out=u, velocity_gradient_out=j, vortex_strength_rate_out=rate)
        ti.sync()
        if (not np.all(u.to_numpy() == 71) or not np.all(j.to_numpy() == 73)
                or not np.all(rate.to_numpy() == 79)):
            raise AssertionError("zero-particle stage overwrote inactive allocated entries")
        report["zero_count_cuda_semantics"] = {
            "target_calls": empty_records, "zero_particle_stage_outputs_unchanged": True,
            "scope": "real CUDA fields/public dispatch and real session zero-source path; no timing claim",
        }
        report["source_fields_unchanged"] = all(np.array_equal(field.to_numpy(), original)
            for field, original in ((xx, x), (gg, gamma), (rr, sigma)))
        if not report["source_fields_unchanged"]:
            raise AssertionError("public dispatch mutated physical source state")
        with archive.open("xb") as stream:
            np.savez(stream, **arrays, oracle_indices=indices)
        report.update(status="complete", archive_sha256=digest(archive),
                      checkpoint_unchanged=digest(args.checkpoint) == identity["checkpoint_sha256"])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if slab is not None:
            try:
                slab.close_mesh_session()
                slab.base.close()
            except BaseException as error:
                report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        report["sources_changed"] = [path for path, before in hashes.items() if digest(path) != before]
        if report["sources_changed"]:
            report["status"] = "failed"
        publish()


if __name__ == "__main__":
    main()
