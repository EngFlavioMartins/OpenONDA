"""UNWIRED wall audit: source-only primary FMM plus saved finite image fields.

The physical primary at arbitrary wall targets uses SOURCE cores, never the
pair-mean particle self contract. CPU direct Gaussian work is bounded and
separate from the isolated GPU primary-target query. No source advancement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from tests.vpm._finite_slab_native_qualification import authenticate_inputs
from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


def source_only_direct(position, strength, core, targets, *, chunk=4096, max_pairs=40_000_000):
    x, gamma, sigma, q = (np.asarray(value, np.float64) for value in (position, strength, core, targets))
    if (x.ndim != 2 or x.shape[1:] != (3,) or gamma.shape != x.shape or sigma.shape != (len(x),)
            or q.ndim != 2 or q.shape[1:] != (3,) or np.any(sigma <= 0)
            or not all(np.isfinite(value).all() for value in (x, gamma, sigma, q))):
        raise ValueError("finite source/target arrays and positive source cores required")
    if (not isinstance(chunk, int) or isinstance(chunk, bool) or not 1 <= chunk <= 16384
            or len(x)*len(q) > max_pairs):
        raise ValueError("bounded direct query work required")
    u, j = np.zeros_like(q), np.zeros((len(q), 3, 3))
    for first in range(0, len(x), chunk):
        last = min(first+chunk, len(x))
        du, dj = gaussian_pairs(q[:, None]-x[None, first:last], gamma[None, first:last], sigma[None, first:last])
        u += du.sum(axis=1)
        j += dj.sum(axis=1)
    return u, j


def finite_wall_normal(position, strength, core, targets, *, zmin, zmax, shells, upper):
    """Exact finite-family normal component via reflection pairing.

    Include physical primary plus both families k=-K..K. At the lower plane,
    even k pairs with odd -k, hence exact zero. At the upper plane, even k
    pairs with odd1-k; only even(-K) and odd(-K) lack their K+1 partners.
    Those two explicit Gaussian sums are the entire finite normal field.
    This avoids calling a finite family equal to the infinite zero-normal
    boundary condition. Evaluation still has ordinary f64 arithmetic error.
    """
    x, gamma, sigma, q = (np.asarray(value, np.float64) for value in (position, strength, core, targets))
    if (not isinstance(shells, int) or isinstance(shells, bool) or shells < 1
            or type(upper) is not bool or not zmax > zmin or q.ndim != 2 or q.shape[1:] != (3,)
            or not np.all(q[:, 2] == (zmax if upper else zmin))):
        raise ValueError("exact geometric plane queries and complete positive shells required")
    result = np.zeros(len(q))
    if not upper:
        return result
    shift = -2*shells*(zmax-zmin)
    for odd in (False, True):
        source, vector = x.copy(), gamma.copy()
        source[:, 2] = shift+(2*zmin-source[:, 2] if odd else source[:, 2])
        if odd:
            vector[:, :2] *= -1
        u, _ = source_only_direct(source, vector, sigma, q)
        result += u[:, 2]
    return result


def _normal_summary(values):
    values = np.asarray(values, np.float64)
    return {"max_absolute": float(np.abs(values).max(initial=0)),
            "l2": float(np.linalg.norm(values)), "values": values.tolist()}


def _imported_source_hashes(root):
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.suffix == ".py" and path.is_relative_to(root):
                paths.add(path)
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "oracle", "wall", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--old-slab-control", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    solution = root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    output = args.output.resolve()
    if output.parent != solution or output.suffix != ".json" or output.exists() or output.with_suffix(".npz").exists():
        raise ValueError("new JSON/NPZ evidence required in ordinary solution directory")
    data = authenticate_inputs(args.checkpoint, args.oracle)
    wall_prefix = args.wall.resolve().with_suffix("")
    wall_report = json.loads(wall_prefix.with_suffix(".json").read_text())
    wall_path = wall_prefix.with_suffix(".npz")
    if (wall_report.get("status") != "complete" or wall_report.get("identity") != data["identity"]
            or not wall_report.get("checkpoint_unchanged") or wall_report.get("sources_changed")
            or hashlib.sha256(wall_path.read_bytes()).hexdigest() != wall_report.get("archive_sha256")
            or wall_report["prepare"]["finite_image_count"] != 513):
        raise ValueError("saved wall fields do not authenticate the unchanged finite513 operator")
    labels = ("wall_lower_f64", "wall_upper_f64", "wall_lower_f32", "wall_upper_f32")
    with np.load(wall_path, allow_pickle=False) as saved:
        fields = {label: {key: saved[f"{label}_{key}"].copy() for key in ("position", "velocity", "gradient")}
                  for label in labels}
    n = len(fields[labels[0]]["position"])
    if n != 24 or any(record["position"].shape != (n, 3) or record["velocity"].shape != (n, 3)
                      or record["gradient"].shape != (n, 3, 3)
                      or not all(np.isfinite(value).all() for value in record.values()) for record in fields.values()):
        raise ValueError("expected24 finite wall queries per plane and precision")
    slab = data["identity"]["configuration"]["induction"]
    for side, plane in (("lower", slab["z_min"]), ("upper", slab["z_max"])):
        intended, rounded = fields[f"wall_{side}_f64"]["position"], fields[f"wall_{side}_f32"]["position"]
        if not np.all(intended[:, 2] == plane) or not np.array_equal(rounded, intended.astype(np.float32).astype(np.float64)):
            raise ValueError("wall coordinate precision mapping changed")
    x, gamma, sigma = data["position"], data["strength"], data["core"]
    queries = np.concatenate([fields[label]["position"] for label in labels])
    report = {"status": "running", "production_admissible": False, "identity": data["identity"],
              "wall_evidence": str(wall_path), "wall_archive_sha256": wall_report["archive_sha256"],
              "primary_core_contract": "source-only arbitrary target; not pair-mean particle self",
              "image_fields": "saved finite513 images with source-core correction, no physical primary",
              "old_slab_control_requested": args.old_slab_control,
              "geometric_truth": "infinite normal0; finite lower normal0; finite upper normal is two unmatched -K images",
              "planes": {}}
    output.write_text(json.dumps(report, indent=2)+"\n")
    base = None
    try:
        started = time.perf_counter()
        direct_u, direct_j = source_only_direct(x, gamma, sigma, queries)
        report["direct_primary_pairs"] = len(x)*len(queries)
        report["direct_primary_seconds"] = time.perf_counter()-started
        end_normal = finite_wall_normal(x, gamma, sigma, fields["wall_upper_f64"]["position"],
                                       zmin=slab["z_min"], zmax=slab["z_max"], shells=128, upper=True)
        # Production imports start only after CPU oracle work. The public
        # target API performs one source-only query, with no freestream.
        import taichi as ti

        from source.solvers.vpm.kernels.base import make_vortex_kernel
        from source.solvers.vpm.physics.base import PhysicsBase
        from source.solvers.vpm.physics.induction.fmm import FMMInduction
        from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction

        report["source_hashes_before"] = _imported_source_hashes(root)
        ti.init(arch=ti.cuda, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2)
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError("explicit CUDA qualification required")
        physics = PhysicsBase(particle_kernel="GAUSSIAN", max_n_particles=len(x), accumulator_dtype=ti.f32)
        base = FMMInduction(stretching_scheme=slab["stretching_scheme"])
        old_slab = None
        if args.old_slab_control:
            old_slab = SlipSlabInduction(base, **{key: slab[key] for key in
                ("z_min", "z_max", "tail_tolerance", "max_shells", "velocity_scale", "gradient_scale")})
            old_slab.bind(physics, kernel=make_vortex_kernel("GAUSSIAN"))
        else:
            base.bind(physics, kernel=make_vortex_kernel("GAUSSIAN"))
        source_x, source_g = (ti.Vector.field(3, ti.f32, shape=len(x)) for _ in range(2))
        source_s = ti.field(ti.f32, shape=len(x))
        query = ti.Vector.field(3, ti.f32, shape=2*n)
        velocity = ti.Vector.field(3, ti.f32, shape=2*n)
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=2*n)
        background = ti.Vector.field(3, ti.f32, shape=())
        source_x.from_numpy(x)
        source_g.from_numpy(gamma)
        source_s.from_numpy(sigma)
        query.from_numpy(queries[2*n:].astype(np.float32))
        background[None] = [0., 0., 0.]
        arguments = {"target_position": query, "source_position": source_x, "source_vortex_strength": source_g,
                     "source_core_radius": source_s, "target_velocity": velocity, "target_velocity_gradient": gradient,
                     "target_count": 2*n, "source_count": len(x), "include_freestream": False,
                     "background_velocity": background}
        started = time.perf_counter()
        base.evaluate_targets(**arguments)
        ti.sync()
        fmm_u, fmm_j = velocity.to_numpy(), gradient.to_numpy()
        report["primary_fmm_cold_seconds"] = time.perf_counter()-started
        arrays = {"direct_primary_velocity": direct_u, "direct_primary_gradient": direct_j,
                  "primary_fmm_velocity": fmm_u, "primary_fmm_gradient": fmm_j,
                  "all_queries": queries, "true_finite_upper_normal": end_normal}
        if old_slab is not None:
            old_slab.evaluate_targets(**arguments)
            ti.sync()
            arrays["old_slab_velocity"], arrays["old_slab_gradient"] = velocity.to_numpy(), gradient.to_numpy()
            report["old_slab_tail"] = old_slab.last_tail
        for side_index, side in enumerate(("lower", "upper")):
            exact_slice, rounded_slice = slice(side_index*n, (side_index+1)*n), slice((side_index+2)*n, (side_index+3)*n)
            fmm_slice = slice(side_index*n, (side_index+1)*n)
            geometric, native = fields[f"wall_{side}_f64"], fields[f"wall_{side}_f32"]
            finite_truth = np.zeros(n) if side == "lower" else end_normal
            mesh_geometric_total = direct_u[exact_slice]+geometric["velocity"]
            mesh_native_total = direct_u[rounded_slice]+native["velocity"]
            hybrid_native_total = fmm_u[fmm_slice].astype(np.float64)+native["velocity"]
            item = {
                "coordinate_offset_max": float(np.abs(native["position"]-geometric["position"]).max()),
                "exact_finite_geometric_normal": _normal_summary(finite_truth),
                "direct_primary_plus_mesh_geometric_normal": _normal_summary(mesh_geometric_total[:, 2]),
                "independent_mesh_geometric_normal_error": _normal_summary(mesh_geometric_total[:, 2]-finite_truth),
                "direct_primary_plus_mesh_native_normal": _normal_summary(mesh_native_total[:, 2]),
                "fmm_primary_plus_mesh_native_normal": _normal_summary(hybrid_native_total[:, 2]),
                "primary_fmm_native_normal_error": _normal_summary(fmm_u[fmm_slice, 2]-direct_u[rounded_slice, 2]),
                "observed_coordinate_shift_normal_effect": _normal_summary(mesh_native_total[:, 2]-mesh_geometric_total[:, 2]),
            }
            if old_slab is not None:
                item["old_slab_native_normal"] = _normal_summary(arrays["old_slab_velocity"][fmm_slice, 2])
            report["planes"][side] = item
            arrays[f"{side}_hybrid_native_velocity"] = hybrid_native_total
            arrays[f"{side}_mesh_geometric_velocity"] = mesh_geometric_total
            arrays[f"{side}_mesh_native_velocity"] = mesh_native_total
        report["source_fields_unchanged"] = (np.array_equal(source_x.to_numpy(), x)
                                              and np.array_equal(source_g.to_numpy(), gamma)
                                              and np.array_equal(source_s.to_numpy(), sigma))
        with output.with_suffix(".npz").open("xb") as stream:
            np.savez(stream, **arrays)
        report.update(status="complete", archive_sha256=hashlib.sha256(output.with_suffix(".npz").read_bytes()).hexdigest())
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if base is not None:
            base.close()
        if "source_hashes_before" in report:
            before = report["source_hashes_before"]
            report["source_hashes_after"] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                              for name in before}
            report["imported_sources_unchanged"] = before == report["source_hashes_after"]
        report["checkpoint_unchanged"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == data["identity"]["checkpoint_sha256"]
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
