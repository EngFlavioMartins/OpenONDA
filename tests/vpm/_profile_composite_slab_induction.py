"""UNWIRED whole saved-state induction: production self-FMM plus mesh images.

Explicit qualification only. No solver advance, provider suppression, native
restart override or runtime backend admission. Each repeat builds a NEW image
owner and measures source readback, analytic tail bound, all finite images,
local correction, transfers, stretching accumulation and owner teardown.
Primary FMM arithmetic and source/core conventions remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fenv-library", type=Path, required=True)
    parser.add_argument("--tail-bound", choices=("pointwise", "aabb"), default="pointwise")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("bounded1..3 repeats required")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if (output.parent != root/"tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
            or output.suffix != ".json"):
        raise ValueError("new evidence must be in the ordinary cylinder solution directory")
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("refusing to overwrite qualification evidence")

    import cupy as cp
    import taichi as ti

    from source.solvers.vpm.kernels.base import make_vortex_kernel
    from source.solvers.vpm.physics.base import PhysicsBase
    from source.solvers.vpm.physics.induction.base import _STRETCHING_MODES
    from source.solvers.vpm.physics.induction.fmm import FMMInduction
    from source.solvers.vpm.physics.induction.stretching import stretching_rate
    from tests.vpm._cupy_slab_field_mesh_fused import SlabFieldMeshFusedGPU
    from tests.vpm._finite_slab_native_qualification import authenticate_inputs, compare_fields
    from tests.vpm._gaussian_core_correction_gpu import GaussianCoreCorrectionGPU
    from tests.vpm._gaussian_tail_aabb_certificate import prepare_tail_source, query_tail_bound
    from tests.vpm._gaussian_tail_arithmetic import add, leading_tail, pairwise_sum, point
    from tests.vpm._gaussian_tail_remainder_interval import tail_remainder
    from tests.vpm._ieee_interval_environment import IntervalEnvironment

    environment = IntervalEnvironment(args.fenv_library)

    data = authenticate_inputs(args.checkpoint, args.oracle)
    config, count = data["identity"]["configuration"], len(data["position"])
    slab = config["induction"]
    baseline_report = json.loads(args.baseline.with_suffix(".json").read_text())
    if baseline_report["checkpoint_sha256"] != data["identity"]["checkpoint_sha256"]:
        raise ValueError("whole-field control checkpoint mismatch")
    baseline = np.load(args.baseline.with_suffix(".npz"), allow_pickle=False)
    if baseline["velocity"].shape != (count, 3) or baseline["gradient"].shape != (count, 3, 3):
        raise ValueError("whole-field control shapes disagree")
    report = {"status": "running", "production_admissible": False, "identity": data["identity"],
              "scope": "Candidate composite induction only; not a complete stage RHS or coupled exchange",
              "source_scope": "unchanged saved arrays, NEW image-owner construction and teardown everyrepeat; primary FMM owner retained",
              "finite_images": sum(len(block["images"]) for block in data["blocks"]),
              "tail_policy": "explicit experimental absolute Gaussian omitted-infinite-tail bound; no completion added; unchanged tolerance/scales/shell cap",
              "tail_bound_evaluation": args.tail_bound,
              "tail_certificate_excludes": "finite mesh/FFT/GPU and primary-FMM errors; these need independent numerical qualification",
              "experimental_auxiliary_parameters": {"tau": .12, "spacing": .035, "order": 10, "cutoff": .6},
              "measurements": [], "baseline_report": str(args.baseline.with_suffix(".json").resolve())}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    source_paths = {Path(module.__file__).resolve() for name, module in sys.modules.items()
                    if name.startswith(("source.", "tests.vpm.")) and getattr(module, "__file__", None)}
    source_paths.add(Path(__file__).resolve())
    # Lazy imports made during GPU cardinal construction are frozen too.
    source_paths.add(Path(__file__).with_name("_cupy_cardinal_stencil.py").resolve())
    source_paths.add(Path(__file__).with_name("_ieee_interval_environment.c").resolve())
    source_paths.add(environment.library_path)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    report["source_hashes"] = hashes

    @ti.kernel
    def accumulate_images(velocity: ti.template(), gradient: ti.template(), rate: ti.template(),
                          strength: ti.template(), image_u: ti.types.ndarray(dtype=ti.f32, ndim=2),
                          image_j: ti.types.ndarray(dtype=ti.f32, ndim=3), n: ti.i32, mode: ti.i32):
        for i in range(n):
            u = ti.Vector([image_u[i, a] for a in ti.static(range(3))])
            j = ti.Matrix([[image_j[i, a, b] for b in ti.static(range(3))] for a in ti.static(range(3))])
            velocity[i] += u
            gradient[i] += j
            rate[i] += stretching_rate(j, strength[i], mode)

    base = None
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2)
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError("CUDA required; no silent CPU fallback")
        physics = PhysicsBase(particle_kernel=config["particle_kernel"],
                              max_n_particles=config["max_n_particles"], accumulator_dtype=ti.f32)
        base = FMMInduction(stretching_scheme=slab["stretching_scheme"])
        base.bind(physics, kernel=make_vortex_kernel(config["particle_kernel"]))
        x_field = ti.Vector.field(3, ti.f32, shape=count)
        gamma_field = ti.Vector.field(3, ti.f32, shape=count)
        sigma_field = ti.field(ti.f32, shape=count)
        velocity, rate = (ti.Vector.field(3, ti.f32, shape=count) for _ in range(2))
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=count)
        for field, key in ((x_field, "position"), (gamma_field, "strength"), (sigma_field, "core")):
            field.from_numpy(data[key])
        images = [image for block in data["blocks"] for image in block["images"]]
        world = [(2*k*(slab["z_max"]-slab["z_min"])+(2*slab["z_min"] if odd else 0.), odd)
                 for k, odd in images]
        for repeat in range(args.repeats):
            ti.sync()
            started = time.perf_counter()
            record = {"repeat": repeat, "cold_execution": repeat == 0}
            base.evaluate_stage(position=x_field, vortex_strength=gamma_field, core_radius=sigma_field,
                                count=count, velocity_out=velocity, velocity_gradient_out=gradient,
                                vortex_strength_rate_out=rate)
            ti.sync()
            record["primary_fmm_seconds"] = time.perf_counter()-started
            part = time.perf_counter()
            x, gamma, sigma = x_field.to_numpy(), gamma_field.to_numpy(), sigma_field.to_numpy()
            record["source_readback_seconds"] = time.perf_counter()-part
            if np.any(x[:, 2] < slab["z_min"]) or np.any(x[:, 2] > slab["z_max"]):
                raise ValueError("physical source outside slab; nothing admitted")
            part = time.perf_counter()
            with environment.ieee():
                bounds = {}
                if args.tail_bound == "aabb":
                    source_moments = prepare_tail_source(x, gamma, sigma, z_min=slab["z_min"], z_max=slab["z_max"])
                    certificate = query_tail_bound(source_moments, x.min(axis=0), x.max(axis=0),
                                                   shells=slab["max_shells"]-1)
                    bounds = {"velocity": certificate.velocity_upper, "gradient": certificate.gradient_upper}
                else:
                    leading = leading_tail(x, gamma, x, z_min=slab["z_min"], z_max=slab["z_max"],
                                           shells=slab["max_shells"]-1)
                    remainder = tail_remainder(x, gamma, sigma, x, z_min=slab["z_min"], z_max=slab["z_max"],
                                               shells=slab["max_shells"]-1)
                    for name, enclosure, residual in (
                        ("velocity", leading.velocity_interval, remainder.velocity),
                        ("gradient", leading.gradient_interval, remainder.gradient)):
                        upper = np.maximum(np.abs(enclosure.lower), np.abs(enclosure.upper)).reshape(count, -1)
                        total = add(pairwise_sum(point(upper.T)), point(residual)).upper
                        bounds[name] = float(total.max())
                for name, scale in (("velocity", slab["velocity_scale"]), ("gradient", slab["gradient_scale"])):
                    budget = np.nextafter(slab["tail_tolerance"]*scale, -np.inf)
                    if bounds[name] > budget:
                        raise RuntimeError(f"certified {name} omitted tail exceeds unchanged tolerance")
            record["tail_certificate_seconds"] = time.perf_counter()-part
            record["absolute_omitted_tail_bounds"] = bounds
            free, total_memory = cp.cuda.runtime.memGetInfo()
            record["gpu_free_before_image_owner_bytes"] = free
            record["gpu_total_bytes"] = total_memory
            if free < 2.5*1024**3:
                raise MemoryError("less than2.5GiB free for bounded coexisting image qualification")
            part = time.perf_counter()
            with (
                SlabFieldMeshFusedGPU(x, gamma, x, zmin=slab["z_min"], zmax=slab["z_max"],
                                     tau=.12, spacing=.035, order=10, dtype="float32",
                                     stencil_backend="gpu") as mesh,
                GaussianCoreCorrectionGPU(mesh.source_x, mesh.source_gamma, sigma, tau=.12,
                                          cutoff=.6, accumulation_dtype="float32") as correction,
            ):
                u, j, mesh_report = mesh.evaluate(images)
                du, dj, correction_report = correction.evaluate(mesh.target_x, world)
                iu = cp.asnumpy(u).astype(np.float64)+cp.asnumpy(du)
                ij = cp.asnumpy(j).astype(np.float64)+cp.asnumpy(dj)
                record["mesh_diagnostics"] = mesh_report
                record["correction_diagnostics"] = correction_report
                del u, j, du, dj
            record["image_build_evaluate_copy_close_seconds"] = time.perf_counter()-part
            part = time.perf_counter()
            accumulate_images(velocity, gradient, rate, gamma_field, iu.astype(np.float32),
                              ij.astype(np.float32), count, _STRETCHING_MODES[slab["stretching_scheme"]])
            ti.sync()
            record["image_upload_and_accumulate_seconds"] = time.perf_counter()-part
            record["whole_candidate_induction_seconds"] = time.perf_counter()-started
            values = {"velocity": velocity.to_numpy(), "gradient": gradient.to_numpy(), "rate": rate.to_numpy()}
            if not all(np.isfinite(value).all() for value in values.values()):
                raise FloatingPointError("nonfinite composite candidate field")
            record["whole_field_difference_from_preserved_fmm"] = {
                key: {"relative_l2": float(np.linalg.norm(value-baseline[key])/np.linalg.norm(baseline[key])),
                      "max_component": float(np.abs(value-baseline[key]).max())}
                for key, value in values.items()}
            indices = data["oracle"]["indices"]
            direct = data["oracle"]["direct"]
            record["image_selected_direct_error"] = compare_fields(iu[indices], ij[indices],
                                                                   direct[:, :3], direct[:, 3:12].reshape(-1, 3, 3))
            report["measurements"].append(record)
            publish()
            print(json.dumps({key: record[key] for key in ("repeat", "whole_candidate_induction_seconds",
                              "primary_fmm_seconds", "tail_certificate_seconds", "image_build_evaluate_copy_close_seconds")}), flush=True)
        with output.with_suffix(".npz").open("xb") as stream:
            np.savez(stream, **values, image_velocity=iu, image_gradient=ij)
        report["source_fields_unchanged"] = all(np.array_equal(field.to_numpy(), data[key])
            for field, key in ((x_field, "position"), (gamma_field, "strength"), (sigma_field, "core")))
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["sources_changed"] = [name for name, digest in hashes.items()
                                     if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
        report["checkpoint_unchanged"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == data["identity"]["checkpoint_sha256"]
        if report["sources_changed"] or not report["checkpoint_unchanged"]:
            report["status"] = "failed"
        publish()
        if base is not None:
            base.close()
    if report["status"] != "complete":
        raise RuntimeError("qualification input or source changed")


if __name__ == "__main__":
    main()
