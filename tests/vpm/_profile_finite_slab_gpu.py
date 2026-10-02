"""UNWIRED saved-state finite-image GPU cost/accuracy screen; no advancement.

Run with the isolated optional CuPy interpreter. All evidence must be new
files in the ordinary cylinder solution directory. First-block direct CPU
work and report I/O are explicitly excluded from GPU operator wall timings.
This is not a complete induction call or an end-to-end solver speed claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._finite_slab_native_qualification import (
    authenticate_inputs,
    compare_fields,
    direct_small_block,
    native_tail_maxima,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", choices=("first", "closing", "all"), default="first")
    parser.add_argument("--spacing", type=float, default=.035)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--correction-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--stencil-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--smooth-variant", choices=("streamed", "fused"), default="streamed")
    parser.add_argument("--coalesce-finite-blocks", action="store_true",
                        help="Evaluate the exact same finite image set in one batch; explicitly excludes tail-controller qualification")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3 or args.spacing not in (.035, .03):
        parser.error("bounded1..3 repeats and preregistered .035/.03 auxiliary spacing required")
    if args.coalesce_finite_blocks and args.blocks != "all":
        parser.error("coalescing requires the entire supplied finite image family")
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    solution = root / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/solution"
    if output.parent != solution or output.suffix != ".json":
        raise ValueError("new JSON evidence must be in the ordinary cylinder solution directory")
    archive_path = output.with_suffix(".npz")
    if output.exists() or archive_path.exists():
        raise FileExistsError("refusing to overwrite GPU qualification evidence")
    numerical_names = ("_profile_finite_slab_gpu.py", "_finite_slab_native_qualification.py",
                       "_cupy_slab_field_mesh.py", "_gaussian_core_correction_gpu.py",
                       "_cupy_cardinal_stencil.py",
                       "_finite_slab_field_mesh_reference.py", "_finite_image_mesh_reference.py",
                       "_slip_periodic_gaussian_oracle.py")
    if args.smooth_variant == "fused":
        numerical_names += ("_cupy_slab_field_mesh_fused.py",)
    hashes = {str(Path(__file__).with_name(name)): hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in numerical_names}
    report = {"status": "running", "production_admissible": False, "source_hashes": hashes,
              "scope": "finite image-only GPU prototype; excludes primary self-FMM and solver work",
              "parameters": vars(args) | {"checkpoint": str(args.checkpoint), "oracle": str(args.oracle),
                                           "output": str(args.output)}, "measurements": []}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        import cupy as cp

        from tests.vpm._cupy_slab_field_mesh import SlabFieldMeshGPU
        from tests.vpm._gaussian_core_correction_gpu import GaussianCoreCorrectionGPU

        smooth_type = SlabFieldMeshGPU
        if args.smooth_variant == "fused":
            from tests.vpm._cupy_slab_field_mesh_fused import SlabFieldMeshFusedGPU
            smooth_type = SlabFieldMeshFusedGPU
        report["runtime"] = {"cupy": cp.__version__, "cupy_path": cp.__file__,
                             "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                             "cuda_driver": cp.cuda.runtime.driverGetVersion()}
        bound_report = solution / "finite-image-correction-census-20261002.json"
        report["separate_local_correction_bound_evidence"] = {
            "report": str(bound_report), "sha256": hashlib.sha256(bound_report.read_bytes()).hexdigest(),
            "tau": .12, "cutoff": .6,
            "scope": "finite513-image omitted local correction only; excludes FFT/interpolation/roundoff"}
        data = authenticate_inputs(args.checkpoint, args.oracle)
        x, gamma, sigma = data["position"], data["strength"], data["core"]
        slab = data["identity"]["configuration"]["induction"]
        zmin, zmax = slab["z_min"], slab["z_max"]
        report.update(identity=data["identity"], oracle_provenance=data["oracle_provenance"],
                      admission_helper_sha256=data["admission_helper_sha256"])
        blocks = data["blocks"]
        if args.blocks == "first":
            blocks = blocks[:1]
        elif args.blocks == "closing":
            blocks = blocks[-1:]
        if args.coalesce_finite_blocks:
            blocks = [{"start": 0, "end": blocks[-1]["end"], "complete": False,
                       "images": [image for block in blocks for image in block["images"]]}]
            report["finite_coalescing_scope"] = (
                "All identical supplied images evaluated in one convolution; summation order changes. "
                "No old block norms are computed and no tail stopping criterion is claimed to pass. "
                "Not admitted to the runtime controller.")
        free, total = cp.cuda.runtime.memGetInfo()
        if free < 3 * 1024**3:
            raise MemoryError("at least3GiB free GPU memory required for isolated bounded qualification")
        report["initial_device_memory"] = {"free": free, "total": total}
        target_indices = data["oracle"]["indices"]
        # Independent direct CPU evidence is calculated before timing the GPU.
        expected = None
        if args.blocks == "first":
            started = time.perf_counter()
            expected = direct_small_block(x, gamma, sigma, x[target_indices], blocks[0]["images"],
                                          zmin=zmin, zmax=zmax)
            report["independent_direct_seconds"] = time.perf_counter() - started
        elif args.blocks == "all":
            exact = data["oracle"]["direct"]
            expected = exact[:, :3], exact[:, 3:12].reshape(-1, 3, 3)
        started = time.perf_counter()
        with smooth_type(x, gamma, x, zmin=zmin, zmax=zmax, tau=.12,
                         spacing=args.spacing, order=10, dtype=args.dtype,
                         stencil_backend=args.stencil_backend) as mesh:
            report["smooth_scope_build_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            with GaussianCoreCorrectionGPU(mesh.source_x, mesh.source_gamma, sigma, tau=.12, cutoff=.6,
                                           accumulation_dtype=args.correction_dtype) as correction:
                report["correction_scope_build_seconds"] = time.perf_counter() - started
                for repeat in range(args.repeats):
                    full_start = time.perf_counter()
                    total_u, total_j = np.zeros((len(x), 3)), np.zeros((len(x), 3, 3))
                    block_reports = []
                    previous_relative, consecutive = float("inf"), 0
                    first_would_stop_at = None
                    for block in blocks:
                        block_start = time.perf_counter()
                        u, j, smooth = mesh.evaluate(block["images"])
                        world = [(2*k*(zmax-zmin)+(2*zmin if odd else 0.), odd)
                                 for k, odd in block["images"]]
                        du, dj, local = correction.evaluate(mesh.target_x, world)
                        # Explicit public copies; private field ownership only.
                        transfer_start = time.perf_counter()
                        uu, jj = cp.asnumpy(u).astype(np.float64), cp.asnumpy(j).astype(np.float64)
                        uu += cp.asnumpy(du)
                        jj += cp.asnumpy(dj)
                        transfer_seconds = time.perf_counter() - transfer_start
                        del u, j, du, dj
                        velocity_max, gradient_max = native_tail_maxima(uu, jj)
                        relative = max(velocity_max / slab["velocity_scale"], gradient_max / slab["gradient_scale"])
                        if block["start"] and block["complete"]:
                            decaying = relative <= previous_relative * 1.2
                            consecutive = consecutive + 1 if relative <= slab["tail_tolerance"] and decaying else 0
                            previous_relative = relative
                            if consecutive >= 2 and first_would_stop_at is None:
                                first_would_stop_at = block["end"]
                        total_u += uu
                        total_j += jj
                        evidence = {"start": block["start"], "end": block["end"], "images": len(world),
                                    "complete_block": block["complete"], "smooth": smooth, "correction": local,
                                    "copy_and_host_combine_seconds": transfer_seconds,
                                    "block_whole_seconds": time.perf_counter()-block_start,
                                    "velocity_max": velocity_max, "gradient_frobenius_max": gradient_max,
                                    "relative": relative, "consecutive_below_gate": consecutive}
                        block_reports.append(evidence)
                        print(json.dumps({"repeat": repeat, "block": block["end"],
                                          "seconds": evidence["block_whole_seconds"], "relative": relative}), flush=True)
                    measurement = {"repeat": repeat, "cold_execution": repeat == 0,
                                   "source_scope": "immutable source/target owner shared across repeats; changed-state scope rebuild not included",
                                   "operator_wall_seconds": time.perf_counter()-full_start,
                                   "blocks": block_reports,
                                   "fixed_supplied_blocks_all_evaluated": True,
                                   "first_would_stop_at": first_would_stop_at,
                                   "tail_norms": "native f32 ordered norm arithmetic; no tolerance changes",
                                   "full_unchanged_tail_gate_met": consecutive >= 2
                                   if args.blocks == "all" and not args.coalesce_finite_blocks else None}
                    if expected is not None:
                        measurement["selected_direct_error"] = compare_fields(total_u[target_indices], total_j[target_indices], *expected)
                    report["measurements"].append(measurement)
                    publish()
        with archive_path.open("xb") as stream:
            np.savez(stream, velocity=total_u, gradient=total_j, indices=target_indices,
                     source_position=x, source_strength=gamma, source_core=sigma)
        report.update(status="complete", archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                      accuracy_scope="selected targets only; no global mesh/roundoff or end-to-end certificate",
                      source_checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == data["identity"]["checkpoint_sha256"])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["sources_changed"] = [name for name, digest in hashes.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
        if report["sources_changed"]:
            report["status"] = "failed"
        publish()
    if report["status"] != "complete":
        raise RuntimeError("qualification sources changed")


if __name__ == "__main__":
    main()
