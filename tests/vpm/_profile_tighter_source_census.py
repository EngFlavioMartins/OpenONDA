"""Bounded immutable-checkpoint source-bound comparison; evaluates no fields."""

# ruff: noqa: I001 -- Archive admission must precede numerical source imports.
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-limit", type=int, default=8192)
    parser.add_argument("--max-shell", type=int, default=2)
    parser.add_argument("--target-stride", type=int, default=64)
    parser.add_argument("--target-offset", type=int, default=0)
    args = parser.parse_args()
    if (args.target_limit < 1 or not 0 <= args.max_shell <= 128 or args.target_stride < 1
            or not 0 <= args.target_offset < args.target_stride):
        parser.error("invalid target prefix, finite shell or explicit sampling pattern")
    root = Path(__file__).resolve().parents[2]
    case = root / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
    output = args.output.resolve()
    if not output.is_relative_to(case / "solution"):
        raise ValueError("qualification output must stay in the normal case solution directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    sys.path.insert(0, str(root))
    from tests.vpm._fmm_tighter_source_census import TighterSourceOrderCensus
    from tests.vpm._fmm_flattened_near_prototype import FROZEN_SOURCE_ROOT, assert_frozen_numerical_imports

    import h5py
    import numpy as np
    import taichi as ti

    from source.solvers.vpm.kernels.base import make_vortex_kernel
    from source.solvers.vpm.physics.base import PhysicsBase
    from source.solvers.vpm.physics.induction.fmm import FMMInduction
    from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator

    paths = [Path(__file__).resolve(), *[
        Path(__file__).with_name(name) for name in (
            "_fmm_tighter_source_census.py", "_fmm_rank_one_remainder.py",
            "_fmm_legacy_descendant_work.py", "_fmm_order_census_prototype.py",
            "_fmm_source_remainder_prototype.py", "_fmm_flattened_near_prototype.py",
        )
    ], *sorted((FROZEN_SOURCE_ROOT / "source").rglob("*.py"))]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with h5py.File(args.checkpoint, "r") as saved:
        configuration = json.loads(saved["solver"].attrs["numerical_configuration"])
        positions = saved["particles/position"][:]
        strengths = saved["particles/vortex_strength"][:]
        cores = saved["particles/core_radius"][:]
        clock = float(saved["solver"].attrs["time"])
    total, targets = len(positions), min(len(positions), args.target_limit)
    report = {
        "status": "running", "source_root": str(FROZEN_SOURCE_ROOT), "source_hashes": hashes,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "particles": total, "target_count": targets, "time": clock,
        "max_shell": args.max_shell, "tile_capacity": 8192, "pair_capacity": 4194304,
        "target_stride": args.target_stride, "target_offset": args.target_offset,
        "configuration": configuration, "blocks": [], "fields_evaluated": False,
        "runtime_admissible_interactions": 0, "convergence_certified": False,
        "adopted_numerical_change": False,
        "cost_warning": "Raw sampled legacy traversal counts; no population extrapolation or speedup claim.",
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def publish():
        output.write_text(json.dumps(report, indent=2) + "\n")

    base = evaluator = census = None
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2)
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError("CUDA required; no silent fallback")
        physics = PhysicsBase(
            particle_kernel=configuration["particle_kernel"], max_n_particles=configuration["max_n_particles"],
            accumulator_dtype=ti.f32,
        )
        base = FMMInduction(stretching_scheme=configuration["induction"]["stretching_scheme"])
        base.bind(physics, kernel=make_vortex_kernel(configuration["particle_kernel"]))
        base._ensure_workspace(total)
        position = ti.Vector.field(3, ti.f32, shape=total)
        strength = ti.Vector.field(3, ti.f32, shape=total)
        core = ti.field(ti.f32, shape=total)
        position.from_numpy(positions)
        strength.from_numpy(strengths)
        core.from_numpy(cores)
        base.workspace.tree.build(position, strength, core, total)
        base.workspace.prepare_source_multipoles(total)
        evaluator = FMMTargetEvaluator(base.workspace, 8192, max_pairs=4194304)
        census = TighterSourceOrderCensus(evaluator, target_stride=args.target_stride, target_offset=args.target_offset)
        minimum, maximum = (configuration["induction"][key] for key in ("z_min", "z_max"))
        length = maximum - minimum
        images = []
        for shell in range(args.max_shell + 1):
            for k in ((0,) if shell == 0 else (-shell, shell)):
                if k:
                    images.append((2 * k * length, False))
                images.append((2 * minimum + 2 * k * length, True))
        shifts, odd = np.zeros(513, np.float32), np.zeros(513, np.int32)
        for index, (shift, reflected) in enumerate(images):
            shifts[index], odd[index] = shift, reflected
        evaluator.image_shift.from_numpy(shifts)
        evaluator.image_odd.from_numpy(odd)
        evaluator.image_count[None], evaluator.block_mode[None] = len(images), 1
        started = perf_counter()
        for start in range(0, targets, 8192):
            count = min(8192, targets - start)
            evaluator.prepare_targets(position, count, target_start=start)
            evaluator.velocity.fill(17)
            evaluator.gradient.fill(19)
            for policy in ("historical", "tight"):
                result = census.run(len(images), policy=policy)
                result.update(target_start=start, target_count=count, image_count=len(images))
                report["blocks"].append(result)
                print(json.dumps({"event": "source_bound_census", **result}), flush=True)
                publish()
            if not np.all(evaluator.velocity.to_numpy() == 17) or not np.all(evaluator.gradient.to_numpy() == 19):
                raise AssertionError("field-free census changed private output sentinels")
        ti.sync()
        report.update(status="complete", census_seconds=perf_counter() - started, private_outputs_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            assert_frozen_numerical_imports()
        except BaseException as error:
            report.update(status="failed", import_error=f"{type(error).__name__}: {error}")
        report["sources_changed"] = [path for path, digest in hashes.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
        if report["sources_changed"]:
            report.update(status="failed", error="qualification source changed during execution")
        cleanup_errors = []
        for owner in (census, evaluator, None if base is None else base.workspace):
            if owner is not None:
                try:
                    owner.destroy()
                except BaseException as error:
                    cleanup_errors.append(f"{type(error).__name__}: {error}")
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
        publish()
    if report["status"] != "complete":
        raise RuntimeError("census did not complete with unchanged immutable source")


if __name__ == "__main__":
    main()
