"""Frozen-native p7/.04 versus hypothetical p12/.1 work census, no fields."""

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
    parser.add_argument("--max-shell", type=int, default=2)
    parser.add_argument("--target-limit", type=int)
    args = parser.parse_args()
    if args.max_shell < 0 or args.max_shell > 128:
        parser.error("shell range must remain inside the preserved finite image family")
    if args.target_limit is not None and args.target_limit < 1:
        parser.error("target limit must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite qualification evidence: {output}")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tests.vpm._fmm_target_order_census import TargetOrderCensusEvaluator, taylor_coefficient
    from tests.vpm._fmm_flattened_near_prototype import (
        FROZEN_SOURCE_ROOT, assert_frozen_numerical_imports,
    )

    import h5py
    import numpy as np
    import taichi as ti

    from source.solvers.vpm.kernels.base import make_vortex_kernel
    from source.solvers.vpm.physics.base import PhysicsBase
    from source.solvers.vpm.physics.induction.fmm import FMMInduction

    paths = [
        Path(__file__).resolve(), Path(__file__).with_name("_fmm_target_order_census.py"),
        Path(__file__).with_name("_fmm_flattened_near_prototype.py"),
        *sorted((FROZEN_SOURCE_ROOT / "source").rglob("*.py")),
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with h5py.File(args.checkpoint, "r") as saved:
        configuration = json.loads(saved["solver"].attrs["numerical_configuration"])
        positions = saved["particles/position"][:]
        strengths = saved["particles/vortex_strength"][:]
        cores = saved["particles/core_radius"][:]
        clock = float(saved["solver"].attrs["time"])
    total = len(positions)
    target_count = min(total, args.target_limit or total)
    report = {
        "status": "running", "source_root": str(FROZEN_SOURCE_ROOT),
        "source_hashes": hashes, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "particles": total, "target_count": target_count, "time": clock,
        "max_shell": args.max_shell, "tile_capacity": 8192,
        "configuration": configuration, "blocks": [],
        "policy": {"baseline": {"order": 7, "ratio": 0.04}, "hypothetical": {"order": 12, "ratio": 0.1}},
        "exact_arithmetic_taylor_coefficients": {
            str(order): [taylor_coefficient(order, d, ratio) for d in (1, 2)]
            for order, ratio in ((7, 0.04), (12, 0.1))
        },
        "fields_evaluated": False, "convergence_tested": False,
        "adopted_numerical_change": False,
        "cost_warning": "Coefficient visits and target terms are different work units, not FLOPs or runtime.",
        "accuracy_warning": "Taylor bound only; f32 and complete core/field error not certified.",
    }

    def publish():
        output.write_text(json.dumps(report, indent=2) + "\n")

    publish()
    base = evaluator = None
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2)
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError("CUDA required; silent CPU fallback is not allowed")
        physics = PhysicsBase(
            particle_kernel=configuration["particle_kernel"],
            max_n_particles=configuration["max_n_particles"], accumulator_dtype=ti.f32,
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
        evaluator = TargetOrderCensusEvaluator(base.workspace, 8192, max_pairs=262144)
        z_min, z_max = (configuration["induction"][key] for key in ("z_min", "z_max"))
        length = z_max - z_min
        image_blocks = []
        shell = 0
        while shell <= args.max_shell:
            end = min(args.max_shell, 0 if shell == 0 else 1 if shell == 1 else 2 * shell - 2)
            images = []
            for current_shell in range(shell, end + 1):
                for k in ((0,) if current_shell == 0 else (-current_shell, current_shell)):
                    if k:
                        images.append((2 * k * length, False))
                    images.append((2 * z_min + 2 * k * length, True))
            image_blocks.append((shell, end, images))
            shell = end + 1
        started = perf_counter()
        for start in range(0, target_count, 8192):
            count = min(8192, target_count - start)
            evaluator.prepare_targets(position, count, target_start=start)
            # Sentinel private outputs make accidental field publication
            # observable without ever constructing a hypothetical field.
            evaluator.velocity.fill(17)
            evaluator.gradient.fill(19)
            for shell, end, images in image_blocks:
                block = evaluator.census_image_block(images)
                block.update(target_start=start, first_shell=shell, last_shell=end)
                report["blocks"].append(block)
                print(json.dumps({"event": "target_order_census", **block}), flush=True)
                publish()
            if not np.all(evaluator.velocity.to_numpy() == 17) or not np.all(evaluator.gradient.to_numpy() == 19):
                raise AssertionError("census modified private field outputs")
        ti.sync()
        total_keys = (
            "old_m2l_packets", "old_monopole_packets", "old_monopole_target_terms",
            "promotable_packets", "promotable_target_terms", "successful_batches",
            "discarded_capacity_attempts", "legacy_subtree_target_jobs_unchanged",
            "p7_old_translation_vector_coefficients", "p12_added_translation_vector_coefficients",
            "p12_global_order_extra_old_translation_coefficients",
            "p12_selective_additional_local_target_coefficient_visits",
            "p12_global_additional_local_target_coefficient_visits",
            "hypothetical_m2l_capacity_exceeding_batches",
        )
        report["totals"] = {key: sum(block[key] for block in report["blocks"]) for key in total_keys}
        report.update(status="complete", census_seconds=perf_counter() - started,
                      private_output_sentinels_unchanged=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        assert_frozen_numerical_imports()
        report["sources_changed"] = [
            path for path, digest in hashes.items()
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
        ]
        if report["sources_changed"]:
            report.update(status="failed", error="Qualification source changed during execution")
        publish()
        if evaluator is not None:
            evaluator.destroy()
        if base is not None and base.workspace is not None:
            base.workspace.destroy()


if __name__ == "__main__":
    main()
