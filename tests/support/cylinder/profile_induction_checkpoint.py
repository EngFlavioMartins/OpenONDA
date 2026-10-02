"""Read-only fixed-state induction qualification against a native VPM checkpoint.

Select an immutable source root to profile an old implementation while the live
source is edited. This does not advance the solution or write scientific samples.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--self-only", action="store_true")
    parser.add_argument(
        "--unwrapped", action="store_true",
        help="Time the public operator without replacing certified backend methods",
    )
    parser.add_argument(
        "--kernel-resources", action="store_true",
        help="Record CUDA launch/register attributes through Taichi's default event profiler",
    )
    parser.add_argument(
        "--exact-reuse", action="store_true",
        help="Qualify the production StageRHS pure-induction cache on identical native sources",
    )
    parser.add_argument(
        "--image-shell",
        type=int,
        help="Profile one finite image shell only; not a complete field or tail qualification",
    )
    parser.add_argument(
        "--target-limit",
        type=int,
        default=8192,
        help="Target prefix for finite-shell screening only",
    )
    parser.add_argument(
        "--legacy-targets",
        action="store_true",
        help="Use the original arbitrary-target API for a finite-shell timing control",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.target_limit < 1:
        parser.error("repeat and target counts must be positive")
    if args.image_shell is not None and (args.image_shell < 0 or args.self_only):
        parser.error(
            "finite image-shell screening requires a nonnegative shell and excludes --self-only"
        )
    if args.legacy_targets and args.image_shell is None:
        parser.error("--legacy-targets requires --image-shell")
    if args.exact_reuse and (args.image_shell is not None or args.repeats < 3):
        parser.error("--exact-reuse requires a complete field and at least three repeats")
    if args.unwrapped and args.image_shell is not None:
        parser.error("--unwrapped requires a complete field, not a finite-shell screen")
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))

    import h5py
    import numpy as np
    import taichi as ti

    from source.solvers.vpm.kernels.base import make_vortex_kernel
    from source.solvers.vpm.physics.base import PhysicsBase
    from source.solvers.vpm.physics.induction.fmm import FMMInduction
    from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction

    for name, module in tuple(sys.modules.items()):
        if name.startswith(("source.", "openonda.")) and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
    with h5py.File(args.checkpoint, "r") as saved:
        config = json.loads(saved["solver"].attrs["numerical_configuration"])
        position = saved["particles/position"][:]
        strength = saved["particles/vortex_strength"][:]
        radius = saved["particles/core_radius"][:]
        clock = float(saved["solver"].attrs["time"])
    count = len(position)
    report = {
        "source_root": str(root),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "particles": count,
        "time": clock,
        "configuration": config,
        "measurements": [],
        "cold_compile_separated": True,
        "status": "running",
        "exact_reuse_qualification": args.exact_reuse,
        "backend_methods_instrumented": not (args.exact_reuse or args.unwrapped),
        "public_operator_timing": args.unwrapped,
        "kernel_resource_profiling": args.kernel_resources,
        "scope": "finite_image_shell_screening"
        if args.image_shell is not None
        else "self_field"
        if args.self_only
        else "complete_induction",
        "source_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "source/solvers/vpm/physics/induction/fmm/device.py",
                "source/solvers/vpm/physics/induction/fmm/targets.py",
                "source/solvers/vpm/physics/induction/fmm/target_geometry.py",
                "source/solvers/vpm/physics/induction/fmm/diagnostics.py",
                "source/solvers/vpm/physics/induction/slip_slab.py",
                "source/solvers/vpm/physics/stage_rhs.py",
                "source/solvers/vpm/physics/induction/reuse.py",
                "source/solvers/vpm/physics/induction/reuse_backends.py",
            )
            if (root / name).is_file()
        },
    }
    output = args.output.resolve()
    if output.with_suffix(".json").exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(f"Refusing to overwrite qualification evidence: {output}")

    def publish():
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")

    publish()
    ti.init(
        arch=ti.cuda, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2,
        kernel_profiler=args.kernel_resources,
    )
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    physics = PhysicsBase(
        particle_kernel=config["particle_kernel"],
        max_n_particles=config["max_n_particles"],
        accumulator_dtype=ti.f32,
    )
    base = FMMInduction(stretching_scheme=config["induction"]["stretching_scheme"])
    slab_config = config["induction"]
    slab = SlipSlabInduction(
        base,
        **{
            key: slab_config[key]
            for key in (
                "z_min",
                "z_max",
                "tail_tolerance",
                "max_shells",
                "velocity_scale",
                "gradient_scale",
            )
        },
    )
    slab.bind(physics, kernel=make_vortex_kernel(config["particle_kernel"]))
    source_position = ti.Vector.field(3, ti.f32, shape=count)
    source_strength = ti.Vector.field(3, ti.f32, shape=count)
    source_radius = ti.field(ti.f32, shape=count)
    velocity = ti.Vector.field(3, ti.f32, shape=count)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=count)
    rate = ti.Vector.field(3, ti.f32, shape=count)
    source_position.from_numpy(position)
    source_strength.from_numpy(strength)
    source_radius.from_numpy(radius)
    base._ensure_workspace(count)
    base.workspace.profile_passes = True
    original_stage = base.evaluate_stage
    original_targets = base.evaluate_targets
    original_blocks = getattr(base, "evaluate_image_block", None)
    record = {}
    screen_images = None
    screen_count = min(count, args.target_limit)
    if args.image_shell is not None:
        from verify_image_operator_checkpoint import image_family

        family = image_family(slab_config["z_min"], slab_config["z_max"], args.image_shell)
        screen_images = family[:1] if args.image_shell == 0 else family[-4:]
        report.update(
            image_shell=args.image_shell, target_count=screen_count, image_count=len(screen_images)
        )
    legacy_screen = screen_images is not None and (args.legacy_targets or original_blocks is None)
    if legacy_screen:
        screen_queries = []
        for shift, odd in screen_images:
            transformed = position[:screen_count].copy()
            transformed[:, 2] = (
                np.float32(shift) - transformed[:, 2]
                if odd
                else transformed[:, 2] - np.float32(shift)
            )
            screen_queries.append(transformed)
        screen_query = ti.Vector.field(3, ti.f32, shape=screen_count * len(screen_images))
        screen_velocity = ti.Vector.field(3, ti.f32, shape=screen_count * len(screen_images))
        screen_gradient = ti.Matrix.field(3, 3, ti.f32, shape=screen_count * len(screen_images))
        screen_query.from_numpy(np.concatenate(screen_queries))
        report["finite_shell_control"] = (
            "Legacy target API timing; host query preparation and final image parity/sum excluded from timed field API"
        )

    def timed_stage(**kwargs):
        ti.sync()
        start = time.perf_counter()
        original_stage(**kwargs)
        ti.sync()
        record["self_seconds"] = time.perf_counter() - start
        record["self_phases"] = dict(base.workspace.last_phase_seconds)

    def timed_targets(**kwargs):
        ti.sync()
        start = time.perf_counter()
        original_targets(**kwargs)
        ti.sync()
        elapsed = time.perf_counter() - start
        record["targets_seconds"] += elapsed
        record["target_calls"] += 1
        record["target_points"] += kwargs["target_count"]
        record["target_call_timings"].append(
            {"targets": kwargs["target_count"], "seconds": elapsed}
        )

    def timed_block(**kwargs):
        ti.sync()
        start = time.perf_counter()
        block = {
            "target_start": kwargs["target_start"],
            "target_count": kwargs["target_count"],
            "image_count": len(kwargs["images"]),
        }
        try:
            original_blocks(**kwargs)
            block["status"] = "complete"
        except BaseException as exc:
            block.update(status="declined-or-failed", error=repr(exc))
            raise
        finally:
            ti.sync()
            block["seconds"] = time.perf_counter() - start
            record["image_block_seconds"] += block["seconds"]
            record["image_block_calls"] += 1
            target = base._target_workspace
            block["diagnostics"] = dict(getattr(target, "last_diagnostics", {}))
            if block["status"] == "complete":
                counts = record["image_block_work"]
                for key in ("m2l_pairs", "near_cell_pairs", "target_count"):
                    counts[key] = counts.get(key, 0) + block["diagnostics"].get(key, 0)
            record["image_blocks"].append(block)
            print(
                json.dumps({"event": "image_block", "repeat": record["repeat"], **block}),
                flush=True,
            )

    if not (args.exact_reuse or args.unwrapped):
        base.evaluate_stage = timed_stage
        base.evaluate_targets = timed_targets
        if original_blocks is not None:
            base.evaluate_image_block = timed_block
    operator = base if args.self_only else slab
    rhs = None
    previous_arrays = None
    previous_hits = 0
    if args.exact_reuse:
        from source.solvers.vpm.physics.induction.base import StageRates, StageState
        from source.solvers.vpm.physics.stage_rhs import StageRHS

        rhs = StageRHS(operator)
        stage_rates = StageRates(velocity, rate, gradient)
    try:
        for repeat in range(args.repeats):
            record = {
                "repeat": repeat,
                "cold": repeat == 0,
                "targets_seconds": 0.0,
                "target_calls": 0,
                "target_points": 0,
                "target_call_timings": [],
                "image_block_seconds": 0.0,
                "image_block_calls": 0,
                "image_block_work": {},
                "image_blocks": [],
            }
            ti.sync()
            start = time.perf_counter()
            if rhs is not None:
                # Deliberately change the requested time. Only the pure
                # autonomous field is reusable; production guards/providers
                # are separately qualified and are never suppressed.
                stage_time = clock + repeat * 0.01
                rhs.evaluate(
                    StageState(source_position, source_strength, source_radius, count, time=stage_time),
                    stage_time,
                    stage_rates,
                )
            elif screen_images is None:
                operator.evaluate_stage(
                    position=source_position,
                    vortex_strength=source_strength,
                    core_radius=source_radius,
                    count=count,
                    velocity_out=velocity,
                    velocity_gradient_out=gradient,
                    vortex_strength_rate_out=rate,
                )
            else:
                with base.fixed_source_targets(
                    source_position, source_strength, source_radius, count
                ):
                    if legacy_screen:
                        timed_targets(
                            source_position=source_position,
                            source_vortex_strength=source_strength,
                            source_core_radius=source_radius,
                            source_count=count,
                            target_position=screen_query,
                            target_count=screen_count * len(screen_images),
                            target_velocity=screen_velocity,
                            target_velocity_gradient=screen_gradient,
                            include_freestream=False,
                            background_velocity=physics._zero_velocity,
                        )
                    else:
                        timed_block(
                            source_position=source_position,
                            source_vortex_strength=source_strength,
                            source_core_radius=source_radius,
                            source_count=count,
                            target_position=source_position,
                            target_start=0,
                            target_count=screen_count,
                            images=screen_images,
                            target_velocity=velocity,
                            target_velocity_gradient=gradient,
                        )
            ti.sync()
            record["total_seconds"] = time.perf_counter() - start
            record["tail"] = slab.last_tail
            record["diagnostics"] = asdict(base.diagnostics)
            if args.unwrapped:
                # No backend wrappers: zero internal call counters above are
                # unobserved, not evidence that no image work was performed.
                for key in (
                    "targets_seconds", "target_calls", "target_points", "target_call_timings",
                    "image_block_seconds", "image_block_calls", "image_block_work", "image_blocks",
                ):
                    record.pop(key, None)
                record["self_phases"] = dict(base.workspace.last_phase_seconds)
            if rhs is not None:
                statistics = rhs.induction_reuse_statistics
                if statistics is None:
                    raise AssertionError("Standard native backend did not create a reuse cache")
                record["induction_reuse"] = asdict(statistics)
                current_arrays = dict(
                    velocity=velocity.to_numpy(), gradient=gradient.to_numpy(), rate=rate.to_numpy()
                )
                if statistics.hits > previous_hits:
                    if previous_arrays is None or any(
                        not np.array_equal(values.view(np.uint8), previous_arrays[name].view(np.uint8))
                        for name, values in current_arrays.items()
                    ):
                        raise AssertionError("A native reuse hit changed a cached output bit")
                    record["hit_outputs_bitwise_equal"] = True
                previous_arrays, previous_hits = current_arrays, statistics.hits
            report["measurements"].append(record)
            publish()
            print(json.dumps(record), flush=True)
        arrays = dict(velocity=velocity.to_numpy(), gradient=gradient.to_numpy())
        if legacy_screen:
            image_velocity = screen_velocity.to_numpy().reshape(len(screen_images), screen_count, 3)
            image_gradient = screen_gradient.to_numpy().reshape(
                len(screen_images), screen_count, 3, 3
            )
            odd = screen_images[:, 1].astype(bool)
            image_velocity[odd, :, 2] *= -1
            image_gradient[odd, :, :2, 2] *= -1
            image_gradient[odd, :, 2, :2] *= -1
            arrays = dict(velocity=image_velocity.sum(axis=0), gradient=image_gradient.sum(axis=0))
        elif screen_images is None:
            arrays["rate"] = rate.to_numpy()
        else:
            arrays = {name: values[:screen_count] for name, values in arrays.items()}
        np.savez(output.with_suffix(".npz"), **arrays)
        if rhs is not None:
            if previous_hits < 1:
                raise AssertionError("Native identical-source qualification obtained no reuse hit")
            report["sources_bitwise_unchanged"] = all(
                np.array_equal(
                    field.to_numpy().view(np.uint8), expected.astype(np.float32).view(np.uint8)
                )
                for field, expected in (
                    (source_position, position), (source_strength, strength), (source_radius, radius)
                )
            )
            if not report["sources_bitwise_unchanged"]:
                raise AssertionError("Read-only induction changed checkpoint sources")
        report["source_files_changed_during_run"] = [
            name for name, digest in report["source_hashes"].items()
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
        ]
        if report["source_files_changed_during_run"]:
            raise RuntimeError("Qualification implementation changed while running")
        if args.kernel_resources:
            # Default CUDA event/Driver API attributes need no restricted
            # hardware performance counters. These are launch/theoretical
            # occupancy attributes, NOT measured SM utilization or spill counts.
            ti.profiler.get_kernel_profiler_total_time()
            report["near_kernel_resources"] = [
                {
                    name: getattr(entry, name)
                    for name in (
                        "name", "kernel_time", "register_per_thread", "shared_mem_per_block",
                        "grid_size", "block_size", "active_blocks_per_multiprocessor",
                    )
                }
                for entry in ti.lang.impl.get_runtime().prog.get_kernel_profiler_records()
                if "evaluate_near_lanes" in entry.name
            ]
        report["status"] = "complete"
        publish()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["incomplete_measurement"] = record
        publish()
        raise
    finally:
        if rhs is not None:
            rhs.close()


if __name__ == "__main__":
    main()
