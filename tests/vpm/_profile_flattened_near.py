"""Isolated frozen-source screen of flattened near accumulators, not a solver."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tests.vpm._fmm_flattened_near_prototype import (
        FROZEN_SOURCE_ROOT,
        assert_frozen_numerical_imports,
        flattened_near_factory,
    )

    source = Path(sys.argv[sys.argv.index("--source-root") + 1]).resolve()
    if source != FROZEN_SOURCE_ROOT:
        raise ValueError(f"this experiment requires the preserved source: {FROZEN_SOURCE_ROOT}")
    if "--exact-reuse" in sys.argv:
        raise ValueError("the experimental target class is deliberately not cache-certified")
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    assets = root / "tests/support/cylinder"
    sys.path.insert(0, str(assets))
    profiler = assets / "profile_induction_checkpoint.py"
    prototype = Path(__file__).with_name("_fmm_flattened_near_prototype.py")
    paths = (prototype, Path(__file__).resolve(), profiler)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    spec = importlib.util.spec_from_file_location("flattened_accumulator_profile", profiler)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failure = None
    try:
        with flattened_near_factory():
            module.main()
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        assert_frozen_numerical_imports()
        if output.with_suffix(".json").exists():
            import taichi as ti

            report = json.loads(output.with_suffix(".json").read_text())
            changed = [
                path for path, digest in hashes.items()
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            ]
            report["flattened_accumulator_qualification"] = {
                "source_hashes": hashes, "sources_changed": changed,
                "source_partition_changed": False,
                "radial_operator_changed": False,
                "summation_order": "one lane subtotal across target ancestors/source subtrees",
                "parity": "unchanged inverse query; physical parity per terminal source",
                "additional_owned_device_bytes": 0,
            }
            if "--kernel-resources" in sys.argv and ti.lang.impl.get_runtime().prog is not None:
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
                    if "evaluate_flattened_near_lanes" in entry.name
                    or "evaluate_near_lanes" in entry.name
                ]
            if changed:
                report.update(status="failed", error="Qualification source changed during execution")
            if failure is not None:
                report.update(status="failed", error=failure)
            output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
