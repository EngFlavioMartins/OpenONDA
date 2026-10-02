"""Isolated qualification driver; never used by a production simulation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    assets = root / "tests/support/cylinder"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(assets))
    from tests.vpm._fmm_specialized_near_prototype import specialized_near_factory

    prototype = root / "tests/vpm/_fmm_specialized_near_prototype.py"
    digest = hashlib.sha256(prototype.read_bytes()).hexdigest()
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    specification = importlib.util.spec_from_file_location(
        "native_operator_qualification", assets / "profile_induction_checkpoint.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    with specialized_near_factory():
        module.main()
    import taichi as ti

    report = json.loads(output.with_suffix(".json").read_text())
    report["qualification_prototype"] = str(prototype)
    report["qualification_prototype_sha256"] = digest
    report["qualification_prototype_changed"] = (
        hashlib.sha256(prototype.read_bytes()).hexdigest() != digest
    )
    if "--kernel-resources" in sys.argv:
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
            if "evaluate_near_class_lanes" in entry.name
        ]
    if report["qualification_prototype_changed"]:
        report["status"] = "failed"
        report["error"] = "Qualification prototype changed during execution"
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "complete":
        raise RuntimeError(report.get("error", "incomplete qualification"))


if __name__ == "__main__":
    main()
