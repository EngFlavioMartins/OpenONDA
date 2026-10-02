"""Isolated instrumentation-only native AABB decision census."""

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
    if "--image-shell" not in sys.argv or "--exact-reuse" in sys.argv:
        raise ValueError("requires finite-shell instrumentation without exact-reuse")
    if "--repeats" not in sys.argv or sys.argv[sys.argv.index("--repeats") + 1] != "1":
        raise ValueError("census requires exactly one repeat")
    from tests.vpm._fmm_aabb_device_census import aabb_census_factory

    prototypes = [
        root / "tests/vpm/_fmm_aabb_census_prototype.py",
        root / "tests/vpm/_fmm_aabb_device_census.py",
        Path(__file__).resolve(),
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in prototypes}
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    specification = importlib.util.spec_from_file_location(
        "aabb_native_qualification", assets / "profile_induction_checkpoint.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    with aabb_census_factory():
        module.main()
    report = json.loads(output.with_suffix(".json").read_text())
    report["qualification_only_source_hashes"] = hashes
    report["qualification_sources_changed"] = [
        str(path) for path in prototypes
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]
    ]
    report["timing_scope_warning"] = "Includes all-point classification validation; NOT an operator speed benchmark"
    report["aabb_decision_operator_adopted"] = False
    if report["qualification_sources_changed"]:
        report["status"] = "failed"
        report["error"] = "Qualification source changed while running"
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "complete":
        raise RuntimeError(report.get("error", "incomplete AABB census"))


if __name__ == "__main__":
    main()
