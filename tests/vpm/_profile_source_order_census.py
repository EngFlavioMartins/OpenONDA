"""One-process, unwired native-tree census around the existing field profiler.

The normal operator still supplies every output. Census duration is explicitly
included in the outer profiler times, so those times MUST NOT be compared as an
operator speed measurement. Run one finite image shell with one repeat only.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter


def main():
    root = Path(__file__).resolve().parents[2]
    assets = root / "tests/support/cylinder"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(assets))
    if "--image-shell" not in sys.argv or "--exact-reuse" in sys.argv:
        raise ValueError("census requires a finite shell and excludes exact-reuse profiling")
    if "--repeats" not in sys.argv or sys.argv[sys.argv.index("--repeats") + 1] != "1":
        raise ValueError("census requires exactly one repeat")

    import taichi as ti

    from source.solvers.vpm.physics.induction import reuse_backends
    from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
    from tests.vpm._fmm_order_census_prototype import SourceOrderFeasibilityCensus

    # First import snapshots the unmodified backend contract before any
    # qualification-only host-method interception. No cached solver is run.
    assert reuse_backends.FMMTargetEvaluator is FMMTargetEvaluator
    originals = FMMTargetEvaluator.evaluate_image_block
    results = []
    prototype_paths = [
        root / "tests/vpm/_fmm_order_census_prototype.py",
        root / "tests/vpm/_fmm_source_remainder_prototype.py",
        Path(__file__).resolve(),
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in prototype_paths}

    def field_then_census(evaluator, images, *args, **kwargs):
        images = tuple(images)
        originals(evaluator, images, *args, **kwargs)
        # Preserve a byte-exact copy of both private output arrays across the
        # census, which may reuse only traversal scratch, never field outputs.
        velocity = evaluator.velocity.to_numpy()
        gradient = evaluator.gradient.to_numpy()
        census = None
        ti.sync()
        started = perf_counter()
        try:
            census = SourceOrderFeasibilityCensus(evaluator)
            result = census.run(len(images))
            import numpy as np

            if not np.array_equal(velocity.view(np.uint8), evaluator.velocity.to_numpy().view(np.uint8)):
                raise AssertionError("census modified velocity output")
            if not np.array_equal(gradient.view(np.uint8), evaluator.gradient.to_numpy().view(np.uint8)):
                raise AssertionError("census modified gradient output")
            result["outputs_bitwise_unchanged"] = True
            result["seconds_including_metadata"] = perf_counter() - started
            results.append(result)
        finally:
            if census is not None:
                census.destroy()

    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    specification = importlib.util.spec_from_file_location(
        "native_census_qualification", assets / "profile_induction_checkpoint.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    FMMTargetEvaluator.evaluate_image_block = field_then_census
    try:
        module.main()
    finally:
        FMMTargetEvaluator.evaluate_image_block = originals
    report = json.loads(output.with_suffix(".json").read_text())
    report["source_order_feasibility_census"] = results
    report["qualification_only_source_hashes"] = hashes
    report["qualification_sources_changed"] = [
        str(path) for path in prototype_paths
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]
    ]
    report["timing_scope_warning"] = "Includes census and host observations; NOT an operator speed benchmark"
    if report["qualification_sources_changed"]:
        report["status"] = "failed"
        report["error"] = "Qualification source changed while running"
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "complete":
        raise RuntimeError(report.get("error", "incomplete census"))


if __name__ == "__main__":
    main()
