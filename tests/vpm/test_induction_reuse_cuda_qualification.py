"""Isolated CUDA qualification of exact reuse and the real StageRHS dispatch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _has_cuda():
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                check=False,
                timeout=5,
            ).returncode
            == 0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.qualification
@pytest.mark.skipif(not _has_cuda(), reason="CUDA device unavailable")
def test_exact_induction_reuse_cuda_in_isolated_runtime(tmp_path, record_property):
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root), environment.get("PYTHONPATH", "")))
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(tmp_path)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report = json.loads((tmp_path / "induction-reuse-cuda.json").read_text())
    assert report["arch"] == "cuda"
    record_property("induction_reuse_cuda", report)
    print(json.dumps(report, sort_keys=True))


def _worker(directory):
    import taichi as ti

    from tests.vpm import test_induction_exact_reuse as checks
    from tests.vpm import test_stage_induction_reuse as integrated

    ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=0.03, offline_cache=False)
    assert ti.lang.impl.current_cfg().arch == ti.cuda
    completed = []

    def run(check, *args):
        check(*args)
        completed.append(check.__name__)

    try:
        for dtype in (ti.f32, ti.f64):
            run(checks.test_exact_reuse_private_outputs_and_state_local_diagnostics, dtype)
        for field in ("position", "strength", "radius"):
            h = checks.Harness()
            try:
                run(checks.test_same_object_mutation_without_revision_forces_miss, h, field)
            finally:
                h.reuse.close()
        for check in (
            checks.test_count_order_signed_zero_and_rollback_are_exact,
            checks.test_identical_stage_zero_copy_hits_but_equal_time_changed_stage_misses,
            checks.test_state_check_rate_disabled_then_full_stage_uses_complete_private_result,
            checks.test_failed_miss_leaves_caller_unpublished_and_entry_invalid,
            checks.test_output_precision_mismatch_bypasses_without_rounding,
            checks.test_growth_rebinds_storage_and_matches_original_backend,
        ):
            h = checks.Harness()
            try:
                run(check, h)
            finally:
                h.reuse.close()
        run(integrated.test_default_dispatch_retains_backend_guard_and_all_providers)
        run(integrated.test_default_slab_dispatch_hits_and_guard_failure_never_publishes)
        run(integrated.test_replacement_and_runtime_disable_release_old_owners)
        report = {"arch": "cuda", "checks": completed, "tests_passed": len(completed)}
        (directory / "induction-reuse-cuda.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        ti.reset()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_induction_reuse_cuda_qualification.py --worker DIRECTORY")
    _worker(Path(sys.argv[2]))
