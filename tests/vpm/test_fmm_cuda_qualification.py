"""Optional isolated CUDA qualification for the production f32 FMM backend."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def _has_cuda() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.qualification
@pytest.mark.skipif(not _has_cuda(), reason="CUDA device unavailable")
@pytest.mark.parametrize("scenario", ("numerics", "targets", "overflow", "restart"))
def test_cuda_fmm_qualification_in_isolated_runtime(tmp_path, scenario):
    """Keep CUDA and CPU Taichi initialization independent across test files."""
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root), environment.get("PYTHONPATH", "")))
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", scenario, str(tmp_path)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def _restart_case(root: Path) -> None:
    from source.solvers.vpm import FMMInduction, ViscousConfig, VPMSolver
    from tests.vpm.test_backup_storage import _add_counter_rotating_pair, _case

    assert "CUDA" in FMMInduction.supported_devices
    solvers = []

    def build(name):
        case = _case(
            root / name,
            max_n_particles=8,
            viscous=ViscousConfig.inviscid(particle_spacing=0.2),
            induction=FMMInduction(),
        )
        case = replace(case, numerics=replace(case.numerics, compute_device="CUDA"))
        with contextlib.redirect_stdout(io.StringIO()):
            solver = VPMSolver(case)
        solvers.append(solver)
        assert solver.compute_device == "CUDA"
        return solver

    try:
        continuous, interrupted = build("continuous"), build("interrupted")
        for solver in (continuous, interrupted):
            _add_counter_rotating_pair(solver)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(4):
                continuous.advance(defer_output=True)
            for _ in range(2):
                interrupted.advance(defer_output=True)
            interrupted.save_backup()
        resumed = build("resumed")
        with contextlib.redirect_stdout(io.StringIO()):
            resumed.load_backup(str(root / "interrupted/solution/vpm/vpm_000002"))
            for _ in range(2):
                resumed.advance(defer_output=True)
        assert continuous.step == resumed.step == 4
        assert continuous.time == resumed.time == 0.04
        np.testing.assert_array_equal(resumed.particle_position, continuous.particle_position)
        np.testing.assert_array_equal(
            resumed.particle_vortex_strength, continuous.particle_vortex_strength
        )
        np.testing.assert_array_equal(resumed.particle_core_radius, continuous.particle_core_radius)
        assert all(s.induction.diagnostics.host_particle_transfers == 0 for s in solvers)
    finally:
        for solver in reversed(solvers):
            solver.close()


def _worker(scenario: str, workdir: Path) -> None:
    import taichi as ti

    from tests.vpm.test_fmm_device import (
        _DeviceFMMHarness,
        test_device_fmm_arbitrary_targets_are_hierarchical_batched_and_ignore_inactive_storage,
        test_device_fmm_is_permutation_translation_and_axis_rotation_covariant,
        test_device_fmm_meets_all_kernel_gates_with_near_pairs_and_far_clusters,
        test_device_hierarchy_handles_edge_cases_and_rebuilds_stage_metadata,
    )

    ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=0.02, offline_cache=False)
    if scenario == "numerics":
        for kernel in ("GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"):
            test_device_fmm_meets_all_kernel_gates_with_near_pairs_and_far_clusters(kernel)
        test_device_fmm_is_permutation_translation_and_axis_rotation_covariant()
    elif scenario == "targets":
        test_device_hierarchy_handles_edge_cases_and_rebuilds_stage_metadata()
        for kernel in ("GAUSSIAN", "WINCKELMANS"):
            test_device_fmm_arbitrary_targets_are_hierarchical_batched_and_ignore_inactive_storage(
                kernel
            )
    elif scenario == "overflow":
        harness = _DeviceFMMHarness(capacity=64)
        harness.induction._ensure_workspace(64)
        harness.induction.workspace.max_pairs = 1
        rng = np.random.default_rng(20260916)
        position = rng.normal(size=(64, 3)).astype(np.float32)
        strength = rng.normal(scale=0.01, size=(64, 3)).astype(np.float32)
        radius = np.full(64, 0.02, dtype=np.float32)
        with pytest.raises(RuntimeError, match="interaction-list capacity was exceeded"):
            harness.evaluate(position, strength, radius)
    elif scenario == "restart":
        _restart_case(workdir)
    else:
        raise ValueError(f"Unknown CUDA FMM scenario {scenario!r}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_fmm_cuda_qualification.py --worker SCENARIO WORKDIR")
    _worker(sys.argv[2], Path(sys.argv[3]))
