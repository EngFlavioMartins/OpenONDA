"""Isolated CUDA qualification of current particle snapshot correctness."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest


def _has_cuda() -> bool:
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
def test_particle_snapshot_cuda_in_isolated_runtime(tmp_path, record_property):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(tmp_path)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report = json.loads((tmp_path / "qualification.json").read_text())
    assert report["arch"] == "cuda"
    record_property("snapshot_cuda", report)
    print(json.dumps(report, sort_keys=True))


def _owner(capacity):
    from source.solvers.vpm.core.solver import VPMSolver
    from source.solvers.vpm.particles.container import Particles
    from source.solvers.vpm.stabilization.manager import StabilizationManager

    particles = Particles(max_n_particles=capacity, float_dtype="f32")
    lineage = SimpleNamespace(reference_vortex_strength=None, reference_lengths=None)
    lineage.on_replacement = MethodType(StabilizationManager.on_replacement, lineage)
    owner = SimpleNamespace(
        particles=particles,
        stabilization=lineage,
        _axisymmetric_orbits_validated=True,
    )
    owner.capture_particle_snapshot = MethodType(VPMSolver.capture_particle_snapshot, owner)
    owner.restore_particle_snapshot = MethodType(VPMSolver.restore_particle_snapshot, owner)
    return owner


def _release(owner):
    for buffer in getattr(owner, "_particle_snapshot_buffers", {}).values():
        buffer.destroy()


def _qualify():
    from tests.vpm import test_particle_device_snapshot as checks

    selected = (
        checks.test_device_snapshot_restores_all_fields_without_host_particle_reads,
        checks.test_nested_slots_are_independent_and_reused_handles_fail_before_mutation,
        checks.test_grown_slot_releases_old_allocation_and_empty_snapshot_restores_count,
        checks.test_snapshot_preserves_refinement_lineage,
        checks.test_snapshot_cannot_restore_into_another_owner,
        checks.test_invalid_snapshot_is_rejected_before_restoring_any_fields,
    )
    completed = []
    for check in selected:
        owner = _owner(32)
        try:
            if check is selected[0]:
                with pytest.MonkeyPatch.context() as patch:
                    check(owner, patch)
            else:
                check(owner)
            completed.append(check.__name__)
        finally:
            _release(owner)
    return {"checks": completed, "copied_fields": "bitwise f32", "tests_passed": len(completed)}


def _worker(directory):
    import taichi as ti

    ti.init(
        arch=ti.cuda,
        default_fp=ti.f32,
        device_memory_fraction=0.15,
        offline_cache=False,
        cpu_max_num_threads=1,
    )
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("CUDA qualification must not silently fall back to CPU")
    result = _qualify()
    result["arch"] = "cuda"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    ti.reset()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_particle_snapshot_cuda_qualification.py --worker OUTPUT_DIR")
    _worker(Path(sys.argv[2]))
