"""Isolated CUDA rollback qualification and a bounded transaction benchmark.

The synthetic benchmark measures only capture/restore, not a coupled timestep.
It neither runs induction nor changes a simulation case or checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import MethodType, SimpleNamespace

import numpy as np
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
@pytest.mark.parametrize("scenario", ("qualification", "benchmark"))
def test_particle_snapshot_cuda_in_isolated_runtime(tmp_path, scenario, record_property):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", scenario, str(tmp_path)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report = json.loads((tmp_path / f"{scenario}.json").read_text())
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
        induction=SimpleNamespace(),
        np_dtype=np.float32,
        _axisymmetric_orbits_validated=True,
    )
    owner.capture_particle_snapshot = MethodType(VPMSolver.capture_particle_snapshot, owner)
    owner.restore_particle_snapshot = MethodType(VPMSolver.restore_particle_snapshot, owner)
    owner.replace_vortex_particles = MethodType(VPMSolver.replace_vortex_particles, owner)
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
        checks.test_snapshot_keeps_legacy_lineage_replacement_semantics,
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


def _benchmark(count=293_340, repetitions=3):
    import taichi as ti

    from source.coupler.vorticity_transfer import _particle_state_snapshot, _restore_particle_state
    from tests.vpm.test_particle_device_snapshot import _assert_payload, _payload

    owner = _owner(count)
    # The legacy adapter deliberately lacks the device-snapshot capability.
    # It exercises the retained, original eleven-array host implementation.
    legacy = SimpleNamespace(
        particles=owner.particles,
        np_dtype=np.float32,
        replace_vortex_particles=owner.replace_vortex_particles,
    )
    initial = {
        name: np.asarray(values, dtype=np.int32 if name.endswith("_id") else np.float32)
        for name, values in _payload(count).items()
    }
    owner.particles.replace_from_numpy(**initial)
    ti.sync()

    def cycle(device):
        ti.sync()
        started = perf_counter()
        snapshot = (
            owner.capture_particle_snapshot(slot="benchmark")
            if device
            else _particle_state_snapshot(legacy)
        )
        ti.sync()
        captured = perf_counter()
        # A real mutation proves restore republishes the original values; this
        # mutation is outside both timed phases and costs each path the same.
        owner.particles.velocity.fill(0.0)
        owner.particles.touch_state()
        ti.sync()
        restore_started = perf_counter()
        if device:
            owner.restore_particle_snapshot(snapshot)
        else:
            _restore_particle_state(legacy, snapshot)
        ti.sync()
        restored = perf_counter()
        return {
            "capture": captured - started,
            "restore": restored - restore_started,
            "total": captured - started + restored - restore_started,
        }

    try:
        # Exclude allocation and first-use JIT from warm throughput results,
        # but report their costs separately. Both paths are warmed explicitly.
        cold = {"legacy": cycle(False), "device": cycle(True)}
        _assert_payload(owner.particles, initial)
        samples = {"legacy": [], "device": []}
        for repetition in range(repetitions):
            for device in (False, True) if repetition % 2 == 0 else (True, False):
                samples["device" if device else "legacy"].append(cycle(device))
                _assert_payload(owner.particles, initial)
        medians = {
            path: {
                phase: float(np.median([row[phase] for row in rows]))
                for phase in ("capture", "restore", "total")
            }
            for path, rows in samples.items()
        }
        assert all(
            np.isfinite(value) and value > 0 for row in medians.values() for value in row.values()
        )
        return {
            "particle_count": count,
            "repetitions": repetitions,
            "precision": "f32",
            "scope": "synchronized capture/restore only; excludes mutation, validation, induction and solver work",
            "validation": "all eleven captured fields bitwise equal; derived vorticity within four f32 ulps relative",
            "lineage_enabled": False,
            "cold_seconds": cold,
            "samples_seconds": samples,
            "median_seconds": medians,
            "transaction_speedup": medians["legacy"]["total"] / medians["device"]["total"],
            "bulk_host_bytes_per_legacy_capture": sum(value.nbytes for value in initial.values()),
            "bulk_host_particle_bytes_per_device_capture": 0,
        }
    finally:
        _release(owner)


def _worker(scenario, directory):
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
    result = _qualify() if scenario == "qualification" else _benchmark()
    result["arch"] = "cuda"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{scenario}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    ti.reset()


if __name__ == "__main__":
    if (
        len(sys.argv) != 4
        or sys.argv[1] != "--worker"
        or sys.argv[2] not in {"qualification", "benchmark"}
    ):
        raise SystemExit(
            "usage: test_particle_snapshot_cuda_qualification.py --worker SCENARIO OUTPUT_DIR"
        )
    _worker(sys.argv[2], Path(sys.argv[3]))
