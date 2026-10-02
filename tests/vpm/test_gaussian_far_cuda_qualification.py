"""Optional isolated CUDA parity and same-pair timing of an unwired prototype.

This does not benchmark tree traversal or promise any whole-solver speedup.
Both alternatives permit the compiler to eliminate duplicate exponentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
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
def test_gaussian_far_prototype_cuda_in_isolated_runtime(tmp_path, record_property):
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
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report = json.loads((tmp_path / "gaussian-far-cuda.json").read_text())
    assert report["arch"] == "cuda"
    record_property("gaussian_far_prototype", report)
    print(json.dumps(report, sort_keys=True))


def _worker(directory):
    import taichi as ti

    from tests.vpm._gaussian_far_prototype import (
        GaussianArithmeticProbe,
        GaussianPairBenchmark,
        adversarial_densities,
        assert_same_values,
        pair_inputs,
    )

    ti.init(
        arch=ti.cuda,
        default_fp=ti.f32,
        fast_math=True,
        device_memory_fraction=0.03,
        offline_cache=False,
    )
    assert ti.lang.impl.current_cfg().arch == ti.cuda
    report = {
        "arch": "cuda",
        "status": "running",
        "operator": "unwired Gaussian shortcut",
        "default_fp": "f32",
        "fast_math": True,
        "comparison": "finite-result bits, infinity signs and NaN classification unchanged",
        "scope": "same monopole pair arithmetic only; not tree traversal or whole solver",
        "arithmetic_checks": [],
        "patterns": {},
    }
    report_path = directory / "gaussian-far-cuda.json"
    try:
        for dtype, np_dtype in ((ti.f32, np.float32), (ti.f64, np.float64)):
            density = adversarial_densities(np_dtype)
            probe = GaussianArithmeticProbe(len(density), dtype)
            probe.density.from_numpy(density)
            probe.evaluate()
            assert_same_values(probe.candidate.to_numpy(), probe.original.to_numpy())
            report["arithmetic_checks"].append(
                {"dtype": str(dtype), "inputs": len(density), "passed": True}
            )
        targets, pairs = 8192, 128
        pair = GaussianPairBenchmark(targets, pairs)
        for pattern in ("far", "near", "mixed"):
            x, gamma, core = pair_inputs(targets, pairs, pattern)
            pair.displacement.from_numpy(x)
            pair.strength.from_numpy(gamma)
            pair.core.from_numpy(core)
            pair.evaluate(False)
            control = pair.velocity.to_numpy(), pair.gradient.to_numpy()
            pair.evaluate(True)
            assert_same_values(pair.velocity.to_numpy(), control[0])
            assert_same_values(pair.gradient.to_numpy(), control[1])
            for _ in range(2):
                pair.evaluate(False)
                pair.evaluate(True)
            ti.sync()
            durations = {"control": [], "candidate": []}
            for repeat in range(11):
                for optimized in (False, True) if repeat % 2 == 0 else (True, False):
                    ti.sync()
                    started = perf_counter()
                    pair.evaluate(optimized)
                    ti.sync()
                    durations["candidate" if optimized else "control"].append(
                        perf_counter() - started
                    )
            old = float(np.median(durations["control"]))
            new = float(np.median(durations["candidate"]))
            report["patterns"][pattern] = {
                "targets": targets,
                "pairs_per_target": pairs,
                "pair_count": targets * pairs,
                "seconds": durations,
                "control_median_seconds": old,
                "candidate_median_seconds": new,
                "same_pair_speedup": old / new,
                "output_bits_equal": True,
            }
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        ti.reset()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_gaussian_far_cuda_qualification.py --worker DIRECTORY")
    _worker(Path(sys.argv[2]))
