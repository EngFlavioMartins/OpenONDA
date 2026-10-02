"""Adversarial CPU tests for an unwired arithmetic-only Gaussian prototype."""

import numpy as np
import pytest
import taichi as ti

from tests.vpm._gaussian_far_prototype import (
    GaussianArithmeticProbe,
    GaussianPairBenchmark,
    adversarial_densities,
    assert_same_values,
    pair_inputs,
)


@pytest.mark.parametrize("dtype,default", [(ti.f32, ti.f32), (ti.f64, ti.f64), (ti.f64, ti.f32)])
@pytest.mark.parametrize("fast_math", [False, True])
def test_underflow_shortcut_matches_all_finite_bits_and_nan_classification(
    dtype, default, fast_math
):
    ti.init(
        arch=ti.cpu,
        cpu_max_num_threads=2,
        default_fp=default,
        fast_math=fast_math,
        offline_cache=False,
    )
    try:
        np_dtype = np.float32 if dtype == ti.f32 else np.float64
        density = adversarial_densities(np_dtype)
        probe = GaussianArithmeticProbe(len(density), dtype)
        probe.density.from_numpy(density)
        probe.evaluate()
        original, candidate = probe.original.to_numpy(), probe.candidate.to_numpy()
        assert_same_values(candidate, original)
        far = (density >= (11 if dtype == ti.f32 else 28)) & (
            density <= 2.0 ** (127 if dtype == ti.f32 else 1023)
        )
        assert np.all(original[far, 1] == 0)
        # Crucially the high-finite/NaN/negative branches still delegate to
        # the original factory, rather than repairing its exceptional values.
        assert np.any(np.isnan(original[:, 0]))
    finally:
        ti.reset()


@pytest.mark.parametrize("pattern", ["far", "near", "mixed"])
def test_same_pair_arithmetic_and_reduction_are_preserved(pattern):
    ti.init(arch=ti.cpu, cpu_max_num_threads=2, default_fp=ti.f32, offline_cache=False)
    try:
        probe = GaussianPairBenchmark(64, 32)
        x, gamma, core = pair_inputs(64, 32, pattern)
        probe.displacement.from_numpy(x)
        probe.strength.from_numpy(gamma)
        probe.core.from_numpy(core)
        probe.evaluate(False)
        expected = probe.velocity.to_numpy(), probe.gradient.to_numpy()
        probe.evaluate(True)
        assert_same_values(probe.velocity.to_numpy(), expected[0])
        assert_same_values(probe.gradient.to_numpy(), expected[1])
    finally:
        ti.reset()
