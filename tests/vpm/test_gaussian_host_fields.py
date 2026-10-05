"""Portable blocked FFT parity, bounded work and source-core correction."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh import host_fields
from source.solvers.vpm.physics.induction.gaussian_mesh.host_fields import GaussianHostImageFields
from tests.vpm._direct_gaussian_reference import direct_finite_images


def _case():
    x = np.array([[-0.075, 0.012, 0.023], [0.018, -0.023, 0.079], [0.086, 0.031, 0.15]])
    gamma = np.array([[0.3, -0.2, 0.7], [-0.25, 0.5, -0.4], [0.1, 0.2, 0.3]])
    sigma = np.array([0.035, 0.04, 0.05])
    q = np.array([[0.012, 0.014, 0.024], [-0.068, 0.053, 0.061], [0.097, -0.036, 0.04]])
    images = [(0, True), (-1, False), (1, False), (1, True)]
    return x, gamma, sigma, q, images


def _field(x, gamma, sigma, q, **kwargs):
    defaults = {
        "zmin": 0.0,
        "zmax": 0.193,
        "tau": 0.12,
        "spacing": 0.035,
        "cutoff": 0.6,
        "order": 4,
        "dtype": "float64",
        "correction_dtype": "float64",
    }
    defaults.update(kwargs)
    return GaussianHostImageFields(x, gamma, sigma, q, **defaults)


@pytest.mark.parametrize(
    "order,dtype,envelope", [(4, "float64", 2e-3), (10, "float64", 2e-5), (4, "float32", 2e-3)]
)
def test_host_fields_match_independent_direct_gaussian_sums(order, dtype, envelope):
    x, gamma, sigma, q, images = _case()
    with _field(x, gamma, sigma, q, order=order, dtype=dtype, correction_dtype=dtype) as field:
        u, j, report = field.evaluate(images)
        exact = direct_finite_images(x, gamma, sigma, q, field._prepared_world_images)[:2]
        for actual, expected in zip((u, j), exact, strict=True):
            assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < envelope
        assert u.dtype == j.dtype == np.dtype(dtype)
        assert report["core_correction_included"] and not report["tail_bound_checked"]
        assert (
            report["peak_estimated_scratch_bytes"] <= field.max_scratch_bytes - field.max_plan_bytes
        )
        assert field.cp.asnumpy(u) is u


def test_memory_partition_preserves_lattice_operator_and_remote_queries():
    x, gamma, sigma, q, images = _case()
    x[:, 0] = [-0.4, 0.0, 0.4]
    q[:, 0] = [-0.3, 0.025, 0.3]
    with _field(x, gamma, sigma, q) as large:
        expected = large.evaluate(images)[:2]
    with _field(
        x,
        gamma,
        sigma,
        q,
        max_scratch_bytes=220_000,
        max_plan_bytes=1,
        max_correction_bytes=20_000,
        max_total_bytes=240_000,
    ) as small:
        u, j, report = small.evaluate(images)
        assert report["fft_blocks"] > 2
        assert report["peak_estimated_scratch_bytes"] < 220_000
        np.testing.assert_allclose(u, expected[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(j, expected[1], rtol=2e-12, atol=2e-12)
        remote = q.copy()
        remote[:, 0] += 10_000.0
        assert small.can_evaluate_targets(remote)
        ru, rj, rr = small.evaluate_prepared(remote)
        assert np.isfinite(ru).all() and np.isfinite(rj).all()
        assert rr["peak_estimated_scratch_bytes"] < 220_000
        assert rr["correction_pairs"] == 0


def test_primary_coincidence_and_strict_correction_cutoff():
    x = np.array([[0.0, 0.0, 0.05]])
    gamma, sigma = np.array([[0.3, -0.2, 0.7]]), np.array([0.04])
    q = np.concatenate((x, x + np.array([0.6, 0.0, 0.0])))
    with _field(x, gamma, sigma, q, source_only_primary=True) as field:
        u, j, report = field.evaluate([(0, False)])
        assert report["correction_pairs"] == 1  # Exact cutoff is excluded.
        assert np.isfinite(u).all() and np.isfinite(j).all()
        np.testing.assert_allclose(u[0], 0.0, atol=2e-14)
        assert np.linalg.norm(j[0]) > 0.0
        assert field._prepared_world_images == ((0.0, False),)


def test_host_has_immutable_sources_and_revokes_failed_preparation():
    x, gamma, sigma, q, images = _case()
    field = _field(x, gamma, sigma, q)
    original = field.host_x.copy()
    x[:] = 0.0
    np.testing.assert_array_equal(field.host_x, original)
    with pytest.raises(ValueError, match="WRITEABLE"):
        field.host_x.setflags(write=True)
    field.prepare(images)
    field.close()
    field.close()
    with pytest.raises(RuntimeError, match="closed"):
        field.evaluate_prepared(q)


def test_host_import_and_execution_do_not_import_cupy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from source.solvers.vpm.physics.induction.gaussian_mesh.host_fields import GaussianHostImageFields; import sys; assert not any(n == 'cupy' or n.startswith('cupy.') for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_minimum_workspace_cap_fails_without_coarsening():
    x, gamma, sigma, q, _ = _case()
    with pytest.raises(MemoryError, match="unchanged cardinal FFT block"):
        _field(
            x,
            gamma,
            sigma,
            q,
            max_scratch_bytes=1024,
            max_plan_bytes=1,
            max_correction_bytes=1024,
            max_total_bytes=2048,
        )


def test_available_host_ram_selects_smaller_unchanged_fft_blocks(monkeypatch):
    x, gamma, sigma, q, images = _case()
    x[:, 0] = [-0.4, 0.0, 0.4]
    q[:, 0] = [-0.3, 0.025, 0.3]
    with _field(x, gamma, sigma, q) as field:
        expected = field.evaluate(images)[:2]
    monkeypatch.setattr(
        host_fields.psutil, "virtual_memory", lambda: SimpleNamespace(available=480_000)
    )
    with _field(x, gamma, sigma, q) as field:
        u, j, report = field.evaluate(images)
        assert report["host_available_bytes"] == 480_000
        assert report["fft_blocks"] > 2
        assert report["peak_estimated_scratch_bytes"] < report["effective_scratch_bytes"] < 240_000
        np.testing.assert_allclose(u, expected[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(j, expected[1], rtol=2e-12, atol=2e-12)


def test_allocation_retry_discards_partial_output_before_smaller_blocks(monkeypatch):
    x, gamma, sigma, q, images = _case()
    with _field(x, gamma, sigma, q) as field:
        expected = field.evaluate(images)[:2]
    original, calls = host_fields._kernels, []

    def fail_after_first_completed_family(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError("injected host allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(host_fields, "_kernels", fail_after_first_completed_family)
    with _field(x, gamma, sigma, q) as field:
        u, j, report = field.evaluate(images)
        assert report["host_memory_retries"] == 1
        np.testing.assert_allclose(u, expected[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(j, expected[1], rtol=2e-12, atol=2e-12)
