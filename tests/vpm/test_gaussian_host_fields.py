"""Portable blocked FFT parity, bounded work and source-core correction."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh import host_fields
from source.solvers.vpm.physics.induction.gaussian_mesh.host_fields import GaussianHostImageFields
from tests.vpm._finite_slab_field_mesh_reference import finite_slab_field_mesh


def _case():
    x = np.array([[-.075, .012, .023], [.018, -.023, .079], [.086, .031, .15]])
    gamma = np.array([[.3, -.2, .7], [-.25, .5, -.4], [.1, .2, .3]])
    sigma = np.array([.035, .04, .05])
    q = np.array([[.012, .014, .024], [-.068, .053, .061], [.097, -.036, .04]])
    images = [(0, True), (-1, False), (1, False), (1, True)]
    return x, gamma, sigma, q, images


def _owner(x, gamma, sigma, q, **kwargs):
    defaults = {"zmin": 0., "zmax": .193, "tau": .12, "spacing": .035,
                "cutoff": .6, "order": 4, "dtype": "float64", "correction_dtype": "float64"}
    defaults.update(kwargs)
    return GaussianHostImageFields(x, gamma, sigma, q, **defaults)


@pytest.mark.parametrize("order,dtype,tolerance", [(4, "float64", 2e-12),
                                                   (10, "float64", 2e-10), (4, "float32", 3e-6)])
def test_host_matches_finite_slab_reference(order, dtype, tolerance):
    x, gamma, sigma, q, images = _case()
    reference = finite_slab_field_mesh(x, gamma, sigma, q, images, zmin=0., zmax=.193,
                                     tau=.12, spacing=.035, order=order, correction_cutoff=.6)
    with _owner(x, gamma, sigma, q, order=order, dtype=dtype, correction_dtype=dtype) as owner:
        u, j, report = owner.evaluate(images)
        np.testing.assert_allclose(u, reference.velocity, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(j, reference.gradient, rtol=tolerance, atol=tolerance)
        assert u.dtype == j.dtype == np.dtype(dtype)
        assert report["core_correction_included"] and not report["tail_certified"]
        assert report["peak_estimated_scratch_bytes"] <= owner.max_scratch_bytes-owner.max_plan_bytes
        assert owner.cp.asnumpy(u) is u


def test_memory_partition_preserves_lattice_operator_and_remote_queries():
    x, gamma, sigma, q, images = _case()
    x[:, 0] = [-.4, 0., .4]
    q[:, 0] = [-.3, .025, .3]
    with _owner(x, gamma, sigma, q) as large:
        expected = large.evaluate(images)[:2]
    with _owner(x, gamma, sigma, q, max_scratch_bytes=220_000, max_plan_bytes=1,
                max_correction_bytes=20_000, max_total_bytes=240_000) as small:
        u, j, report = small.evaluate(images)
        assert report["fft_blocks"] > 2
        assert report["peak_estimated_scratch_bytes"] < 220_000
        np.testing.assert_allclose(u, expected[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(j, expected[1], rtol=2e-12, atol=2e-12)
        remote = q.copy()
        remote[:, 0] += 10_000.
        assert small.can_evaluate_targets(remote)
        ru, rj, rr = small.evaluate_prepared(remote)
        assert np.isfinite(ru).all() and np.isfinite(rj).all()
        assert rr["peak_estimated_scratch_bytes"] < 220_000
        assert rr["correction_pairs"] == 0


def test_primary_coincidence_and_strict_correction_cutoff():
    x = np.array([[0., 0., .05]])
    gamma, sigma = np.array([[.3, -.2, .7]]), np.array([.04])
    q = np.concatenate((x, x+np.array([.6, 0., 0.])))
    with _owner(x, gamma, sigma, q, source_only_primary=True) as owner:
        u, j, report = owner.evaluate([(0, False)])
        assert report["correction_pairs"] == 1  # Exact cutoff is excluded.
        assert np.isfinite(u).all() and np.isfinite(j).all()
        np.testing.assert_allclose(u[0], 0., atol=2e-14)
        assert np.linalg.norm(j[0]) > 0.
        assert owner._prepared_world_images == ((0., False),)


def test_host_has_immutable_sources_and_revokes_failed_preparation():
    x, gamma, sigma, q, images = _case()
    owner = _owner(x, gamma, sigma, q)
    original = owner.host_x.copy()
    x[:] = 0.
    np.testing.assert_array_equal(owner.host_x, original)
    with pytest.raises(ValueError, match="WRITEABLE"):
        owner.host_x.setflags(write=True)
    owner.prepare(images)
    owner.close()
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.evaluate_prepared(q)


def test_host_import_and_execution_do_not_import_cupy():
    result = subprocess.run([sys.executable, "-c", "from source.solvers.vpm.physics.induction.gaussian_mesh.host_fields import GaussianHostImageFields; import sys; assert not any(n == 'cupy' or n.startswith('cupy.') for n in sys.modules)"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_minimum_workspace_cap_fails_without_coarsening():
    x, gamma, sigma, q, _ = _case()
    with pytest.raises(MemoryError, match="unchanged cardinal FFT block"):
        _owner(x, gamma, sigma, q, max_scratch_bytes=1024, max_plan_bytes=1,
               max_correction_bytes=1024, max_total_bytes=2048)


def test_available_host_ram_selects_smaller_unchanged_fft_blocks(monkeypatch):
    x, gamma, sigma, q, images = _case()
    x[:, 0] = [-.4, 0., .4]
    q[:, 0] = [-.3, .025, .3]
    with _owner(x, gamma, sigma, q) as owner:
        expected = owner.evaluate(images)[:2]
    monkeypatch.setattr(host_fields.psutil, "virtual_memory", lambda: SimpleNamespace(available=480_000))
    with _owner(x, gamma, sigma, q) as owner:
        u, j, report = owner.evaluate(images)
        assert report["host_available_bytes"] == 480_000
        assert report["fft_blocks"] > 2
        assert report["peak_estimated_scratch_bytes"] < report["effective_scratch_bytes"] < 240_000
        np.testing.assert_allclose(u, expected[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(j, expected[1], rtol=2e-12, atol=2e-12)


def test_allocation_retry_discards_partial_output_before_smaller_blocks(monkeypatch):
    x, gamma, sigma, q, images = _case()
    with _owner(x, gamma, sigma, q) as owner:
        expected = owner.evaluate(images)[:2]
    original, calls = host_fields._kernels, []
    def fail_after_first_completed_family(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError("injected host allocation failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(host_fields, "_kernels", fail_after_first_completed_family)
    with _owner(x, gamma, sigma, q) as owner:
        u, j, report = owner.evaluate(images)
        assert report["host_memory_retries"] == 1
        np.testing.assert_allclose(u, expected[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(j, expected[1], rtol=2e-12, atol=2e-12)
