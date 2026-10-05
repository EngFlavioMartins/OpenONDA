"""CUDA block execution through CPU fakes; no device or numerical run."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh import blocked_fields, fields
from source.solvers.vpm.physics.induction.gaussian_mesh.blocked_fields import (
    GaussianBlockedCUDAFields,
)
from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import (
    finite_images,
    slab_coordinates,
)


def _data():
    x = np.array([[-0.3, 0.01, 0.02], [0.0, -0.02, 0.08], [0.3, 0.03, 0.16]])
    g = np.array([[0.3, -0.2, 0.7], [-0.25, 0.5, -0.4], [0.1, 0.2, 0.3]])
    sigma = np.array([0.035, 0.04, 0.05])
    q = np.array([[-0.25, 0.04, 0.0], [0.01, 0.02, 0.04], [0.2, -0.04, 0.193], [0.31, 0.05, 0.1]])
    return x, g, sigma, q


def _options(**kwargs):
    options = {
        "zmin": 0.0,
        "zmax": 0.193,
        "tau": 0.12,
        "spacing": 0.035,
        "cutoff": 0.6,
        "order": 4,
        "dtype": "float64",
        "correction_dtype": "float64",
    }
    options.update(kwargs)
    return options


def _field(x, g, q, world):
    """Independent nonconstant pair kernel with an analytic full gradient."""
    values = np.zeros((len(q), 12))
    for shift, odd in world:
        for position, gamma in zip(x, g, strict=True):
            p, strength = position.copy(), gamma.copy()
            p[2] = shift - p[2] if odd else shift + p[2]
            if odd:
                strength[:2] *= -1
            r = q - p
            den = 1.0 + np.sum(r * r, axis=1)
            cross = np.cross(strength, r)
            values[:, :3] += cross / den[:, None] ** 1.5
            skew = np.array(
                [
                    [0.0, -strength[2], strength[1]],
                    [strength[2], 0.0, -strength[0]],
                    [-strength[1], strength[0], 0.0],
                ]
            )
            jac = skew / den[:, None, None] ** 1.5
            jac -= 3.0 * cross[:, :, None] * r[:, None, :] / den[:, None, None] ** 2.5
            values[:, 3:] += jac.reshape(-1, 9)
    return values


def _fake(monkeypatch, *, phase="construct", budget=2, fault=None, cleanup=False):
    state = SimpleNamespace(live=0, peak=0, attempted=[], completed=[], closes=0)

    class Leaf:
        def __init__(self, x, gamma, sigma, q, **options):
            self.closed = False
            state.live += 1
            state.peak = max(state.peak, state.live)
            self.x, self.gamma, self.q, self.options = x, gamma, q, options
            self.cp = SimpleNamespace(asnumpy=np.asarray)
            state.attempted.append((x.copy(), q.copy(), options["_lattice_origin"].copy()))
            self.large = len(x) * len(q) > budget
            if phase == "construct" and self.large:
                raise MemoryError("injected CUDA capacity")

        def prepare(self, images):
            self.images = tuple(images)
            _, _, cells = slab_coordinates(
                self.x, self.options["zmin"], self.options["zmax"], self.options["spacing"]
            )
            _, self._prepared_world_images, _ = finite_images(
                self.images,
                self.options["zmin"],
                self.options["zmax"],
                cells,
                self.options["max_images"],
                include_primary=self.options["source_only_primary"],
            )
            if phase == "prepare" and self.large:
                raise MemoryError("injected CUDA capacity")
            return {
                "estimated_field_bytes": 100,
                "pool_reserved_bytes": 80,
                "plan_peak_work_bytes": 20,
                "inverse_transforms": 12,
            }

        def evaluate_prepared(self, q):
            if phase == "query" and self.large:
                raise MemoryError("injected CUDA capacity")
            if fault is not None and state.completed:
                raise fault
            state.completed.append((self.x.copy(), q.copy()))
            value = _field(self.x, self.gamma, q, self._prepared_world_images)
            return (
                value[:, :3],
                value[:, 3:].reshape(-1, 3, 3),
                {
                    "inverse_transforms": 0,
                    "correction": {
                        "accepted_pairs": len(self.x) * len(q),
                        "candidate_pairs": len(self.x) * len(q),
                    },
                },
            )

        def close(self):
            if cleanup:
                raise RuntimeError("injected stream drain failure")
            if not self.closed:
                self.closed = True
                state.live -= 1
                state.closes += 1

    monkeypatch.setattr(fields, "GaussianImageFields", Leaf)
    return state


@pytest.mark.parametrize("phase", ["construct", "prepare", "query"])
@pytest.mark.parametrize("primary", [False, True])
def test_recursive_blocks_preserve_all_pairs_images_weights_and_gradients(
    monkeypatch, phase, primary
):
    state = _fake(monkeypatch, phase=phase)
    x, gamma, sigma, q = _data()
    images = (
        ((0, False), (0, True), (-1, False), (1, True))
        if primary
        else ((0, True), (-1, False), (1, True))
    )
    with GaussianBlockedCUDAFields(
        x, gamma, sigma, q, **_options(source_only_primary=primary)
    ) as field:
        prepared = field.prepare(images)
        u, j, report = field.evaluate_prepared(q)
        expected = _field(x, gamma, q, field._prepared_world_images)
        np.testing.assert_allclose(u, expected[:, :3], rtol=3e-15, atol=3e-15)
        np.testing.assert_allclose(j, expected[:, 3:].reshape(-1, 3, 3), rtol=3e-15, atol=3e-15)
        assert field.execution_backend == prepared["backend"] == "cupy_cuda"
        assert report["fft_blocks"] > 1 and report["block_splits"] > 1
        assert report["correction_pairs"] == len(x) * len(q)
        assert report["inverse_transforms"] == report["fft_blocks"] * 12
        assert report["correction_candidates"] == len(x) * len(q)
        assert report["peak_pool_reserved_bytes"] == 80
        assert report["peak_estimated_field_bytes"] == 100
        assert report["peak_plan_work_bytes"] == 20
        lattice_x, _, _ = slab_coordinates(x, 0.0, 0.193, 0.035)
        reflected = lattice_x.copy()
        reflected[:, 2] *= -1
        lattice_q, _, _ = slab_coordinates(q, 0.0, 0.193, 0.035)
        origin = np.floor(np.concatenate((lattice_x, reflected, lattice_q)).min(axis=0)) - 4
        assert all(np.array_equal(record[2], origin) for record in state.attempted)
        pairs = {(tuple(a), tuple(b)): 0 for a in x for b in q}
        for sources, targets in state.completed:
            for a in sources:
                for b in targets:
                    pairs[tuple(a), tuple(b)] += 1
        assert set(pairs.values()) == {1}
        assert state.live == 0 and state.peak == 1
    assert state.closes == len(state.attempted)


def test_exact_query_cache_is_immutable_and_new_remote_query_rebuilds(monkeypatch):
    state = _fake(monkeypatch)
    monkeypatch.setattr(
        blocked_fields.psutil, "virtual_memory", lambda: SimpleNamespace(available=10**9)
    )
    x, gamma, sigma, q = _data()
    with GaussianBlockedCUDAFields(x, gamma, sigma, q, **_options()) as field:
        field.prepare(((0, True), (1, False)))
        u, j, _ = field.evaluate_prepared(q)
        saved = u.copy(), j.copy()
        u[:] = j[:] = 0.0
        count = len(state.attempted)
        u, j, report = field.evaluate_prepared(q.copy())
        np.testing.assert_array_equal(u, saved[0])
        np.testing.assert_array_equal(j, saved[1])
        assert report["exact_query_output_hit"] and report["fft_blocks"] == 0
        assert report["inverse_transforms"] == report["correction_pairs"] == 0
        assert report["peak_pool_reserved_bytes"] == 0
        assert len(state.attempted) == count
        remote = q + [10000.0, 0.0, 0.0]
        assert field.can_evaluate_targets(remote)
        u, j, report = field.evaluate_prepared(remote)
        expected = _field(x, gamma, remote, field._prepared_world_images)
        np.testing.assert_allclose(u, expected[:, :3], rtol=3e-15, atol=3e-15)
        np.testing.assert_allclose(j, expected[:, 3:].reshape(-1, 3, 3), rtol=3e-15, atol=3e-15)
        assert not report["exact_query_output_hit"] and len(state.attempted) > count


def test_coincident_clouds_split_by_count_and_low_host_memory_skips_cache(monkeypatch):
    state = _fake(monkeypatch, budget=1)
    monkeypatch.setattr(
        blocked_fields.psutil, "virtual_memory", lambda: SimpleNamespace(available=0)
    )
    x, gamma, sigma, q = _data()
    x[:] = [0.0, 0.0, 0.08]
    q[:] = [0.0, 0.0, 0.1]
    with GaussianBlockedCUDAFields(x, gamma, sigma, q, **_options()) as field:
        field.prepare(((0, True), (1, False)))
        u, j, report = field.evaluate_prepared(q)
        expected = _field(x, gamma, q, field._prepared_world_images)
        np.testing.assert_allclose(u, expected[:, :3], rtol=3e-15, atol=3e-15)
        np.testing.assert_allclose(j, expected[:, 3:].reshape(-1, 3, 3), rtol=3e-15, atol=3e-15)
        assert report["fft_blocks"] == len(x) * len(q)
        assert report["correction_pairs"] == len(x) * len(q)
        assert state.peak == 1 and field._cached_values is None


@pytest.mark.parametrize(
    "error", [RuntimeError("kernel failure"), MemoryError("irreducible capacity")]
)
def test_failure_after_successful_leaf_publishes_no_partial_field(monkeypatch, error):
    state = _fake(monkeypatch, budget=1, fault=error)
    with GaussianBlockedCUDAFields(*_data(), **_options()) as field:
        field.prepare(((0, True),))
        with pytest.raises(type(error), match="kernel failure|unchanged cardinal"):
            field.evaluate_prepared(field.host_targets)
        assert state.completed and state.live == 0
        assert field._prepared_images is None and field._cached_values is None


def test_cleanup_failure_retains_leaf_and_cannot_be_retried_as_memory_pressure(monkeypatch):
    _fake(monkeypatch, budget=0, cleanup=True)
    field = GaussianBlockedCUDAFields(*_data(), **_options())
    field.prepare(((0, True),))
    with pytest.raises(RuntimeError, match="cleanup remains uncertain"):
        field.evaluate_prepared(field.host_targets)
    assert field._failed_field is not None
    with pytest.raises(RuntimeError, match="cleanup remains uncertain"):
        field.close()


def test_inputs_and_prepared_images_are_immutable_and_invalid_prepare_revokes(monkeypatch):
    _fake(monkeypatch)
    x, gamma, sigma, q = _data()
    with GaussianBlockedCUDAFields(x, gamma, sigma, q, **_options()) as field:
        saved = field.host_x.copy()
        x[:] = 0.0
        np.testing.assert_array_equal(field.host_x, saved)
        with pytest.raises(ValueError, match="WRITEABLE"):
            field.host_x.setflags(write=True)
        images = [(0, True)]
        field.prepare(images)
        images[:] = [(0, False)]
        assert field._prepared_images == ((0, True),)
        with pytest.raises(ValueError, match="primary excluded"):
            field.prepare(images)
        assert field._prepared_images is None


def test_blocked_import_does_not_import_cupy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from source.solvers.vpm.physics.induction.gaussian_mesh.blocked_fields import GaussianBlockedCUDAFields; import sys; assert not any(n == 'cupy' or n.startswith('cupy.') for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
