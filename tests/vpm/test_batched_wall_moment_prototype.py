"""The isolated batch prototype must preserve the production scalar contract."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest

from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin, _m4_prime_1d


def asset():
    directory = (
        Path(__file__).resolve().parents[2]
        / "tests/support/cylinder"
    )
    sys.path.insert(0, str(directory))
    spec = spec_from_file_location(
        "wall_moment_batch_test", directory / "qualify_wall_moment_batch.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wall():
    owner = _GridDiffusionMixin.__new__(_GridDiffusionMixin)
    owner._body_mask_active = True
    owner._slip_slab_bounds = None
    owner._body_classifier = lambda points: points[:, 0] < 0
    owner._body_segment_classifier = None
    return owner


def stencils(count=128):
    rng = np.random.default_rng(49)
    points = rng.uniform([0.001, -0.4, -0.4], [0.12, 0.4, 0.4], (count, 3))
    spacing, origin, shape = 0.1, np.array([-1.0, -1.0, -1.0]), (25, 25, 25)
    offsets = np.stack(np.meshgrid(*([np.arange(-1, 3)] * 3), indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    calls = []
    for point in points:
        fraction = (point - origin) / spacing
        indices = np.floor(fraction).astype(np.int64) + offsets
        weights = np.prod(_m4_prime_1d(fraction - indices), axis=1)
        fluid = (origin + spacing * indices)[:, 0] >= 0
        calls.append((point, indices, weights, fluid, origin, spacing, shape))
    return calls


def test_grouped_radius_one_matches_actual_scalar_operator():
    module, owner, calls = asset(), wall(), stencils()
    reference = [owner._wall_moment_correction(*call) for call in calls]
    candidate, diagnostics = module.batched_radius_one(owner, calls)
    comparison = module.compare_results(reference, candidate)
    assert comparison["maximum_absolute_correction_difference"] < 2e-15
    assert comparison["f32_deposit_entries_changed"] == 0
    assert diagnostics["batched_radius_one"] == len(calls)
    assert diagnostics["scalar_fallback"] == 0
    for call, result in zip(calls, candidate, strict=True):
        point, indices, weights, fluid, origin, spacing, _ = call
        relative = (origin + spacing * result[0] - point) / spacing
        constraints = np.vstack((np.ones(fluid.sum()), relative.T))
        corrected = weights[fluid] + result[1]
        assert np.linalg.cond(constraints @ constraints.T) <= 1e10
        assert np.sum(np.abs(corrected)) <= 2.0
        assert np.max(np.abs(constraints @ corrected - [1, 0, 0, 0])) <= 1e-10


def test_rank_deficiency_uses_actual_radius_expansion():
    module, owner = asset(), wall()
    call = list(stencils(1)[0])
    # Keep many nodes, but only one x plane: zeroth/x-moment rank deficiency.
    call[3] = call[1][:, 0] == 10
    reference = [owner._wall_moment_correction(*call)]
    assert reference[0][2] > 1
    candidate, diagnostics = module.batched_radius_one(owner, [call])
    assert diagnostics["scalar_fallback"] == 1
    assert (
        module.compare_results(reference, candidate)["maximum_absolute_correction_difference"] == 0
    )


def test_insufficient_support_preserves_scalar_failure():
    module, owner = asset(), wall()
    owner._body_classifier = lambda points: np.ones(len(points), dtype=bool)
    call = list(stencils(1)[0])
    call[3] = np.zeros(64, dtype=bool)
    with pytest.raises(RuntimeError, match="insufficient visible fluid support"):
        owner._wall_moment_correction(*call)
    with pytest.raises(RuntimeError, match="insufficient visible fluid support"):
        module.batched_radius_one(owner, [call])


def test_condition_boundary_uses_scalar_oracle(monkeypatch):
    module, owner, calls = asset(), wall(), stencils(1)
    expected = owner._wall_moment_correction(*calls[0])
    oracle_calls = []
    owner._wall_moment_correction = lambda *args: oracle_calls.append(args) or expected
    monkeypatch.setattr(module.np.linalg, "cond", lambda values: np.full(len(values), 1e10))
    candidate, diagnostics = module.batched_radius_one(owner, calls)
    assert len(oracle_calls) == diagnostics["scalar_fallback"] == 1
    assert candidate[0] is expected


@pytest.mark.parametrize("solution", [0.0, 1e12, np.nan])
def test_moment_boundedness_and_finiteness_failures_use_scalar(monkeypatch, solution):
    module, owner, calls = asset(), wall(), stencils(1)
    expected = owner._wall_moment_correction(*calls[0])
    oracle_calls = []
    owner._wall_moment_correction = lambda *args: oracle_calls.append(args) or expected
    monkeypatch.setattr(module.np.linalg, "solve", lambda matrix, rhs: np.full(rhs.shape, solution))
    result, diagnostics = module.batched_radius_one(owner, calls)
    assert diagnostics["scalar_fallback"] == len(oracle_calls) == 1
    assert result[0] is expected
