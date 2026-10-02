"""Failure injection for the production owner used by the thin wrapper."""

import pytest

from source.solvers.vpm.physics.induction.fmm import targets
from tests.vpm import _fmm_topdown_near_prototype as prototype
from tests.vpm.test_fmm_device import _DeviceFMMHarness


def test_topdown_post_allocation_failure_releases_all_owned_scratch(monkeypatch):
    harness = _DeviceFMMHarness(capacity=1)
    real_owner = targets._OwnedFields
    owners = []

    class FailingOwner(real_owner):
        def __init__(self):
            super().__init__()
            owners.append(self)

        def finalize(self):
            super().finalize()
            raise RuntimeError("injected ancestry post-allocation failure")

    monkeypatch.setattr(targets, "_OwnedFields", FailingOwner)
    evaluator = prototype.TopdownNearEvaluator.__new__(prototype.TopdownNearEvaluator)
    with pytest.raises(RuntimeError, match="injected ancestry post-allocation failure"):
        evaluator.__init__(harness.induction.workspace, 3)
    assert len(owners) == 1
    assert owners[0].tree is None
    assert evaluator.tree is evaluator._fields is evaluator._pair_fields is None
    evaluator.destroy()


def test_topdown_path_cleanup_failure_still_releases_base_owners():
    harness = _DeviceFMMHarness(capacity=1)
    evaluator = prototype.TopdownNearEvaluator(harness.induction.workspace, 3)
    original_owner = evaluator._fields

    class FailingCleanup:
        def destroy(self):
            original_owner.destroy()
            raise RuntimeError("injected ancestry cleanup failure")

    evaluator._fields = FailingCleanup()
    with pytest.raises(RuntimeError, match="injected ancestry cleanup failure"):
        evaluator.destroy()
    assert original_owner.tree is None
    assert evaluator.tree is evaluator._fields is evaluator._pair_fields is None
    evaluator.destroy()
