"""Reflected-FMM geometry ownership, alias admission, fallback and reuse checks.

WINCKELMANS exercises reflected FMM targets; Gaussian slabs use the separately
qualified finite mesh operator.
"""

from dataclasses import asdict

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.fmm.diagnostics import FMMDiagnostics
from source.solvers.vpm.physics.induction.fmm.target_geometry import (
    TargetGeometryCache,
    TargetGeometrySession,
    TargetGeometryStorage,
    certified_geometry,
)
from tests.vpm._fmm_geometry_harness import Harness
from tests.vpm.test_fmm_target_geometry_roundoff import field_repeat_evidence


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    owns_runtime = ti.lang.impl.get_runtime().prog is None
    if owns_runtime:
        ti.init(arch=ti.cpu, cpu_max_num_threads=2, offline_cache=False)
    yield
    if owns_runtime:
        ti.reset()


@pytest.mark.parametrize("kernel", ["WINCKELMANS"])
def test_production_geometry_preserves_fields_tail_and_allocation_across_scopes(kernel):
    h = Harness(kernel)
    try:
        h.slab.base.max_image_geometry_bytes = 0
        control, control_tail = h.run()
        h.slab.base.max_image_geometry_bytes = 64 * 1024 * 1024
        before = asdict(h.slab.base.diagnostics)
        result, tail = h.run()
        if ti.lang.impl.current_cfg().arch == ti.cuda:
            # CUDA atomic interaction-list ordering is not bitwise even for
            # repeated fresh builds. The independent repeat/geometry suite
            # verifies that fact and exact geometry transport separately.
            field_repeat_evidence(h, kernel, [(control, control_tail)], [(result, tail)])
        else:
            for actual, expected in zip(result, control, strict=True):
                np.testing.assert_array_equal(actual, expected)
            assert tail == control_tail
        cache = h.slab.base._image_geometry_cache
        assert certified_geometry(cache) and cache.active is None and not cache.scope_open
        owner = cache.storage
        diagnostics = h.slab.base.diagnostics
        assert (
            diagnostics.image_target_geometry_builds - before["image_target_geometry_builds"] == 3
        )
        assert (
            diagnostics.image_target_geometry_restores - before["image_target_geometry_restores"]
            == 6
        )
        assert diagnostics.image_geometry_bytes == 16 * 96
        h.positions[:, 0] += 0.19
        h.x.from_numpy(h.positions)
        changed, changed_tail = h.run()
        assert cache.storage is owner and diagnostics.image_geometry_allocations == 1
        assert (
            diagnostics.image_target_geometry_builds - before["image_target_geometry_builds"] == 6
        )
        h.slab.base.max_image_geometry_bytes = 0
        fresh, fresh_tail = h.run()
        if ti.lang.impl.current_cfg().arch == ti.cuda:
            field_repeat_evidence(h, kernel, [(fresh, fresh_tail)], [(changed, changed_tail)])
        else:
            for actual, expected in zip(changed, fresh, strict=True):
                np.testing.assert_array_equal(actual, expected)
            assert changed_tail == fresh_tail
        assert diagnostics.image_geometry_bytes == 0
    finally:
        h.slab.base.close()


def test_active_scope_rejects_rebind_close_and_complete_field_reuse():
    from source.solvers.vpm.physics.induction.reuse_backends import StandardFMMReuseContract

    h = Harness("WINCKELMANS")
    try:
        h.run()
        base, cache = h.slab.base, h.slab.base._image_geometry_cache
        with cache.scope(h.x, h.n, 4):
            assert not certified_geometry(cache)
            assert StandardFMMReuseContract(h.slab)() is None
            with pytest.raises(RuntimeError, match="scope"):
                base.close()
            with pytest.raises(RuntimeError, match="scope"):
                base.bind(h.slab.physics)
        assert certified_geometry(cache)
        owner = cache.storage
        base.close()
        assert owner.owner is None and base._image_geometry_cache is None
        assert base.workspace is None and base._target_workspace is None
        base.close()
        # Explicit FMM teardown does not destroy any caller-owned state.
        np.testing.assert_array_equal(h.x.to_numpy(), h.positions)
    finally:
        h.slab.base.close()


def test_custom_backend_method_binding_falls_back_without_cached_geometry(monkeypatch):
    h = Harness("WINCKELMANS")
    try:
        h.run()
        base = h.slab.base
        before = asdict(base.diagnostics)
        evaluate = base.evaluate_image_block

        def custom(**kwargs):
            return evaluate(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(base, "evaluate_image_block", custom)
            h.run()
        assert (
            base.diagnostics.image_target_geometry_restores
            == before["image_target_geometry_restores"]
        )
        assert (
            base.diagnostics.image_target_geometry_builds - before["image_target_geometry_builds"]
            == 9
        )
        assert (
            base.diagnostics.image_geometry_fallback_scopes
            - before["image_geometry_fallback_scopes"]
            == 1
        )
    finally:
        h.slab.base.close()


def test_geometry_cache_is_bypassed_by_real_output_component_alias():
    h = Harness("WINCKELMANS")
    try:
        h.run()
        base = h.slab.base
        before = base.diagnostics.image_geometry_fallback_scopes
        with base._fixed_image_targets(
            h.x,
            h.n,
            4,
            read_fields=(h.x, h.gamma, h.radius),
            write_fields=(h.x.get_scalar_field(2),),
        ):
            assert not base._image_geometry_cache.scope_open
        assert base.diagnostics.image_geometry_fallback_scopes == before + 1
        geometry = base._image_geometry_cache.storage
        for position, outputs in (
            (geometry.centre, (h.velocity,)),
            (h.x, (geometry.position,)),
            (h.x, (geometry.centre.get_scalar_field(0),)),
        ):
            with base._fixed_image_targets(
                position,
                h.n,
                4,
                read_fields=(position, h.gamma, h.radius),
                write_fields=outputs,
            ):
                assert not base._image_geometry_cache.scope_open
        assert base.diagnostics.image_geometry_fallback_scopes == before + 4
    finally:
        h.slab.base.close()


def test_geometry_storage_layout_rebinding_declines_pure_induction_reuse(monkeypatch):
    from source.solvers.vpm.physics.induction.reuse_backends import StandardFMMReuseContract

    h = Harness("WINCKELMANS")
    try:
        h.run()
        cache = h.slab.base._image_geometry_cache
        with monkeypatch.context() as patch:
            patch.setattr(cache.storage, "left", cache.storage.right)
            assert not certified_geometry(cache)
            assert StandardFMMReuseContract(h.slab)() is None
        assert certified_geometry(cache)
    finally:
        h.slab.base.close()


def test_empty_or_over_budget_scope_does_not_allocate_and_always_revokes():
    diagnostics = FMMDiagnostics()
    cache = TargetGeometryCache(96, diagnostics)
    for count in (0, 2):
        with cache.scope(object(), count, 4):
            assert cache.scope_open and cache.active is None and cache.storage is None
        assert not cache.scope_open
    assert diagnostics.image_geometry_fallback_scopes == 2
    cache.close()


def test_clean_allocation_decline_uses_original_prepare_without_poisoning(monkeypatch):
    from types import SimpleNamespace

    from source.solvers.vpm.physics.induction.fmm import target_geometry

    def declined(capacity):
        raise MemoryError("constructor cleaned partial allocation")

    prepared = []
    target = SimpleNamespace(
        prepare_targets=lambda *args, **kwargs: prepared.append((args, kwargs))
    )
    cache = TargetGeometryCache(96 * 32, FMMDiagnostics())
    with monkeypatch.context() as patch:
        patch.setattr(target_geometry, "TargetGeometryStorage", declined)
        with cache.scope(object(), 11, 4):
            assert cache.active is None and not cache.poisoned
            cache.prepare(target, object(), 3, target_start=8)
    assert len(prepared) == 1 and cache.storage is None
    assert cache.diagnostics.image_geometry_fallback_scopes == 1
    cache.close()


def test_session_class_override_is_not_certified_even_between_scopes(monkeypatch):
    cache = TargetGeometryCache(0, FMMDiagnostics())
    assert certified_geometry(cache)
    with monkeypatch.context() as patch:
        patch.setattr(TargetGeometrySession, "prepare", lambda *args, **kwargs: None)
        assert not certified_geometry(cache)
    cache.close()


@pytest.mark.parametrize(
    "cls,method",
    [
        (TargetGeometrySession, "prepare"),
        (TargetGeometryStorage, "capture"),
        (TargetGeometryStorage, "__init__"),
    ],
)
def test_class_override_before_first_scope_uses_original_rebuild(monkeypatch, cls, method):
    from source.solvers.vpm.physics.induction.reuse_backends import StandardFMMReuseContract

    h = Harness("WINCKELMANS")
    calls = []
    try:
        base = h.slab.base
        assert base._image_geometry_cache is None and certified_geometry(None)
        with monkeypatch.context() as patch:
            patch.setattr(cls, method, lambda *args, **kwargs: calls.append((args, kwargs)))
            assert not certified_geometry(None)
            assert StandardFMMReuseContract(h.slab)() is None
            with base._fixed_image_targets(
                h.x,
                h.n,
                4,
                read_fields=(h.x, h.gamma, h.radius),
                write_fields=(h.velocity,),
            ):
                assert base._image_geometry_cache is None
                # The existing target preparation remains the fallback; no
                # delegated bank method is invoked, even on the first scope.
                target = base._replace_target_workspace(4, 256)
                target.prepare_targets(h.x, 3, target_start=8)
                np.testing.assert_array_equal(target.tree.position.to_numpy()[:3], h.positions[8:])
        assert not calls and base.diagnostics.image_geometry_fallback_scopes == 1
    finally:
        h.slab.base.close()


def test_geometry_class_rebinding_fails_closed_before_first_scope(monkeypatch):
    from source.solvers.vpm.physics.induction.fmm import target_geometry

    with monkeypatch.context() as patch:
        patch.setattr(target_geometry, "TargetGeometryStorage", lambda count: None)
        assert not certified_geometry(None)


def test_session_constructor_failure_never_publishes_open_lease(monkeypatch):
    cache = TargetGeometryCache(96 * 16, FMMDiagnostics())
    try:
        with cache.scope(object(), 11, 4):
            pass
        storage = cache.storage
        with monkeypatch.context() as patch:

            def fail(*args, **kwargs):
                raise MemoryError("session construction failed")

            patch.setattr(TargetGeometrySession, "__init__", fail)
            with (
                pytest.raises(MemoryError, match="session construction failed"),
                cache.scope(object(), 11, 4),
            ):
                pytest.fail("failed session must not yield")
        assert not cache.scope_open and cache.active is None and cache.storage is storage
        assert certified_geometry(cache)
        cache.close()
        assert storage.owner is None and cache.closed
    finally:
        cache.close()


def test_revoked_session_cannot_publish_during_or_after_another_scope():
    h = Harness("WINCKELMANS")
    try:
        h.run()
        cache, target = h.slab.base._image_geometry_cache, h.slab.base._target_workspace
        with cache.scope(h.x, h.n, 4):
            stale = cache.active
            stale.prepare(target, h.x, 3, target_start=8)
        with pytest.raises(RuntimeError, match="no longer active"):
            stale.prepare(target, h.x, 3, target_start=8)
        assert target._prepared_count == 0
        with cache.scope(h.x, h.n, 4):
            current = cache.active
            current.prepare(target, h.x, 4)
            with pytest.raises(RuntimeError, match="no longer active"):
                stale.prepare(target, h.x, 3, target_start=8)
            current.prepare(target, h.x, 4)
            assert target._prepared_count == 4
            np.testing.assert_array_equal(target.tree.position.to_numpy()[:4], h.positions[:4])
    finally:
        h.slab.base.close()


def test_source_workspace_replacement_invalidates_tiles_without_destroying_bank():
    h = Harness("WINCKELMANS")
    try:
        h.run()
        base, cache = h.slab.base, h.slab.base._image_geometry_cache
        storage, previous_source = cache.storage, base.workspace
        with cache.scope(h.x, h.n, 4):
            session = cache.active
            session.prepare(base._target_workspace, h.x, 3, target_start=8)
            assert session.records
            base._replace_workspace(previous_source.max_n_particles, previous_source.max_pairs)
            assert not session.records and cache.storage is storage
            target = base._replace_target_workspace(4, 256)
            session.prepare(target, h.x, 3, target_start=8)
            session.prepare(target, h.x, 4)
            before = base.diagnostics.image_target_geometry_restores
            session.prepare(target, h.x, 3, target_start=8)
            assert base.diagnostics.image_target_geometry_restores == before + 1
            np.testing.assert_array_equal(target.tree.position.to_numpy()[:3], h.positions[8:])
    finally:
        h.slab.base.close()


def test_failed_explicit_workspace_destroy_revokes_backend_handle(monkeypatch):
    h = Harness("WINCKELMANS")
    base, workspace = h.slab.base, h.slab.base.workspace
    destroy = workspace.destroy
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                workspace, "destroy", lambda: (_ for _ in ()).throw(RuntimeError("destroy failed"))
            )
            with pytest.raises(RuntimeError, match="destroy failed"):
                base.close()
        assert base.workspace is None
    finally:
        destroy()
        base.close()
