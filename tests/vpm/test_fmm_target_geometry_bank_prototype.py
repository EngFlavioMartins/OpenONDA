"""Pure controller tests: no solver imports, Taichi runtime or device work."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.vpm._fmm_target_geometry_bank_prototype import (
    GeometryAllocationPool,
    ImmutableTargetGeometryBank,
    fields_are_disjoint,
)


class Store:
    instances = []

    def __init__(self, count):
        self.capacity = count
        self.bytes, self.closed = 96 * count, False
        self.tiles = {}
        self.fail_capture = self.fail_restore = False
        self.instances.append(self)

    def capture(self, target, start, count):
        if self.fail_capture:
            raise RuntimeError("partial capture")
        self.tiles[start] = target.geometry.copy()
        return count

    def restore(self, target, start, count, descriptor):
        assert target._prepared_count == 0
        assert count == descriptor
        target.geometry = self.tiles[start].copy()
        if self.fail_restore:
            raise RuntimeError("partial restore")
        target.paths += 1

    def close(self):
        self.closed = True
        self.tiles.clear()


def original(target, position, count, *, target_start=0):
    target._prepared_count = 0
    target.geometry = position[target_start:target_start + count].copy()
    target.builds += 1
    target.paths += 1
    target._prepared_count = count


def harness(count=19, *, cap=None, allocate=Store):
    points = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    target = SimpleNamespace(binding=object(), geometry=None, builds=0, paths=0, _prepared_count=0)
    bank = ImmutableTargetGeometryBank(
        points, count, 8, max_bytes=count * 96 if cap is None else cap,
        allocate=allocate, layout=lambda t: (id(t), id(t.binding)),
    )
    return points, target, bank


def test_repeated_blocks_reuse_all_tiles_with_remainder_and_preserve_order():
    points, target, bank = harness()
    visited = []
    for shell in range(4):
        for start in range(0, len(points), 8):
            length = min(8, len(points) - start)
            bank.prepare(target, original, points, length, target_start=start)
            np.testing.assert_array_equal(target.geometry, points[start:start + length])
            visited.append((shell, start))
    assert visited == [(shell, start) for shell in range(4) for start in (0, 8, 16)]
    assert target.builds == 3 and target.paths == 12
    assert bank.record["builds"] == 3 and bank.record["hits"] == 9
    assert bank.record["peak_payload_bytes"] == 96 * len(points)
    store = bank.store
    bank.close()
    assert store.closed and not bank.saved and bank.store is None
    assert bank.record["status"] == "complete"
    with pytest.raises(RuntimeError, match="scope already ended"):
        bank.prepare(target, original, points, 8)


def test_byte_limit_falls_back_without_allocation_or_numerical_change():
    def forbidden_allocate(count):
        raise AssertionError("memory cap must be checked before allocation")

    points, target, bank = harness(cap=19 * 96 - 1, allocate=forbidden_allocate)
    for _ in range(3):
        bank.prepare(target, original, points, 8)
        np.testing.assert_array_equal(target.geometry, points[:8])
    assert target.builds == 3 and bank.record["fallbacks"] == 3
    assert bank.store is None and bank.record["hits"] == 0
    bank.close()


def test_layout_change_invalidates_every_tile_before_any_hit():
    points, target, bank = harness()
    for start in (0, 8):
        bank.prepare(target, original, points, 8, target_start=start)
    target.binding = object()
    for start in (8, 0):
        bank.prepare(target, original, points, 8, target_start=start)
    assert target.builds == 4 and bank.record["hits"] == 0
    assert bank.record["layout_invalidations"] == 1
    bank.prepare(target, original, points, 8)
    assert bank.record["hits"] == 1
    bank.close()


@pytest.mark.parametrize("kind", ["field", "partition"])
def test_outside_query_disables_scope_without_reusing_incompatible_geometry(kind):
    points, target, bank = harness()
    bank.prepare(target, original, points, 8)
    store = bank.store
    supplied = points + 100 if kind == "field" else points
    length = 8 if kind == "field" else 7
    bank.prepare(target, original, supplied, length)
    np.testing.assert_array_equal(target.geometry, supplied[:length])
    assert store.closed and bank.record["disabled_reason"] == "query-outside-immutable-scope"
    bank.prepare(target, original, points, 8)
    assert target.builds == 3 and bank.record["hits"] == 0
    bank.close()


@pytest.mark.parametrize("failure", ["capture", "restore", "original", "synchronize"])
def test_partial_failure_never_leaves_prepared_or_saved_geometry(failure):
    points, target, bank = harness()
    bank.prepare(target, original, points, 8)
    store = bank.store

    def failing_original(*args, **kwargs):
        raise RuntimeError("original failed")

    def failing_sync():
        raise RuntimeError("synchronize failed")

    if failure == "capture":
        store.fail_capture = True
    elif failure == "restore":
        store.fail_restore = True
    elif failure == "synchronize":
        bank.synchronize = failing_sync
    start = 8 if failure in ("capture", "original") else 0
    with pytest.raises(RuntimeError):
        bank.prepare(target, failing_original if failure == "original" else original,
                     points, 8, target_start=start)
    assert not bank.saved and store.closed and not bank.enabled
    assert target._prepared_count == 0 and bank.record["status"] == "failed"
    bank.close()


def test_allocation_unavailable_preserves_successful_current_preparation():
    def fail_allocate(count):
        raise MemoryError("owned allocator cleaned itself")

    points, target, bank = harness(allocate=fail_allocate)
    bank.prepare(target, original, points, 8)
    assert target._prepared_count == 8
    np.testing.assert_array_equal(target.geometry, points[:8])
    bank.prepare(target, original, points, 8)
    assert target.builds == 2 and bank.record["hits"] == 0
    assert bank.record["disabled_reason"] == "allocation-unavailable"
    bank.close()


def test_oversized_allocator_is_closed_and_fails_before_record_publication():
    def too_large(count):
        result = Store(count)
        result.bytes += 1
        return result

    points, target, bank = harness(allocate=too_large)
    with pytest.raises(RuntimeError, match="declared memory bound"):
        bank.prepare(target, original, points, 8)
    assert Store.instances[-1].closed and not bank.saved
    assert target._prepared_count == 0
    bank.close()


def test_new_scope_does_not_reuse_old_geometry_or_storage():
    points, target, bank = harness()
    bank.prepare(target, original, points, 8)
    first = bank.store
    bank.close()
    points[:] += 17  # Permitted only after the immutable scope has ended.
    next_bank = ImmutableTargetGeometryBank(
        points, len(points), 8, max_bytes=96 * len(points), allocate=Store,
        layout=lambda t: (id(t), id(t.binding)),
    )
    next_bank.prepare(target, original, points, 8)
    assert next_bank.store is not first and next_bank.record["hits"] == 0
    np.testing.assert_array_equal(target.geometry, points[:8])
    next_bank.close()


def test_storage_alias_guard_checks_all_components_and_unknown_bindings():
    vector = SimpleNamespace(members=(10, 11, 12))
    unrelated = SimpleNamespace(members=(20, 21, 22))
    scalar_alias = SimpleNamespace(members=(12,))
    distinct_wrapper = SimpleNamespace(members=(10, 11, 12))
    def identify(field):
        return getattr(field, "members", None)

    assert fields_are_disjoint([vector, None], [unrelated], identify)
    assert not fields_are_disjoint([vector], [scalar_alias], identify)
    assert not fields_are_disjoint([vector], [distinct_wrapper], identify)
    assert not fields_are_disjoint([vector], [object()], identify)
    assert not fields_are_disjoint([object()], [unrelated], identify)


def test_source_workspace_replacement_invalidates_geometry_records():
    points, target, bank = harness()
    target.source = object()
    bank.layout = lambda value: (id(value), id(value.binding), id(value.source))
    bank.prepare(target, original, points, 8)
    target.source = object()
    bank.prepare(target, original, points, 8)
    assert target.builds == 2 and bank.record["hits"] == 0
    assert bank.record["layout_invalidations"] == 1
    bank.close()


@pytest.mark.parametrize("kind", ["field", "partition"])
def test_failed_owner_destruction_on_outside_query_revokes_prepared_geometry(kind):
    points, target, bank = harness()
    bank.prepare(target, original, points, 8)
    assert target._prepared_count == 8
    store = bank.store

    def failed_close():
        raise RuntimeError("owned store destruction failed")

    store.close = failed_close
    supplied = points.copy() if kind == "field" else points
    length = 8 if kind == "field" else 7
    with pytest.raises(RuntimeError, match="owned store destruction failed"):
        bank.prepare(target, original, supplied, length)
    assert target._prepared_count == 0
    assert bank.record["status"] == "failed"
    assert not bank.saved and bank.store is None and not bank.enabled
    assert target.builds == 1  # No fallback may proceed after failed cleanup.
    bank.close()


def pooled_bank(pool, backend, points):
    return ImmutableTargetGeometryBank(
        points, len(points), 8, max_bytes=pool.max_bytes,
        allocate=lambda count: pool.acquire(backend, count),
        layout=lambda target: (id(target), id(target.binding)),
    )


def test_pool_reuses_physical_owner_but_never_logical_geometry_across_scopes():
    points, target, _ = harness()
    backend = object()
    pool = GeometryAllocationPool(96 * 64, Store)
    physical = None
    for turn in range(3):
        points[:] += turn  # Different immutable geometry for every new scope.
        bank = pooled_bank(pool, backend, points)
        bank.prepare(target, original, points, 8)
        lease = bank.store
        if physical is None:
            physical = lease.storage
        assert lease.storage is physical
        np.testing.assert_array_equal(target.geometry, points[:8])
        assert bank.record["builds"] == 1 and bank.record["hits"] == 0
        bank.prepare(target, original, points, 8)
        assert bank.record["hits"] == 1
        bank.close()
        assert not physical.closed and not bank.saved
        with pytest.raises(RuntimeError, match="no longer active"):
            lease.restore(target, 0, 8, 8)
    assert pool.record["allocations"] == 1 and pool.record["reuses"] == 2
    assert pool.record["scope_acquisitions"] == pool.record["scope_releases"] == 3
    assert physical.capacity == 32 and pool.record["peak_payload_bytes"] == 96 * 32
    pool.close()
    assert physical.closed and pool.record["current_payload_bytes"] == 0


def test_pool_growth_has_one_owner_and_bounded_capacity_headroom():
    backend = object()
    pool = GeometryAllocationPool(96 * 31, Store)
    first = pool.acquire(backend, 11)
    first_owner = first.storage
    assert first.capacity == 16
    first.close()
    second = pool.acquire(backend, 17)
    assert first_owner.closed and second.capacity == 31
    assert len(pool.entries) == 1
    assert pool.record["peak_payload_bytes"] == 96 * 31
    assert pool.record["allocations"] == 2 and pool.record["growths"] == 1
    second.close()
    pool.close()


def test_pool_global_byte_cap_and_backend_slot_cap_fall_back_without_eviction():
    first_backend, second_backend = object(), object()
    pool = GeometryAllocationPool(96 * 24, Store, max_backends=2)
    first = pool.acquire(first_backend, 9)
    first_owner = first.storage
    first.close()
    second = pool.acquire(second_backend, 7)
    assert second.capacity == 8
    second.close()
    with pytest.raises(MemoryError, match="slots exhausted"):
        pool.acquire(object(), 1)
    with pytest.raises(MemoryError, match="global byte cap"):
        pool.acquire(second_backend, 9)
    assert not first_owner.closed and len(pool.entries) == 2
    assert pool.record["current_payload_bytes"] == 96 * 24
    pool.close()


def test_live_lease_prohibits_growth_destruction_and_nested_ownership():
    backend = object()
    pool = GeometryAllocationPool(96 * 64, Store)
    lease = pool.acquire(backend, 8)
    for action in (lambda: pool.acquire(backend, 17), lambda: pool.acquire(object(), 1), pool.close):
        with pytest.raises(RuntimeError, match="scope|leases"):
            action()
    assert not lease.storage.closed
    lease.close()
    pool.close()


def test_failed_capture_revokes_lease_but_preserves_pool_for_fresh_scope():
    points, target, _ = harness()
    backend, pool = object(), GeometryAllocationPool(96 * 32, Store)
    bank = pooled_bank(pool, backend, points)
    bank.prepare(target, original, points, 8)
    physical = bank.store.storage
    physical.fail_capture = True
    with pytest.raises(RuntimeError, match="partial capture"):
        bank.prepare(target, original, points, 8, target_start=8)
    assert pool.active is None and not physical.closed and not bank.saved
    bank.close()
    physical.fail_capture = False
    points[:] += 11
    next_bank = pooled_bank(pool, backend, points)
    next_bank.prepare(target, original, points, 8)
    assert next_bank.record["hits"] == 0
    np.testing.assert_array_equal(target.geometry, points[:8])
    next_bank.close()
    pool.close()


def test_failed_growth_destruction_poisoned_pool_never_claims_recovered_memory():
    backend = object()
    pool = GeometryAllocationPool(96 * 32, Store)
    lease = pool.acquire(backend, 8)
    physical = lease.storage
    lease.close()
    close = physical.close

    def fail_close():
        raise RuntimeError("destruction failed")

    physical.close = fail_close
    with pytest.raises(RuntimeError, match="destruction failed"):
        pool.acquire(backend, 17)
    assert pool.poisoned and pool.record["current_payload_bytes"] == 96 * 8
    with pytest.raises(RuntimeError, match="closed or failed"):
        pool.acquire(backend, 8)
    physical.close = close
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.close()
    assert pool.record["current_payload_bytes"] == pool.record["uncertain_payload_bytes"] == 96 * 8
    physical.close()  # Test fixture cleanup does not rewrite uncertain pool telemetry.


def test_failed_growth_allocation_has_no_active_lease_or_count_keyed_old_owner():
    def allocator(count):
        if count > 8:
            raise MemoryError("factory cleaned partial owner")
        return Store(count)

    backend, pool = object(), GeometryAllocationPool(96 * 32, allocator)
    lease = pool.acquire(backend, 8)
    physical = lease.storage
    lease.close()
    with pytest.raises(MemoryError, match="partial owner"):
        pool.acquire(backend, 17)
    assert physical.closed and not pool.entries and pool.active is None
    assert pool.record["current_payload_bytes"] == 0
    pool.close()


def test_uncertain_allocator_cleanup_is_fatal_and_prevents_further_allocation():
    def failed_allocate(count):
        raise RuntimeError("partial allocation cleanup failed")

    pool = GeometryAllocationPool(96 * 32, failed_allocate)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        pool.acquire(object(), 8)
    assert pool.poisoned and pool.active is None
    with pytest.raises(RuntimeError, match="closed or failed"):
        pool.acquire(object(), 8)
    pool.close()


def test_growth_close_memoryerror_is_fatal_to_the_controller_not_a_clean_decline():
    backend = object()
    pool = GeometryAllocationPool(96 * 32, Store)
    points, target, _ = harness(count=8)
    bank = pooled_bank(pool, backend, points)
    bank.prepare(target, original, points, 8)
    physical, close = bank.store.storage, bank.store.storage.close
    bank.close()

    def failed_close():
        raise MemoryError("old owner destruction failed")

    physical.close = failed_close
    larger = np.arange(19 * 3, dtype=np.float32).reshape(19, 3)
    growing = pooled_bank(pool, backend, larger)
    with pytest.raises(RuntimeError, match="destruction failed"):
        growing.prepare(target, original, larger, 8)
    assert target._prepared_count == 0 and growing.record["status"] == "failed"
    assert growing.record["disabled_reason"] != "allocation-unavailable"
    assert pool.poisoned and pool.record["current_payload_bytes"] == 96 * 8
    growing.close()
    physical.close = close
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.close()
    assert pool.record["current_payload_bytes"] == pool.record["uncertain_payload_bytes"] == 96 * 8
    physical.close()


def test_invalid_capacity_cleanup_memoryerror_is_fatal_to_the_controller():
    def invalid_factory(count):
        storage = Store(count)
        storage.capacity += 1

        def failed_close():
            raise MemoryError("invalid owner destruction failed")

        storage.close = failed_close
        return storage

    pool = GeometryAllocationPool(96 * 32, invalid_factory)
    points, target, _ = harness()
    bank = pooled_bank(pool, object(), points)
    with pytest.raises(RuntimeError, match="allocation cleanup failed"):
        bank.prepare(target, original, points, 8)
    assert target._prepared_count == 0 and bank.record["status"] == "failed"
    assert bank.record["disabled_reason"] != "allocation-unavailable"
    assert pool.poisoned and pool.active is None
    bank.close()
    pool.close()
