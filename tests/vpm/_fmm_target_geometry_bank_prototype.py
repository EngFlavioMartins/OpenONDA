"""Unwired, scope-owned target geometry reuse; no source or image approximations.

The production slab visits every target tile for each convergence block. This
qualification prototype retains the *first actual prepared geometry* of each
tile only until that same ``_images`` call returns. Source work, shell ordering,
global convergence tests, regularization and output publication are unchanged.
It does not infer immutable targets from the existing immutable-source scope.

Storage is 96 bytes/capacity slot (plus bounded host tile descriptors): positions,
Morton indices, leaf depths, minimal node geometry/topology, initial target
cells. Interaction lists, local coefficients, images, outputs and source
moments are never retained. Root-first ancestry is rebuilt after restoration;
banking its worst-case 97 entries/target would waste considerably more memory.

One allocation per backend can survive consecutive image scopes; every logical
tile descriptor is nevertheless discarded at scope exit. This distinction is
essential on Taichi 1.7.4: destroying an SNode tree globally clears compiled
functions. Physical owners are released only on bounded growth between scopes
or final factory teardown, never on an ordinary logical-scope exit.

The controller is independent of Taichi so failure/lifetime tests need no
device imports. The factory below is isolated-process qualification only:
standard reuse adapters are imported *before* temporary method replacement so
they cannot accidentally certify the prototype as a production operator.
"""

from contextlib import contextmanager
from inspect import signature
from time import perf_counter


def fields_are_disjoint(read_fields, write_fields, identify):
    """Fail closed for unknown storage and scalar aliases inside vector fields."""
    readers, writers = set(), set()
    for fields, result in ((read_fields, readers), (write_fields, writers)):
        for field in fields:
            if field is None:
                continue
            members = identify(field)
            if not members:
                return False
            result.update(members)
    return not readers.intersection(writers)


class GeometryAllocationPool:
    """Globally capped payload, at most one physical owner per backend.

    Capacities use power-of-two headroom only when the global cap permits it.
    No count-keyed historical owners accumulate. Acquisition/reallocation and
    destruction are forbidden while any scope holds a lease. Out-of-budget
    acquisition raises MemoryError for the controller's unchanged fallback.
    """

    def __init__(self, max_bytes, allocate, *, max_backends=8):
        if max_bytes < 0 or max_backends < 1:
            raise ValueError("invalid geometry allocation pool bounds")
        self.max_bytes, self.max_backends, self.allocate = max_bytes, max_backends, allocate
        self.entries, self.active = {}, None
        self.quarantined = set()
        self.closed, self.poisoned = False, False
        self.record = {
            "memory_cap_bytes": max_bytes, "max_backends": max_backends,
            "current_payload_bytes": 0, "peak_payload_bytes": 0,
            "allocations": 0, "reuses": 0, "growths": 0, "releases": 0,
            "scope_acquisitions": 0, "scope_releases": 0,
            "allocation_seconds": 0.0, "physical_release_seconds": 0.0,
            "uncertain_payload_bytes": 0,
        }

    def _retire(self, key):
        if self.active is not None:
            raise RuntimeError("cannot destroy geometry storage during an image scope")
        entry = self.entries[key]
        if key in self.quarantined:
            # Some owners clear their Python handle before the device destroy
            # completes. A subsequent no-op close is not evidence of recovery.
            raise RuntimeError("Failed geometry owner remains quarantined; recovery is unconfirmed")
        began = perf_counter()
        try:
            entry[1].close()
        except BaseException as error:
            # Do not claim that failed destruction recovered the payload or
            # permit later acquisition against an uncertain physical budget.
            self.poisoned = True
            self.quarantined.add(key)
            self.record["uncertain_payload_bytes"] += entry[1].bytes
            raise RuntimeError("Geometry storage destruction failed; memory recovery is uncertain") from error
        finally:
            self.record["physical_release_seconds"] += perf_counter() - began
        self.record["current_payload_bytes"] -= entry[1].bytes
        self.record["releases"] += 1
        del self.entries[key]

    def acquire(self, backend, count):
        if self.closed or self.poisoned:
            raise RuntimeError("geometry pool is closed or failed")
        if self.active is not None:
            raise RuntimeError("nested geometry allocation leases are unsupported")
        if count < 1:
            raise ValueError("geometry allocation needs a positive active count")
        key = id(backend)
        entry = self.entries.get(key)
        if entry is not None and entry[0] is not backend:
            raise RuntimeError("geometry backend identity changed unexpectedly")
        if entry is not None and entry[1].capacity >= count:
            self.record["reuses"] += 1
        else:
            if entry is None and len(self.entries) >= self.max_backends:
                raise MemoryError("bounded geometry backend slots exhausted")
            available = self.max_bytes - self.record["current_payload_bytes"]
            if entry is not None:
                available += entry[1].bytes
            capacity_limit = available // 96
            if count > capacity_limit:
                raise MemoryError("geometry payload would exceed global byte cap")
            capacity = min(1 << (count - 1).bit_length(), capacity_limit)
            if entry is not None:
                self._retire(key)
                self.record["growths"] += 1
            began = perf_counter()
            try:
                storage = self.allocate(capacity)
            except MemoryError:
                # This exception is the allocator's explicit clean-failure
                # contract. A constructor whose cleanup also failed must
                # propagate a different error, never a recoverable decline.
                raise
            except BaseException:
                self.poisoned = True
                raise
            finally:
                self.record["allocation_seconds"] += perf_counter() - began
            if storage.bytes != 96 * capacity or storage.capacity != capacity:
                try:
                    storage.close()
                except BaseException as error:
                    raise RuntimeError("Invalid geometry allocation cleanup failed") from error
                finally:
                    self.poisoned = True
                raise RuntimeError("geometry allocator violated its bounded capacity contract")
            self.entries[key] = entry = (backend, storage)
            self.record["allocations"] += 1
            self.record["current_payload_bytes"] += storage.bytes
            self.record["peak_payload_bytes"] = max(
                self.record["peak_payload_bytes"], self.record["current_payload_bytes"]
            )
        lease = _GeometryLease(self, key, entry[1])
        self.active = lease
        self.record["scope_acquisitions"] += 1
        return lease

    def _release_scope(self, lease):
        if self.active is not lease:
            raise RuntimeError("geometry lease is not the active scope")
        self.active = None
        self.record["scope_releases"] += 1

    def close(self):
        if self.closed:
            return
        if self.active is not None:
            raise RuntimeError("cannot close geometry pool during an image scope")
        failures = []
        for key in tuple(self.entries):
            try:
                self._retire(key)
            except BaseException as error:
                failures.append(error)
        self.closed = True
        if failures:
            for later in failures[1:]:
                failures[0].add_note(f"Another geometry owner failed to close: {later!r}")
            raise failures[0]


class _GeometryLease:
    """Scope revocation does not destroy its retained physical allocation."""

    def __init__(self, pool, key, storage):
        self.pool, self.key, self.storage = pool, key, storage
        self.capacity, self.bytes, self.closed = storage.capacity, storage.bytes, False

    def _validate(self):
        if self.closed or self.pool.active is not self:
            raise RuntimeError("geometry lease is no longer active")

    def capture(self, *args):
        self._validate()
        return self.storage.capture(*args)

    def restore(self, *args):
        self._validate()
        return self.storage.restore(*args)

    def close(self):
        if not self.closed:
            self.pool._release_scope(self)
            self.closed = True


class ImmutableTargetGeometryBank:
    """One explicitly immutable target prefix and fixed tile partition.

    ``allocate(count)`` must return an owned store with capture/restore/close.
    Its capture/restore routines preserve prepared geometry, not source state.
    The owner must keep target fields unchanged for this short scope. Unknown
    queries invalidate/disable reuse, rather than guess at their ownership.
    """

    def __init__(self, position, count, tile_capacity, *, max_bytes, allocate,
                 layout, synchronize=lambda: None):
        if count < 0 or tile_capacity < 1 or max_bytes < 0:
            raise ValueError("invalid geometry bank bounds")
        self.position, self.count, self.tile_capacity = position, count, tile_capacity
        self.allocate, self.layout, self.synchronize = allocate, layout, synchronize
        self.max_bytes = max_bytes
        self.required_bytes = 96 * count
        self.store, self.saved, self.binding = None, {}, None
        self.enabled, self.closed = bool(count and self.required_bytes <= max_bytes), False
        self.record = {
            "target_count": count, "tile_capacity": tile_capacity,
            "required_bytes": self.required_bytes, "memory_cap_bytes": max_bytes,
            "peak_payload_bytes": 0, "builds": 0, "hits": 0, "fallbacks": 0,
            "layout_invalidations": 0, "prepare_seconds": 0.0,
            "capture_seconds": 0.0, "restore_seconds": 0.0,
            "disabled_reason": None if self.enabled else "empty-or-memory-cap",
            "status": "running",
        }

    def _timed(self, name, action):
        self.synchronize()
        began = perf_counter()
        try:
            return action()
        finally:
            self.synchronize()
            self.record[name] += perf_counter() - began

    def _disable(self, reason):
        self.enabled = False
        self.record["disabled_reason"] = reason
        self.saved.clear()
        store, self.store = self.store, None
        if store is not None:
            store.close()

    def prepare(self, target, original, position, count, *, target_start=0):
        if self.closed:
            raise RuntimeError("geometry bank scope already ended")
        count, start = int(count), int(target_start)
        valid_tile = (
            position is self.position and 0 <= start < self.count
            and start % self.tile_capacity == 0
            and count == min(self.tile_capacity, self.count - start)
        )
        try:
            if self.enabled and not valid_tile:
                # Destruction itself can fail. Keep invalidation inside this
                # transaction so no previously prepared tile stays publishable
                # after an out-of-scope request or failed owner cleanup.
                self._disable("query-outside-immutable-scope")
            if not self.enabled:
                self.record["fallbacks"] += 1
                result = self._timed("prepare_seconds", lambda: original(
                    target, position, count, target_start=start
                ))
                self.record["builds"] += 1
                return result
            binding = self.layout(target)
            if binding != self.binding:
                if self.binding is not None:
                    self.record["layout_invalidations"] += 1
                self.saved.clear()
                self.binding = binding
            if start in self.saved:
                target._prepared_count = 0
                self._timed("restore_seconds", lambda: self.store.restore(
                    target, start, count, self.saved[start]
                ))
                # Restore certifies all geometry and the rebuilt bounded paths
                # before allowing a subsequent evaluator to publish anything.
                target._prepared_count = count
                self.record["hits"] += 1
                return None
            self._timed("prepare_seconds", lambda: original(
                target, position, count, target_start=start
            ))
            self.record["builds"] += 1
            if target._prepared_count != count:
                raise RuntimeError("original target preparation did not complete")
            if self.store is None:
                try:
                    self.store = self.allocate(self.count)
                except MemoryError:
                    # The store constructor owns partial-allocation cleanup.
                    # Current geometry is already valid; future calls rebuild.
                    self._disable("allocation-unavailable")
                    return None
                if self.store.bytes > self.max_bytes:
                    raise RuntimeError("geometry store exceeds its declared memory bound")
                self.record["peak_payload_bytes"] = self.store.bytes
            descriptor = self._timed("capture_seconds", lambda: self.store.capture(
                target, start, count
            ))
            # Never publish a record after partial capture or synchronization.
            self.saved[start] = descriptor
            return None
        except BaseException as error:
            target._prepared_count = 0
            self.record.update(status="failed", error=repr(error))
            try:
                self._disable("failed-prepare-or-restore")
            except BaseException as cleanup_error:
                error.add_note(f"Geometry bank cleanup also failed: {cleanup_error!r}")
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._disable(self.record["disabled_reason"])
        if self.record["status"] == "running":
            self.record["status"] = "complete"


@contextmanager
def target_geometry_bank_factory(*, max_bytes=64 * 1024 * 1024):
    """Install a bounded qualification-only bank around actual slab scopes.

    No generic FMM query is cached outside ``SlipSlabInduction._images``.
    This is deliberately not a public cache API or a production integration.
    """
    import taichi as ti
    from taichi.lang.field import ScalarField
    from taichi.lang.matrix import MatrixField

    from source.solvers.vpm.physics.induction import reuse_backends
    from source.solvers.vpm.physics.induction.fmm.device import FMMInduction
    from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
    from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction
    from source.solvers.vpm.physics.induction.treecode.lbvh import _OwnedFields

    assert reuse_backends.FMMTargetEvaluator is FMMTargetEvaluator
    previous_images = SlipSlabInduction._images
    previous_prepare = FMMTargetEvaluator.prepare_targets
    image_signature = signature(previous_images)
    active, records = [], []

    def storage_members(field):
        if type(field) not in (ScalarField, MatrixField):
            return None
        try:
            return tuple(
                (member.ptr.snode().get_snode_tree_id(), member.ptr.snode().id)
                for member in field._get_field_members()
            )
        except (AttributeError, RuntimeError, TypeError):
            return None

    def admitted_scope(slab, arguments):
        if type(slab) is not SlipSlabInduction or type(slab.base) is not FMMInduction:
            return False
        read = [arguments.get(name) for name in (
            "target_position", "source_position", "source_strength", "source_radius", "stage_strength"
        )]
        write = [arguments.get(name) for name in ("velocity", "gradient", "stage_rate")]
        # Include current shell/backend/tree scratch, not just explicit outputs.
        # Fresh owned allocations made later cannot alias existing read fields.
        base, target = slab.base, slab.base._target_workspace
        owners = (slab, base, base.workspace, getattr(base.workspace, "tree", None),
                  target, getattr(target, "tree", None))
        for owner in owners:
            if owner is not None:
                write.extend(value for value in vars(owner).values()
                             if type(value) in (ScalarField, MatrixField))
        return fields_are_disjoint(read, write, storage_members)

    @ti.data_oriented
    class Storage:
        def __init__(self, count):
            self.capacity, self.bytes, self.owner = count, 96 * count, _OwnedFields()
            try:
                self.position = self.owner.vector(3, dtype=ti.f32, shape=count)
                self.sorted_indices = self.owner.scalar(dtype=ti.i32, shape=count)
                self.leaf_depth = self.owner.scalar(dtype=ti.i32, shape=count)
                self.leaf_nodes = self.owner.scalar(dtype=ti.i32, shape=count)
                self.centre = self.owner.vector(3, dtype=ti.f32, shape=2 * count)
                self.half_size = self.owner.scalar(dtype=ti.f32, shape=2 * count)
                self.particle_count = self.owner.scalar(dtype=ti.i32, shape=2 * count)
                self.left = self.owner.scalar(dtype=ti.i32, shape=2 * count)
                self.right = self.owner.scalar(dtype=ti.i32, shape=2 * count)
                self.parent = self.owner.scalar(dtype=ti.i32, shape=2 * count)
                self.bound_radius = self.owner.scalar(dtype=ti.f32, shape=2 * count)
                self.owner.finalize()
            except BaseException:
                try:
                    self.close()
                except BaseException as cleanup_error:
                    raise RuntimeError(
                        "Partial geometry allocation cleanup failed; memory recovery is uncertain"
                    ) from cleanup_error
                raise

        @ti.kernel
        def _capture(self, target: ti.template(), start: ti.i32, count: ti.i32, leaves: ti.i32):
            for i in range(count):
                self.position[start + i] = target.tree.position[i]
                self.sorted_indices[start + i] = target.tree.sorted_indices[i]
                self.leaf_depth[start + i] = target.tree.node_depth[i]
            for i in range(leaves):
                self.leaf_nodes[start + i] = target.leaf_nodes[i]
            for i in range(2 * count - 1):
                j = 2 * start + i
                self.centre[j] = target.tree.node_centre[i]
                self.half_size[j] = target.tree.node_half_size[i]
                self.particle_count[j] = target.tree.node_particle_count[i]
                self.left[j], self.right[j] = target.tree.node_left[i], target.tree.node_right[i]
                self.parent[j] = target.tree.node_parent[i]
                self.bound_radius[j] = target.node_bound_radius[i]

        @ti.kernel
        def _restore(self, target: ti.template(), start: ti.i32, count: ti.i32, leaves: ti.i32):
            for i in range(count):
                target.tree.position[i] = self.position[start + i]
                target.tree.sorted_indices[i] = self.sorted_indices[start + i]
                target.tree.node_depth[i] = self.leaf_depth[start + i]
            for i in range(leaves):
                target.leaf_nodes[i] = self.leaf_nodes[start + i]
            for i in range(2 * count - 1):
                j = 2 * start + i
                target.tree.node_centre[i] = self.centre[j]
                target.tree.node_half_size[i] = self.half_size[j]
                target.tree.node_particle_count[i] = self.particle_count[j]
                target.tree.node_left[i], target.tree.node_right[i] = self.left[j], self.right[j]
                target.tree.node_parent[i] = self.parent[j]
                target.node_bound_radius[i] = self.bound_radius[j]
            target.leaf_count[None] = leaves
            target.tree.n_particles_total[None] = count

        def capture(self, target, start, count):
            leaves = int(target.leaf_count[None])
            if not 0 < leaves <= count:
                raise RuntimeError("invalid prepared target cell count")
            self._capture(target, start, count, leaves)
            return leaves

        def restore(self, target, start, count, leaves):
            self._restore(target, start, count, leaves)
            target._prepare_target_paths(count)
            if int(target.target_path_error[None]):
                raise RuntimeError("restored target hierarchy exceeds its ancestry bound")

        def close(self):
            owner, self.owner = self.owner, None
            if owner is not None:
                owner.destroy()

    def layout(target):
        fields = (
            target.tree.position, target.tree.sorted_indices, target.tree.node_depth,
            target.tree.node_centre, target.tree.node_half_size, target.tree.node_particle_count,
            target.tree.node_left, target.tree.node_right, target.tree.node_parent,
            target.node_bound_radius, target.leaf_nodes, target.leaf_count,
            target.target_path, target.target_path_length, target.target_path_error,
        )
        if target._fields.tree is None or target.tree._field_owner.tree is None:
            raise RuntimeError("target geometry owner is closed")
        return (
            id(target), id(target._fields.tree), id(target.tree._field_owner.tree),
            id(target.source), id(target.source._field_owner.tree),
            id(target.source.tree), id(target.source.tree._field_owner.tree),
            target.max_targets, target.target_path_capacity,
            target.tree.max_tree_depth_guard, target.tree.max_leaf_size,
            tuple((id(f), str(f.dtype), f.shape, getattr(f, "n", 1)) for f in fields),
        )

    def prepare(target, position, count, *, target_start=0):
        if not active or type(target) is not FMMTargetEvaluator:
            return previous_prepare(target, position, count, target_start=target_start)
        return active[-1].prepare(
            target, previous_prepare, position, count, target_start=target_start
        )

    def images(slab, *args, **kwargs):
        if active:
            raise RuntimeError("qualification does not allow nested immutable-target scopes")
        arguments = image_signature.bind(slab, *args, **kwargs).arguments
        count = int(arguments["target_count"])
        tile = min(slab.physics.max_evaluation_points,
                   getattr(slab.base, "max_image_block_targets", slab.physics.max_evaluation_points))
        bank = ImmutableTargetGeometryBank(
            arguments["target_position"], count, tile, max_bytes=max_bytes,
            allocate=lambda required: pool.acquire(slab.base, required),
            layout=layout, synchronize=ti.sync,
        )
        if not admitted_scope(slab, arguments):
            bank._disable("unsupported-or-aliased-scope")
        records.append(bank.record)
        active.append(bank)
        try:
            return previous_images(slab, *args, **kwargs)
        except BaseException as error:
            bank.record.update(status="failed", error=repr(error))
            try:
                bank.close()
            except BaseException as cleanup_error:
                error.add_note(f"Geometry scope cleanup also failed: {cleanup_error!r}")
            raise
        finally:
            active.pop()
            bank.close()
            bank.record["allocation_pool_after_scope"] = dict(pool.record)

    pool = GeometryAllocationPool(max_bytes, Storage)
    FMMTargetEvaluator.prepare_targets = prepare
    SlipSlabInduction._images = images
    original_error = None
    try:
        yield records
    except BaseException as error:
        original_error = error
        raise
    finally:
        SlipSlabInduction._images = previous_images
        FMMTargetEvaluator.prepare_targets = previous_prepare
        try:
            pool.close()
        except BaseException as cleanup_error:
            if original_error is None:
                raise
            original_error.add_note(f"Geometry pool final cleanup also failed: {cleanup_error!r}")
        finally:
            if records:
                records[-1]["allocation_pool_at_factory_exit"] = dict(pool.record)


__all__ = [
    "GeometryAllocationPool", "ImmutableTargetGeometryBank", "fields_are_disjoint",
    "target_geometry_bank_factory",
]
