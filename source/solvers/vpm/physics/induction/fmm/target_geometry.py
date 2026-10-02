"""Bounded immutable-target geometry scratch for standard image evaluation.

Only geometry lives across image convergence blocks. Every logical descriptor
expires when the enclosing immutable-target scope ends; the capped physical
allocation survives to avoid Taichi's global JIT invalidation on SNode destroy.
No interaction lists, sources, physical outputs, or convergence decisions are
cached here. Ordinary target queries retain their fresh-preparation contract.
"""

from contextlib import contextmanager
import inspect

import taichi as ti
from taichi.lang.field import ScalarField
from taichi.lang.matrix import MatrixField

from ..treecode.lbvh import _OwnedFields

_BYTES_PER_TARGET = 96
_STORAGE_FIELDS = (
    "position", "sorted_indices", "leaf_depth", "leaf_nodes", "centre", "half_size",
    "particle_count", "left", "right", "parent", "bound_radius",
)


def method_bindings(cls):
    return {
        name: inspect.getattr_static(cls, name)
        for name, value in vars(cls).items()
        if not name.startswith("__") and (callable(value) or isinstance(value, property))
    }


def standard_methods(instance, cls, methods):
    return type(instance) is cls and all(
        name not in vars(instance) and inspect.getattr_static(cls, name) is method
        for name, method in methods.items()
    )


def storage_members(field):
    """Identify every placed component, including aliases via another wrapper."""
    if type(field) not in (ScalarField, MatrixField):
        return None
    try:
        return tuple(
            (member.ptr.snode().get_snode_tree_id(), member.ptr.snode().id)
            for member in field._get_field_members()
        )
    except (AttributeError, RuntimeError, TypeError):
        return None


def disjoint_fields(read_fields, write_fields):
    read, write = set(), set()
    for fields, result in ((read_fields, read), (write_fields, write)):
        for field in fields:
            if field is not None:
                members = storage_members(field)
                if not members:
                    return False
                result.update(members)
    return not read.intersection(write)


def scratch_fields(*owners):
    return tuple(
        value for owner in owners if owner is not None
        for value in vars(owner).values() if type(value) in (ScalarField, MatrixField)
    )


def geometry_layout(target):
    fields = (
        target.tree.position, target.tree.sorted_indices, target.tree.node_depth,
        target.tree.node_centre, target.tree.node_half_size, target.tree.node_particle_count,
        target.tree.node_left, target.tree.node_right, target.tree.node_parent,
        target.node_bound_radius, target.leaf_nodes, target.leaf_count,
        target.target_path, target.target_path_length, target.target_path_error,
    )
    for owner in (target._fields, target.tree._field_owner,
                  target.source._field_owner, target.source.tree._field_owner):
        if owner.tree is None:
            raise RuntimeError("image geometry refers to a closed workspace")
    return (
        id(target), id(target._fields.tree), id(target.tree._field_owner.tree),
        id(target.source), id(target.source._field_owner.tree),
        id(target.source.tree), id(target.source.tree._field_owner.tree),
        target.max_targets, target.target_path_capacity,
        target.tree.max_tree_depth_guard, target.tree.max_leaf_size,
        tuple((id(f), str(f.dtype), f.shape, getattr(f, "n", 1)) for f in fields),
    )


@ti.data_oriented
class TargetGeometryStorage:
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


class TargetGeometryCache:
    """One owned allocation per FMM backend, with an explicit short-lived lease."""

    def __init__(self, max_bytes, diagnostics):
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError("image geometry byte cap must be nonnegative")
        self.diagnostics = diagnostics
        self.storage = None
        self.active = None
        self.scope_open = False
        self.poisoned = False
        self.closed = False

    @property
    def allocated_bytes(self):
        return 0 if self.storage is None else self.storage.bytes

    def invalidate(self):
        if self.active is not None:
            self.active.invalidate()

    def _retire(self):
        if self.scope_open:
            raise RuntimeError("cannot release image geometry during an immutable-target scope")
        if self.poisoned:
            raise RuntimeError("image geometry memory recovery is uncertain")
        if self.storage is not None:
            ti.sync()
            try:
                self.storage.close()
            except BaseException as error:
                self.poisoned = True
                raise RuntimeError("image geometry destruction failed; recovery is uncertain") from error
            self.storage = None
            self.diagnostics.image_geometry_bytes = 0

    def _ensure_storage(self, count):
        if self.storage is not None and self.storage.capacity >= count:
            return True
        capacity = min(1 << (count - 1).bit_length(), self.max_bytes // _BYTES_PER_TARGET)
        self._retire()
        try:
            self.storage = TargetGeometryStorage(capacity)
        except MemoryError:
            # The constructor guarantees cleanup for this recoverable case;
            # uncertain cleanup is deliberately raised as RuntimeError.
            self.storage = None
            return False
        except BaseException:
            self.poisoned = True
            raise
        self.diagnostics.image_geometry_allocations += 1
        self.diagnostics.image_geometry_bytes = self.storage.bytes
        self.diagnostics.peak_image_geometry_bytes = max(
            self.diagnostics.peak_image_geometry_bytes, self.storage.bytes
        )
        return True

    @contextmanager
    def scope(self, position, count, tile_capacity):
        if self.closed or self.poisoned:
            raise RuntimeError("image geometry owner is closed or failed")
        if self.scope_open:
            raise RuntimeError("nested immutable-target geometry scopes are unsupported")
        if count < 0 or tile_capacity < 1:
            raise ValueError("invalid immutable-target prefix or tile capacity")
        admitted = 0 < count <= self.max_bytes // _BYTES_PER_TARGET
        if admitted:
            admitted = self._ensure_storage(count)
        if not admitted:
            self.diagnostics.image_geometry_fallback_scopes += 1
        # Construction can fail before a lease exists. Publish the open state
        # only once its complete session is available for unconditional cleanup.
        session = TargetGeometrySession(self, position, count, tile_capacity) if admitted else None
        self.active = session
        self.scope_open = True
        try:
            yield
        finally:
            self.invalidate()
            self.active = None
            self.scope_open = False

    def prepare(self, target, position, count, *, target_start=0):
        if self.closed or self.poisoned:
            raise RuntimeError("image geometry owner is closed or failed")
        if self.active is None:
            target.prepare_targets(position, count, target_start=target_start)
            self.diagnostics.image_target_geometry_builds += 1
        else:
            self.active.prepare(target, position, count, target_start=target_start)

    def close(self):
        if self.closed:
            return
        self._retire()
        self.closed = True


class TargetGeometrySession:
    """Logical geometry only; never retained after a scope, exception or rebind."""

    def __init__(self, owner, position, count, tile_capacity):
        self.owner, self.position = owner, position
        self.count, self.tile_capacity = int(count), int(tile_capacity)
        self.records, self.binding = {}, None
        self.enabled = True

    def invalidate(self):
        self.records.clear()
        self.binding = None

    def prepare(self, target, position, count, *, target_start=0):
        count, start = int(count), int(target_start)
        try:
            if self.owner.active is not self or not self.owner.scope_open:
                raise RuntimeError("immutable-target geometry lease is no longer active")
            if not (position is self.position and 0 <= start < self.count
                    and start % self.tile_capacity == 0
                    and count == min(self.tile_capacity, self.count - start)):
                self.invalidate()
                self.enabled = False
            if self.enabled:
                layout = geometry_layout(target)
                if layout != self.binding:
                    self.invalidate()
                    self.binding = layout
                if start in self.records:
                    target._prepared_count = 0
                    self.owner.storage.restore(target, start, count, self.records[start])
                    target._prepared_count = count
                    self.owner.diagnostics.image_target_geometry_restores += 1
                    return
            target.prepare_targets(position, count, target_start=start)
            self.owner.diagnostics.image_target_geometry_builds += 1
            if self.enabled:
                if target._prepared_count != count:
                    raise RuntimeError("target geometry preparation did not complete")
                descriptor = self.owner.storage.capture(target, start, count)
                self.records[start] = descriptor
        except BaseException:
            target._prepared_count = 0
            self.invalidate()
            self.enabled = False
            raise


_GEOMETRY_CLASSES = (TargetGeometryCache, TargetGeometrySession, TargetGeometryStorage)
_GEOMETRY_METHODS = {
    cls: {**method_bindings(cls), "__init__": inspect.getattr_static(cls, "__init__")}
    for cls in _GEOMETRY_CLASSES
}


def certified_geometry(cache):
    """Operational admission for complete-field reuse and immutable scopes."""
    # Certify delegated methods before the first allocation as well as between
    # leases. An absent owner must not admit an overridden constructor/copy path.
    if (TargetGeometryCache, TargetGeometrySession, TargetGeometryStorage) != _GEOMETRY_CLASSES:
        return False
    for cls, methods in _GEOMETRY_METHODS.items():
        if any(inspect.getattr_static(cls, name, None) is not method
               for name, method in methods.items()):
            return False
    if cache is None:
        return True
    if not standard_methods(cache, TargetGeometryCache, _GEOMETRY_METHODS[TargetGeometryCache]):
        return False
    if cache.closed or cache.poisoned or cache.scope_open or cache.active is not None:
        return False
    storage = cache.storage
    if storage is None:
        return True
    valid = (
        standard_methods(storage, TargetGeometryStorage, _GEOMETRY_METHODS[TargetGeometryStorage])
        and storage.owner is not None and storage.owner.tree is not None
        and storage.owner.tree.prog is ti.lang.impl.get_runtime().prog
        and not storage.owner.tree.destroyed
        and storage.capacity > 0 and storage.bytes == _BYTES_PER_TARGET * storage.capacity
        and storage.bytes <= cache.max_bytes
    )
    if not valid:
        return False
    members = set()
    tree_id = storage.owner.tree.id
    for name in _STORAGE_FIELDS:
        field = getattr(storage, name)
        field_members = storage_members(field)
        width = 3 if name in ("position", "centre") else 1
        length = storage.capacity if name in _STORAGE_FIELDS[:4] else 2 * storage.capacity
        dtype = ti.f32 if name in ("position", "centre", "half_size", "bound_radius") else ti.i32
        if (
            not field_members or len(field_members) != width or field.shape != (length,)
            or field.dtype != dtype or any(owner != tree_id for owner, _ in field_members)
            or members.intersection(field_members)
        ):
            return False
        members.update(field_members)
    return True
