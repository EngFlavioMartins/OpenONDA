"""Unwired fair same-prefix image tiling qualification; no numerical changes."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from time import perf_counter


class ImageTileDispatcher:
    """Split queries but publish the complete output only after every tile succeeds."""

    def __init__(self, capacity, *, allocate, copy, synchronize):
        self.capacity = capacity
        self.allocate, self.copy, self.synchronize = allocate, copy, synchronize
        self.storage, self.records = {}, []

    def __call__(self, backend, original, **kwargs):
        count, start = int(kwargs["target_count"]), int(kwargs["target_start"])
        if count <= 0 or start < 0:
            raise ValueError("qualification requires a positive prefix and nonnegative start")
        saved = self.storage.get(id(backend))
        if saved is None or saved.count < count:
            fresh = self.allocate(count, min(count, self.capacity))
            if saved is not None:
                saved.close()
            self.storage[id(backend)] = saved = fresh
        source = (
            kwargs["source_position"], kwargs["source_vortex_strength"],
            kwargs["source_core_radius"], kwargs["source_count"],
        )
        context = (
            nullcontext() if getattr(backend, "_fixed_source_key", None) is not None
            else backend.fixed_source_targets(*source)
        )
        record = {"target_count": count, "target_start": start, "tile_capacity": self.capacity,
                  "private_scratch_bytes": saved.bytes, "tiles": [], "status": "running"}
        self.records.append(record)
        try:
            with context:
                for offset in range(0, count, self.capacity):
                    length = min(count - offset, self.capacity)
                    arguments = dict(kwargs, target_start=start + offset, target_count=length,
                                     target_velocity=saved.tile_velocity,
                                     target_velocity_gradient=saved.tile_gradient)
                    self.synchronize()
                    began = perf_counter()
                    original(backend, **arguments)
                    self.synchronize()
                    seconds = perf_counter() - began
                    # Backend outputs always start at zero, independently of query start.
                    self.copy(saved.tile_velocity, saved.velocity, 0, offset, length)
                    self.copy(saved.tile_gradient, saved.gradient, 0, offset, length)
                    target = backend._target_workspace
                    record["tiles"].append({
                        "target_start": start + offset, "target_count": length,
                        "seconds": seconds,
                        "diagnostics": deepcopy(getattr(target, "last_diagnostics", {})),
                        "backend_estimated_bytes": backend._estimate_memory_bytes(),
                        "pair_capacity": target.max_pairs,
                    })
            self.synchronize()
            for source_field, target_field in (
                (saved.velocity, kwargs.get("target_velocity")),
                (saved.gradient, kwargs.get("target_velocity_gradient")),
            ):
                if target_field is not None:
                    self.copy(source_field, target_field, 0, 0, count)
            self.synchronize()
            record["status"] = "complete"
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise

    def close(self):
        for saved in self.storage.values():
            saved.close()
        self.storage.clear()


@contextmanager
def image_tiling_factory(capacity):
    """Temporarily override only tile capacity/dispatch in an isolated process.

The caller must compare the same complete target prefix in both variants.
Source MAC, expansion orders, tail bounds and the 4,194,304-pair hard cap remain
unchanged. Neither variant is a production solver or exact-reuse qualification.
"""
    if capacity not in (8192, 32768):
        raise ValueError("screen only the agreed 8192/32768 tile capacities")
    import taichi as ti

    from source.solvers.vpm.physics.induction import reuse_backends
    from source.solvers.vpm.physics.induction.fmm import device
    from source.solvers.vpm.physics.induction.treecode.lbvh import _OwnedFields

    backend_type = device.FMMInduction
    assert reuse_backends.FMMInduction is backend_type
    assert device._MAX_IMAGE_PAIR_CAPACITY == 4194304
    previous_cap, previous = backend_type.max_image_block_targets, backend_type.evaluate_image_block

    @ti.kernel
    def copy(source: ti.template(), target: ti.template(), first: ti.i32, offset: ti.i32, count: ti.i32):
        for index in range(count):
            target[offset + index] = source[first + index]

    class Storage:
        def __init__(self, count, tile):
            self.count, self.bytes = count, 4 * 12 * (count + tile)
            self.owner = _OwnedFields()
            try:
                self.velocity = self.owner.vector(3, dtype=ti.f32, shape=count)
                self.gradient = self.owner.matrix(3, 3, dtype=ti.f32, shape=count)
                self.tile_velocity = self.owner.vector(3, dtype=ti.f32, shape=tile)
                self.tile_gradient = self.owner.matrix(3, 3, dtype=ti.f32, shape=tile)
                self.owner.finalize()
            except BaseException:
                self.close()
                raise

        def close(self):
            owner, self.owner = self.owner, None
            if owner is not None:
                owner.destroy()

    dispatcher = ImageTileDispatcher(capacity, allocate=Storage, copy=copy, synchronize=ti.sync)

    def tiled(backend, **kwargs):
        return dispatcher(backend, previous, **kwargs)

    backend_type.max_image_block_targets = capacity
    backend_type.evaluate_image_block = tiled
    try:
        yield dispatcher.records
    finally:
        backend_type.evaluate_image_block = previous
        backend_type.max_image_block_targets = previous_cap
        dispatcher.close()
