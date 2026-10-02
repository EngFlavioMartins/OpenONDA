"""Bounded exact host geometry evidence for grid diffusion.

Only explicitly certified immutable geometry queries can reuse prior grid
results. World coordinates are matched bitwise; particle fields and physical
solutions are never cached. Device residency is owned by the grid operator.
"""

from dataclasses import dataclass
import math
from types import MethodType

import numpy as np


def _same_callable(first, second):
    return first is second or (
        type(first) is MethodType and type(second) is MethodType
        and first.__self__ is second.__self__ and first.__func__ is second.__func__
    )


@dataclass(frozen=True)
class ImmutableBodyGeometryQueries:
    """Internal capability supplied by a certified immutable geometry owner.

    Ordinary user callbacks never acquire this capability implicitly. The
    producer must revalidate its implementation and current geometric content
    on every identity query; a revision label alone is insufficient.
    """

    contains: object
    blocks: object
    identity: object
    initial_content: object
    initial_runtime: object

    def key(self, contains, blocks, revision):
        if (type(self) is not ImmutableBodyGeometryQueries
                or not _same_callable(self.contains, contains)
                or not _same_callable(self.blocks, blocks)):
            return None
        identity = self.identity()
        if identity is None:
            return None
        current_revision, content, runtime = identity
        if content != self.initial_content:
            raise RuntimeError(
                "Certified static body geometry changed in place; construct a fresh geometry "
                "owner and reconfigure body queries before continuing"
            )
        if runtime != self.initial_runtime:
            # A VTK Modified() can be harmless bookkeeping. Unknown internal
            # input changes also alter these generations; both decline reuse
            # rather than pretending the original immutable query is intact.
            return None
        configured = _immutable_token(revision)
        if configured is None or _immutable_token(current_revision) != configured:
            return None
        return _immutable_token((configured, content))


def _immutable_token(value):
    if type(value) in (str, bytes, bool, int):
        return (type(value).__name__, value)
    if type(value) is float and math.isfinite(value):
        return ("float", np.float64(value).tobytes())
    if type(value) is tuple:
        parts = tuple(_immutable_token(item) for item in value)
        return None if any(part is None for part in parts) else ("tuple", parts)
    return None


def _matching_axis(previous, current):
    """Map only byte-identical coordinates, including the sign of zero."""
    dtype = np.uint32 if previous.dtype == np.float32 else np.uint64
    old_bits, new_bits = previous.view(dtype), current.view(dtype)
    values, first = np.unique(old_bits, return_index=True)
    places = np.searchsorted(values, new_bits)
    valid = places < len(values)
    valid[valid] &= values[places[valid]] == new_bits[valid]
    current_ids = np.flatnonzero(valid)
    return current_ids, first[places[valid]]


def _query_chunks(selected):
    flat = selected.reshape(-1)
    for start in range(0, len(flat), 65536):
        ids = np.flatnonzero(flat[start:start + 65536]) + start
        if len(ids):
            yield np.column_stack(np.unravel_index(ids, selected.shape))


@dataclass(frozen=True)
class _Record:
    key: tuple
    axes32: tuple
    axes64: tuple
    mask: np.ndarray
    links: np.ndarray

    @property
    def nbytes(self):
        return self.mask.nbytes + self.links.nbytes + sum(
            axis.nbytes for axis in (*self.axes32, *self.axes64)
        )


class BodyGridGeometryCache:
    def __init__(self, *, max_bytes=64 * 1024 * 1024):
        self.record = None
        self.max_bytes = max_bytes

    @property
    def max_bytes(self):
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, max_bytes):
        value = int(max_bytes)
        if value < 0:
            raise ValueError("negative geometry evidence cap")
        self._max_bytes = value
        if self.record is not None and self.record.nbytes > self.max_bytes:
            self.record = None

    def prepare(self, origin, spacing, shape, *, contains, blocks, geometry_key):
        shape = tuple(int(n) for n in shape)
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError("positive three-dimensional grid required")
        origin = np.asarray(origin, dtype=np.float32).reshape(3)
        spacing = float(spacing)
        if not np.all(np.isfinite(origin)) or not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("finite origin and positive spacing required")
        axes64 = tuple(origin[d] + np.arange(n, dtype=np.int64) * spacing
                       for d, n in enumerate(shape))
        axes32 = tuple(axis.astype(np.float32) for axis in axes64)
        if any(not np.all(np.isfinite(axis)) for axis in (*axes32, *axes64)):
            raise ValueError("nonfinite generated grid")
        token = _immutable_token(geometry_key)
        cacheable = token is not None
        key = (token, np.float64(spacing).tobytes())
        if self.record is not None and self.record.nbytes > self.max_bytes:
            self.record = None
        old = self.record if cacheable and self.record is not None and self.record.key == key else None
        stats = {"nodes": math.prod(shape), "membership_reused": 0,
                 "membership_queried": 0, "links_reused": 0, "links_queried": 0,
                 "retained_bytes": 0, "qualified": cacheable}
        mask = np.empty(shape, dtype=bool)
        known = np.zeros(shape, dtype=bool)
        links = np.zeros(shape, dtype=np.int32)
        try:
            if old is not None:
                matches = [_matching_axis(a, b) for a, b in zip(old.axes32, axes32, strict=True)]
                new_ix = np.ix_(*(new for new, _ in matches))
                old_ix = np.ix_(*(prior for _, prior in matches))
                mask[new_ix] = old.mask[old_ix]
                known[new_ix] = True
                stats["membership_reused"] = int(np.count_nonzero(known))
            for missing in _query_chunks(~known):
                points = np.column_stack([axes32[d][missing[:, d]] for d in range(3)]).astype(np.float64)
                values = np.asarray(contains(points), dtype=bool).reshape(-1)
                if len(values) != len(missing):
                    raise RuntimeError("body classifier must return one flag per node")
                mask[tuple(missing.T)] = values
                stats["membership_queried"] += len(missing)
            if blocks is not None:
                self._links(old, axes64, shape, mask, links, spacing, blocks, stats)
            candidate = _Record(key, axes32, axes64, mask, links)
            # The previous owned record is no longer needed. Release it before
            # copying the new payload, rather than retaining two capped banks
            # during publication. Required output arrays are separate scratch.
            self.record = None
            old = None
            if cacheable and candidate.nbytes <= self.max_bytes:
                self.record = _Record(key, tuple(a.copy() for a in axes32),
                                      tuple(a.copy() for a in axes64), mask.copy(), links.copy())
                stats["retained_bytes"] = candidate.nbytes
            else:
                self.record = None
            return mask, links, stats
        except BaseException:
            self.record = None
            raise

    @staticmethod
    def _links(old, axes64, shape, mask, links, spacing, blocks, stats):
        for axis in range(3):
            lower, upper = [slice(None)] * 3, [slice(None)] * 3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            eligible = np.zeros(shape, dtype=bool)
            eligible[tuple(lower)] = ~mask[tuple(lower)] & ~mask[tuple(upper)]
            reused = np.zeros(shape, dtype=bool)
            if old is not None:
                matches = [_matching_axis(a, b) for a, b in zip(old.axes64, axes64, strict=True)]
                # The old final grid node never evaluated its outward edge.
                current, prior = matches[axis]
                valid = (current + 1 < shape[axis]) & (prior + 1 < old.mask.shape[axis])
                matches[axis] = current[valid], prior[valid]
                new_ix = np.ix_(*(new for new, _ in matches))
                old_ix = np.ix_(*(previous for _, previous in matches))
                next_old = [previous.copy() for _, previous in matches]
                next_old[axis] += 1
                old_eligible = ~old.mask[old_ix] & ~old.mask[np.ix_(*next_old)]
                admit = eligible[new_ix] & old_eligible
                values = (old.links[old_ix] >> axis) & 1
                links[new_ix] |= np.where(admit, values << axis, 0).astype(np.int32)
                reused[new_ix] = admit
            for missing in _query_chunks(eligible & ~reused):
                starts = np.column_stack([axes64[d][missing[:, d]] for d in range(3)])
                ends = starts.copy()
                # Deliberately not axes64[axis][i+1]: original endpoint is
                # formed by adding h to the already-rounded start.
                ends[:, axis] += spacing
                values = np.asarray(blocks(starts, ends), dtype=bool).reshape(-1)
                if len(values) != len(missing):
                    raise RuntimeError("segment classifier must return one flag per edge")
                links[tuple(missing.T)] |= values.astype(np.int32) << axis
                stats["links_queried"] += len(missing)
            stats["links_reused"] += int(np.count_nonzero(reused))
