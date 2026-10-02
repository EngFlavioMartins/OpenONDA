"""Unwired exact host geometry reuse across changing diffusion-grid extents.

The caller supplies the SAME folded membership and segment callbacks used by
the original algorithm. No geometry is inferred from a nominal lattice index,
body bounds, a fitted shape, or particle data. Membership inputs retain the
original f32-then-f64 encoding; link starts/endpoints retain original f64
arithmetic. Only explicitly qualified immutable, pointwise callback bindings
may reuse evidence. Unrecognized callbacks and mutable revision tokens use the
fresh path. This is an experimental host-only reference, not a production API.

Retained payload is capped independently of required output arrays. A failed
query invalidates retained evidence; a complete new record is published only
after all queries succeed. Returned arrays never alias retained evidence.
"""

from dataclasses import dataclass
import math

import numpy as np


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


class ExactBodyGridCache:
    def __init__(self, *, qualified_bindings=(), max_bytes=64 * 1024 * 1024):
        # Explicit prototype qualification, not trust inferred from callable
        # identity or a caller-supplied revision alone. Production integration
        # would certify its standard immutable geometry owner at configuration.
        self.qualified_bindings = tuple(qualified_bindings)
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError("negative geometry evidence cap")
        self.record = None

    def prepare(self, origin, spacing, shape, *, contains, blocks, revision, slab=None):
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
        revision_key = _immutable_token(revision)
        slab_key = ("none",) if slab is None else _immutable_token(tuple(slab))
        qualified = any(contains is a and blocks is b for a, b in self.qualified_bindings)
        cacheable = qualified and revision_key is not None and slab_key is not None
        key = (id(contains), id(blocks), revision_key, slab_key, np.float64(spacing).tobytes())
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
