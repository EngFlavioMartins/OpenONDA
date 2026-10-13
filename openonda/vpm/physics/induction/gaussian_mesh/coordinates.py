"""Slab-commensurate auxiliary coordinates, without physical source snapping."""

from collections.abc import Iterable, Sequence
from itertools import islice
import math
from numbers import Integral

import numpy as np

from .runtime import positive_integer


def slab_coordinates(
    position: np.ndarray, zmin: float, zmax: float, spacing: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Lattice coordinates, step sizes and slab cells of physical sources."""
    if np.iscomplexobj(position):
        raise ValueError("real positions required")
    points = np.asarray(position, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or not np.isfinite(points).all()
        or not all(math.isfinite(value) for value in (zmin, zmax, spacing))
        or spacing <= 0
        or zmax <= zmin
    ):
        raise ValueError("finite positions, ordered slab bounds and positive spacing required")
    width = zmax - zmin
    if not math.isfinite(width) or not math.isfinite(width / spacing):
        raise ValueError("nonfinite slab-to-grid ratio")
    count = math.ceil(width / spacing)
    if not 1 <= count <= 2**30:
        raise ValueError("slab-to-grid cell count outside index range")
    steps = np.array([spacing, spacing, width / count])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = points / steps
        result[:, 2] = count * ((points[:, 2] - zmin) / width - 0.5)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite normalized slab coordinates")
    return result, steps, count


# Saved (index, reflected), world (z shift, reflected) and lattice (z shift,
# reflected) image descriptors.
ImageDescriptors = tuple[
    tuple[tuple[int, bool], ...], tuple[tuple[float, bool], ...], tuple[tuple[int, bool], ...]
]


def finite_images(
    images: Iterable[tuple[int, bool]],
    zmin: float,
    zmax: float,
    cells: int,
    max_images: int,
    *,
    include_primary: bool = False,
) -> ImageDescriptors:
    """Bounded immutable descriptors plus exact integer/world image shifts."""
    if type(include_primary) is not bool:
        raise ValueError("primary permission must be an explicit boolean")
    cells = positive_integer(cells, "slab cells")
    if not math.isfinite(zmin) or not math.isfinite(zmax) or zmax <= zmin:
        raise ValueError("finite ordered slab bounds required")
    cap = positive_integer(max_images, "max_images")
    descriptors = tuple(islice(iter(images), cap + 1))
    if not descriptors or len(descriptors) > cap:
        raise ValueError("nonempty finite image list within cap required")
    world: list[tuple[float, bool]] = []
    integer: list[tuple[int, bool]] = []
    saved: list[tuple[int, bool]] = []
    for k, odd in descriptors:
        if isinstance(k, bool) or not isinstance(k, Integral) or type(odd) not in (bool, np.bool_):
            raise ValueError("integer image index and boolean reflection required")
        if k == 0 and not odd and not include_primary:
            raise ValueError("physical primary excluded from image-only field")
        shift = (2 * int(k) - int(odd)) * cells
        value = 2 * int(k) * (zmax - zmin) + (2 * zmin if odd else 0.0)
        if abs(shift) > 2**52 or not math.isfinite(value):
            raise ValueError("image shift exceeds finite exact-coordinate range")
        saved.append((int(k), bool(odd)))
        integer.append((shift, bool(odd)))
        world.append((value, bool(odd)))
    if include_primary and saved.count((0, False)) != 1:
        raise ValueError("source-only whole field requires primary exactly once")
    return tuple(saved), tuple(world), tuple(integer)


def image_tail_edges(
    integer: Sequence[tuple[int, bool]], cells: int, *, include_primary: bool
) -> dict[bool, tuple[int, int]]:
    """Lattice z positions beyond which each source family's image rows are omitted.

    Retained rows of a family must be consecutive, one period apart; the even
    family may lack only the physical primary. Rows beyond these midpoint
    edges are added by the closed-form image tail completion.
    """
    edges: dict[bool, tuple[int, int]] = {}
    for odd in (False, True):
        rows = sorted(shift for shift, parity in integer if parity == odd)
        if not rows:
            raise ValueError("image tail completion requires retained rows in both families")
        expected = [
            shift
            for shift in range(rows[0], rows[-1] + 1, 2 * cells)
            if odd or include_primary or shift != 0
        ]
        if rows != expected:
            raise ValueError("image tail completion requires consecutive retained shells")
        edges[odd] = (rows[0] - cells, rows[-1] + cells)
    return edges
