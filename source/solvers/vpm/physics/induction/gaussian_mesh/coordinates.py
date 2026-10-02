"""Slab-commensurate auxiliary coordinates, without physical source snapping."""

from itertools import islice
import math
from numbers import Integral

import numpy as np

from .runtime import positive_integer


def slab_coordinates(position, zmin, zmax, spacing):
    if np.iscomplexobj(position):
        raise ValueError("real positions required")
    points = np.asarray(position, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or not all(math.isfinite(value) for value in (zmin, zmax, spacing))
            or spacing <= 0 or zmax <= zmin):
        raise ValueError("finite positions, ordered slab bounds and positive spacing required")
    width = zmax-zmin
    if not math.isfinite(width) or not math.isfinite(width/spacing):
        raise ValueError("nonfinite slab-to-grid ratio")
    count = math.ceil(width/spacing)
    if not 1 <= count <= 2**30:
        raise ValueError("slab-to-grid cell count outside index range")
    steps = np.array([spacing, spacing, width/count])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = points/steps
        result[:, 2] = count*((points[:, 2]-zmin)/width-.5)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite normalized slab coordinates")
    return result, steps, count


def finite_images(images, zmin, zmax, cells, max_images, *, include_primary=False):
    """Bounded immutable descriptors plus exact integer/world image shifts."""
    if type(include_primary) is not bool:
        raise ValueError("primary permission must be an explicit boolean")
    cells = positive_integer(cells, "slab cells")
    if not math.isfinite(zmin) or not math.isfinite(zmax) or zmax <= zmin:
        raise ValueError("finite ordered slab bounds required")
    cap = positive_integer(max_images, "max_images")
    descriptors = tuple(islice(iter(images), cap+1))
    if not descriptors or len(descriptors) > cap:
        raise ValueError("nonempty finite image list within cap required")
    world, integer, saved = [], [], []
    for k, odd in descriptors:
        if (isinstance(k, bool) or not isinstance(k, Integral)
                or type(odd) not in (bool, np.bool_)):
            raise ValueError("integer image index and boolean reflection required")
        if k == 0 and not odd and not include_primary:
            raise ValueError("physical primary excluded from image-only field")
        shift = (2*int(k)-int(odd))*cells
        value = 2*int(k)*(zmax-zmin)+(2*zmin if odd else 0.)
        if abs(shift) > 2**52 or not math.isfinite(value):
            raise ValueError("image shift exceeds finite exact-coordinate range")
        saved.append((int(k), bool(odd)))
        integer.append((shift, bool(odd)))
        world.append((value, bool(odd)))
    if include_primary and saved.count((0, False)) != 1:
        raise ValueError("source-only whole field requires primary exactly once")
    return tuple(saved), tuple(world), tuple(integer)
