"""Fixed-shape host arrays for chunked transfers between NumPy and Taichi fields.

Taichi's Vulkan/Metal external-array staging can cache one device buffer per
distinct ndarray shape. Long runs whose particle count changes at every
remeshing or regeneration step would leak cached staging buffers if transfers
used arrays of shape ``(N, ...)`` or tail slices of varying size. Every transfer
therefore goes through one array of a fixed chunk shape per field and
direction, and the live count is passed to the kernel as a scalar.
"""

import numpy as np

_FAMILY_SHAPES = {"vector": (3,), "scalar": (), "matrix": (3, 3), "int": ()}


class HostStagingBuffers:
    """One fixed-shape host array per field, transfer direction and family."""

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self._buffers: dict[tuple[str, str, int], np.ndarray] = {}

    def get(self, family: str, field: object, direction: str, dtype: type) -> np.ndarray:
        """Return the array of ``chunk_size`` entries for ``field`` and ``direction``."""
        if family not in _FAMILY_SHAPES:
            raise ValueError(f"Unknown host transfer buffer family {family!r}")
        key = (family, direction, id(field))
        if key not in self._buffers:
            shape = (self.chunk_size, *_FAMILY_SHAPES[family])
            self._buffers[key] = np.empty(shape, dtype=dtype)
        return self._buffers[key]
