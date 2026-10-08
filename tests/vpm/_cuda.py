"""Skip CUDA Gaussian-mesh checks where the optional CUDA runtime is not usable."""

import pytest

from openonda.vpm.physics.induction.gaussian_mesh.availability import (
    GaussianMeshUnavailableError,
    require_gaussian_mesh_runtime,
)


def require_gaussian_cuda():
    """Return CuPy when a device, CuPy and its NVRTC headers are all available."""
    cupy = pytest.importorskip("cupy")
    if not cupy.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    try:
        require_gaussian_mesh_runtime()
    except GaussianMeshUnavailableError as error:
        pytest.skip(str(error))
    return cupy
