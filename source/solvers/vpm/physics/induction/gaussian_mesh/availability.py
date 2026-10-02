"""Lazy installation admission for the explicitly selected CUDA mesh backend.

No solver fields, allocation pools, FFT plans, device selection or stream
selection are changed here. The restoring FENV scope leaves all caller math
controls and flags as they were. Kernel/FFT execution is an explicit separate
installation check, not an import-time probe or a numerical fallback.
"""

from importlib import import_module
from pathlib import Path
import re


class GaussianMeshUnavailableError(RuntimeError):
    """An explicitly selected optional backend is not installed or supported."""


def _cupy_version(version):
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:$|[.+-])", str(version))
    if match is None or not (14, 2, 0) <= tuple(map(int, match.groups())) < (15, 0, 0):
        raise GaussianMeshUnavailableError(
            "Gaussian mesh requires qualified CuPy >=14.2,<15 with CUDA 12. "
            "Install OpenONDA[gaussian-mesh-cuda12] in this Python environment."
        )
    return str(version)


def require_gaussian_host_runtime():
    """Admit portable IEEE certificates without importing a CUDA dependency."""
    from ....numerics.ieee import _bridge, ieee_arithmetic, require_round_to_nearest
    from ..gaussian_tail._interval import _platform

    try:
        bridge = _bridge()
        require_round_to_nearest()
        with ieee_arithmetic():
            _platform()
    except (ImportError, AttributeError, RuntimeError) as error:
        raise GaussianMeshUnavailableError(
            "Gaussian mesh requires OpenONDA's installed _fenv C extension and "
            "restoring IEEE arithmetic support. Reinstall OpenONDA with a C compiler "
            "available (python -m pip install --force-reinstall --no-deps .), then run "
            "python -m openonda.verify_install --with-gaussian-mesh. "
            "Plain FMM backends do not require this extension."
        ) from error
    return {"execution_backend": "cpu", "fenv_extension": str(bridge.__file__)}


def require_gaussian_mesh_runtime():
    """Validate optional CUDA libraries without selecting a device.

    Explicit CUDA selection raises an actionable dependency error. Portable
    selection may execute the identical finite field operator on the host.
    """
    host = require_gaussian_host_runtime()
    try:
        cp = import_module("cupy")
    except (ImportError, OSError) as error:
        raise GaussianMeshUnavailableError(
            "Gaussian mesh requires optional CuPy/CUDA libraries in this interpreter. "
            "Install OpenONDA[gaussian-mesh-cuda12]; do not install multiple CuPy "
            "CUDA variants together. A compatible NVIDIA driver and CUDA 12 runtime, "
            "cuFFT and NVRTC are required."
        ) from error
    version = _cupy_version(cp.__version__)
    try:
        # CuPy 14.2's assemble_cupy_compiler_options uses this same public
        # resolver for NVRTC's runtime headers. A library-only installation
        # is insufficient even when the driver, runtime and FFT are present.
        pathfinder = import_module("cuda.pathfinder")
        header_directory = pathfinder.find_nvidia_header_directory("cudart")
        if (not isinstance(header_directory, str) or not header_directory
                or not (Path(header_directory) / "cuda_runtime.h").is_file()):
            raise RuntimeError("CUDA runtime headers were not found")
    except Exception as error:
        raise GaussianMeshUnavailableError(
            "Gaussian mesh NVRTC compilation requires discoverable CUDA 12 runtime "
            "headers (cuda_runtime.h). Install the compatible toolkit headers, or "
            "explicitly select cupy-cuda12x[ctk]. For a toolkit layout not discovered "
            "automatically, set CUDA_PATH to its installation prefix in the case's "
            "environment before starting Python, then rerun python -m "
            "openonda.verify_install --with-gaussian-mesh. No environment or backend "
            "was changed."
        ) from error
    try:
        cuda_version = int(cp.cuda.runtime.runtimeGetVersion())
        driver_version = int(cp.cuda.runtime.driverGetVersion())
        devices = int(cp.cuda.runtime.getDeviceCount())
        nvrtc = import_module("cupy_backends.cuda.libs.nvrtc")
        nvrtc_version = tuple(int(value) for value in nvrtc.getVersion())
        cufft = import_module("cupy.cuda.cufft")
        if not callable(getattr(cufft, "PlanNd", None)):
            raise RuntimeError("owned cuFFT PlanNd API is unavailable")
        if devices < 1:
            raise RuntimeError("no CUDA device is available")
        if not 12000 <= cuda_version < 13000 or nvrtc_version[0] != 12:
            raise RuntimeError("this optional backend requires the qualified CUDA 12 runtime/NVRTC")
    except Exception as error:
        raise GaussianMeshUnavailableError(
            "Gaussian mesh CUDA 12 runtime/NVRTC/cuFFT admission failed. Check the "
            "NVIDIA driver and compatible CUDA toolkit libraries; run python -m "
            "openonda.verify_install --with-gaussian-mesh for the explicit kernel/FFT "
            "check. No fallback backend was selected."
        ) from error
    return {"cupy_version": version, "cuda_runtime_version": cuda_version,
            "cuda_driver_version": driver_version, "nvrtc_version": nvrtc_version,
            "cuda_devices": devices, "cuda_header_directory": header_directory,
            "fenv_extension": host["fenv_extension"]}
