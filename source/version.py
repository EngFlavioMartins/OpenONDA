"""OpenONDA version information."""

__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Supported Python version
PYTHON_REQUIRES = "==3.11.*"

# Minimum required dependencies
MIN_NUMPY_VERSION = "1.20.0"
MIN_SCIPY_VERSION = "1.7.0"
MIN_MATPLOTLIB_VERSION = "3.3.0"
MIN_TAICHI_VERSION = "1.7.3"
