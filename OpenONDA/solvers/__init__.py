# OpenONDA/solvers/__init__.py

# Just initialize the submodules, no imports here to avoid circular dependencies
from .FVM import fvmModule
from .VPM import vpmModule

# Expose the modules for easy access
__all__ = ["fvmModule", "vpmModule"]
