"""Optional compiled numerical support; project metadata lives in pyproject.toml.

Failure to build this small helper leaves existing solver backends available.
Certified Gaussian-tail users explicitly require it at admission. Nothing is
compiled on first solver use.
"""

import sys

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "source.solvers.vpm.numerics._fenv",
            sources=["source/solvers/vpm/numerics/_fenv.c"],
            libraries=[] if sys.platform == "win32" else ["m"],
            extra_compile_args=[] if sys.platform == "win32" else ["-fno-fast-math"],
            optional=True,
        )
    ]
)
