#!/usr/bin/python

import os
from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy


""" 
-------------------------------
To clean up the installation:
-------------------------------
python setup.py clean --all
find . -type f -name "*.c" -delete
find . -type f -name "*.cpp" -delete
find . -type f -name "*.so" -delete

-------------------------------
To rebuid the extension:
-------------------------------
python built_cython.py build_ext --inplace
"""


# Check for OpenFOAM environment variables
try:
    FOAM_SRC = os.environ['FOAM_SRC']
    FOAM_LIBBIN = os.environ['FOAM_LIBBIN']
    FOAM_USER_LIBBIN = os.environ['FOAM_USER_LIBBIN']
except KeyError:
    raise RuntimeError("OpenFOAM environment variables are not set. Source OpenFOAM before installation.")

# Define Cython extension for Eulerian solver
ext_modules = [
    Extension(
        "fvmModule",
        language="c++",
        sources=[
            "foamSolverWrapper.pyx",
            "./cpp/solver/foamSolverCore.C",
            "./cpp/solver/foamSolverBridge.C"
        ],
        library_dirs=[FOAM_LIBBIN, FOAM_USER_LIBBIN],
        libraries=[
            "finiteVolume",
            "fvOptions",
            "meshTools",
            "sampling",
            "turbulenceModels",
            "incompressibleTurbulenceModels",
            "incompressibleTransportModels",
            "dynamicMesh",
            "dynamicFvMesh",
            "topoChangerFvMesh",
            "atmosphericModels",
            "regionFaModels",
            "finiteArea",
            "dl",
            "m",
            "pimpleStepperFoamBC",
            "pimpleStepperFoamFvModels"
        ],
        
        include_dirs=[
            FOAM_SRC + "/finiteVolume/lnInclude",
            FOAM_SRC + "/meshTools/lnInclude",
            FOAM_SRC + "/sampling/lnInclude",
            FOAM_SRC + "/TurbulenceModels/turbulenceModels/lnInclude",
            FOAM_SRC + "/TurbulenceModels/incompressible/lnInclude",
            FOAM_SRC + "/transportModels",
            FOAM_SRC + "/transportModels/incompressible/singlePhaseTransportModel",
            FOAM_SRC + "/dynamicMesh/lnInclude",
            FOAM_SRC + "/dynamicFvMesh/lnInclude",
            FOAM_SRC + "/regionFaModels/lnInclude",
            FOAM_SRC + "/OpenFOAM/lnInclude",
            FOAM_SRC + "/OSspecific/POSIX/lnInclude",
            "./cpp/solver",
            "./cpp/boundaryConditions/lnInclude",
            "./cpp/customLibraries/customFvModels/lnInclude",
            "./lnInclude",
            ".",
            numpy.get_include()
        ],

        extra_link_args=[
            "-m64", "-Dlinux64", "-DWM_ARCH_OPTION=64", "-DWM_DP",
            "-DWM_LABEL_SIZE=32", "-Wall", "-Wextra", "-Wno-old-style-cast",
            "-Wnon-virtual-dtor", "-Wno-unused-parameter", "-Wno-invalid-offsetof",
            "-O3", "-DNoRepository", "-ftemplate-depth-100", "-fPIC", "-shared"
        ],
        extra_compile_args=[
            "-m64", "-Dlinux64", "-DWM_ARCH_OPTION=64", "-DWM_DP",
            "-DWM_LABEL_SIZE=32", "-Wall", "-Wextra", "-Wno-old-style-cast",
            "-Wnon-virtual-dtor", "-Wno-unused-parameter", "-Wno-invalid-offsetof",
            "-O3", "-DNoRepository", "-ftemplate-depth-100", "-std=c++17", "-fPIC"
        ]
    )
]

Options.docstrings = True
Options.embed_pos_in_docstring = True

# Setup function with corrected compiler_directives
setup(
    name="fvmModule",
    ext_modules=cythonize(
        ext_modules,
        language_level="3",
        compiler_directives={
            'embedsignature': True,
            'binding': True
        }
    )
)
