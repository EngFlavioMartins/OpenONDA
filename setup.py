from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os
import numpy
import shutil
from setuptools.command.build_ext import build_ext

root = os.getcwd()

# Check for OpenFOAM environment variables
try:
    FOAM_SRC = os.environ['FOAM_SRC']
    FOAM_LIBBIN = os.environ['FOAM_LIBBIN']
    FOAM_USER_LIBBIN = os.environ['FOAM_USER_LIBBIN']
    print(">>> OpenFOAM environment variables are correctly set. Continuing...")
except KeyError:
    raise RuntimeError(">>> OpenFOAM environment variables are not set. Source OpenFOAM before installation.")

# Define Cython extension for Eulerian solver
ext_modules = [
    Extension(
        "OpenONDA.solvers.FVM.fvmModule",
        language="c++",
        sources=[
            "OpenONDA/solvers/FVM/foamSolverWrapper.pyx",
            "OpenONDA/solvers/FVM/cpp/solver/foamSolverCore.C",
            "OpenONDA/solvers/FVM/cpp/solver/foamSolverBridge.C"
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
            "OpenONDA/solvers/FVM/cpp/boundaryConditions/lnInclude",
            "OpenONDA/solvers/FVM/cpp/customFvModels/lnInclude",
            "OpenONDA/solvers/FVM/cpp/solver",
            "OpenONDA/solvers/FVM",
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
            "-O3", "-DNoRepository", "-ftemplate-depth-100", "-std=c++14", "-fPIC"
        ]
    )
]

setup(
    name="OpenONDA",
    version="0.0.1",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=["numpy", "scipy", "cython"],
    ext_modules=cythonize(
        ext_modules,
        language_level="3",
        show_all_warnings=True,
        annotate=True
    ),
    include_path=["OpenONDA/solvers/FVM"],
)
