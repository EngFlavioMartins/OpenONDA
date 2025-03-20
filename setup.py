# setup.py (Modified)

from setuptools import setup, find_packages
import subprocess
import os
import sys
import glob

root = os.getcwd()

def compile_openfoam():
    """
    Compile OpenFOAM boundary conditions and custom libraries.
    """
    print("# ======================================== #")
    print("# Compiling OpenFOAM boundary conditions   #")
    print("# ======================================== #")

    try:
        os.chdir(os.path.join(root, "OpenONDA/solvers/FVM/cpp/boundaryConditions"))
        print(subprocess.run(["wclean"], check=True, capture_output=True, text=True).stdout)
        print(subprocess.run(["wmake"], check=True, capture_output=True, text=True).stdout)

        print("# ======================================== #")
        print("# Compiling OpenFOAM custom libraries      #")
        print("# ======================================== #")

        os.chdir(os.path.join(root, "OpenONDA/solvers/FVM/cpp/customLibraries/customFvModels"))

        for lib_dir in os.listdir():
            lib_path = os.path.join(os.getcwd(), lib_dir)
            if os.path.isdir(lib_path):
                os.chdir(lib_path)
                print(subprocess.run(["wclean"], check=True, capture_output=True, text=True).stdout)
                print(subprocess.run(["wmake", "libso"], check=True, capture_output=True, text=True).stdout)

        os.chdir(root)
        print("OpenFOAM compilation complete.")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise RuntimeError("OpenFOAM compilation failed. Ensure OpenFOAM is sourced and configured correctly.")

def compile_cython():
    """Compile Cython-based Eulerian solver."""
    cython_dir = os.path.join(root, "OpenONDA/solvers/FVM")

    os.chdir(cython_dir)

    for file in glob.glob("fvmModule*.so") + glob.glob("fvmModule*.cpp"):
        os.remove(file)
        print(f"Removed {file}")

    print(subprocess.run([sys.executable, "build_cython.py", "build_ext", "--inplace"], check=True, capture_output=True, text=True).stdout)

    so_files = glob.glob("fvmModule*.so")
    if not so_files:
        raise RuntimeError("Cython compilation failed: No .so file found.")

    new_so_file = "fvmModule.so"
    print(subprocess.run(["mv", so_files[0], new_so_file], check=True, capture_output=True, text=True).stdout)

    foam_user_libbin = os.environ.get("FOAM_USER_LIBBIN")
    if not foam_user_libbin:
        raise RuntimeError("FOAM_USER_LIBBIN environment variable is not set. Please set it to the correct OpenFOAM library directory.")

    print(subprocess.run(["cp", new_so_file, os.path.join(foam_user_libbin, new_so_file)], check=True, capture_output=True, text=True).stdout)

    os.chdir(root)
    print("Cython compilation complete.")

def main():
    compile_openfoam()
    compile_cython()

    setup(
        name="OpenONDA",
        version="0.0.1",
        packages=find_packages(where="."),
        package_dir={"": "."},
        install_requires=["numpy", "scipy", "cython"],
    )

if __name__ == "__main__":
    main()