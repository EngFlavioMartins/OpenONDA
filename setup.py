# setup.py (Modified)

from setuptools import setup, find_packages

import os
import sys
import glob
import shutil
import subprocess

root = os.getcwd()

def compile_cython():
    """Compile Cython-based Eulerian solver."""

    print("# ======================================== #")
    print("# Compiling OpenFOAM custom libraries      #")
    print("# ======================================== #")
    
    cython_dir = os.path.join(root, "OpenONDA/solvers/FVM")

    os.chdir(cython_dir)

    print(f">>> Cleaning up previous installations")
    for file in glob.glob("fvmModule*.so") + glob.glob("fvmModule*.cpp"):
        os.remove(file)
        print(f">>> Removed {file}")

    print(f">>> Building cython")
    
    env = os.environ.copy()  # Copy the current environment

    subprocess.run([sys.executable, "build_cython.py", "build_ext", "--inplace"], env=env)

    so_files = glob.glob("fvmModule*.so")
    if not so_files:
        raise RuntimeError(">>> Cython compilation failed: No .so file found.")

    new_so_file = "fvmModule.so"
    print(subprocess.run(["mv", so_files[0], new_so_file], check=True, capture_output=True, text=True).stdout)

    foam_user_libbin = os.environ.get("FOAM_USER_LIBBIN")
    if not foam_user_libbin:
        print("Warning: FOAM_USER_LIBBIN environment variable is not set. Skipping .so copy.")
    else:
        print(subprocess.run(["cp", new_so_file, os.path.join(foam_user_libbin, new_so_file)], check=True, capture_output=True, text=True).stdout)

    os.chdir(root)
    print("Cython compilation complete.")


# =======================
# Fix libstdc++ in OpenONDA Conda environment
# =======================
def fix_libstdcpp():
    conda_env_path = os.path.expanduser("~/anaconda3/envs/OpenONDA/lib")
    system_libstdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

    if os.path.isfile(system_libstdcpp):
        print("Copying system libstdc++.so.6 to Conda environment...")
        shutil.copy(system_libstdcpp, conda_env_path)
        print(">>> libstdc++.so.6 successfully copied!")
    else:
        print(">>> Warning: System libstdc++.so.6 not found! Check your installation.")


def main():

    compile_cython()
    fix_libstdcpp()

    setup(
        name="OpenONDA",
        version="0.0.1",
        packages=find_packages(where="."),
        package_dir={"": "."},
        install_requires=["numpy", "scipy", "cython"],
    )

if __name__ == "__main__":
    main()