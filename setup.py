from setuptools import setup, find_packages
import subprocess
import os
import sys
import glob

root = os.getcwd()

def compile_openfoam():
    """
    Compile OpenFOAM boundary conditions and custom libraries.
    This function navigates to the appropriate directories within
    the openONDA package, cleans previous builds, and compiles
    the necessary OpenFOAM boundary conditions and custom libraries.
    """
    print("# ======================================== #")
    print("# Compiling OpenFOAM boundary conditions   #")
    print("# ======================================== #")
    
    try:
        # Navigate to the boundary conditions directory
        os.chdir(os.path.join(root, "openONDA/solvers/FVM/cpp/boundaryConditions"))
        
        # Clean and compile boundary conditions
        subprocess.run(["wclean"], check=True)
        subprocess.run(["wmake"], check=True)

        print("# ======================================== #")
        print("# Compiling OpenFOAM custom libraries      #")
        print("# ======================================== #")
        
        # Navigate to the custom libraries directory
        os.chdir(os.path.join(root, "openONDA/solvers/FVM/cpp/customLibraries/customFvModels"))

        # Loop through directories and compile custom libraries
        for lib_dir in os.listdir():
            lib_path = os.path.join(os.getcwd(), lib_dir)
            if os.path.isdir(lib_path):
                os.chdir(lib_path)
                subprocess.run(["wclean"], check=True)
                subprocess.run(["wmake", "libso"], check=True)
                
        # Return to the root directory
        os.chdir(root)
        print("OpenFOAM compilation complete.")
        
    except subprocess.CalledProcessError:
        raise RuntimeError("OpenFOAM compilation failed. Ensure OpenFOAM is sourced.")

def compile_cython():
    """Compile Cython-based Eulerian solver."""
    cython_dir = os.path.join(root, "openONDA/solvers/FVM")

    os.chdir(cython_dir)
    
    # Remove old .so and .cpp files before compilation
    for file in glob.glob("fvmModule*.so") + glob.glob("fvmModule*.cpp"):
        os.remove(file)
        print(f"Removed {file}")

    subprocess.run([sys.executable, "build_cython.py", "build_ext", "--inplace"], check=True)

    # Rename and move the compiled shared object
    so_files = glob.glob("fvmModule*.so")
    if not so_files:
        raise RuntimeError("Cython compilation failed: No .so file found.")

    new_so_file = "fvmModule.so"
    subprocess.run(["mv", so_files[0], new_so_file], check=True)

    # Copy to FOAM_USER_LIBBIN
    foam_user_libbin = os.environ.get("FOAM_USER_LIBBIN")
    if not foam_user_libbin:
        raise RuntimeError("FOAM_USER_LIBBIN environment variable is not set.")

    subprocess.run(["cp", new_so_file, os.path.join(foam_user_libbin, new_so_file)], check=True)

    os.chdir(root)
    print("Cython compilation complete.")

# ==================================================== #
# Main Execution
# ==================================================== #
def main():
    # Step 2: Compile OpenFOAM boundary conditions and libraries
    compile_openfoam()

    # Step 3: Compile Cython solver
    compile_cython()

    # Step 4: Python setup (setuptools)
    setup(
        name="openONDA",
        version="0.0.1",
        packages=find_packages(where="."),  # Make sure all packages under the root are detected
        package_dir={"": "."},  # Ensure packages are correctly mapped to the root
        install_requires=["numpy", "scipy", "cython"],
    )

if __name__ == "__main__":
    main()
