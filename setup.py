from setuptools import setup, find_packages
import subprocess
import os
import sys
import glob
import time 

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
                
        # Change to the target directory
        os.chdir(os.path.join(root, "openONDA/solvers/FVM"))


        # Return to the root directory
        os.chdir(root)
        print("OpenFOAM compilation complete.")
        
    except subprocess.CalledProcessError:
        # Handle any errors during compilation
        raise RuntimeError("OpenFOAM compilation failed. Ensure OpenFOAM is sourced.")


def compile_cython():
    """
    Compile Cython-based Eulerian solver.
    This function compiles the Cython-based code for the Eulerian solver
    by running the setup script with build_ext.
    """
    os.chdir(os.path.join(os.getcwd(), "openONDA/solvers/FVM"))
    
    subprocess.run([sys.executable, "build_cython.py", "build_ext", "--inplace"], check=True)
    
    # Copy the shared object to FOAM_USER_LIBBIN
    foam_user_libbin = os.environ.get("FOAM_USER_LIBBIN")
    
    so_files = glob.glob("fvmModule*.so")
    
    if so_files:
            so_file = so_files[0]
            
    # Rename the shared object file
    new_so_file = "fvmModule.so"
    subprocess.run(["mv", so_file, new_so_file], check=True)

    # Copy the shared object to FOAM_USER_LIBBIN
    foam_user_libbin = os.environ.get("FOAM_USER_LIBBIN")
    
    if foam_user_libbin:
        subprocess.run(["cp", new_so_file, os.path.join(foam_user_libbin, new_so_file)], check=True)
    else:
        raise RuntimeError("FOAM_USER_LIBBIN environment variable is not set.")

    os.chdir(root)
    print("Cython compilation complete.")
    
    
# ==================================================== #
# Check if OpenFOAM is properly set up and modify bashrc
# ==================================================== #
def get_openfoam_paths():
    """Retrieve OpenFOAM installation paths dynamically."""
    try:
        # Run a shell command to get OpenFOAM's environment variables
        output = subprocess.check_output(
            "bash -c 'source /usr/lib/openfoam/openfoam2406/etc/bashrc && env'",
            shell=True,
            text=True
        )

        env_vars = {}
        for line in output.split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key] = value

        wm_project_dir = env_vars.get("WM_PROJECT_DIR", "/usr/lib/openfoam/openfoam2406")
        foam_version = env_vars.get("WM_PROJECT_VERSION", "openfoam2406")

        return wm_project_dir, foam_version

    except subprocess.CalledProcessError:
        print("Error: Could not retrieve OpenFOAM environment variables.")
        return None, None


def add_to_bashrc():
    """Dynamically add OpenFOAM environment variables to .bashrc"""
    bashrc_path = os.path.expanduser("~/.bashrc")

    wm_project_dir, foam_version = get_openfoam_paths()
    if not wm_project_dir:
        print("Could not determine OpenFOAM installation directory. Exiting...")
        return

    # Define the lines to be added, dynamically using OpenFOAM's detected paths
    bashrc_lines = [
        f'source {wm_project_dir}/etc/bashrc',
        f'alias of{foam_version}="source {wm_project_dir}/etc/bashrc"',
        f'export PYTHONPATH="${{PYTHONPATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export PYTHONPATH="${{PYTHONPATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/bin"',
        f'export WM_PROJECT_DIR={wm_project_dir}',
        f'export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:{wm_project_dir}/src/OpenFOAM/lnInclude',
        f'export LIBRARY_PATH=$LIBRARY_PATH:{wm_project_dir}/platforms/linux64GccDPInt32/lib',
        f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{wm_project_dir}/platforms/linux64GccDPInt32/lib',
    ]

    # Read existing .bashrc content
    if os.path.exists(bashrc_path):
        with open(bashrc_path, "r") as file:
            bashrc_content = file.read()
    else:
        bashrc_content = ""

    # Append only missing lines
    with open(bashrc_path, "a") as file:
        for line in bashrc_lines:
            if line not in bashrc_content:
                file.write(f"\n{line}")

    print("\nOpenFOAM environment variables have been added to ~/.bashrc.")
    print("Run 'source ~/.bashrc' or restart your terminal to apply changes.")


# ==================================================== #
# Main Execution: Compile OpenFOAM, Cython, and Update .bashrc
# ==================================================== #

# Compile OpenFOAM boundary conditions and libraries
compile_openfoam()

# Compile Cython solver
compile_cython()

# Add necessary environment variables to .bashrc
add_to_bashrc()

# Make sure to come back to this directory
os.chdir(root)

# Standard Python setup (setuptools)
setup(
    name="openONDA",
    version="0.0.1",
    packages=find_packages(where="."),  # Make sure all packages under the root are detected
    package_dir={"": "."},  # Ensure packages are correctly mapped to the root
    install_requires=["numpy", "scipy", "cython"],
)
