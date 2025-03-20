import os
import subprocess

# ======================================= #

def find_openfoam_paths():
    """Attempts to find OpenFOAM installation paths automatically."""
    potential_dirs = ["/usr/lib/openfoam", "/opt/openfoam", os.path.expanduser("~/OpenFOAM")]
    for directory in potential_dirs:
        bashrc_path = os.path.join(directory, "etc/bashrc")
        foam_version_path = os.path.join(directory, "bin/foamVersion")
        if os.path.exists(bashrc_path) and os.path.exists(foam_version_path):
            try:
                result = subprocess.run([foam_version_path], capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    return directory, version.split(" ")[1]
            except Exception as e:
                print(f">>> Error getting OpenFOAM version: {e}")
    return None, None

# ======================================= #

def get_openfoam_paths():
    """Gets OpenFOAM paths, automatically or via user input."""
    wm_project_dir, foam_version = find_openfoam_paths()
    if wm_project_dir and foam_version:
        print(f"OpenFOAM found at {wm_project_dir}, version {foam_version}")
        return wm_project_dir, foam_version
    else:
        print(">>> OpenFOAM not found automatically.")
        return input("Enter the OpenFOAM installation directory (e.g., /usr/lib/openfoam2406): "), input("Enter the OpenFOAM version: ")

# ======================================= #

def fix_libstdcpp():
    """Fix libstdc++ issue."""
    default_conda_env = "~/anaconda3/envs/openONDA"
    default_conda_path = os.path.expanduser(default_conda_env)

    if os.path.exists(default_conda_path):
        conda_env_path = default_conda_path
        print(f">>> Default Anaconda environment found: {default_conda_env}")
    else:
        conda_env_path = input(">>> Default Anaconda environment not found. Enter the Anaconda environment path (e.g., ~/anaconda3/envs/openONDA): ")

    conda_lib_path = os.path.join(conda_env_path, "lib")
    system_libstdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    conda_libstdcpp = os.path.join(conda_lib_path, "libstdc++.so.6")

    if os.path.exists(conda_libstdcpp):
        print(">>> Fixing libstdc++ issue...")
        os.system(f"mv {conda_libstdcpp} {conda_libstdcpp}.backup")
        os.system(f"ln -s {system_libstdcpp} {conda_libstdcpp}")
    else:
        print(f">>> Warning: {conda_libstdcpp} not found. libstdc++ fix may not be necessary.")

# ======================================= #

def verify_openfoam_env(wm_project_dir):
    """Verifies OpenFOAM environment."""
    try:
        os.environ["WM_PROJECT_DIR"] = wm_project_dir
        os.environ["FOAM_APPBIN"] = os.path.join(wm_project_dir, "platforms/linux64GccDPInt32Opt/bin")
        os.environ["FOAM_LIBBIN"] = os.path.join(wm_project_dir, "platforms/linux64GccDPInt32Opt/lib")
        os.environ["CPLUS_INCLUDE_PATH"] = os.path.join(wm_project_dir, "src/OpenFOAM/lnInclude")
        result = os.system("foamVersion")
        if result == 0:
            print(">>> OpenFOAM environment verified.")
        else:
            print(">>> OpenFOAM environment verification failed.")
    except KeyError:
        print(">>> OpenFOAM environment variables not set.")

# ======================================= #

def add_to_bashrc(wm_project_dir, foam_version):
    """Adds OpenFOAM variables to .bashrc."""
    bashrc_path = os.path.expanduser("~/.bashrc")
    bashrc_lines = [
        f'source {wm_project_dir}/etc/bashrc',
        f'alias of{foam_version}="source {wm_project_dir}/etc/bashrc"',
        f'export WM_PROJECT_DIR="{wm_project_dir}"',
        f'export CPLUS_INCLUDE_PATH="${{CPLUS_INCLUDE_PATH}}:{wm_project_dir}/src/OpenFOAM/lnInclude"',
        f'export LIBRARY_PATH="${{LIBRARY_PATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export PYTHONPATH="${{PYTHONPATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export PYTHONPATH="${{PYTHONPATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/bin"'
    ]
    if os.path.exists(bashrc_path):
        with open(bashrc_path, "r") as file:
            bashrc_content = file.read()
    else:
        bashrc_content = ""
    with open(bashrc_path, "a") as file:
        for line in bashrc_lines:
            if line not in bashrc_content:
                file.write(f"\n{line}")
    print("\nOpenFOAM environment variables added to ~/.bashrc.")
    print("If no errors appeared, please run 'source ~/.bashrc' or restart your terminal to apply changes.")
    print("Don't forget to go back into the conda environment with 'conda activate openONDA'")
    print("Please check if the following lines were added to your to your ~/.bashrc:")
    print(f"source {wm_project_dir}/etc/bashrc")
    print(f"alias of{foam_version}=\"source {wm_project_dir}/etc/bashrc\"")

# ======================================= #

if __name__ == "__main__":
    wm_project_dir, foam_version = get_openfoam_paths()

    if not wm_project_dir or not foam_version:
        print("Invalid OpenFOAM path or version. Exiting...")
        exit(1)

    verify_openfoam_env(wm_project_dir)
    fix_libstdcpp()
    add_to_bashrc(wm_project_dir, foam_version)
    
    print(">>> Please check if the following lines were added to your to your ~/.bashrc:")
    print(f">>> source {wm_project_dir}/etc/bashrc")
    print(f">>> alias of{foam_version}=\"source {wm_project_dir}/etc/bashrc\"")
    print(f"If no errors appeared so far, you can continue with the installation!")
    print(">>> Then run 'source ~/.bashrc' or restart your terminal.")
