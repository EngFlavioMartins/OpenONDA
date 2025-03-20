import os
import subprocess

# ======================================= #

def get_openfoam_paths():
    """Gets OpenFOAM installation paths from the environment."""
    
    wm_project_dir = os.environ.get("WM_PROJECT_DIR")
    
    if wm_project_dir:
        foam_version = os.path.basename(wm_project_dir).replace("openfoam", "")
        print(f">>> OpenFOAM found: {wm_project_dir}, version {foam_version}")
        return wm_project_dir, foam_version
    else:
        print(">>> Error: WM_PROJECT_DIR is not set. Ensure OpenFOAM is sourced before running this script.")
        exit(1)

# ======================================= #

def fix_libstdcpp():
    """Fix libstdc++ issue."""
    default_conda_env = "~/anaconda3/envs/OpenONDA"
    default_conda_path = os.path.expanduser(default_conda_env)

    if os.path.exists(default_conda_path):
        conda_env_path = default_conda_path
        print(f">>> Default Anaconda environment found: {default_conda_env}")
    else:
        conda_env_path = input(">>> Default Anaconda environment not found. Enter the Anaconda environment path (e.g., ~/anaconda3/envs/OpenONDA): ")

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

def add_to_bashrc(wm_project_dir, foam_version):
    """Adds OpenFOAM variables to .bashrc permanently."""
    bashrc_path = os.path.expanduser("~/.bashrc")
    bashrc_lines = [
        f'# >>> OpenFOAM {foam_version} setup >>>',
        f'export WM_PROJECT_DIR="{wm_project_dir}"',
        f'export CPLUS_INCLUDE_PATH="${{CPLUS_INCLUDE_PATH}}:{wm_project_dir}/src/OpenFOAM/lnInclude"',
        f'export LIBRARY_PATH="${{LIBRARY_PATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export PYTHONPATH="${{PYTHONPATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export PYTHONPATH="${{PYTHONPATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/bin"',
        f'# <<< OpenFOAM {foam_version} setup <<<'
    ]

    # Check if already present
    if os.path.exists(bashrc_path):
        with open(bashrc_path, "r") as file:
            bashrc_content = file.read()
    else:
        bashrc_content = ""

    with open(bashrc_path, "a") as file:
        for line in bashrc_lines:
            if line not in bashrc_content:
                file.write(f"\n{line}")

    print("\n>>> OpenFOAM environment variables added to ~/.bashrc.")
# ======================================= #

if __name__ == "__main__":

    wm_project_dir, foam_version = get_openfoam_paths()

    print("\n>>> Verifying OpenFOAM environment...")
    verify_env_cmd = f"bash -c 'source {wm_project_dir}/etc/bashrc && echo $WM_PROJECT_DIR'"
    env_result = subprocess.run(verify_env_cmd, shell=True, capture_output=True, text=True)
    retrieved_wm_project_dir = env_result.stdout.strip()

    if retrieved_wm_project_dir == wm_project_dir:
        print(">>> OpenFOAM environment verified successfully!")
    else:
        print(">>> Warning: OpenFOAM environment may not be properly set.")

    fix_libstdcpp()
    add_to_bashrc(wm_project_dir, foam_version)

    print("\n>>> Installation completed!")
    print(">>> Please check your ~/.bashrc file to ensure OpenFOAM is sourced correctly.")
    print(f">>> You can now use 'of{foam_version}' to source OpenFOAM quickly.")
    print(">>> Run 'source ~/.bashrc' or restart your terminal before proceeding.")
