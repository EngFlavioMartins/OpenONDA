import os

def get_openfoam_paths():
    """Dummy function to return OpenFOAM installation directory and version.
       Replace this with your actual logic to determine OpenFOAM paths.
    """
    wm_project_dir = "/usr/lib/openfoam/openfoam2406"  # Replace with actual logic if needed
    foam_version = "2406"  # Replace with actual logic if needed
    return wm_project_dir, foam_version

def fix_libstdcpp():
    """Fix libstdc++ issue by ensuring the correct version is used."""
    conda_lib_path = os.path.expanduser("~/anaconda3/envs/openONDA/lib")
    system_libstdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    conda_libstdcpp = os.path.join(conda_lib_path, "libstdc++.so.6")
    
    if os.path.exists(conda_libstdcpp):
        print("Fixing libstdc++ issue...")
        os.system(f"mv {conda_libstdcpp} {conda_libstdcpp}.backup")  # Backup the old file
        os.system(f"ln -s {system_libstdcpp} {conda_libstdcpp}")  # Link system version

def add_to_bashrc():
    """Dynamically add OpenFOAM environment variables to .bashrc"""
    bashrc_path = os.path.expanduser("~/.bashrc")

    wm_project_dir, foam_version = get_openfoam_paths()
    if not wm_project_dir:
        print("Could not determine OpenFOAM installation directory. Exiting...")
        return

    # Define the lines to be added, dynamically using OpenFOAM's detected paths
    bashrc_lines = [
        f'conda activate openONDA',
        f'source {wm_project_dir}/etc/bashrc',
        f'alias of{foam_version}="source {wm_project_dir}/etc/bashrc"',
        f'export WM_PROJECT_DIR="{wm_project_dir}"',
        f'export CPLUS_INCLUDE_PATH="${{CPLUS_INCLUDE_PATH}}:{wm_project_dir}/src/OpenFOAM/lnInclude"',
        f'export LIBRARY_PATH="${{LIBRARY_PATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export PYTHONPATH="${{PYTHONPATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/lib"',
        f'export PYTHONPATH="${{PYTHONPATH}}:{wm_project_dir}/platforms/linux64GccDPInt32Opt/bin"'
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

    print("\n>>> OpenFOAM environment variables have been added to ~/.bashrc.")
    print(">>> Please run 'source ~/.bashrc' or restart your terminal to apply changes.")

if __name__ == "__main__":
    fix_libstdcpp()  # Apply fix before setting environment variables
    add_to_bashrc()
