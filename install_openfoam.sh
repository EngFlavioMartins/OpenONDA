#!/bin/bash

CURRENT_DIR=$(pwd)

# ======================
# Install dependencies
# ======================

install_dependencies(){
    echo '# ----------------------------------------------- #'
    echo 'Update package list, install libboost and CUDA'
    echo '# ----------------------------------------------- #'
    sudo apt update
    sudo apt install -y libboost-all-dev
    sudo apt install nvidia-cuda-toolkit -y
    sudo apt install curl -y
}


# ======================
# Install OpenFOAM
# ======================
install_openfoam(){ 
    echo ' '
    echo '# ----------------------------------------------- #'
    echo '>>> Installing OpenFOAM'
    echo '# ----------------------------------------------- #'
    curl https://dl.openfoam.com/add-debian-repo.sh | sudo bash

    sudo apt-get update
    sudo apt-get install openfoam2406-default

    echo ' '
    echo '# ----------------------------------------------- #'
    echo '>>> Adding OpenFOAM to bashrc'
    echo '# ----------------------------------------------- #'

    WM_PROJECT_VERSION="2406"
    WM_PROJECT_DIR="/usr/lib/openfoam/openfoam2406"

    if ! grep -Fxq '# >>> OpenFOAM setup >>>' ~/.bashrc; then
        echo '# >>> OpenFOAM setup >>>' >> ~/.bashrc
        echo 'source ${WM_PROJECT_DIR}/etc/bashrc' >> ~/.bashrc
        echo "alias of${WM_PROJECT_VERSION}=\"source \${WM_PROJECT_DIR}/etc/bashrc\"" >> ~/.bashrc

        echo 'export WM_PROJECT_VERSION=${WM_PROJECT_VERSION}' >> ~/.bashrc
        echo 'export WM_PROJECT_DIR=${WM_PROJECT_DIR}' >> ~/.bashrc
        echo 'export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${WM_PROJECT_DIR}/src/OpenFOAM/lnInclude"' >> ~/.bashrc
        echo 'export LIBRARY_PATH="${LIBRARY_PATH}:${WM_PROJECT_DIR}/platforms/linux64GccDPInt32Opt/lib"' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${WM_PROJECT_DIR}/platforms/linux64GccDPInt32Opt/lib"' >> ~/.bashrc
        echo 'export PYTHONPATH="${PYTHONPATH}:${WM_PROJECT_DIR}/platforms/linux64GccDPInt32Opt/lib"' >> ~/.bashrc
        echo 'export PYTHONPATH="${PYTHONPATH}:${WM_PROJECT_DIR}/platforms/linux64GccDPInt32Opt/bin"' >> ~/.bashrc
        echo "# <<< OpenFOAM ${WM_PROJECT_VERSION} setup <<<" >> ~/.bashrc
    fi

    # Source bashrc to apply the changes
    source ~/.bashrc

    echo ' '
    echo '# ----------------------------------------------- #'
    echo '>>> OpenFOAM setup complete. Paths available.'
    echo '# ----------------------------------------------- #'
}


# =======================
# Verify environment:
# =======================
verify_env(){
    echo ' '
    echo '# ----------------------------------------------- #'
    echo '>>> Verifying environment:'
    echo '# ----------------------------------------------- #'
    source /usr/lib/openfoam/openfoam2406/etc/bashrc && echo $WM_PROJECT_DIR

    # Dummy code: how to check if the environment is properly set?
    if [ -d "$WM_PROJECT_DIR" ]; then
        echo '>>> OpenFOAM environment verified successfully!'
    else
        echo '>>> Warning: OpenFOAM environment may not be properly set.'
    fi

    # Source to apply the changes immediately
    source /usr/lib/openfoam/openfoam2406/etc/bashrc
}

# =======================
# Compile custom OpenFOAM
# =======================
compile_custom_openfoam() {
    echo ' '
    echo '# ----------------------------------------------- #'
    echo ">>> Compiling custom OpenFOAM boundary conditions"
    echo '# ----------------------------------------------- #'
    cd "$CURRENT_DIR/OpenONDA/solvers/FVM/cpp/boundaryConditions" || exit 1
    wclean
    wmake

    echo ' '
    echo '# ----------------------------------------------- #'
    echo ">>> Compiling OpenFOAM custom libraries"
    echo '# ----------------------------------------------- #'
    cd "$CURRENT_DIR/OpenONDA/solvers/FVM/cpp/customFvModels" || exit 1
    wclean
    wmake libso
    cd "$CURRENT_DIR"
    echo ">>> OpenFOAM compilation complete."
}

# =======================
# Compile custom OpenFOAM
# =======================
build_custom_solver() {
    echo ' '
    echo '# ----------------------------------------------- #'
    echo ">>> Building the custom solver"
    echo '# ----------------------------------------------- #'
    cd "$CURRENT_DIR/OpenONDA/solvers/FVM/" || exit 1
    # Build the Cython extension
    python build_cython.py build_ext --inplace || { echo "Cython build failed"; exit 1; }

    # Rename the shared object file
    mv fvmModule*.so fvmModule.so || { echo "Failed to rename the .so file"; exit 1; }
    cd "$CURRENT_DIR"
    echo ">>> Build fvmModule.so file"
}

# Execute steps sequentially
echo ' '
echo '# ================================================ #'
echo '# Stating the installation...'
echo '# ================================================ #'
install_dependencies
install_openfoam
#build_custom_solver
verify_env
compile_custom_openfoam

echo ' '
echo '# ================================================ #'
echo '# Installation complete!'
echo '# ================================================ #'
echo '>>> You can now use "of2406" to source OpenFOAM quickly.'
echo '>>> Run "source ~/.bashrc" or restart your terminal before proceeding.'
