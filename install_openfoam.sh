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
    sudo apt install nvidia-cuda-toolkit
}


# ======================
# Install OpenFOAM
# ======================
install_openfoam(){ 
    echo '# ----------------------------------------------- #'
    echo '>>> Installing OpenFOAM'
    echo '# ----------------------------------------------- #'
    curl https://dl.openfoam.com/add-debian-repo.sh | sudo bash

    sudo apt-get update
    sudo apt-get install openfoam2406-default

    echo '# ----------------------------------------------- #'
    echo '>>> Adding OpenFOAM to bashrc'
    echo '# ----------------------------------------------- #'

    WM_PROJECT_VERSION="2406"
    WM_PROJECT_DIR="/usr/lib/openfoam/openfoam2406"

    if ! grep -Fxq '# >>> OpenFOAM setup >>>' ~/.bashrc; then
        echo '# >>> OpenFOAM setup >>>' >> ~/.bashrc
        echo "source ${WM_PROJECT_DIR}/etc/bashrc" >> ~/.bashrc
        echo "alias of2406=\"source ${WM_PROJECT_DIR}/etc/bashrc\"" >> ~/.bashrc

        echo "export WM_PROJECT_VERSION=${WM_PROJECT_VERSION}" >> ~/.bashrc
        echo "export WM_PROJECT_DIR=${WM_PROJECT_DIR}" >> ~/.bashrc
        echo "export CPLUS_INCLUDE_PATH=\"${CPLUS_INCLUDE_PATH}:${WM_PROJECT_DIR}/src/OpenFOAM/lnInclude\"" >> ~/.bashrc
        echo "export LIBRARY_PATH=\"${LIBRARY_PATH}:${WM_PROJECT_DIR}/platforms/linux64GccDPInt32Opt/lib\"" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}:${WM_PROJECT_DIR}/platforms/linux64GccDPInt32Opt/lib\"" >> ~/.bashrc
        echo "# <<< OpenFOAM ${WM_PROJECT_VERSION} setup <<<" >> ~/.bashrc
    fi

    # Source bashrc to apply the changes
    source ~/.bashrc

    echo '# ----------------------------------------------- #'
    echo '>>> OpenFOAM setup complete. Paths available.'
    echo '# ----------------------------------------------- #'
}


# =======================
# Verify environment:
# =======================
verify_env(){
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
# Fix libstdc++ in OpenONDA Conda environment
# =======================
fix_libstdcpp(){
    CONDA_ENV_PATH="$HOME/anaconda3/envs/OpenONDA/lib"
    SYSTEM_LIBSTDCPP="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

    if [ -f "$SYSTEM_LIBSTDCPP" ]; then
        echo 'Copying system libstdc++.so.6 to Conda environment...'
        cp "$SYSTEM_LIBSTDCPP" "$CONDA_ENV_PATH/"
        echo '>>> libstdc++.so.6 successfully copied!'
    else
        echo '>>> Warning: System libstdc++.so.6 not found! Check your installation.'
    fi
}


# =======================
# Compile custom OpenFOAM
# =======================
compile_custom_openfoam() {
    echo '# ----------------------------------------------- #'
    echo ">>> Compiling custom OpenFOAM boundary conditions"
    echo '# ----------------------------------------------- #'
    cd "$CURRENT_DIR/OpenONDA/solvers/FVM/cpp/boundaryConditions" || exit 1
    wclean
    wmake

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
# Compile Cython
# =======================
compile_cython() {
    echo '# ----------------------------------------------- #'
    echo ">>> Compiling Cython-based PIMPLE solver"
    echo '# ----------------------------------------------- #'
    CYTHON_DIR="$CURRENT_DIR/OpenONDA/solvers/FVM"
    cd "$CYTHON_DIR" || exit 1

    echo ">>> Cleaning up previous installations"
    rm -f fvmModule*.so fvmModule*.cpp

    if [ ! -f "build_cython.py" ]; then
        echo "Error: build_cython.py not found!"
        exit 1
    fi

    echo '# ----------------------------------------------- #'
    echo ">>> Building Cython"
    echo '# ----------------------------------------------- #'
    python3 build_cython.py build_ext --inplace

    SO_FILE=$(ls fvmModule*.so 2>/dev/null | head -n 1)
    if [[ -z "$SO_FILE" ]]; then
        echo "Error: Cython compilation failed: No .so file found."
        exit 1
    fi

    NEW_SO_FILE="fvmModule.so"
    mv "$SO_FILE" "$NEW_SO_FILE"

    if [[ -z "$FOAM_USER_LIBBIN" || ! -d "$FOAM_USER_LIBBIN" ]]; then
        echo "Error: FOAM_USER_LIBBIN is not set or does not exist."
        exit 1
    fi
    
    cp "$NEW_SO_FILE" "$FOAM_USER_LIBBIN/$NEW_SO_FILE"
    cd "$ROOT_DIR"
    echo '# ----------------------------------------------- #'
    echo ">>> Cython compilation complete."
    echo '# ----------------------------------------------- #'
}


# Execute steps sequentially
echo '# ================================================ #'
echo '# Stating the installation...'
echo '# ================================================ #'
install_dependencies
install_openfoam
# fix_libstdcpp
verify_env
compile_custom_openfoam
compile_cython

echo '# ================================================ #'
echo '# Installation complete!'
echo '# ================================================ #'
echo '>>> You can now use "of2406" to source OpenFOAM quickly.'
echo '>>> Run "source ~/.bashrc" or restart your terminal before proceeding.'
