#!/bin/bash

# ======================
# Install dependencies
# ======================

# Update package list and install libboost-all-dev
sudo apt update
sudo apt install -y libboost-all-dev

# ======================
# Install OpenFOAM
# ======================

# Add the repository
curl https://dl.openfoam.com/add-debian-repo.sh | sudo bash

# Update the repository information
sudo apt-get update

# Install preferred package. Eg,
sudo apt-get install openfoam2406-default

# Add OpenFOAM sourcing and alias to ~/.bashrc
echo '# >>> OpenFOAM setup >>>'  >> ~/.bashrc
echo 'source /usr/lib/openfoam/openfoam2406/etc/bashrc' >> ~/.bashrc
echo 'alias of2406="source /usr/lib/openfoam/openfoam2406/etc/bashrc"' >> ~/.bashrc
echo 'export WM_PROJECT_VERSION=2406' >> ~/.bashrc
echo 'export WM_PROJECT_DIR="/usr/lib/openfoam/openfoam2406"'  >> ~/.bashrc

# =======================
# Verify environment:
# =======================
echo 'Verifying environment:'
source /usr/lib/openfoam/openfoam2406/etc/bashrc && echo $WM_PROJECT_DIR

# Dummy code: how to check if the environment is properly set?
if [ -d "$WM_PROJECT_DIR" ]; then
    echo '>>> OpenFOAM environment verified successfully!'
else
    echo '>>> Warning: OpenFOAM environment may not be properly set.'
fi

# Source to apply the changes immediately
source /usr/lib/openfoam/openfoam2406/etc/bashrc

# =======================
# Fix libstdc++ in OpenONDA Conda environment
# =======================
CONDA_ENV_PATH="$HOME/anaconda3/envs/OpenONDA/lib"
SYSTEM_LIBSTDCPP="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

if [ -f "$SYSTEM_LIBSTDCPP" ]; then
    echo 'Copying system libstdc++.so.6 to Conda environment...'
    cp "$SYSTEM_LIBSTDCPP" "$CONDA_ENV_PATH/"
    echo '>>> libstdc++.so.6 successfully copied!'
else
    echo '>>> Warning: System libstdc++.so.6 not found! Check your installation.'
fi

# =======================
# Add paths to bashrc
# =======================
echo 'export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${WM_PROJECT_DIR}/src/OpenFOAM/lnInclude"'  >> ~/.bashrc 
echo 'export LIBRARY_PATH="${LIBRARY_PATH}:${WM_PROJECT_DIR}/platforms/linux64GccDPInt32Opt/lib"'  >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${WM_PROJECT_DIR}/platforms/linux64GccDPInt32Opt/lib"'  >> ~/.bashrc
echo '# <<< OpenFOAM ${WM_PROJECT_VERSION} setup <<<'  >> ~/.bashrc

echo '>>> You can now use "of2406" to source OpenFOAM quickly.'
echo '>>> Run "source ~/.bashrc" or restart your terminal before proceeding.'
