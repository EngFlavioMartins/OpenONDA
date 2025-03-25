curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

chmod u+x Anaconda3-2024.10-1-Linux-x86_64.sh

bash Anaconda3-2024.10-1-Linux-x86_64.sh

source ~/.bashrc

conda env create -f ./documentation/openonda_environment.yml

conda activate OpenONDA


