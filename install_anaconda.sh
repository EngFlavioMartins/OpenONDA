sudo apt update

if ! command -v curl &> /dev/null; then
    sudo apt install -y curl
if

curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

chmod u+x Anaconda3-2024.10-1-Linux-x86_64.sh

bash Anaconda3-2024.10-1-Linux-x86_64.sh

source ~/.bashrc

