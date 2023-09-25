#1) install conda
#2) install conda env from yaml file
#3) determine if we need alias path to msms executable and stride executable
#4) aws configure cli tool so we can write pickle files out to aws s3

sudo apt-get update

# install anaconda, will require user input
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
sha256sum Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh

source ~/.bashrc

# install anaconda environment from yml file
conda env create -f lib/cpi_preprocessing_environment.yml

source activate cpi_preprocessing

sudo apt install awscli
aws configure

