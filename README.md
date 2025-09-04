# Overview
This code performs quick image analysis on web page screenshots using offline batched VLM prompting (using [LMdeploy](https://lmdeploy.readthedocs.io/)).

# Installation
Running these scripts will require linux and a NVIDIA GPU. VRAM usage for InternVL3_5 is around 20GB. Make sure NVIDIA drivers are setup and utilities correctly (see below)

## NVIDIA drivers setup on Ubuntu 24.04 LTS (2025-09-04)

```bash
# install using ubuntu drivers
sudo apt-get -y install ubuntu-drivers-common

# optionally: list available drivers
sudo ubuntu-drivers list --gpgpu

# use server version 570 or (535 should also be fine) - note: it will take some minutes at "EFI variables are not supported on this system", but then finish correctly
sudo apt-get install -y nvidia-driver-570-server nvidia-utils-570-server 

# reboot activates nvidia kernel drivers
sudo reboot 

# test
nvidia-smi

# optionally(?) install libcudnn, see here:
# - https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu
# - https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit nvidia-gds
```

## Host setup notes
- uv install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Uses python 3.10 (for lmdeploy)
- `uv sync`

# Running
