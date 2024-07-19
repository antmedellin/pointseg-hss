# # # docker build -t pointseg_hss .  

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt upgrade -y

RUN apt install -y \
    git \
    build-essential \
    wget \
    unzip \
    pkg-config \
    cmake \
    python3-pip \
    sudo \
    g++ \
    ca-certificates \
    libgl1-mesa-glx \
    imagemagick \
    libmagickwand-dev

# Install PyTorch, torchvision, and cuml with specific versions and extra indexes
# RUN pip install torch==1.12.1 torchvision cuml-cu11 --extra-index-url https://download.pytorch.org/whl/cu116 --extra-index-url https://pypi.nvidia.com


RUN pip install torch==1.12.1 torchvision --no-cache-dir --extra-index-url https://pypi.nvidia.com cuml-cu11

# RUN pip install cuml-cu11 --extra-index-url https://download.pytorch.org/whl/cu116 --extra-index-url https://pypi.nvidia.com

# # Upgrade pip to ensure compatibility with extra-index-url
# RUN pip install --upgrade pip

# # Install cuml-cu11 with the correct extra index URL
# RUN pip install cuml-cu11  --extra-index-url https://download.pytorch.org/whl/cu116 --extra-index-url https://pypi.nvidia.com


# RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cuml-cu11


# Install various Python packages
RUN pip install omegaconf torchmetrics==0.10.3 fvcore iopath xformers==0.0.12 submitit



# Install openmim
RUN pip install openmim

# Uninstall mmcv-full if it exists and install a specific version

# # Update setuptools and install six
# RUN pip3 install --upgrade setuptools six

# # Then try running the mim command again
# RUN mim uninstall -y mmcv-full && mim install mmcv-full==1.5.3

RUN mim uninstall -y mmcv-full && \
    mim install mmcv-full==1.5.3

# Uninstall mmsegmentation if it exists and install a specific version
RUN pip uninstall -y mmsegmentation && \
    mim install mmsegmentation==0.27.0


RUN pip install ipykernel

RUN apt install -y python3-tk libgdal-dev gdal-bin

RUN pip install albumentations transformers evaluate

RUN mim install mmengine 

RUN mim install mmcv==2.0.0rc4

RUN pip uninstall mmsegmentation   -y 
# --upgrade # need to use one included in the repo

RUN pip install \
    ftfy \
    spectral

RUN useradd -m dino_user

RUN echo "dino_user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/dino_user

USER dino_user

SHELL ["/bin/bash", "-c"]

WORKDIR /home