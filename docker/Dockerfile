FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# originally from https://github.com/anibali/docker-pytorch/blob/master/cuda-9.0/Dockerfile

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    libturbojpeg \
    cmake \
    software-properties-common \
    gcc \
    g++

# Create a working directory
RUN mkdir /app
WORKDIR /app

# All users can use /app as their home directory
ENV HOME=/app
RUN chmod 777 /app

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=$HOME/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN $HOME/miniconda/bin/conda install conda-build \
 && $HOME/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && $HOME/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=$HOME/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 9.0-specific steps
RUN conda install -y -c pytorch \
    cuda90=1.0 \
    magma-cuda90=2.3.0 \
    "pytorch=0.4.1=py36_cuda9.0.176_cudnn7.1.2_1" \
    torchvision=0.2.1

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0
RUN pip install h5py-cache==1.0

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    libsm6 libxext6 libxrender-dev \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

# Install additional libraries
# conda/pip
RUN conda install -y scikit-learn nb_conda jupyter numba cython
RUN pip install pandas imgaug matplotlib jupyter scikit-image pillow seaborn jpeg4py shapely flask tqdm tensorboardX

ENV CUDA_HOME="/usr/local/cuda"

RUN pip install --upgrade azureml-sdk[notebooks,automl]

# Set the default command to python3
CMD ["python3"]
