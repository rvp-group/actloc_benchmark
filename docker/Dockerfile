FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# install Miniconda
RUN cd /tmp && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
RUN conda init bash && . ~/.bashrc

RUN conda create -n flash_env python=3.10 -y

# install PyTorch and dependencies in the environment (same as your manual process)
RUN conda run -n flash_env conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# install additional packages with pip
RUN conda run -n flash_env pip install flash-attn --no-build-isolation
RUN conda run -n flash_env pip install open3d

# set up the environment to activate flash_env by default
RUN echo "conda activate flash_env" >> ~/.bashrc

# for data
RUN mkdir -p /workspace/example_data

WORKDIR /workspace

# default command to start bash with the environment activated
CMD ["/bin/bash"]


