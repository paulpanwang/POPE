# get the development image from nvidia cuda 11.3, ubuntu16.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu16.04

LABEL name="POPE" \
    maintainer="Tiancheng <athinkingneal@gmail.com>" \
    lastupdate="2024-02-20" 

# create workspace folder and set it as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set the timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# update package lists and install git, wget, vim, libgl1-mesa-glx, and libglib2.0-0
RUN apt-get update && \
    apt-get install -y git wget vim libgl1-mesa-glx libglib2.0-0 unzip

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# initialize conda
RUN conda init bash

# create and activate conda environment
RUN conda create -n pope python=3.9 && echo "source activate pope" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/pope/bin:$PATH

# clone the popo repository
RUN git clone https://github.com/paulpanwang/POPE.git && \
    cd POPE

# set the working directory to the repository
WORKDIR /workspace/POPE

# install the dependencies
RUN pip install -r requirements.txt
RUN python -m pip install --no-cache torch==1.10.1+cu113 torchvision -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install scikit-learn pandas einops


# download the model weights
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  -O weights/sam_vit_h_4b8939.pth && \
wget  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth -O weights/dinov2_vits14.pth

# download the datasets
RUN cd data && wget https://huggingface.co/datasets/paulpanwang/POPE_Dataset/resolve/main/LM_dataset.zip && \
wget https://huggingface.co/datasets/paulpanwang/POPE_Dataset/resolve/main/onepose.zip && \
wget https://huggingface.co/datasets/paulpanwang/POPE_Dataset/resolve/main/onepose_plusplus.zip && \
wget https://huggingface.co/datasets/paulpanwang/POPE_Dataset/resolve/main/ycbv.zip && \
unzip LM_dataset.zip && \
unzip onepose.zip && \
unzip onepose_plusplus.zip && \
unzip ycbv.zip && \
rm LM_dataset.zip onepose.zip onepose_plusplus.zip ycbv.zip


