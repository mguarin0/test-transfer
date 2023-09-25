FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04
RUN mkdir  -p /home/pipgcn
COPY . /home/pipgcn
WORKDIR /home/pipgcn
ENV PL_DATA=./pl_data
ENV PL_OUT=./pl_out
ENV PL_EXPERIMENTS=./pl_experiments
ENV CUDA_VISIBLE_DEVICES=2
RUN apt-get update && apt-get install -y \
    vim \
    git-core \
    wget \
    python2.7 \
    python-pip \
    && pip install -r requirements.txt 
RUN gunzip $PL_DATA/*.gz

