FROM ubuntu:latest

# install basics
RUN apt-get update -y \
 && apt-get install wget -y 

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda \
 && rm Miniconda3-latest-Linux-x86_64.sh


ENV PATH=/miniconda/bin/conda:$PATH
RUN ls

# Create a Python 3.7 environment
RUN conda install -y conda-build \
 && conda create -y --name py37 python=3.7 \
 && conda clean -ya \
 && conda init

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN git clone https://github.com/JelenaBanjac/microbiome-toolbox.git
RUN microbiome-toolbox \
 && conda env create -f environment.yml
RUN echo "source activate microbiome" > ~/.bashrc


WORKDIR /microbiome-toolbox

