FROM ubuntu:latest

# install basics
RUN apt-get update -y \
 && apt-get install wget -y 

# Install python
RUN apt install -y python && \
    apt install -y python3-pip

# Update pip
RUN pip install --upgrade pip

# download microbiome-toolbox repository
RUN git clone https://github.com/JelenaBanjac/microbiome-toolbox.git && \
    cd microbiome-toolbox

# install microbiome-toolbox
RUN pip install -e .

WORKDIR /microbiome-toolbox

