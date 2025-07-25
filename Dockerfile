FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.10 python3-pip pipx vim make wget

# Set up python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Make a virtual env that we can safely install into
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install poetry

# Set the working directory to the user's home directory
WORKDIR /home

# Copy your project files into the image (adjust as needed)
COPY . /home

# Download DGL wheel
RUN mkdir -p /home/include/dgl && \
    cd /home/include/dgl && \
    wget https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl

# Build USAlign binaries
RUN cd /home/include/USalign && make

# Install Python dependencies with Poetry
RUN cd /home && poetry install

RUN poetry run pip install biotite

ENTRYPOINT ["/bin/bash"]
