# ARG GITHUB_TOKEN

# Use a base image with CUDA and Python pre-installed
# FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
# FROM docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and essential tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-dev \
#     python3.12-venv \
#     git \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     && rm -rf /var/lib/apt/lists/*

    # Create virtual environment
RUN python3 -m venv /opt/venv

# Activate venv for all following RUN commands
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Jupyter and the llms library
# RUN pip3 install --upgrade pip --break-system-packages
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install jupyterlab notebook 

# Step 1: Install PyTorch ecosystem first
# RUN pip3 install --no-cache-dir \
#     torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/cu126 \
#     --break-system-packages
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# Step 2: Install standard dependencies from PyPI
RUN pip3 install --no-cache-dir \
    pillow \
    matplotlib \
    opencv-python-headless \
    numpy \
    transformers \
    faiss-cpu \
    pybktree \
    imagehash \
    inference-sdk \
    PyMuPDF \
    tqdm \
    sentence-transformers \
    accelerate \
    bitsandbytes \
    flashrag-dev \
    # vllm>=0.4.1 \
    pyserini \
    packaging \
    ninja \
    wheel \
    setuptools 


# Step 3: Install Flash Attention (now able to safely find torch)
# RUN pip3 install --no-cache-dir \
#     flash-attn  --no-build-isolation \
#     --break-system-packages

RUN pip3 install ultralytics 

RUN pip3 install -U sentencepiece protobuf
RUN rm -rf /root/.cache/huggingface/hub/models--THUDM--CogVideoX-2b

# Register kernel with Jupyter
RUN python -m ipykernel install --name docker-env --display-name "Python (docker-env)"

# Create a working directory
WORKDIR /notebooks

# Expose the port for Jupyter
EXPOSE 8008

# Command to run Jupyter Notebook on container start
# --ip=0.0.0.0 allows connections from outside the container
# --allow-root is necessary if running as the default root user
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8008", "--no-browser", "--allow-root"]



#docker build -t rag-gpu-jupyter .
# from root dir with data 
#docker run --gpus all -it -p 8888:8888 -v "$(pwd):/notebooks" -e GOOGLE_APPLICATION_CREDENTIALS="/notebooks/service-account-key.json" --env-file "$(pwd)/.env"  rag-gpu-jupyter


