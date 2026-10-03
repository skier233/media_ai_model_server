FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.12 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Set Python 3.12 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip dependencies
COPY install/requirements.txt .
COPY install/requirements-base.txt .
RUN python3.12 -m pip install -r requirements.txt

# No system FFmpeg needed: PyAV bundles its own libav* libraries and all
# decoding (video frames and audio) runs in-process through it.

# Copy the wheel file and install it
COPY dist/ai_processing-0.0.0-cp312-cp312-linux_x86_64.whl /tmp/
RUN python3.12 -m pip install /tmp/ai_processing-0.0.0-cp312-cp312-linux_x86_64.whl

# Expose the port FastAPI runs on
EXPOSE 8000

# Set the working directory
WORKDIR /app

# Command to run the server.py script
CMD ["python3.12", "server.py"]