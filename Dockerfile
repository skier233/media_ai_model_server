# No nvidia/cuda base image: the torch cu132 wheels ship their own CUDA 13.2
# runtime, cuDNN and cuBLAS via the nvidia-* pip packages, and TensorRT comes
# from tensorrt_cu13_libs. The host driver (R580+) is injected at run time by
# the NVIDIA Container Toolkit (`--gpus all` / the compose deploy block).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # These are what the nvidia/cuda base used to set. `video` is required so the
    # toolkit also mounts libnvcuvid for the h264_cuvid / hevc_cuvid decoders.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# No system FFmpeg or OpenCV: PyAV bundles its own libav* (including the NVDEC
# decoders) and all decoding runs in-process through it; resizing falls back to PIL.

COPY install/requirements.txt install/requirements-base.txt ./
RUN pip install -r requirements.txt

COPY dist/ai_processing-0.0.0-cp312-cp312-linux_x86_64.whl /tmp/
RUN pip install /tmp/ai_processing-0.0.0-cp312-cp312-linux_x86_64.whl && rm /tmp/*.whl

EXPOSE 8000
WORKDIR /app
CMD ["python", "server.py"]
