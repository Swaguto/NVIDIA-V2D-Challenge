FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYOPENGL_PLATFORM=egl \
    PYTHONPATH=/opt/FoundationPose:/opt/FoundationPose/mycpp/build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ffmpeg libboost-all-dev libeigen3-dev \
    libegl1 libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel ninja && \
    python -m pip install \
      'numpy<2' 'opencv-python-headless<4.12' scipy pyarrow trimesh \
      pyrender imageio scikit-image scikit-learn omegaconf pyyaml \
      einops kornia transformers huggingface_hub accelerate \
      fvcore iopath pybind11 warp-lang

# FoundationPose uses both compiled rasterization and PyTorch3D geometry.
RUN TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' \
    python -m pip install --no-build-isolation \
      'git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0' && \
    TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' \
    python -m pip install --no-build-isolation \
      'git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8'

COPY reconstruction/modules/v2d_foundation_pose/lib/FoundationPose /opt/FoundationPose
RUN cmake -S /opt/FoundationPose/mycpp -B /opt/FoundationPose/mycpp/build \
      -DPYTHON_EXECUTABLE=/opt/conda/bin/python && \
    cmake --build /opt/FoundationPose/mycpp/build -j 4

RUN python -c 'import cv2, pyarrow, pyrender, scipy, torch, transformers, trimesh; import nvdiffrast.torch; import pytorch3d; from estimater import FoundationPose; print("Track 3 imports OK")'

WORKDIR /workspace
CMD ["sleep", "infinity"]
