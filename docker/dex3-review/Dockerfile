# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
FROM ${PYTHON_IMAGE} AS base
ARG TARGETARCH
RUN test "$TARGETARCH" = amd64
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    NVIDIA_DRIVER_CAPABILITIES=all NVIDIA_VISIBLE_DEVICES=all
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git ffmpeg fonts-dejavu-core build-essential \
    libgl1 libegl1 libglu1-mesa libopengl0 libvulkan1 libx11-6 libxext6 \
    libxi6 libxrender1 libxrandr2 libxcursor1 libxinerama1 libxt6 \
    libsm6 libice6 libglib2.0-0 libgomp1 libnss3 libasound2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/v2d

FROM base AS ik-runtime
COPY docker/dex3-review/ik-runtime-lock.txt /opt/locks/ik-runtime-lock.txt
RUN python -m venv /opt/ik && /opt/ik/bin/pip install --no-cache-dir --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cpu -r /opt/locks/ik-runtime-lock.txt
FROM base AS sim-runtime
COPY docker/dex3-review/setup_sim6.py docker/dex3-review/sim6-runtime-lock.txt /opt/locks/
RUN --mount=type=cache,target=/root/.cache/pip PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PIP_RESUME_RETRIES=20 /usr/local/bin/python /opt/locks/setup_sim6.py --lock /opt/locks/sim6-runtime-lock.txt \
    --prefix /opt/sim --lab /opt/IsaacLab
FROM base AS ik
COPY --from=ik-runtime /opt/ik /opt/ik
COPY --from=ik-runtime /opt/locks/ik-runtime-lock.txt /opt/locks/ik-runtime-lock.txt
COPY robotic_grounding/source/robotic_grounding/robotic_grounding /opt/v2d/robotic_grounding/source/robotic_grounding/robotic_grounding
COPY robotic_grounding/scripts/retarget /opt/v2d/robotic_grounding/scripts/retarget
COPY scripts/dex3_review /opt/v2d/scripts/dex3_review
ENV PYTHONPATH=/opt/v2d/robotic_grounding/source/robotic_grounding PATH=/opt/ik/bin:$PATH
RUN /opt/ik/bin/python -c 'import pinocchio,pink,viser,torch; print("IK imports passed")'
ARG VCS_REF=unknown
LABEL org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.source="https://github.com/Swaguto/NVIDIA-V2D-Challenge"
ENTRYPOINT ["/usr/local/bin/python", "/opt/v2d/scripts/dex3_review/container.py"]
CMD ["doctor", "--runtime", "ik"]

# Inherit the large simulator layers to avoid copying the full SDK a second time.
FROM sim-runtime AS review
COPY --from=ik /opt/ik /opt/ik
COPY --from=ik /opt/locks/ik-runtime-lock.txt /opt/locks/ik-runtime-lock.txt
COPY --from=ik /opt/v2d /opt/v2d
ENV PYTHONPATH=/opt/v2d/robotic_grounding/source/robotic_grounding PATH=/opt/ik/bin:$PATH
ENTRYPOINT ["/usr/local/bin/python", "/opt/v2d/scripts/dex3_review/container.py"]
# The original IK and simulator environments intentionally keep separate NumPy/Torch closures.
ENV V2D_SIM_PYTHON=/opt/sim/bin/python V2D_LAB=/opt/IsaacLab
RUN mkdir -p /outputs /cache /data && \
    /opt/sim/bin/python -c 'import cv2; from PIL import Image; print("Media imports passed")'
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="V2D hand continuity and Dex3 review" \
    org.opencontainers.image.source="https://github.com/Swaguto/NVIDIA-V2D-Challenge" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.description="Pinned kinematic preview runtime; physical grasp and policy evaluation remain unvalidated"
CMD ["doctor", "--runtime", "all"]
