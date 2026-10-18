# Code-only fresh perception runtime. Models and episode inputs are private mounts.
FROM ghcr.io/swaguto/track3-runpod@sha256:46cc00a62d1076595b71401a96d17fbe53488afe1fe4d7a9784bf3512ac3e908
ENV GIT_LFS_SKIP_SMUDGE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYOPENGL_PLATFORM=egl V2D_OPT_HAND_SHAPE=0 SAM2_BUILD_CUDA=0
RUN python -m pip install --no-cache-dir --no-deps open3d==0.19.0
RUN git clone --filter=blob:none --no-checkout https://github.com/geopavlakos/hamer.git /workspace/hamer \
    && git -C /workspace/hamer checkout --detach 3a01849f4148352e9260b69bf28b65d1671a4905 \
    && git clone --filter=blob:none --no-checkout https://github.com/ZhengdiYu/Dyn-HaMR.git /workspace/Dyn-HaMR \
    && git -C /workspace/Dyn-HaMR checkout --detach fa9cd7412c205fd15ee4139c8caacf79bf6167e6 \
    && git clone --filter=blob:none --no-checkout https://github.com/ZhengdiYu/hamer.git /workspace/Dyn-HaMR/third-party/hamer \
    && git -C /workspace/Dyn-HaMR/third-party/hamer checkout --detach 2a5b8ee852d87fdb56d65df7fb0e473c0d4f7648
COPY code/ /workspace/pr32/
COPY context-manifest.json /opt/locks/scored-code-context.json
RUN git -C /workspace/Dyn-HaMR apply --ignore-space-change \
    /workspace/pr32/reconstruction/validation/issue6/dynhamr-native-fixed-shape.patch
COPY hand-requirements.txt /opt/locks/scored-hand-requirements.txt
RUN python -m venv --system-site-packages /workspace/hand-runtime \
    && /workspace/hand-runtime/bin/python -m pip install --no-cache-dir \
      -r /opt/locks/scored-hand-requirements.txt \
    && /workspace/hand-runtime/bin/python -m pip install --no-cache-dir --no-deps --no-build-isolation \
      git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da \
      git+https://github.com/hassony2/manopth.git@4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b \
      git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4 \
      git+https://github.com/lixiny/manotorch.git@a2a70c591f91551078b7bb2af9b5d9f275b626e0 \
      -e /workspace/hamer -e /workspace/Dyn-HaMR \
      -e /workspace/pr32/reconstruction/modules/v2d_common \
      -e /workspace/pr32/reconstruction/modules/v2d_mv \
      -e /workspace/pr32/reconstruction/modules/v2d_hamer/lib \
      -e /workspace/pr32/reconstruction/modules/v2d_issue6 \
      -e /workspace/pr32/reconstruction/modules/v2d_sam2/lib \
      -e /workspace/pr32/reconstruction/modules/v2d_hand_alignment/lib
ENV PYTHONPATH=/workspace/pr32/reconstruction/modules/v2d_issue6:/opt/FoundationPose \
    V2D_FOUNDATIONPOSE=/opt/FoundationPose V2D_FP_SHIM="" \
    V2D_RUNTIME=/workspace/pr32/reconstruction/modules/v2d_stereo_depth/lib
RUN /workspace/hand-runtime/bin/python -c \
    'import hamer.models,manopth,manotorch,sam2,mediapipe,issue6.worker; print("Scored hand imports verified; inference untested")' \
    && /workspace/hand-runtime/bin/python -m pip freeze > /opt/locks/scored-hand-installed.txt \
    && python -m pip freeze > /opt/locks/scored-foundationpose-installed.txt
RUN python scripts/track3/check_geometry_runtime.py --out /opt/locks/metric-tsdf-runtime.json
ARG VCS_REF
RUN V2D_BUILD_REVISION=${VCS_REF} python -c \
    'import os,json; m=json.load(open("/opt/locks/scored-code-context.json")); assert m["source_commit"] == os.environ["V2D_BUILD_REVISION"]; assert not m["model_weights_included"] and not m["private_footage_included"] and not m["credentials_included"]'
LABEL org.opencontainers.image.source="https://github.com/Swaguto/NVIDIA-V2D-Challenge" \
      org.opencontainers.image.revision="${VCS_REF}"
WORKDIR /workspace/pr32
ENTRYPOINT ["/opt/conda/bin/python"]
CMD ["scripts/track3/batch_public.py", "--help"]
