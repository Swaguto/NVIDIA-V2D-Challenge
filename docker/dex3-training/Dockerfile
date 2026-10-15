# syntax=docker/dockerfile:1.7
# Build context is produced by make_context.py, excluding all private inputs.
FROM ghcr.io/astral-sh/uv:0.11.21@sha256:ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946 AS uvbin
FROM ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:8585588357d898929bf6f80f0011441cbcab07a3d95e2f5f4d0e24bc5c177190
COPY --from=uvbin /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    VIRTUAL_ENV=/opt/flash-env UV_PROJECT_ENVIRONMENT=/opt/flash-env
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.11.15 && uv venv --python 3.11.15 /opt/flash-env
WORKDIR /opt/flash-chord
COPY runtime/pyproject.toml runtime/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --active --frozen --extra train --no-install-project
COPY runtime/ /opt/flash-chord/
COPY extension/ /opt/dex3-extension/
COPY context-manifest.json /opt/dex3-extension/image-context-manifest.json
RUN /opt/flash-env/bin/python /opt/dex3-extension/install_extension.py --runtime /opt/flash-chord
ARG VCS_REF
LABEL org.opencontainers.image.source="https://github.com/Swaguto/NVIDIA-V2D-Challenge" \
      org.opencontainers.image.revision="${VCS_REF}" \
      io.v2d.flash-chord.commit="8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef"
RUN V2D_BUILD_REVISION=${VCS_REF} /opt/flash-env/bin/python -c \
    'import json,os; m=json.load(open("/opt/dex3-extension/image-context-manifest.json")); assert m["source_code_commit"] == os.environ["V2D_BUILD_REVISION"] and not m["selected_source_dirty"]'
ENV PATH="/opt/flash-env/bin:${PATH}" \
    PYTHONPATH=/opt/flash-chord/src:/opt/dex3-extension \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
    FLASH_CHORD_CACHE_DIR=/root/.cache/flash-chord
ENTRYPOINT ["/opt/flash-env/bin/python"]
CMD ["/opt/dex3-extension/train.py", "--config-dir", "/opt/dex3-extension/configs", "experiment=dex3_pilot"]
