# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Hermetic base image for the bisection agent's ``docker-reconstruct`` mode.
#
# Deliberately does NOT bake Isaac Sim or any pinned runtime stack: that is the
# whole point of per-commit reconstruction. The image only provides the OS/CUDA
# userspace, git, and uv. The container's entrypoint then runs the *same*
# ``local-reconstruct`` runner we use on the host, which builds a fresh uv venv
# and pip-installs each candidate commit's own pinned Isaac Sim + modular stack.
#
# The host GPU driver is provided at run time via ``docker run --gpus all``.
#
# The ubuntu24.04 base ships system python3.12, which the runner needs for
# ``tomllib``; uv still fetches each commit's own interpreter for the venv.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t isaaclab-bisection-agent:dev .
FROM ghcr.io/astral-sh/uv:0.11.20@sha256:eaa5f1a3305307aaf9e67fe2bbba1d85ebbb2d8a63bce23af21797bfafbe0f8b AS uv

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:92906d87596d638d35015c6353053121bd299d25943b875763321653884ba924

ARG AGENT_VERSION=0.1.0a1
LABEL org.opencontainers.image.title="IsaacLab Bisection Agent"
LABEL org.opencontainers.image.version="${AGENT_VERSION}"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

ENV DEBIAN_FRONTEND=noninteractive
# uv hardlinks from its global cache by default; across a bind-mounted cache that
# can span filesystems, so copy instead of failing.
ENV UV_LINK_MODE=copy
ENV PYTHONPATH=/opt/isaaclab-bisection-agent/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        python3 \
        libgl1 \
        libglu1-mesa \
        libxrandr2 \
        libxi6 \
        libxcursor1 \
        libxinerama1 \
        libxt6 \
        libgmp-dev \
        libgomp1 \
        ncurses-term \
    && rm -rf /var/lib/apt/lists/*

# uv provides both the environment manager and (via ``--python X.Y``) the era's
# interpreter, so the base image need not ship multiple Python versions. Copying
# from a versioned multi-architecture image keeps the tool version reproducible.
COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /workspace
COPY src /opt/isaaclab-bisection-agent/src
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
