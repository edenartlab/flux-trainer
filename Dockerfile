# syntax=docker/dockerfile:1
#
# The trainer's environment. Modal builds its image from this file (see
# modal_app.py); `docker build .` gives the same environment locally.
#
# It is the recipe of the last Vertex AI image with three differences:
#   - every Python package is pinned (requirements.lock) instead of taking
#     whatever is newest on the day of the build;
#   - the model weights are not in the image, they are on a Modal Volume that
#     is mounted at /models (./models is a link to it);
#   - no credentials are built in, and none are needed to build it.
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04@sha256:eaaccb3528ceca110601131434ab467e41d694a41e8c9bf280fb27ac18fcb29b
ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# install deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-dev \
    libmagic1 \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app/flux-trainer

# Python packages, exactly as listed. /usr/bin/python3 is named explicitly:
# Modal adds a second Python under /usr/local for its own client, and the
# trainer must not end up in that one.
COPY requirements.lock /tmp/requirements.lock
RUN /usr/bin/python3 -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock
RUN /usr/bin/python3 -m pip install --no-cache-dir --no-deps torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# sd-scripts at the commit every production LoRA was trained with. It stays a
# git checkout: the commit hash is written into each LoRA's metadata.
RUN git clone https://github.com/kohya-ss/sd-scripts.git \
    && cd sd-scripts \
    && git checkout 75933d70a15ac423f396c9d3df04d0d35fae6130 \
    && /usr/bin/python3 -m pip install --no-cache-dir --no-deps -e .

# ./models is the model Volume; `python` for the trainer is always Python 3.10.
RUN ln -s /models /app/flux-trainer/models \
    && mkdir -p /opt/trainer-bin \
    && ln -s /usr/bin/python3 /opt/trainer-bin/python \
    && ln -s /usr/bin/python3 /opt/trainer-bin/python3

# The trainer's code is not copied here. modal_app.py adds this checkout as the
# image's last layer (never a network clone, so the image contains exactly the
# deployed commit); for a local `docker build`, mount or COPY it to
# /app/flux-trainer.
#
# No ENTRYPOINT: Modal starts its own process and runs
# `python3 eden_trainer.py --job_dir=...` from modal_app.py.
