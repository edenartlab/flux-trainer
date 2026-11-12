# syntax=docker/dockerfile:1
# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# No credentials are built into this image, and none are needed to build it:
# every model it downloads is public. Build arguments and ENV values are
# recorded in the image layers and can be read by anyone who can pull it, so do
# not add any.
#
# At run time the trainer is started with --job_dir (see job_io.py). It reads
# its arguments from that directory and writes its results back to it, so it
# needs no database or storage credentials. The only setting the launcher
# passes is OPENAI_API_KEY, as a per-job environment variable.

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

# The trainer's code comes from the build context (COPY below), never from a
# network clone, so the image contains exactly the reviewed commit.
WORKDIR /app/flux-trainer

# install dependencies
RUN pip install timm==1.0.9 triton==3.1.0 bitsandbytes==0.44.0 requests==2.32.3 tqdm==4.67.1 pymongo==4.11.1 mediapipe==0.10.21 huggingface_hub==0.28.1 python-dotenv==1.0.1 boto3==1.36.21 python-magic==0.4.27 openai==1.61.0 toml==0.10.2
# Pinned to the release that was current when the last working image was built
# (March 2025); an unpinned install now resolves to a newer, untested version.
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# clone and setup sd-scripts
# this will install specific versions of many dependencies (einops, transformers, accelerate, ...)
RUN git clone https://github.com/kohya-ss/sd-scripts.git \
    && cd sd-scripts \
    && git checkout sd3 \
    ## new hash:
    && git checkout 75933d70a15ac423f396c9d3df04d0d35fae6130 \
    ## old hash:
    ## && git checkout e89653975ddf429cdf0c0fd268da0a5a3e8dba1f \ 
    && pip install --no-cache-dir -r requirements.txt \
    && cd ..

# copy download script and download models from huggingface
# (FLUX weights, text encoders, tokenizers, Florence-2 and CLIPSeg; all public)
COPY download_models.py eden_utils.py /app/flux-trainer/
RUN python3 download_models.py

# copy the rest of the files
COPY . /app/flux-trainer/

# Set the default command to python
ENTRYPOINT ["python3", "eden_trainer.py"]
