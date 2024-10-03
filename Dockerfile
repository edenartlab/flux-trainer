# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# env vars
ARG HF_TOKEN \
    MONGO_URI MONGO_DB_NAME_STAGE MONGO_DB_NAME_PROD \
    AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION_NAME \
    AWS_BUCKET_NAME_STAGE AWS_BUCKET_NAME_PROD \
    OPENAI_API_KEY
ENV HF_TOKEN=${HF_TOKEN} \
    MONGO_URI=${MONGO_URI} \
    MONGO_DB_NAME_STAGE=${MONGO_DB_NAME_STAGE} \
    MONGO_DB_NAME_PROD=${MONGO_DB_NAME_PROD} \
    AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    AWS_REGION_NAME=${AWS_REGION_NAME} \
    AWS_BUCKET_NAME_STAGE=${AWS_BUCKET_NAME_STAGE} \
    AWS_BUCKET_NAME_PROD=${AWS_BUCKET_NAME_PROD} \
    OPENAI_API_KEY=${OPENAI_API_KEY}

# install deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libglib2.0-0 \
    python3 \
    python3-pip \
    libmagic1 \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# clone the repository and set the working directory to flux-trainer
RUN git clone https://github.com/edenartlab/flux-trainer.git
WORKDIR /app/flux-trainer

# install dependencies
RUN pip install timm==1.0.9 requests tqdm pymongo huggingface_hub python-dotenv boto3 python-magic openai
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# clone and setup sd-scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git \
    && cd sd-scripts \
    && git checkout sd3 \
    && git checkout 8bea039a8d9503a3fe696c445ca992301be1d6fd \
    && pip install --no-cache-dir -r requirements.txt \
    && cd ..

# copy download script and download models from huggingface
COPY download_models.py /app/flux-trainer/
RUN HF_TOKEN=${HF_TOKEN} python3 download_models.py

# copy the rest of the files
COPY . /app/flux-trainer/

# Set the default command to python
ENTRYPOINT ["python3", "eden_trainer.py"]
