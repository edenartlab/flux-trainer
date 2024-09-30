# Use an official Python runtime as a parent image
#FROM python:3.10-slim
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# env vars
ARG HF_TOKEN
ARG MONGO_URI
ARG MONGO_DB_NAME_STAGE
ENV HF_TOKEN=${HF_TOKEN} \
    MONGO_URI=${MONGO_URI} \
    MONGO_DB_NAME_STAGE=${MONGO_DB_NAME_STAGE}

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
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# clone the repository and set the working directory to flux-trainer
RUN git clone https://github.com/edenartlab/flux-trainer.git
WORKDIR /app/flux-trainer

# copy download script
COPY download_models.py /app/flux-trainer/

# install dependencies
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install timm==1.0.9 requests tqdm pymongo huggingface_hub python-dotenv
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# clone and setup sd-scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git \
    && cd sd-scripts \
    && git checkout sd3 \
    && git checkout a2ad7e5644f08141fe053a2b63446d70d777bdcf \
    && pip install --no-cache-dir -r requirements.txt \
    && cd ..

# download models from huggingface
RUN HF_TOKEN=${HF_TOKEN} python3 download_models.py

# last steps
RUN apt-get update && apt-get install -y libmagic1
RUN pip install boto3 python-magic

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION_NAME
ARG AWS_BUCKET_NAME_STAGE
ARG AWS_BUCKET_NAME_PROD
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    AWS_REGION_NAME=${AWS_REGION_NAME} \
    AWS_BUCKET_NAME_STAGE=${AWS_BUCKET_NAME_STAGE} \
    AWS_BUCKET_NAME_PROD=${AWS_BUCKET_NAME_PROD}


# copy the rest of the files
COPY . /app/flux-trainer/


# Set the default command to python
ENTRYPOINT ["python3", "main.py", "--config", "template/train_config.json", "--name", "test"]
# ENTRYPOINT ["python3", "main.py"]