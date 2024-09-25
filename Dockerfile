# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

# Allow passing Hugging Face token as a build argument
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository and set the working directory to flux-trainer
RUN git clone https://github.com/edenartlab/flux-trainer.git
WORKDIR /app/flux-trainer
RUN . /app/flux-trainer

# Install the main requirements
RUN pip install --no-cache-dir -r requirements.txt

# Clone and setup sd-scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git \
    && cd sd-scripts \
    && git checkout sd3 \
    && git checkout a2ad7e5644f08141fe053a2b63446d70d777bdcf \
    && pip install --no-cache-dir -r requirements.txt \
    && cd ..

# Install additional dependencies for the project
RUN pip install --no-cache-dir huggingface_hub python-dotenv

EXPOSE 8080
ENV PYTHONUNBUFFERED=1

# Run download_models.py once to trigger model downloads, using the Hugging Face token
# RUN HF_TOKEN=${HF_TOKEN} python3 download_models.py
# Set the default command to python
CMD ["python3", "-u", "main.py", "--config", "template/train_config.json"]