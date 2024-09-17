# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository and set the working directory to flux-trainer
RUN git clone https://github.com/edenartlab/flux-trainer.git
WORKDIR /app/flux-trainer

# Copy .env file from local machine to the container
COPY .env /app/flux-trainer/.env

# Install the main requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and setup sd-scripts
RUN git clone https://github.com/kohya-ss/sd-scripts.git \
    && cd sd-scripts \
    && git checkout sd3 \
    && git checkout cefe52629e1901dd8192b0487afd5e9f089e3519 \
    && pip install --no-cache-dir -r requirements.txt \
    && cd ..

# Install additional dependencies for the project
RUN pip install --no-cache-dir huggingface_hub python-dotenv

# Run download_models.py once to trigger model downloads
RUN python3 download_models.py

# Set the default command to python
CMD ["python3", "main.py", "--config", "templates/train_config.json"]