# flux-trainer
Flux LoRA trainer and full-finetuning.

For now this is a utility wrapper around [kohya](https://github.com/kohya-ss/sd-scripts/tree/sd3) that deals with:
- dataset downloading
- automatic captioning
- easily passing in training args
- running inference on checkpoints with sample prompts
- packaging and uploading outputs into a .tar file to upload

## Setup instructions:

### 1. Setup the environment:
```
conda create --name flux python=3.10
conda activate flux
git clone https://github.com/edenartlab/flux-trainer.git
cd flux-trainer
pip install -r requirements.txt
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
git checkout cefe52629e1901dd8192b0487afd5e9f089e3519
git checkout sd3
pip install -r requirements.txt
cd ..
```

### 2. Download the Flux models into ./models:
- [FLUX denoiser](https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors) and [FLUX VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors) from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main)
- [clip_l.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors) and [T5_fp16.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors) from [here](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main)

## Training instructions:
1. Create a folder of training images
2. adjust train_config.yaml
3. run main.py (in the root of flux-trainer)
