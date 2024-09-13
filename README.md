# FLUX-trainer
Flux LoRA trainer and full-finetuning.

For now this is a utility wrapper around [kohya](https://github.com/kohya-ss/sd-scripts/tree/sd3) that deals with:
- dataset preparation and cleaning
- automatic captioning (using Florence2)
- easily passing in training args through config.json files
- running sample inference using sample prompts provided in a .txt file
- packaging and uploading outputs into a .tar file to upload (TODO)

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
- [FLUX denoiser](https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors) and [FLUX vae](https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors) from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main)
- [clip_l.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors) and [T5_fp16.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors) from [here](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main)

Easiest way to download these models is:
1. `pip install huggingface_hub`
2. Grab your Huggingface token (account --> settings --> Access Tokens)
3. `huggingface-cli login`

And then run:
```
mkdir models
cd models
huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --repo-type model --local-dir .
huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors  --repo-type model --local-dir .
huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --repo-type model --local-dir .
huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --repo-type model --local-dir .
```

### 3. Run a training job:
1. Create a folder of training images
2. make a copy of `template/train_config.json` and adjust with your training setup.
3. Optionally adjust `template/eval_prompts.txt`
3. run `python main.py --config /path/to/train_config.json`
4. All the logs, samples and .safetensors files will appear under ./results
