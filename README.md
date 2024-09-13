# flux-trainer
Flux LoRA trainer and full-finetuning.

For now this is a utility wrapper around [kohya](https://github.com/kohya-ss/sd-scripts/tree/sd3) that deals with:
- dataset downloading
- automatic captioning
- easily passing in training args
- running inference on checkpoints with sample prompts
- packaging and uploading outputs into a .tar file to upload

## Setup instructions:
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

## Training instructions:
1. Create a folder of training images
2. adjust train_config.yaml
3. run main.py (in the root of flux-trainer)
