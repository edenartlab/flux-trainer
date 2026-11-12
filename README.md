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
git checkout a2ad7e5644f08141fe053a2b63446d70d777bdcf
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

If you have all these models already downloaded somewhere else, you can just point to their paths in your `train_config.json` 

### 3. Run a training job:
1. Create a folder of training images
2. make a copy of `template/train_config.json` and adjust with your training setup.
3. Optionally adjust `template/eval_prompts.txt`
3. run `python main.py --config /path/to/train_config.json`
4. All the logs, samples and .safetensors files will appear under ./results


## Building with docker:

`docker build -t flux-trainer .`

No credentials are needed to build the image and none are stored in it: every model it downloads is public. The image is about 50 GB, so build it on a machine with at least 150 GB of free disk. To build it with Google Cloud Build and push it to Artifact Registry, see `cloudbuild.yaml`.

## Running as a job (no database or storage credentials):

The container is started with one argument, a job directory:

`python3 eden_trainer.py --job_dir gs://<bucket>/<path>` (or a local directory)

- `<job_dir>/args.json` is read at start: `{"task_id": "...", "args": {"name": ..., "lora_training_urls": [...], ...}}`
- `<job_dir>/output/` receives the LoRA, up to three checkpoint versions and a thumbnail
- `<job_dir>/result.json` is written last: `{"status": "completed", "lora": "output/...", "checkpoint_versions": [...], "thumbnail": "output/...", "lora_trigger_text": ..., "lora_mode": ...}` or `{"status": "failed", "error": "..."}`. The process exits with a non-zero code on failure.

On Vertex AI the bucket is available as a mounted directory (`/gcs/<bucket>`) using the job's own service account, so the trainer needs no keys for it. The only setting the launcher passes is `OPENAI_API_KEY` (used to detect the training mode and describe the concept), as an environment variable of that one job. The launcher registers the model from `result.json`; the trainer does not write to any database. See `job_io.py`.

Add `--local_test` for a 10-step run that exercises the whole path cheaply.

Tests for the job hand-off: `python -m unittest discover -s tests`

The older `--task_id` / `--db` mode, in which the trainer itself reads and writes the database and the media bucket, still exists but needs `MONGO_URI`, `MONGO_DB_NAME_*`, `AWS_*` and `AWS_BUCKET_NAME_*` in the environment. It is not used by the hosted service any more.
