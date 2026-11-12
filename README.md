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
pip install --no-deps -r requirements.lock
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
git checkout 75933d70a15ac423f396c9d3df04d0d35fae6130
pip install --no-deps -e .
cd ..
```

### 2. Download the models into ./models:
```
python download_models.py
```
This needs no Hugging Face token. It fetches the de-distilled FLUX.1-dev transformer, the FLUX autoencoder, the two text encoders and their tokenizers, CLIPSeg and Florence-2, each at a pinned revision.

If you have all these models already downloaded somewhere else, you can just point to their paths in your `train_config.json`

### 3. Run a training job:
1. Create a folder of training images
2. make a copy of `template/train_config.json` and adjust with your training setup.
3. Optionally adjust `template/eval_prompts.txt`
3. run `python main.py --config /path/to/train_config.json`
4. All the logs, samples and .safetensors files will appear under ./results


## Hosted service: Modal

In production the trainer runs as a [Modal](https://modal.com) app, defined in `modal_app.py`. Nothing is kept running between trainings.

```
modal deploy modal_app.py                 # deploys the app flux-trainer-PROD
modal run modal_app.py::download_models   # once: fills the model Volume
modal run modal_app.py::smoke_test        # short end-to-end run on the GPU
```

- **Image:** built from the `Dockerfile`. Every Python package is pinned in `requirements.lock`, which is the environment of the last Vertex AI image (March 2025) reproduced by `modal_resolve_env.py`. sd-scripts is pinned to the commit recorded in every production LoRA.
- **Models:** on the Modal Volume `flux-trainer-models`, mounted read-only at `./models`. `download_models.py` fetches them at pinned revisions and checks the weight files against their SHA-256. No Hugging Face token is needed: the one gated file (`ae.safetensors` from FLUX.1-dev) is published byte-identically elsewhere.
- **GPU:** one A100 40 GB, the same class as before.
- **Secret:** `flux-trainer-secrets-PROD`, holding only `OPENAI_API_KEY` (used to detect the training mode and describe the concept). The trainer has no database or storage credentials and cannot reach other apps in the Modal workspace.
- **Contract:** the caller starts `train({"task_id": ..., "args": {...}})` and gets the trainer's `result.json` back. The files it names are on the Volume `flux-trainer-jobs-PROD` under `<task id>/output/`; `<task id>/started.json` appears when the container starts working. The caller (Eden's API) copies the files to its media bucket, registers the model and removes the directory.

## Building with docker:

`docker build -t flux-trainer .`

No credentials are needed to build the image and none are stored in it. The image holds the environment only: mount this checkout at `/app/flux-trainer` and put the models in `./models` with `python3 download_models.py`.

## Running as a job (no database or storage credentials):

The trainer is started with one argument, a job directory:

`python3 eden_trainer.py --job_dir <directory>`

- `<job_dir>/args.json` is read at start: `{"task_id": "...", "args": {"name": ..., "lora_training_urls": [...], ...}}`
- `<job_dir>/output/` receives the LoRA, up to three checkpoint versions and a thumbnail
- `<job_dir>/result.json` is written last: `{"status": "completed", "lora": "output/...", "checkpoint_versions": [...], "thumbnail": "output/...", "lora_trigger_text": ..., "lora_mode": ..., "config": {...}}` or `{"status": "failed", "error": "..."}`. The process exits with a non-zero code on failure.

The only setting the trainer needs is `OPENAI_API_KEY`. The launcher registers the model from `result.json`; the trainer does not write to any database. See `job_io.py`.

Add `--local_test` for a 10-step run that exercises the whole path cheaply.

Tests for the job hand-off: `python -m unittest discover -s tests`

The older `--task_id` / `--db` mode, in which the trainer itself reads and writes the database and the media bucket, still exists but needs `MONGO_URI`, `MONGO_DB_NAME_*`, `AWS_*` and `AWS_BUCKET_NAME_*` in the environment. It is not used by the hosted service any more.
