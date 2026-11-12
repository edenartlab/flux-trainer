"""The FLUX LoRA trainer as a Modal app.

    modal deploy modal_app.py                       # deploy flux-trainer-PROD
    modal run modal_app.py::download_models         # once: fill the model Volume
    modal run modal_app.py::smoke_test              # short end-to-end run on the GPU

What runs is unchanged from the Vertex AI container: the image is built from the
same Dockerfile recipe, and `train` starts the same command the container's
entrypoint did (`python3 eden_trainer.py`), in the same directory, on the same
GPU class (one A100 40 GB). Only the plumbing around it differs:

- The model weights are not in the image; they live on the Volume
  `flux-trainer-models`, mounted read-only where the code expects `./models`.
- The trainer has no database or storage credentials. The caller passes the
  task's arguments to `train`, which hands them to the trainer as a job
  directory (see job_io.py) on the Volume `flux-trainer-jobs-<ENV>`:

      <task id>/started.json   written when the container starts working
      <task id>/output/...     the LoRA, its checkpoint versions, a thumbnail

  `train` returns the trainer's result.json. The caller (Eden's API) copies the
  files to its media bucket and registers the model.
- Its only secret is `flux-trainer-secrets-<ENV>`: OPENAI_API_KEY, for the mode
  detection and the concept description.
- Nothing is kept warm: every run gets a fresh container that is gone when the
  run ends.
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

ENV = os.environ.get("FLUX_TRAINER_ENV", "PROD")

TRAINER_DIR = "/app/flux-trainer"
MODELS_DIR = "/models"  # TRAINER_DIR/models is a symlink to this (Dockerfile)
JOBS_DIR = "/jobs"

# The trainer runs under the image's own Python 3.10 (/usr/bin/python3), which
# holds the pinned training environment. Modal's client runs under a separate
# Python that Modal adds to the image, so the two never share packages.
TRAINER_PYTHON = "/usr/bin/python3"
TRAINER_BIN = "/opt/trainer-bin"  # python -> /usr/bin/python3, first on PATH

# Hard stop for a single run. A default run (3000 steps) takes about 75
# minutes and the longest allowed one (5000 steps) about two hours, so anything
# near this limit is stuck and only costs money.
MAX_RUN_SECONDS = 4 * 3600

app = modal.App(f"flux-trainer-{ENV}")

REPO_DIR = Path(__file__).parent

# The environment comes from the Dockerfile (only requirements.lock is sent as
# build context, so it is rebuilt only when the environment changes); the
# trainer's code is added as the last layer, from this checkout.
image = (
    modal.Image.from_dockerfile(
        REPO_DIR / "Dockerfile",
        context_dir=REPO_DIR,
        ignore=lambda path: path.name != "requirements.lock",
        add_python="3.11",
    )
    .add_local_dir(
        REPO_DIR,
        TRAINER_DIR,
        copy=True,
        ignore=[
            ".git",
            ".env",
            "results",
            "xander",
            "models",
            "datasets",
            "tests",
            "**/__pycache__",
            "sd-scripts",
        ],
    )
    .env({"FLUX_TRAINER_ENV": ENV})
)

models_volume = modal.Volume.from_name("flux-trainer-models", create_if_missing=True)
jobs_volume = modal.Volume.from_name(f"flux-trainer-jobs-{ENV}", create_if_missing=True)


def _trainer_env(extra=None):
    """Environment of the trainer process: the container's, without anything
    that belongs to Modal's own Python or identifies the Modal container."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MODAL_", "PYTHON"))
    }
    env["PATH"] = f"{TRAINER_BIN}:{env.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra or {})
    return env


def _run(cmd, env):
    """Run a command in the trainer directory, streaming its output to the log."""
    process = subprocess.Popen(
        cmd, cwd=TRAINER_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


@app.function(
    image=image,
    volumes={MODELS_DIR: models_volume},
    cpu=4,
    memory=16384,
    timeout=3 * 3600,
)
def download_models() -> dict:
    """Fill the model Volume. Safe to run again: files that are already there
    and have the right SHA-256 are kept."""
    code = _run([TRAINER_PYTHON, "download_models.py", MODELS_DIR], _trainer_env())
    models_volume.commit()
    if code != 0:
        raise RuntimeError(f"download_models.py exited with {code}")
    with open(os.path.join(MODELS_DIR, "manifest.json")) as f:
        return json.load(f)


@app.function(
    image=image,
    gpu="A100-40GB",
    # The Vertex machine (a2-highgpu-1g) had 12 vCPUs and 85 GB of memory.
    cpu=8,
    memory=65536,
    volumes={MODELS_DIR: models_volume.read_only(), JOBS_DIR: jobs_volume},
    secrets=[modal.Secret.from_name(f"flux-trainer-secrets-{ENV}")],
    timeout=MAX_RUN_SECONDS,
    retries=0,
    max_containers=2,
    # Nothing waits around: a container that has no run goes away at once.
    scaledown_window=5,
    # One run per container, like one Vertex job per training: no state is
    # carried from one user's training to the next.
    single_use_containers=True,
    # The container cannot look up or call anything else in the workspace.
    restrict_modal_access=True,
)
def train(job: dict) -> dict:
    """Run one training.

    job is {"task_id": "...", "args": {...}} with the arguments of Eden's
    `flux_trainer` tool. Returns the trainer's result.json: {"status":
    "completed", "lora": "output/<file>", "checkpoint_versions": [...],
    "thumbnail": ..., "lora_trigger_text": ..., "lora_mode": ..., "config":
    {...}, "run_time": seconds}, or {"status": "failed", "error": "..."}. File
    names are relative to <task id>/ on the jobs Volume.
    """
    task_id = str(job.get("task_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", task_id):
        return {"status": "failed", "error": "Invalid task id"}
    if not isinstance(job.get("args"), dict):
        return {"status": "failed", "error": "The job has no arguments"}
    if not os.path.isfile(os.path.join(MODELS_DIR, "manifest.json")):
        return {"status": "failed", "error": "The trainer's model Volume is empty"}

    job_dir = os.path.join(JOBS_DIR, task_id)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "args.json"), "w") as f:
        json.dump({"task_id": task_id, "args": job["args"]}, f)
    with open(os.path.join(job_dir, "started.json"), "w") as f:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat()}, f)
    jobs_volume.commit()

    code = _run(
        [TRAINER_PYTHON, "eden_trainer.py", f"--job_dir={job_dir}"],
        # The models are all on the Volume, at pinned revisions; never look for
        # newer ones on the Hub.
        _trainer_env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
    )

    result_path = os.path.join(job_dir, "result.json")
    if os.path.isfile(result_path):
        with open(result_path) as f:
            result = json.load(f)
    else:
        result = {"status": "failed", "error": f"The trainer exited with code {code}"}

    # The arguments are not needed any more; the caller removes the rest of the
    # directory once it has copied the output files.
    os.remove(os.path.join(job_dir, "args.json"))
    jobs_volume.commit()
    return result


@app.function(image=image, gpu="A100-40GB", volumes={MODELS_DIR: models_volume.read_only()}, timeout=900)
def describe_environment() -> dict:
    """What the trainer process actually runs on: interpreter, packages, GPU,
    sd-scripts commit and the model files. For checking a deployment."""

    def out(cmd):
        done = subprocess.run(cmd, cwd=TRAINER_DIR, env=_trainer_env(), capture_output=True, text=True)
        return (done.stdout + done.stderr).strip()

    manifest_path = os.path.join(MODELS_DIR, "manifest.json")
    return {
        "python": out(["python", "-VV"]),
        "pip_freeze": out(["python", "-m", "pip", "freeze"]).splitlines(),
        "torch": out(
            [
                "python",
                "-c",
                "import torch; print(torch.__version__, torch.version.cuda, "
                "torch.backends.cudnn.version(), torch.cuda.get_device_name(0), "
                "torch.cuda.get_device_properties(0).total_memory)",
            ]
        ),
        "sd_scripts_commit": out(["git", "-C", "sd-scripts", "rev-parse", "HEAD"]),
        "gpu": out(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "models": json.load(open(manifest_path)) if os.path.isfile(manifest_path) else None,
    }


# Five photos of one dog: the public DreamBooth example set.
SMOKE_TEST_IMAGES = [
    "https://huggingface.co/datasets/diffusers/dog-example/resolve/7aac740a23542bde4c568131b815a9ed08a7e192/" + name
    for name in (
        "alvan-nee-9M0tSjb-cpA-unsplash.jpeg",
        "alvan-nee-Id1DBHv4fbg-unsplash.jpeg",
        "alvan-nee-bQaAJCbNq3g-unsplash.jpeg",
        "alvan-nee-brFsZ7qszSY-unsplash.jpeg",
        "alvan-nee-eoqnr8ikwFE-unsplash.jpeg",
    )
]


@app.local_entrypoint()
def smoke_test(steps: int = 100, images: int = 5, mode: str = "object", rank: int = 8, seed: int = 1):
    """Train on a small public example dataset and list what came back.

    With mode "auto" (and no caption prefix) this also exercises the OpenAI
    calls. Costs GPU time: about 10 minutes for 100 steps.
    """
    task_id = datetime.now(timezone.utc).strftime("smoke-%Y%m%d-%H%M%S")
    args = {
        "name": "smoke",
        "lora_training_urls": SMOKE_TEST_IMAGES[:images],
        "mode": mode,
        "max_train_steps": steps,
        "lora_rank": rank,
        "learning_rate": 5e-05,
        "seed": seed,
    }
    if mode != "auto":
        args["caption_prefix"] = "a photo of TOK dog"
    result = train.remote({"task_id": task_id, "args": args})
    print(json.dumps(result, indent=2))
    print(f"\nFiles on the Volume flux-trainer-jobs-{ENV} under {task_id}/:")
    for entry in jobs_volume.listdir(task_id, recursive=True):
        print(f"  {entry.path}  {entry.size}")
