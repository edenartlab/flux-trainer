"""File-based hand-off between the trainer and whatever launched it.

A job started with ``--job_dir`` gets everything it needs from, and returns
everything it produces to, one directory:

    <job_dir>/args.json      input:  {"task_id": "...", "args": {...}}
    <job_dir>/result.json    output: {"status": "completed" | "failed", ...}
    <job_dir>/output/...     output: the LoRA, checkpoint versions, thumbnail

The trainer therefore needs no database or storage credentials. On Vertex AI
the directory is given as ``gs://bucket/path``; Vertex mounts every bucket the
job's service account can use at ``/gcs/<bucket>``, so plain file operations
are enough and no Cloud Storage client is needed.

This module only uses the standard library so that it can be tested without the
training dependencies.
"""

import glob
import json
import os
import shutil
from typing import Any, Dict, Optional

ARGS_FILE = "args.json"
RESULT_FILE = "result.json"
OUTPUT_DIR = "output"

# Number of intermediate checkpoints returned next to the final LoRA.
MAX_CHECKPOINT_VERSIONS = 3


def local_path(job_dir: str) -> str:
    """Translate a job directory to a path on the local file system."""
    if not job_dir.startswith("gs://"):
        return job_dir

    bucket, _, prefix = job_dir[len("gs://"):].partition("/")
    mount_root = os.environ.get("GCS_MOUNT_ROOT", "/gcs")
    bucket_path = os.path.join(mount_root, bucket)
    if not bucket or not os.path.isdir(bucket_path):
        raise RuntimeError(
            f"Bucket '{bucket}' is not mounted at {bucket_path}. On Vertex AI this "
            "means the job's service account cannot use the bucket."
        )
    return os.path.join(bucket_path, prefix.strip("/"))


def read_args(job_path: str) -> Dict[str, Any]:
    """Return the contents of args.json: {"task_id": ..., "args": {...}}."""
    with open(os.path.join(job_path, ARGS_FILE), "r") as f:
        job = json.load(f)
    if not isinstance(job.get("args"), dict):
        raise ValueError(f"{ARGS_FILE} must contain an 'args' object")
    return job


def write_result(job_path: str, result: Dict[str, Any]) -> str:
    """Write result.json. It is written last, so its presence means the output
    files it names are complete."""
    os.makedirs(job_path, exist_ok=True)
    path = os.path.join(job_path, RESULT_FILE)
    tmp_path = os.path.join(job_path, f".{RESULT_FILE}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp_path, path)
    return path


def checkpoint_step(path: str) -> int:
    """Step number of a file named <output_name>-step<digits>.safetensors."""
    filename = os.path.basename(path)
    return int(filename.split("-step")[1].split(".")[0])


def export_outputs(
    job_path: str,
    config: Dict[str, Any],
    thumbnail_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Copy the training outputs into <job_dir>/output and describe them.

    The returned dictionary is the body of result.json. File names in it are
    relative to the job directory.
    """
    output_dir = str(config["output_dir"])
    output_name = config["output_name"]
    target_dir = os.path.join(job_path, OUTPUT_DIR)
    os.makedirs(target_dir, exist_ok=True)

    def export(source: str, name: Optional[str] = None) -> str:
        name = name or os.path.basename(source)
        shutil.copyfile(source, os.path.join(target_dir, name))
        return f"{OUTPUT_DIR}/{name}"

    lora_path = os.path.join(output_dir, f"{output_name}.safetensors")
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"Training produced no LoRA at {lora_path}")

    result: Dict[str, Any] = {
        "status": "completed",
        "lora": export(lora_path),
        "checkpoint_versions": [],
        "lora_trigger_text": config.get("lora_trigger_text"),
        "lora_mode": config.get("mode"),
    }

    versions = glob.glob(os.path.join(output_dir, f"{output_name}-step*.safetensors"))
    versions.sort(key=checkpoint_step, reverse=True)
    for path in versions[:MAX_CHECKPOINT_VERSIONS]:
        result["checkpoint_versions"].append(
            {"step": checkpoint_step(path), "file": export(path)}
        )

    if thumbnail_path and os.path.isfile(thumbnail_path):
        extension = os.path.splitext(thumbnail_path)[1] or ".jpg"
        result["thumbnail"] = export(thumbnail_path, f"thumbnail{extension}")

    prompts = config.get("custom_validation_prompts")
    if isinstance(prompts, list) and prompts:
        result["custom_validation_prompts"] = prompts

    result["config"] = config
    return result
