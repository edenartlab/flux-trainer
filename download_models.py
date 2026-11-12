"""Download every model the trainer needs into ./models.

On Modal this runs once, into the `flux-trainer-models` Volume (see
modal_app.py: `modal run modal_app.py::download_models`). It needs no access
token: every file is public. Each repository is pinned to the revision that was
current when the last production image was built (March 2025), and the large
weight files are checked against their SHA-256.

The layout is the one the trainer has always used:

    models/ae.safetensors
    models/clip_l.safetensors
    models/t5xxl_fp16.safetensors
    models/flux-dev-de-distill-diffusers/transformer/diffusion_pytorch_model-0000N-of-00003.safetensors
    models/openai_clip-vit-large-patch14/        tokenizer, as sd-scripts caches it
    models/google_t5-v1_1-xxl/                   tokenizer, as sd-scripts caches it
    models/models--CIDAS--clipseg-rd64-refined/  Hugging Face cache layout
    models/models--microsoft--Florence-2-large/  Hugging Face cache layout
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download

# (repository, revision, file in the repository, path under models/, SHA-256)
WEIGHT_FILES = [
    # The FLUX autoencoder. The original copy in black-forest-labs/FLUX.1-dev
    # needs an access token; this is the same file, byte for byte (the Git LFS
    # pointer of the original, which is public, names the same SHA-256).
    (
        "Comfy-Org/Lumina-Image-2",
        "5b072540ef86570fecb8249c505f23d5bdeb88cd",
        "split_files/vae/ae.safetensors",
        "ae.safetensors",
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
    ),
    (
        "comfyanonymous/flux_text_encoders",
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5",
        "clip_l.safetensors",
        "clip_l.safetensors",
        "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
    ),
    (
        "comfyanonymous/flux_text_encoders",
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5",
        "t5xxl_fp16.safetensors",
        "t5xxl_fp16.safetensors",
        "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",
    ),
    # The de-distilled FLUX.1-dev transformer that training starts from.
    (
        "InstantX/flux-dev-de-distill-diffusers",
        "9029417540765d379d3cce4abb1f388121907398",
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "flux-dev-de-distill-diffusers/transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "8a99f3d0262748309658a6fa5371e5a7557f888367a32a2dd4d5616475c85fd8",
    ),
    (
        "InstantX/flux-dev-de-distill-diffusers",
        "9029417540765d379d3cce4abb1f388121907398",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
        "flux-dev-de-distill-diffusers/transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "eb9948ec50b4fed2fa4d651101bc024502d74dfa1028a75778eb07a6992ebcfe",
    ),
    (
        "InstantX/flux-dev-de-distill-diffusers",
        "9029417540765d379d3cce4abb1f388121907398",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
        "flux-dev-de-distill-diffusers/transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "2d1e825e357857d6621ffcda32b56ee911d2bbfa2869af6b36afacd45eae69fa",
    ),
]

# Tokenizers, saved where sd-scripts looks for them when it is given
# --tokenizer_cache_dir (<dir>/<model id with "/" replaced by "_">).
TOKENIZERS = [
    ("CLIPTokenizer", "openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
    ("T5TokenizerFast", "google/t5-v1_1-xxl", "3db67ab1af984cf10548a73467f0e5bca2aaaeb2"),
]

# Models loaded with from_pretrained(..., cache_dir="./models").
CACHED_REPOS = [
    # CLIPSeg, for the foreground masks (face and object modes).
    ("CIDAS/clipseg-rd64-refined", "999e0328d9e10b484360c477313983f9afdd7050"),
    # Florence-2, only used when caption_mode asks for it. Pinned to the last
    # revision before March 2025; later revisions changed its remote code.
    ("microsoft/Florence-2-large", "00d2f1570b00c6dea5df998f5635db96840436bc"),
]

MANIFEST = "manifest.json"


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(models_dir):
    files = {}
    for repo, revision, filename, target, expected in WEIGHT_FILES:
        target_path = os.path.join(models_dir, target)
        if not (os.path.isfile(target_path) and sha256_of(target_path) == expected):
            print(f"Downloading {repo}/{filename}...", flush=True)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with tempfile.TemporaryDirectory(dir=models_dir) as tmp_dir:
                downloaded = hf_hub_download(
                    repo, filename, revision=revision, local_dir=tmp_dir
                )
                actual = sha256_of(downloaded)
                if actual != expected:
                    raise RuntimeError(
                        f"{repo}/{filename} has SHA-256 {actual}, expected {expected}"
                    )
                shutil.move(downloaded, target_path)
        files[target] = {
            "repo": repo,
            "revision": revision,
            "sha256": expected,
            "size": os.path.getsize(target_path),
        }
        print(f"OK {target} {expected}", flush=True)
    return files


def download_tokenizers(models_dir):
    import transformers

    saved = {}
    for class_name, model_id, revision in TOKENIZERS:
        path = os.path.join(models_dir, model_id.replace("/", "_"))
        if not os.path.exists(path):
            print(f"Downloading tokenizer {model_id}...", flush=True)
            tokenizer = getattr(transformers, class_name).from_pretrained(
                model_id, revision=revision
            )
            tokenizer.save_pretrained(path)
        saved[os.path.basename(path)] = {"repo": model_id, "revision": revision}
    return saved


def download_cached_repos(models_dir):
    saved = {}
    for repo, revision in CACHED_REPOS:
        print(f"Downloading {repo}...", flush=True)
        snapshot_download(repo, revision=revision, cache_dir=models_dir)
        # from_pretrained() without a revision resolves "main" through this
        # file; point it at the pinned revision so nothing newer is ever used.
        repo_dir = os.path.join(models_dir, "models--" + repo.replace("/", "--"))
        os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
        with open(os.path.join(repo_dir, "refs", "main"), "w") as f:
            f.write(revision)
        saved[os.path.basename(repo_dir)] = {"repo": repo, "revision": revision}
    return saved


def main(models_dir="./models"):
    os.makedirs(models_dir, exist_ok=True)
    manifest = {
        "weights": download_weights(models_dir),
        "tokenizers": download_tokenizers(models_dir),
        "cached_repos": download_cached_repos(models_dir),
    }
    # Written last: its presence means the directory is complete.
    with open(os.path.join(models_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:2])
