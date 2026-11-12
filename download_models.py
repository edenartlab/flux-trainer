import torch
import os
import hashlib
import shutil
import subprocess
import tempfile

from eden_utils import clipseg_mask_generator
from PIL import Image

#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    try:
        imports.remove("flash_attn")
    except:
        pass
    return imports

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@torch.no_grad()
def download_florence(models_dir):
    print("Downloading florence2...", flush=True)
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", attn_implementation="sdpa", device_map=device, torch_dtype=torch_dtype, trust_remote_code=True, cache_dir=models_dir)
            
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True, cache_dir=models_dir)

@torch.no_grad()
def download_clipseg(models_dir):
    print("Downloading clipseg...", flush=True)
    # make a random dummy pil image:
    dummy_pil_image_list = [Image.new('RGB', (512, 512))]
    dummy_mask = clipseg_mask_generator(dummy_pil_image_list, ["test"])

def run_command(command):
    """Execute a command and return its output and error status"""
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"Command failed with error: {process.stderr}")
        return False
    return True

# The FLUX autoencoder. The original copy in black-forest-labs/FLUX.1-dev needs
# an access token; the same file (checked by its SHA-256) is published without
# one, so no token is used at build time and none can end up in the image.
AE_REPO = "Comfy-Org/Lumina_Image_2.0_Repackaged"
AE_FILE = "split_files/vae/ae.safetensors"
AE_SHA256 = "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_autoencoder(models_dir):
    from huggingface_hub import hf_hub_download

    print("Downloading the FLUX autoencoder...", flush=True)
    target = os.path.join(models_dir, "ae.safetensors")
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded = hf_hub_download(AE_REPO, AE_FILE, local_dir=tmp_dir)
        actual = sha256_of(downloaded)
        if actual != AE_SHA256:
            raise RuntimeError(f"ae.safetensors has SHA-256 {actual}, expected {AE_SHA256}")
        shutil.copyfile(downloaded, target)

def download_tokenizers(models_dir):
    """
    Save the two tokenizers where sd-scripts looks for them when it is given
    --tokenizer_cache_dir (<dir>/<model id with "/" replaced by "_">).
    """
    from transformers import CLIPTokenizer, T5TokenizerFast

    print("Downloading tokenizers...", flush=True)
    for tokenizer_class, model_id in (
        (CLIPTokenizer, "openai/clip-vit-large-patch14"),
        (T5TokenizerFast, "google/t5-v1_1-xxl"),
    ):
        path = os.path.join(models_dir, model_id.replace("/", "_"))
        if not os.path.exists(path):
            tokenizer_class.from_pretrained(model_id).save_pretrained(path)

def download_flux(models_dir):
    download_autoencoder(models_dir)

    # List of models and paths to download
    models_to_download = [
        #('black-forest-labs/FLUX.1-dev', 'flux1-dev.safetensors'),
        ('comfyanonymous/flux_text_encoders', 'clip_l.safetensors'),
        ('comfyanonymous/flux_text_encoders', 't5xxl_fp16.safetensors')
    ]

    # Download each model
    print("Downloading flux...", flush=True)
    for repo, filename in models_to_download:
        if not run_command(f'huggingface-cli download {repo} {filename} --repo-type model --local-dir "{models_dir}"'):
            raise RuntimeError(f"Failed to download {filename} from {repo}")

    # Download the entire flux-dev-de-distill-diffusers repository
    flux_distill_dir = os.path.join(models_dir, "flux-dev-de-distill-diffusers/transformer")
    os.makedirs(flux_distill_dir, exist_ok=True)
    print("Downloading flux-dev-de-distill-diffusers repository...", flush=True)
    
    # Try different approaches for downloading the safetensors files
    commands = [
        f'huggingface-cli download InstantX/flux-dev-de-distill-diffusers --repo-type model --local-dir "{flux_distill_dir}" --include "*.safetensors"'
    ]
    
    success = False
    for command in commands:
        print(f"Trying command: {command}")
        if run_command(command):
            success = True
            break
    
    if not success:
        raise RuntimeError("Failed to download flux-dev-de-distill-diffusers files")
    
    print(f"Successfully downloaded .safetensors files to {flux_distill_dir}!", flush=True)

if __name__ == "__main__":
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    download_clipseg(models_dir)
    download_florence(models_dir)
    download_flux(models_dir)
    download_tokenizers(models_dir)
