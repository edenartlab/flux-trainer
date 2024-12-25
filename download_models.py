import torch
import os
import subprocess
from dotenv import load_dotenv

#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

from utils import clipseg_mask_generator
from PIL import Image

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

def download_flux(models_dir):
    # Load .env file
    load_dotenv()

    # Get the Hugging Face token
    hf_token = os.getenv('HF_TOKEN')

    # Ensure that the token is set
    if not hf_token:
        raise ValueError("Hugging Face token not found. Make sure it's set in the .env file or set as an environment variable.")
    else:
        print("Hugging Face token successfully loaded!", flush=True)

    # Log in to Hugging Face CLI using the token
    if not run_command(f'huggingface-cli login --token {hf_token}'):
        raise RuntimeError("Failed to login to Hugging Face")

    # List of models and paths to download
    models_to_download = [
        ('black-forest-labs/FLUX.1-dev', 'ae.safetensors'),
        ('black-forest-labs/FLUX.1-dev', 'flux1-dev.safetensors'),
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
        f'huggingface-cli download InstantX/flux-dev-de-distill-diffusers --repo-type model --local-dir "{flux_distill_dir}" --include *.safetensors'
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