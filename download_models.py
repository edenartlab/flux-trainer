
import torch, os
from dotenv import load_dotenv

#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@torch.no_grad()
def download_florence(models_dir):
    print("Downloading florence2...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", attn_implementation="sdpa", device_map=device, torch_dtype=torch_dtype, trust_remote_code=True, cache_dir=models_dir)
            
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True, cache_dir=models_dir)


def download_flux(models_dir):
    # Load .env file
    load_dotenv()

    # Get the Hugging Face token
    hf_token = os.getenv('HF_TOKEN')

    # Ensure that the token is set
    if not hf_token:
        raise ValueError("Hugging Face token not found. Make sure it's set in the .env file or set as an environment variable.")
    else:
        print("Hugging Face token successfully loaded!")

    # Log in to Hugging Face CLI using the token
    os.system(f'huggingface-cli login --token {hf_token}')

    # List of models and paths to download
    models_to_download = [
        ('black-forest-labs/FLUX.1-dev', 'ae.safetensors'),
        ('black-forest-labs/FLUX.1-dev', 'flux1-dev.safetensors'),
        ('comfyanonymous/flux_text_encoders', 'clip_l.safetensors'),
        ('comfyanonymous/flux_text_encoders', 't5xxl_fp16.safetensors')
    ]

    # Download each model
    print("Downloading flux...")
    for repo, filename in models_to_download:
        os.system(f'huggingface-cli download {repo} {filename} --repo-type model --local-dir {models_dir}')


if __name__ == "__main__":
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    download_florence(models_dir)
    download_flux(models_dir)
