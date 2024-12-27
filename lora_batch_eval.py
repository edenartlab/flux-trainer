# Minimum Inference Code for FLUX
# add kohya scripts to path:

import sys
import os
import importlib.util

import argparse
import datetime
import math
import random
from typing import Callable, List, Optional
import einops
import numpy as np

import json
import torch
from tqdm import tqdm
from PIL import Image
import accelerate
from transformers import CLIPTextModel
from safetensors.torch import load_file

from typing import List, Dict, Optional
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

# add kohya sd-scripts to path:
current_file_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_file_directory, "sd-scripts"))

from minimal_inference_utils import *
import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, sd3_utils, strategy_flux
from library.utils import setup_logging, str_to_dtype
from library import device_utils
from library.device_utils import init_ipex, get_preferred_device
from networks import oft_flux

init_ipex()
setup_logging()

def encode_prompts(
    clip_l: CLIPTextModel,
    t5xxl,
    tokenize_strategy, 
    encoding_strategy,
    prompts: List[str],
    negative_prompts: Optional[List[str]] = None,
    device: torch.device = torch.device("cuda"),
) -> List[Dict[str, torch.Tensor]]:
    """Encode a list of prompts into conditioning vectors."""
    # prepare fp8 models
    if is_fp8(clip_l_dtype) and (not hasattr(clip_l, "fp8_prepared") or not clip_l.fp8_prepared):
        logger.info(f"prepare CLIP-L for fp8: set to {clip_l_dtype}, set embeddings to {torch.bfloat16}")
        clip_l.to(clip_l_dtype)  # fp8
        clip_l.text_model.embeddings.to(dtype=torch.bfloat16)
        clip_l.fp8_prepared = True

    if is_fp8(t5xxl_dtype) and (not hasattr(t5xxl, "fp8_prepared") or not t5xxl.fp8_prepared):
        logger.info(f"prepare T5xxl for fp8: set to {t5xxl_dtype}")

        def prepare_fp8(text_encoder, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    hidden_gelu = module.act(module.wi_0(hidden_states))
                    hidden_linear = module.wi_1(hidden_states)
                    hidden_states = hidden_gelu * hidden_linear
                    hidden_states = module.dropout(hidden_states)
                    hidden_states = module.wo(hidden_states)
                    return hidden_states
                return forward

            for module in text_encoder.modules():
                if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                    module.to(target_dtype)
                if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                    module.forward = forward_hook(module)

        t5xxl.to(t5xxl_dtype)
        prepare_fp8(t5xxl.encoder, torch.bfloat16)
        t5xxl.fp8_prepared = True

    # Move models to device
    logger.info("Encoding prompts...")
    clip_l = clip_l.to(device)
    t5xxl = t5xxl.to(device)

    def encode_single(prpt: str):
        tokens_and_masks = tokenize_strategy.tokenize(prpt)
        with torch.no_grad():
            if is_fp8(clip_l_dtype):
                with accelerator.autocast():
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
            else:
                with torch.autocast(device_type=device.type, dtype=clip_l_dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)

            if is_fp8(t5xxl_dtype):
                with accelerator.autocast():
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=t5xxl_dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy, [None, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                    )
        return l_pooled, t5_out, txt_ids, t5_attn_mask

    conditioning_vectors = []
    for idx, prompt in enumerate(prompts):
        l_pooled, t5_out, txt_ids, t5_attn_mask = encode_single(prompt)
        
        # Handle negative prompt if provided
        if negative_prompts and idx < len(negative_prompts):
            neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask = encode_single(negative_prompts[idx])
        else:
            neg_l_pooled, neg_t5_out, neg_t5_attn_mask = None, None, None

        # NaN check
        if torch.isnan(l_pooled).any():
            raise ValueError(f"NaN in l_pooled for prompt {idx}")
        if torch.isnan(t5_out).any():
            raise ValueError(f"NaN in t5_out for prompt {idx}")

        conditioning_vectors.append({
            'l_pooled': l_pooled.cpu(),
            't5_out': t5_out.cpu(),
            'txt_ids': txt_ids.cpu(),
            't5_attn_mask': t5_attn_mask.cpu() if args.apply_t5_attn_mask else None,
            'neg_l_pooled': neg_l_pooled.cpu() if neg_l_pooled else None,
            'neg_t5_out': neg_t5_out.cpu() if neg_t5_out else None,
            'neg_t5_attn_mask': neg_t5_attn_mask.cpu() if args.apply_t5_attn_mask else None
        })

    if args.offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
    device_utils.clean_memory()

    return conditioning_vectors

def generate_images(
    model,
    ae,
    accelerator,
    conditioning_vectors: List[Dict[str, torch.Tensor]],
    seeds: Optional[List[int]],
    image_width: int,
    image_height: int,
    steps: Optional[int],
    guidance: float,
    cfg_scale: float,
    device: torch.device = torch.device("cuda"),
    is_schnell: bool = False,
) -> List[Image.Image]:
    """Generate images from conditioning vectors."""
    if seeds is None:
        seeds = [random.randint(0, 2**32 - 1) for _ in range(len(conditioning_vectors))]
    elif len(seeds) != len(conditioning_vectors):
        raise ValueError("Number of seeds must match number of conditioning vectors")

    packed_latent_height, packed_latent_width = math.ceil(image_height / 16), math.ceil(image_width / 16)
    noise_dtype = torch.float32 if is_fp8(dtype) else dtype
    
    # Move model to device once
    model = model.to(device)
    if steps is None:
        steps = 4 if is_schnell else 50

    denoised_latents = []
    generated_images = []
    for idx, (cond_vector, seed) in enumerate(zip(conditioning_vectors, seeds)):
        logger.info(f"Generating image {idx + 1}/{len(conditioning_vectors)} with seed {seed}")
        
        # Generate noise for this image
        noise = torch.randn(
            1,
            packed_latent_height * packed_latent_width,
            16 * 2 * 2,
            device=device,
            dtype=noise_dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        
        img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(device)

        # Generate the image
        x = do_sample(
            accelerator,
            model,
            noise,
            img_ids,
            cond_vector['l_pooled'].to(device),
            cond_vector['t5_out'].to(device),
            cond_vector['txt_ids'].to(device),
            steps,
            guidance,
            cond_vector['t5_attn_mask'],
            is_schnell,
            device,
            flux_dtype,
            cond_vector['neg_l_pooled'],
            cond_vector['neg_t5_out'],
            cond_vector['neg_t5_attn_mask'],
            cfg_scale,
        )

        # Unpack
        x = x.float()
        x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
        denoised_latents.append(x)

    if args.offload:
        model = model.cpu()
    device_utils.clean_memory()

    # Decode
    print(f"Decoding {len(denoised_latents)} images...")
    ae = ae.to(device)

    for x in denoised_latents:
        with torch.no_grad():
            if is_fp8(ae_dtype):
                with accelerator.autocast():
                    x = ae.decode(x)
            else:
                with torch.autocast(device_type=device.type, dtype=ae_dtype):
                    x = ae.decode(x)

        x = x.clamp(-1, 1)
        x = x.permute(0, 2, 3, 1)
        img = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])
        generated_images.append(img)
    
    if args.offload:
        ae = ae.cpu()

    return generated_images


def load_flux(args, device):
    logger.info("Loading FLUX from disk...")
    loading_device = "cpu" if args.offload else device
    
    # load clip_l
    logger.info(f"Loading clip_l from {args.clip_l}...")
    clip_l = flux_utils.load_clip_l(args.clip_l, clip_l_dtype, loading_device)
    clip_l.eval()

    logger.info(f"Loading t5xxl from {args.t5xxl}...")
    t5xxl = flux_utils.load_t5xxl(args.t5xxl, t5xxl_dtype, loading_device)
    t5xxl.eval()

    # DiT
    is_schnell, model = flux_utils.load_flow_model(args.ckpt_path, None, loading_device)
    model.eval()
    logger.info(f"Casting model to {flux_dtype}")
    model.to(flux_dtype)  # make sure model is dtype

    t5xxl_max_length = 256 if is_schnell else 512
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

    # AE
    ae = flux_utils.load_ae(args.ae, ae_dtype, loading_device)
    ae.eval()

    return clip_l, t5xxl, model, ae, is_schnell, tokenize_strategy, encoding_strategy

def load_lora(lora_filepaths, config_filepath, args, clip_l, t5xxl, model, ae, device, lora_scale):
    print("Loading LoRA weights into Flux...")
    lora_models: List[lora_flux.LoRANetwork] = []

    for weights_file in lora_filepaths:
        weights_sd = load_file(weights_file)
        is_lora = is_oft = False
        for key in weights_sd.keys():
            if key.startswith("lora"):
                is_lora = True
            if key.startswith("oft"):
                is_oft = True
            if is_lora or is_oft:
                break

        module = lora_flux if is_lora else oft_flux
        lora_model, _ = module.create_network_from_weights(lora_scale, None, ae, [clip_l, t5xxl], model, weights_sd, True)

        if args.merge_lora_weights:
            print("Merging LoRA weights into Flux model...")
            lora_model.merge_to([clip_l, t5xxl], model, weights_sd)
        else:
            lora_model.apply_to([clip_l, t5xxl], model)
            info = lora_model.load_state_dict(weights_sd, strict=True)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.eval()
            lora_model.to(device)

        lora_models.append(lora_model)

    with open(config_filepath, 'r', encoding='utf-8') as f:
        training_config_data = json.load(f)
        trigger_words = training_config_data["lora_trigger_text"]

    return lora_models, trigger_words

import re
def prep_prompt(prompt, trigger_words, verbose = True):
    # replace "TOK" with the trigger words from the LoRA:
    prompt = prompt.replace("TOK", trigger_words)
    
    # Pattern matches any combination of --w, --h, --s, --d followed by numbers
    # at the end of the string
    pattern = r'\s*(?:--[whsd]\s+\d+\s*)+$'
    prompt = re.sub(pattern, '', prompt).strip()

    print(f"Final prompt: {prompt}")

    return prompt.strip()

def evaluate_lora_checkpoint(lora_filepath, config_filepath, args, device, lora_scale):
    use_fp8 = [is_fp8(d) for d in [dtype, clip_l_dtype, t5xxl_dtype, ae_dtype, flux_dtype]]
    if any(use_fp8):
        accelerator = accelerate.Accelerator(mixed_precision="bf16")
    else:
        accelerator = None

    clip_l, t5xxl, model, ae, is_schnell, tokenize_strategy, encoding_strategy = load_flux(args, device)
    lora_model, trigger_words = load_lora([lora_filepath], config_filepath, args, clip_l, t5xxl, model, ae, device, lora_scale)

    prompts = []
    if not Path(args.prompt_file).exists():
        raise ValueError(f"Prompt file not found: {args.prompt_file}")
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    prompts = [prep_prompt(p, trigger_words) for p in prompts]

    logger.info(f"Encoding {len(prompts)} prompts...")
    conditioning_vectors = encode_prompts(
        clip_l,
        t5xxl,
        tokenize_strategy, 
        encoding_strategy,
        prompts=prompts
    )

    images = generate_images(
        model,
        ae,
        accelerator,
        conditioning_vectors,
        seeds=list(range(len(prompts))),
        image_width=args.width,
        image_height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        cfg_scale=args.cfg_scale,
        device=device,
        is_schnell=is_schnell
    )

    # Save images
    print(f"Saving {len(images)} images to {args.output_dir}...")
    n_megapixels = args.width * args.height / 1e6

    for idx, img in enumerate(images):
        output_path = os.path.join(args.output_dir, f"{idx}_{os.path.basename(lora_filepath)}_{n_megapixels:.1f}MP.jpg")
        img.save(output_path)


if __name__ == "__main__":
    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True, 
                        help="Path to either a single LoRA safetensors file or a directory containing LoRA weights")
    parser.add_argument("--clip_l", type=str, required=False)
    parser.add_argument("--t5xxl", type=str, required=False)
    parser.add_argument("--ae", type=str, required=False)
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--prompt_file", type=str, help="Path to text file containing prompts, one per line")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", help="base dtype")
    parser.add_argument("--clip_l_dtype", type=str, default=None, help="dtype for clip_l")
    parser.add_argument("--ae_dtype", type=str, default=None, help="dtype for ae")
    parser.add_argument("--t5xxl_dtype", type=str, default=None, help="dtype for t5xxl")
    parser.add_argument("--flux_dtype", type=str, default=None, help="dtype for flux")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=25, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--lora_scale", type=float, default=0.9)
    args = parser.parse_args()

    if not args.output_dir:
        base_name = os.path.basename(os.path.dirname(args.lora_path) if os.path.isfile(args.lora_path) else args.lora_path)
        args.output_dir = f"evals/{base_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = str_to_dtype(args.dtype)
    clip_l_dtype = str_to_dtype(args.clip_l_dtype, dtype)
    t5xxl_dtype = str_to_dtype(args.t5xxl_dtype, dtype)
    ae_dtype = str_to_dtype(args.ae_dtype, dtype)
    flux_dtype = str_to_dtype(args.flux_dtype, dtype)
    logger.info(f"Dtypes for clip_l, t5xxl, ae, flux: {clip_l_dtype}, {t5xxl_dtype}, {ae_dtype}, {flux_dtype}")

    if os.path.isfile(args.lora_path):
        # Single LoRA evaluation
        config_filepath = os.path.join(os.path.dirname(args.lora_path), "config.json")
        evaluate_lora_checkpoint(args.lora_path, config_filepath, args, device, args.lora_scale)
    else:
        # Directory evaluation - first check if it's a direct directory with safetensors
        lora_files = [f for f in os.listdir(args.lora_path) if f.endswith('.safetensors') and '-step000' in f]
        
        if lora_files:
            # Direct directory with LoRA files
            config_filepath = os.path.join(args.lora_path, "config.json")
            for file in lora_files:
                lora_filepath = os.path.join(args.lora_path, file)
                print(f"Running on {lora_filepath}")
                evaluate_lora_checkpoint(lora_filepath, config_filepath, args, device, args.lora_scale)
        else:
            # Root directory with subfolders
            subfolders = sorted(os.listdir(args.lora_path))
            
            for i, subfolder in enumerate(subfolders):
                print("-----------------------------------------")
                print(f"Evaluating LoRAs from {subfolder} ({i+1} of {len(subfolders)})")
                print("-----------------------------------------")
                training_run_folder = os.path.join(args.lora_path, subfolder)
                lora_filepaths = sorted([
                    os.path.join(training_run_folder, f) 
                    for f in os.listdir(training_run_folder) 
                    if f.endswith(".safetensors") and "-step000" in f
                ])
                
                config_filepath = os.path.join(training_run_folder, "config.json")

                for lora_filepath in lora_filepaths:
                    print(f"Running on {lora_filepath}")
                    evaluate_lora_checkpoint(lora_filepath, config_filepath, args, device, args.lora_scale)