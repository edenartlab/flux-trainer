
import argparse
import datetime
import math
import os
import random
from typing import Callable, List, Optional
import einops
import numpy as np

import torch
from tqdm import tqdm
from PIL import Image
import accelerate
from transformers import CLIPTextModel
from safetensors.torch import load_file

from library import device_utils
from library.device_utils import init_ipex, get_preferred_device
from networks import oft_flux

init_ipex()

from library.utils import str_to_dtype
import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, sd3_utils, strategy_flux

from library.utils import setup_logging, str_to_dtype

def is_fp8(dt):
    return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]

def print_gpu_memory(info_str = ""):
    """Print the GPU memory usage in GB"""
    # Get the current GPU device
    device = torch.cuda.current_device()
    
    # Get memory stats
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert bytes to GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3    # Convert bytes to GB
    
    print(info_str)
    print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
    print(f"GPU Memory Reserved: {memory_reserved:.2f} GB")


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    neg_txt: Optional[torch.Tensor] = None,
    neg_vec: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # prepare classifier free guidance
    if neg_txt is not None and neg_vec is not None:
        b_img_ids = torch.cat([img_ids, img_ids], dim=0)
        b_txt_ids = torch.cat([txt_ids, txt_ids], dim=0)
        b_txt = torch.cat([neg_txt, txt], dim=0)
        b_vec = torch.cat([neg_vec, vec], dim=0)
        if t5_attn_mask is not None and neg_t5_attn_mask is not None:
            b_t5_attn_mask = torch.cat([neg_t5_attn_mask, t5_attn_mask], dim=0)
        else:
            b_t5_attn_mask = None
    else:
        b_img_ids = img_ids
        b_txt_ids = txt_ids
        b_txt = txt
        b_vec = vec
        b_t5_attn_mask = t5_attn_mask

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((b_img_ids.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # classifier free guidance
        if neg_txt is not None and neg_vec is not None:
            b_img = torch.cat([img, img], dim=0)
        else:
            b_img = img

        pred = model(
            img=b_img,
            img_ids=b_img_ids,
            txt=b_txt,
            txt_ids=b_txt_ids,
            y=b_vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            txt_attention_mask=b_t5_attn_mask,
        )

        # classifier free guidance
        if neg_txt is not None and neg_vec is not None:
            pred_uncond, pred = torch.chunk(pred, 2, dim=0)
            pred = pred_uncond + cfg_scale * (pred - pred_uncond)

        img = img + (t_prev - t_curr) * pred

    return img


def do_sample(
    accelerator: Optional[accelerate.Accelerator],
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    l_pooled: torch.Tensor,
    t5_out: torch.Tensor,
    txt_ids: torch.Tensor,
    num_steps: int,
    guidance: float,
    t5_attn_mask: Optional[torch.Tensor],
    is_schnell: bool,
    device: torch.device,
    flux_dtype: torch.dtype,
    neg_l_pooled: Optional[torch.Tensor] = None,
    neg_t5_out: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
):
    timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

    if isinstance(device, str):
        device = torch.device(device)

    # denoise initial noise
    if accelerator:
        with accelerator.autocast(), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
            )
    else:
        with torch.autocast(device_type=device.type, dtype=flux_dtype), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
            )

    return x


    timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

    if isinstance(device, str):
        device = torch.device(device)

    # denoise initial noise
    if accelerator:
        with accelerator.autocast(), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
            )
    else:
        with torch.autocast(device_type=device.type, dtype=flux_dtype), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
            )

    return x