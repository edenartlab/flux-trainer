import logging
import sys
import argparse
from pathlib import Path
import time, json
from utils import construct_config, prep_dataset, run_job
from typing import Optional, List, Tuple, Union, Literal, Dict, Any
from download_dataset import *

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout
)

def construct_train_command(config: Dict[str, Any]) -> List[str]:
    """Construct the training command."""
    root_dir = Path(__file__).resolve().parent

    if config['full_finetune']:

        if float(config['learning_rate']) > 1.0e-4:
            print("WARNING: learning rate is higher than 1.0e-4, this is not recommended for full finetuning.")
            sys.exit(1)

        cmd = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_cpu_threads_per_process", "1",
            "--num_processes", "1",  # run on 1 gpu, remove this line for multi-gpu training
            str(root_dir / "sd-scripts" / "flux_train.py"),
            "--dataset_config", config['dataset_config'],
            "--pretrained_model_name_or_path", config['MODEL_PATH'],
            "--clip_l", config['CLIP_L_PATH'],
            "--t5xxl", config['T5XXL_PATH'],
            "--ae", config['AE_PATH'],
            "--cache_latents_to_disk",
            "--save_model_as", "safetensors",
            "--sdpa",
            #"--loss_type", "huber",
            *(['--alpha_mask'] if config['alpha_mask'] else []),
            "--persistent_data_loader_workers",
            "--max_data_loader_n_workers", "2",
            "--seed", config['seed'],
            "--gradient_checkpointing",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--optimizer_type", "adafactor",
            "--optimizer_args", "relative_step=False", "scale_parameter=False", "warmup_init=False",
            "--fused_backward_pass",  
            "--blocks_to_swap", "8", 
            "--full_bf16", 
            "--learning_rate", config['learning_rate'],
            "--noise_offset", config['noise_offset'],
            "--noise_offset_random_strength",
            "--ip_noise_gamma", config['ip_noise_gamma'],
            "--ip_noise_gamma_random_strength",
            #"--lr_scheduler", "cosine",
            "--lr_scheduler", "constant_with_warmup",
            "--lr_warmup_steps", "0.05",
            #"--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--max_grad_norm", "0.0", 
            "--text_encoder_batch_size", "4",
            "--highvram",
            "--max_train_steps", config['max_train_steps'],
            "--save_every_n_steps", config['save_every_n_steps'],
            "--sample_every_n_steps", config['sample_every_n_steps'],
            "--sample_prompts", config['eval_prompts'],
            *(['--sample_at_first'] if config['sample_at_first'] else []),
            "--output_dir", str(config["output_dir"]),
            "--output_name", config["output_name"],
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", "3.1582",
            "--model_prediction_type", "raw",
            "--guidance_scale", "1.0"
        ]
    else: # LoRA
        cmd = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_cpu_threads_per_process", "1",
            "--num_processes", "1",  # run on 1 gpu, remove this line for multi-gpu training
            str(root_dir / "sd-scripts" / "flux_train_network.py"),
            "--dataset_config", config['dataset_config'],
            "--pretrained_model_name_or_path", config['MODEL_PATH'],
            "--clip_l", config['CLIP_L_PATH'],
            "--t5xxl", config['T5XXL_PATH'],
            "--ae", config['AE_PATH'],
            "--save_model_as", "safetensors",
            "--sdpa",
            #"--loss_type", "huber",
            *(['--alpha_mask'] if config['alpha_mask'] else []),
            "--persistent_data_loader_workers",
            "--max_data_loader_n_workers", "2",
            "--seed", config['seed'],
            "--gradient_checkpointing",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--fp8_base",
            "--gradient_accumulation_steps", config["gradient_accumulation_steps"],
            "--network_train_unet_only",
            "--network_module", "networks.lora_flux",
            "--network_dim", config['lora_rank'],
            "--optimizer_type", "adamw8bit",
            "--optimizer_args", "weight_decay=0.1", 
            "--learning_rate", config['learning_rate'],
            "--noise_offset", config['noise_offset'],
            "--noise_offset_random_strength",
            "--ip_noise_gamma", config['ip_noise_gamma'],
            "--ip_noise_gamma_random_strength",
            "--lr_scheduler", "cosine_with_restarts",
            "--lr_scheduler_num_cycles", "3",
            #"--lr_scheduler", "constant_with_warmup",
            #"--lr_warmup_steps", "0.05",
            "--network_alpha", config["lora_rank"],
            "--min_snr_gamma", "5",
            "--cache_latents_to_disk",
            "--cache_text_encoder_outputs_to_disk",
            "--max_grad_norm", "0.0", 
            "--text_encoder_batch_size", "4",
            "--highvram",
            "--max_train_steps", config['max_train_steps'],
            "--save_every_n_steps", config['save_every_n_steps'],
            "--sample_every_n_steps", config['sample_every_n_steps'],
            "--sample_prompts", config['eval_prompts'],
            *(['--sample_at_first'] if config['sample_at_first'] else []),
            "--output_dir", str(config["output_dir"]),
            "--output_name", config["output_name"],
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", "3.1582",
            "--model_prediction_type", "raw",
            "--guidance_scale", config["guidance_scale"]
        ]

    return cmd

"""

TODO:


later:
- add automatic dataset augmentation on start
- test out better segmentation models for masking: https://github.com/microsoft/X-Decoder or https://github.com/IDEA-Research/OpenSeeD 
- caption with GPT4V and re-inject trigger token
- style token trigger: "in the style of TOK" / "TOK"
- https://github.com/kohya-ss/sd-scripts/issues/1702#issuecomment-2556661896
- test guidance_scale > 1.0 with captions!!
- sweep --min_snr_gamma values


OPTIONAL ARGS:
"mode"
"masking_prompt"
"caption_prefix"

args to test out from sd-scripts/library/train_util.py:

--min_snr_gamma 5
loss_type
gradient_accumulation_steps
color_aug                                                                                               
face_crop_aug_range: None                                                                                                    
random_crop: False   
face_crop_aug_range
zero_terminal_snr
adaptive_noise_scale
max_grad_norm

"""

def run_trainer(args, verbose = True):
    start_time = time.time()

    # Step 1: Load the training config from the provided file
    config = construct_config(args.config)

    # Step 2: Download the dataset from the URL provided
    if args.dataset_url:
        download_dataset(config["dataset_path"], [args.dataset_url])

    # Step 3: Prepare dataset:
    config = prep_dataset(config)

    # Step 4: Construct and run the training command
    cmd = construct_train_command(config)

    if verbose:
        print(" ========= Final Train Config: ==========")
        print(config)

    run_job(cmd, config)

    # add total runtime to the config.json in config['output_dir’]:
    try:
        config['total_runtime'] = time.time() - start_time
        with open(config['output_dir'] / 'config.json', 'w') as f:
            json.dump(config, dict(config), indent=4)
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='Training script for flux network.')
   
    # Add arguments for dataset URL and config file
    parser.add_argument('--dataset_url', default=None, help="URL of the dataset to download and train on")
    parser.add_argument('--config', type=str, default="template/train_config.json", help='Path to the training config file (JSON).')

    # Parse the arguments
    args = parser.parse_args()
    run_trainer(args)

if __name__ == "__main__":
    main()

