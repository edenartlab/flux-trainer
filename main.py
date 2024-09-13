import os
import logging
import argparse
from utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def construct_train_command(config: Dict[str, Any]) -> List[str]:
    """Construct the training command."""
    root_dir = Path(__file__).resolve().parent

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
        "--cache_latents_to_disk",
        "--save_model_as", "safetensors",
        "--sdpa",
        "--persistent_data_loader_workers",
        "--max_data_loader_n_workers", "2",
        "--seed", config['seed'],
        "--gradient_checkpointing",
        "--mixed_precision", "bf16",
        "--save_precision", "bf16",
        "--network_module", "networks.lora_flux",
        "--network_dim", config['lora_rank'],
        "--optimizer_type", "adamw8bit",
        "--learning_rate", config['learning_rate'],
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--fp8_base",
        "--text_encoder_batch_size", "4",
        "--highvram",
        "--max_train_steps", config['max_train_steps'],
        "--save_every_n_steps", config['save_every_n_steps'],
        "--sample_every_n_steps", config['sample_every_n_steps'],
        "--sample_prompts", config['eval_prompts'],
        "--sample_at_first",
        "--output_dir", str(config["output_dir"]),
        "--output_name", config["output_name"],
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "3.1582",
        "--model_prediction_type", "raw",
        "--guidance_scale", "1.0",
    ]
    return cmd

def main():
    parser = argparse.ArgumentParser(description='Training script for flux network.')
    parser.add_argument('--config', type=str, required=True, help='Path to the training config file (JSON).')
    args = parser.parse_args()

    # Load provided config:
    config = construct_config(args.config)
    
    # Preprocess the dataset
    prep_dataset(config["dataset_path"], hard_prep=False)
    florence_caption_dataset(config["dataset_path"], caption_mode=config["caption_mode"])
    
    # Construct and run the training command
    cmd = construct_train_command(config)
    run_job(cmd, config)

if __name__ == "__main__":
    main()