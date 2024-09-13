import json
import subprocess
import os
from datetime import datetime
from utils import florence_caption_dataset, prep_dataset
import toml

def construct_config(config_path, toml_path, dataset_path, caption_prefix = "", caption_suffix = "", mode = "style"):
    with open(config_path, 'r') as f:
        config = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config["timestamp"] = timestamp
    return config

def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

def construct_command(config):
    script_dir = get_script_directory()
    
    output_name = f"{os.path.basename(config['DATASET_CONFIG']).rsplit('.', 1)[0]}_{config["timestamp"]}"
    output_dir = f"results/{output_name}"

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_cpu_threads_per_process", "1",
        "--num_processes", "1",
        os.path.join(script_dir, "sd-scripts", "flux_train_network.py"),
        "--dataset_config", config['DATASET_CONFIG'],
        "--pretrained_model_name_or_path", config['MODEL_PATH'],
        "--clip_l", config['CLIP_L_PATH'],
        "--t5xxl", config['T5XXL_PATH'],
        "--ae", config['AE_PATH'],
        "--cache_latents_to_disk",
        "--save_model_as", "safetensors",
        "--sdpa",
        "--persistent_data_loader_workers",
        "--max_data_loader_n_workers", "2",
        "--seed", "42",
        "--gradient_checkpointing",
        "--mixed_precision", "bf16",
        "--save_precision", "bf16",
        "--network_module", "networks.lora_flux",
        "--network_dim", "24",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", "2.0e-4",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--fp8_base",
        "--text_encoder_batch_size", "4",
        "--highvram",
        "--max_train_steps", "20000",
        "--save_every_n_steps", "250",
        "--sample_every_n_steps", "250",
        "--sample_prompts", config['EVAL_PROMPTS'],
        "--sample_at_first",
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "3.1582",
        "--model_prediction_type", "raw",
        "--guidance_scale", "1.0",
       #"--debug_dataset"
    ]

    return cmd

def run_job(cmd, config):

    log_file = f"training_log_{config["timestamp"]}.txt"

    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')  # Print to console
            f.write(line)  # Write to log file
            f.flush()  # Ensure it's written immediately

    if process.wait() != 0:
        print(f"Command failed with return code {process.returncode}")


def main():
    toml_path = "dataseet512.toml"
    dataset_path = "/data/xander/Projects/cog/xander_eden_stuff/clipxdata/clipx_dataset"

    config = construct_config('train_config.json')
    cmd = construct_command(config)
    run_job(cmd, config)

if __name__ == "__main__":
    main()