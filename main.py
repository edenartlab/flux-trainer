import json
import subprocess
import os
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

def construct_command(config):
    script_dir = get_script_directory()
    flux_train_network_path = os.path.join(script_dir, "sd-scripts", "flux_train_network.py")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_name = f"{os.path.basename(config['DATASET_CONFIG']).rsplit('.', 1)[0]}_{timestamp}"
    output_dir = f"results/{output_name}"

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_cpu_threads_per_process", "1",
        "--num_processes", "1",
        flux_train_network_path,
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
        "--learning_rate", "1.0e-4",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--fp8_base",
        "--text_encoder_batch_size", "4",
        "--highvram",
        "--max_train_epochs", "36",
        "--save_every_n_epochs", "1",
        "--sample_every_n_epochs", "1",
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

def main():
    config = load_config('train_config.json')
    cmd = construct_command(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"training_log_{timestamp}.txt"

    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')  # Print to console
            f.write(line)  # Write to log file
            f.flush()  # Ensure it's written immediately

    if process.wait() != 0:
        print(f"Command failed with return code {process.returncode}")

if __name__ == "__main__":
    main()