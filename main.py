import json
import subprocess
import os
from datetime import datetime
from utils import florence_caption_dataset, prep_dataset
import toml

def update_image_dir_in_dict(data, new_image_dir):
    """
    Recursively update every instance of 'image_dir' in the given data structure.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "image_dir":
                data[key] = new_image_dir
            elif isinstance(value, (dict, list)):
                update_image_dir_in_dict(value, new_image_dir)
    elif isinstance(data, list):
        for item in data:
            update_image_dir_in_dict(item, new_image_dir)

def construct_toml(config):
    # Load the existing dataset configuration
    with open(config["dataset_toml"], 'r') as file:
        toml_data = toml.load(file)

    # Update the image_dir in all datasets.subsets sections
    if 'datasets' in toml_data:
        for dataset in toml_data['datasets']:
            if 'subsets' in dataset:
                for subset in dataset['subsets']:
                    subset['image_dir'] = config['dataset_path']
    print(f"All instances of 'image_dir' in dataset.toml updated to: {config['dataset_path']}")
    
    # Set the flip_aug parameter based on the mode
    if config["mode"] == "face":
        print("Disabling flip augmentation for face mode.")
        toml_data['general']['flip_aug'] = False
    else:
        toml_data['general']['flip_aug'] = True

    # Reorder sections to make sure [general] is at the top
    reordered_toml_data = {}
    if 'general' in toml_data:
        reordered_toml_data['general'] = toml_data.pop('general')  # Move general to the top
    reordered_toml_data.update(toml_data)  # Add the rest of the sections

    # Write to the TOML file
    toml_file_path = os.path.join(config["output_dir"], "dataset.toml")
    with open(toml_file_path, 'w') as file:
        toml.dump(reordered_toml_data, file)
    
    # Manually adjust the output indentation by re-reading and fixing structure
    with open(toml_file_path, 'r') as file:
        lines = file.readlines()

    with open(toml_file_path, 'w') as file:
        for line in lines:
            if line.startswith("[[datasets.subsets]]"):
                file.write("  " + line)  # Add indentation for subsets
            else:
                file.write(line)

    config["dataset_config"] = toml_file_path

    return config


def construct_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config["timestamp"] = timestamp
    config["output_name"] = f"{os.path.basename(os.path.basename(config['dataset_path']))}_{timestamp}"
    config["output_dir"] = f"results/{config['output_name']}"

    # Save the adjusted config inside the output directory:
    os.makedirs(config["output_dir"], exist_ok=True)
    with open(os.path.join(config["output_dir"], "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    config = construct_toml(config)

    return config

def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

def construct_train_command(config):

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_cpu_threads_per_process", "1",
        "--num_processes", "1",
        os.path.join(get_script_directory(), "sd-scripts", "flux_train_network.py"),
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
        "--sample_prompts", config['eval_prompts'],
        "--sample_at_first",
        "--output_dir", config["output_dir"],
        "--output_name", config["output_name"],
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "3.1582",
        "--model_prediction_type", "raw",
        "--guidance_scale", "1.0",
       #"--debug_dataset"
    ]

    return cmd

def run_job(cmd, config):
    timestamp = config["timestamp"]
    output_dir = config["output_dir"]
    log_file = f"{output_dir}/training_log_{timestamp}.txt"

    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')  # Print to console
            f.write(line)  # Write to log file
            f.flush()  # Ensure it's written immediately

    if process.wait() != 0:
        print(f"Command failed with return code {process.returncode}")


def main():
    config = construct_config("train_config.json")
    prep_dataset(config["dataset_path"])
    florence_caption_dataset(config["dataset_path"], caption_mode = config["caption_mode"])
    cmd = construct_train_command(config)
    run_job(cmd, config)

"""
cp -r /data/xander/Projects/cog/xander_eden_stuff/worlds/test_imgs /data/xander/Projects/cog/GitHub_repos/flux-trainer
"""

if __name__ == "__main__":
    main()