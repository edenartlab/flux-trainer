import random
import os
import ast
import json
import shutil
import time
from tqdm import tqdm
from datetime import datetime

# Setup the base experiment config
exp_name = "banny_final_sweep"
n_exp = 100  # how many random experiment settings to generate
min_hamming_distance = 1  # min params that have to be different from previous experiments
nohup = True

# Define training hyperparameters and their possible values
hyperparameters = {
    "dataset_path": [
        "/data/xander/Projects/cog/GitHub_repos/flux-trainer/datasets/banny_best"
    ],
    "caption_mode": [None],
    "mode": ["auto"],
    "caption_prefix": ["a yellow cartoon banana character, with expressive eyes, spotted skin and pink lips wearing a black spiked collar"],
    "masking_prompt": ["a yellow banana character"],
    "dataset_toml": [
        #"template/dataset_template_512_bs1.toml",
        "template/dataset_template_512_bs4.toml"
    ],
    "eval_prompts": ["template/eval_prompts_TOK.txt"],  # Keeping this constant
    "full_finetune": [False],
    "sample_at_first": [False],
    "lora_rank": ["2", "4", "8"],
    "network_alpha": ["8", "16", "32"],
    "learning_rate": ["0.5e-4", "1e-4"],
    "ip_noise_gamma": ["0.1"],
    "noise_offset": ["0.1"],

    "max_train_steps": ["1500"],
    "save_every_n_steps": ["500"],
    "sample_every_n_steps": ["1500"],
    "gradient_accumulation_steps": ["1"],
    "seed": ["1"],  # Keeping this constant
    "MODEL_PATH": [
        "models/flux-dev-de-distill-diffusers",
        #"models/flux1-dev.safetensors"
    ],
    "CLIP_L_PATH": ["models/clip_l.safetensors"], 
    "T5XXL_PATH": ["models/t5xxl_fp16.safetensors"],
    "AE_PATH": ["models/ae.safetensors"]
}

#############################################

def hamming_distance(dict1, dict2):
    """Calculate number of different values between two dictionaries."""
    distance = 0
    for key in dict1.keys():
        if dict1[key] != dict2.get(key, None):
            distance += 1
    return distance

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
exp_name += f"_{timestamp}"
output_sh_path = f"{exp_name}.sh"

# Create output directories
config_output_dir = f"gridsearch_configs/{exp_name}"
shutil.rmtree(config_output_dir, ignore_errors=True)
os.makedirs(config_output_dir, exist_ok=True)

# Set random seed
random.seed(int(1000 * time.time()))

# Keep track of scheduled experiments
scheduled_experiments = set()

# Generate experiments
try_sampling_n_times = 120
for exp_index in tqdm(range(n_exp)):
    resamples = 0
    
    while resamples < try_sampling_n_times:
        # Generate random experiment settings
        experiment_settings = {name: random.choice(values) for name, values in hyperparameters.items()}
        
        resamples += 1

        # Check minimum distance from existing experiments
        min_distance = float('inf')
        for str_experiment_settings in scheduled_experiments:
            existing_experiment_settings = dict(sorted(ast.literal_eval(str_experiment_settings)))
            distance = hamming_distance(experiment_settings, existing_experiment_settings)
            min_distance = min(min_distance, distance)

        if min_distance >= min_hamming_distance:
            str_experiment_settings = str(sorted(experiment_settings.items()))
            scheduled_experiments.add(str_experiment_settings)

            # Add output paths
            output_name = f"{exp_name}_{exp_index:03d}"
            experiment_settings["output_name"] = output_name
            
            # Save config to JSON file
            config_filename = f"{config_output_dir}/{exp_name}_{exp_index:03d}.json"
            os.makedirs(os.path.dirname(config_filename), exist_ok=True)
            
            with open(config_filename, "w") as f:
                json.dump(experiment_settings, f, indent=4)
            break

    if resamples >= try_sampling_n_times:
        print(f"\nCould not find a new experiment setting after {try_sampling_n_times} attempts")
        break

print(f"\n---> Saved {len(scheduled_experiments)} experiment configurations to {config_output_dir}")

def generate_sh_script(folder_path, output_sh_path):
    """Generate shell script to run all experiments."""
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    
    with open(output_sh_path, 'w') as sh_file:
        sh_file.write("#!/bin/bash\n\n")
        
        # Get total number of files for handling the last line differently
        total_files = len(json_files)
        
        for i, json_file in enumerate(json_files):
            file_path = os.path.join(folder_path, json_file)
            base_command = f"nohup python main.py --config {file_path}"
            log_path = file_path.replace('.json', '.log')
            
            # Add command with log redirection
            command = f"{base_command} > {log_path} 2>&1"
            
            # Add line continuation if not the last line
            if i < total_files - 1:
                command += " ; \\\n"
            else:
                command += "\n"
                
            sh_file.write(command)

generate_sh_script(config_output_dir, output_sh_path)
print(f"\n---> Saved the executable shell script to {output_sh_path}")