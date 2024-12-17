import random
import os
import ast
import json
import shutil
import time
from tqdm import tqdm

def hamming_distance(dict1, dict2):
    """Calculate number of different values between two dictionaries."""
    distance = 0
    for key in dict1.keys():
        if dict1[key] != dict2.get(key, None):
            distance += 1
    return distance

# Setup the base experiment config
exp_name = "banny"
output_dir = "results_grid"
n_exp = 100  # how many random experiment settings to generate
min_hamming_distance = 3  # min params that have to be different from previous experiments
nohup = True
output_sh_path = f"gridsearch_configs/{exp_name}.sh"


# Define training hyperparameters and their possible values
hyperparameters = {
    "dataset_path": [
        "/data/xander/Projects/cog/GitHub_repos/flux-trainer/datasets/bannyv2",
        #"/data/xander/Projects/cog/GitHub_repos/flux-trainer/datasets/banny_small"
    ],
    "caption_mode": [None, "<CAPTION>"],
    "prep_dataset": [True],  # Keeping this constant
    "mode": ["object"],
    "caption_prefix": [
        "Banny, the yellow cartoon bananaman"
    ],
    "caption_suffix": [""],  # Keeping this constant
    "dataset_toml": [
        "template/dataset_template_512_bs1.toml",
        "template/dataset_template_512_bs2.toml",
        #"template/dataset_template_1024_bs1.toml",
        #"template/dataset_template_1024_bs2.toml",
    ],
    "eval_prompts": ["template/eval_prompts_TOK.txt"],  # Keeping this constant
    "full_finetune": [False],
    "lora_rank": ["4", "8"],
    "learning_rate": ["1.0e-4", "3e-4"],
    "max_train_steps": ["3000"],
    "save_every_n_steps": ["1000"],
    "sample_every_n_steps": ["1000"],
    "seed": ["42"],  # Keeping this constant
    "MODEL_PATH": [
        "models/flux-dev-de-distill-diffusers",
        #"models/flux1-dev.safetensors"
    ],
    "CLIP_L_PATH": ["models/clip_l.safetensors"], 
    "T5XXL_PATH": ["models/t5xxl_fp16.safetensors"],
    "AE_PATH": ["models/ae.safetensors"]
}

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
        
        # Add timestamp and output paths
        timestamp = time.strftime("%Y%m%d_%H%M")
        output_name = f"{exp_name}_{timestamp}_{exp_index:03d}.zip"
        experiment_settings["timestamp"] = timestamp
        experiment_settings["output_name"] = output_name
        experiment_settings["output_dir"] = f"{output_dir}/{output_name}"
        
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
        
        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            command = f"python main.py --config {file_path}\n"
            
            if nohup:
                command = f"nohup {command} > {file_path.replace('.json', '.log')} 2>&1 &\n"
            
            sh_file.write(command)

generate_sh_script(config_output_dir, output_sh_path)
print(f"\n---> Saved the executable shell script to {output_sh_path}")