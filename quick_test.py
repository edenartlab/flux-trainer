#!/usr/bin/env python3
"""
Quick test script for running full training workflows without database interactions.
This script imports functions from eden_trainer.py and executes them with test parameters.

USAGE:
    ./run_interactive.sh
    python3 quick_test.py

    To use custom arguments, modify the DEFAULT_ARGS dictionary below instead of 
    passing command-line arguments.
"""
import os
import sys
import json
import random
from datetime import datetime, timezone
import logging

#############################################################################
# CONFIGURATION - MODIFY THESE VALUES DIRECTLY INSTEAD OF USING CLI ARGS
#############################################################################

DEFAULT_ARGS = {
    # Options
    "skip_uploads": False,  # Set to False if you want to upload results
    "debug": False,  # Set to True for more verbose output

    # Model parameters
    "name": f"quick-test-{datetime.now().strftime('%m%d-%H%M')}",  # Will be auto-generated with timestamp
    "mode": "face",  # Options: face, object, style, etc.
    "config": "template/train_config.json",  # Base config file

    "lora_training_urls": [
        "https://edenartlab-stage-data.s3-accelerate.amazonaws.com/cab12b589bc622e78847eb0e94465dfcb377b50ff786c6489ee460758f731f18.jpeg",
        "https://edenartlab-stage-data.s3-accelerate.amazonaws.com/c8abb2a5416d8e8be612e3e570f6db5aac80f73b7bc113c0bd891b55e99863be.jpeg",
        "https://edenartlab-stage-data.s3-accelerate.amazonaws.com/821c79b42638b21c0dd9d5a1d17403c58e01caa037f636f0b539fd4183dfb67f.jpeg",
        "https://edenartlab-stage-data.s3-accelerate.amazonaws.com/8941c0ffdbd9f911724af65e09ccc353b362c05786085b0cf90d3fcd0338a95c.jpeg"
    ],
    
    # Training parameters
    "lora_rank": "8",
    "learning_rate": "0.5e-4",
    "max_train_steps": "20",  # Use a small number for quick testing
    "seed": None,  # Set to an integer for reproducible results or None for random

    # Development overrides - these will override settings from the base config
    "dev_overrides": {
        "caption_prefix": "",
        "dataset_toml": "template/dataset_template_512_bs2.toml",
        "eval_prompts": "template/grid_prompts_small.txt",
        "full_finetune": False,
        "sample_at_first": False
    }
}

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout
)

# Import functions from eden_trainer.py
# We use this approach to avoid modifying the original code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    # Import all the necessary functions
    from eden_trainer import (
        construct_config, 
        download_dataset, 
        prep_dataset, 
        construct_train_command, 
        run_job,
        get_filename_from_url
    )
    import eden_utils
except ImportError as e:
    print(f"Failed to import from eden_trainer.py: {e}")
    print("Make sure you're running this script from the project directory.")
    sys.exit(1)

def main():
    start_time = datetime.now(timezone.utc)
    args = DEFAULT_ARGS
    
    # Set debug level if requested
    if args["debug"]:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Prepare task arguments
        task_args = {
            "lora_training_urls": args["lora_training_urls"] if isinstance(args["lora_training_urls"], list) 
                                  else args["lora_training_urls"].split(','),
            "name": args["name"],
            "mode": args["mode"],
            "lora_rank": args["lora_rank"],
            "learning_rate": args["learning_rate"],
            "max_train_steps": args["max_train_steps"],
            "seed": args["seed"]
        }
        
        print("Task arguments:")
        print(json.dumps(task_args, indent=2, default=str))
        
        # Load the default configuration
        with open(args["config"], 'r') as f:
            config_json = json.load(f)
        
        # Update config with task arguments
        for key, value in task_args.items():
            if key == "seed" and value is None:
                config_json[key] = str(random.randint(0, 2147483648))
            elif isinstance(value, list):
                config_json[key] = value
            else:
                config_json[key] = str(value)
        
        # Apply development-specific overrides
        for key, value in args["dev_overrides"].items():
            config_json[key] = value
        
        # Configure sampling and saving intervals
        max_steps = int(config_json["max_train_steps"])
        config_json["sample_every_n_steps"] = str(max_steps)  # Sample once during training
        config_json["save_every_n_steps"] = str(max(1, max_steps // 4))    # Save 4x during training
        
        # Write the final config
        with open("quick_test_config.json", 'w') as f:
            json.dump(config_json, f, indent=2)
        
        print("\nFinal training configuration:")
        print(json.dumps(config_json, indent=2))
        
        # Load the training config
        config = construct_config("quick_test_config.json")
        
        # Download the dataset from the URLs provided
        print("\nDownloading dataset...")
        download_dataset(config["dataset_path"], task_args["lora_training_urls"])
        
        # Preprocess the dataset
        print("\nPreprocessing dataset...")
        config = prep_dataset(config)
        
        # Construct the training command
        print("\nConstructing training command...")
        cmd = construct_train_command(config)
        print(f"Training command: {cmd}")
        
        # Run the training job
        print("\nRunning training job...")
        run_job(cmd, config)
        
        # If we're not skipping uploads, handle result processing
        if not args["skip_uploads"]:
            print("\nUploading results...")
            # Generate thumbnails with sample images
            sample_dir = os.path.join(config["output_dir"], "sample")
            thumbnail_url = eden_utils.combine_samples_into_grid(sample_dir, db="STAGE")
            print(f"Thumbnail url: {thumbnail_url}")
            
            # Upload the LoRA checkpoint
            file_url, _ = eden_utils.upload_file(
                f"{config['output_dir']}/{config['output_name']}.safetensors",
                file_type=".safetensors",
                db="STAGE"
            )
            print("Uploaded main LoRA checkpoint, file_url:", file_url)
            
            # Extract filenames from urls
            thumbnail_filename = get_filename_from_url(thumbnail_url)
            lora_filename = get_filename_from_url(file_url)
            
            print(f"\nResults:")
            print(f"- LoRA filename: {lora_filename}")
            print(f"- Thumbnail: {thumbnail_filename}")
        else:
            print("\nSkipping result uploads.")
            print(f"Output directory: {config['output_dir']}")
            print(f"Main checkpoint: {config['output_dir']}/{config['output_name']}.safetensors")
        
        finish_time = datetime.now(timezone.utc)
        run_time = (finish_time - start_time).total_seconds()
        print(f"\nTraining completed in {run_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())