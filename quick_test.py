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
import shutil
from datetime import datetime, timezone
import logging

# Import the modular functions from eden_trainer
from eden_trainer import (
    run_training_workflow,
    get_filename_from_url
)

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
    "custom_validation_prompts": ["a person with blue eyes", "a person smiling"],  # Optional validation prompts

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
}

def cleanup_temp_files(results):
    """
    Clean up temporary files and directories created during the training process.
    
    Args:
        results (dict): Dictionary containing training results with paths to output files
    """
    try:
        # Files to clean up
        temp_files = [
            "tmp_train_config.json",   # Temporary config file
        ]
        
        # Remove temporary files
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
        
        # Clean up temporary dataset directory if it exists
        dataset_path = "dataset_path"  # Default path used in training config
        if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
            shutil.rmtree(dataset_path)
            print(f"Removed temporary directory: {dataset_path}")
            
        print("Cleanup completed successfully.")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")
        
def main():
    # Set up logging
    if DEFAULT_ARGS["debug"]:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
    
    # Prepare task arguments from the DEFAULT_ARGS
    task_args = {
        "lora_training_urls": DEFAULT_ARGS["lora_training_urls"],
        "name": DEFAULT_ARGS["name"],
        "mode": DEFAULT_ARGS["mode"],
        "lora_rank": DEFAULT_ARGS["lora_rank"],
        "learning_rate": DEFAULT_ARGS["learning_rate"],
        "max_train_steps": DEFAULT_ARGS["max_train_steps"],
        "seed": DEFAULT_ARGS["seed"]
    }
    
    # Add custom_validation_prompts if provided (even if empty)
    # The validation on whether to save an empty list happens in save_model_to_db
    if "custom_validation_prompts" in DEFAULT_ARGS:
        task_args["custom_validation_prompts"] = DEFAULT_ARGS["custom_validation_prompts"]
    
    print("Task arguments:")
    print(json.dumps(task_args, indent=2, default=str))
    
    # Run the training workflow
    try:
        results = run_training_workflow(
            task_args=task_args,
            config_path=DEFAULT_ARGS["config"],
            local_test=True,
            skip_db_updates=DEFAULT_ARGS["skip_uploads"] # Use the setting from DEFAULT_ARGS
        )
        
        # Print results summary
        if results["status"] == "success":
            print("\nTraining completed successfully!")
            
            print("\nSkipping result uploads.")
            print(f"Output directory: {results['output_dir']}")
            print(f"Main checkpoint: {results['output_dir']}/{results['output_name']}.safetensors")
            
            print(f"\nTraining completed in {results['run_time']:.2f} seconds")
        else:
            print(f"\nTraining failed: {results.get('error', 'Unknown error')}")
            return 1
            
        # Clean up temporary files
        cleanup_temp_files(results)
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())