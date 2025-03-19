from datetime import datetime, timezone
import logging
import json
import random
import sys
import argparse
import os
from urllib.parse import urlparse
import glob
import eden_utils
from main import *
from bson import ObjectId
from eden_utils import get_collection
from typing import Dict, Any, List, Optional, Union

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout
)

def get_filename_from_url(url):
    """Extract the filename from a URL."""
    path = urlparse(url).path
    return os.path.basename(path)

def prepare_training_config(
    base_config_path: str, 
    task_args: Dict[str, Any], 
    local_test: bool = False
) -> Dict[str, Any]:
    """
    Prepare the training configuration by merging base config with task args.
    
    Args:
        base_config_path: Path to the base configuration JSON file
        task_args: Dictionary of task arguments to override the base config
        local_test: If True, overrides certain settings for local testing
        
    Returns:
        Dictionary containing the final configuration
    """
    # Get default base args
    with open(base_config_path, 'r') as f:
        config_json = json.load(f)

    # Update config_json with all values from task_args
    for key, value in task_args.items():
        # Convert all values to strings, handle seed specially
        if key == "seed" and value is None:
            config_json[key] = str(random.randint(0, 2147483648))
        elif key == "custom_validation_prompts" and value is not None:
            # Handle custom_validation_prompts as a list of strings
            config_json[key] = value
        else:
            config_json[key] = str(value)

    if local_test:
        print("=====================================================")
        print(f"WARNING: Running in local test mode, overwriting some training args!")
        print("=====================================================")
        
        # Override settings for local testing
        overwrite_dict = {
            "caption_prefix": "",
            "dataset_toml": "template/dataset_template_512_bs2.toml",
            "eval_prompts": "template/grid_prompts_small.txt",
            "mode": "face",
            "full_finetune": False,
            "sample_at_first": False,
            "lora_rank": "8",
            "learning_rate": "0.5e-4",
            "max_train_steps": "10",
            "save_every_n_steps": "10",
            "sample_every_n_steps": "2000",
            "gradient_accumulation_steps": "1",
            "noise_offset": "0.1",
            "ip_noise_gamma": "0.1",
            "MODEL_PATH": "models/flux-dev-de-distill-diffusers" 
        }

        # Update config_json with overwrite_dict values
        for key, value in overwrite_dict.items():
            config_json[key] = value

    # Configure sampling and saving intervals
    max_steps = int(config_json["max_train_steps"])
    config_json["sample_every_n_steps"] = str(2 * max_steps)  # Sample once after training
    config_json["save_every_n_steps"] = str(max_steps // 4)   # Save 4 checkpoints during training

    print(f"Final training arguments for job:")
    print(json.dumps(config_json, indent=2))

    return config_json

def save_config_to_file(config: Dict[str, Any], output_path: str = "tmp_train_config.json") -> str:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration
        
    Returns:
        Path to the saved configuration file
    """
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    return output_path

def upload_checkpoints(
    config: Dict[str, Any], 
    db: str = "STAGE"
) -> Dict[str, Any]:
    """
    Upload LoRA checkpoints and create thumbnail.
    
    Args:
        config: Training configuration
        db: Database to use (STAGE or PROD)
        
    Returns:
        Dictionary with uploaded filenames and URLs
    """
    result = {}
    
    # Generate/upload thumbnail
    if 0:  # Generate thumbnails with full FLUX model (doesnt work)
        print("Starting thumbnail generation subprocess...")
        thumbnail_url = eden_utils.create_thumbnail(config, db=db)
    elif 1:  # just upload default EDEN thumbnail
        grid_path = "EDEN.jpg"
        print("Uploading EDEN default thumbnail to db...")
        thumbnail_url, _ = eden_utils.upload_file(str(grid_path), db=db)
    else:  # Generate thumbnails with sample images
        sample_dir = os.path.join(config["output_dir"], "sample")
        thumbnail_url = eden_utils.combine_samples_into_grid(sample_dir, db=db)

    print(f"Thumbnail url: {thumbnail_url}")
    result["thumbnail_url"] = thumbnail_url

    # Upload main LoRA checkpoint
    file_url, _ = eden_utils.upload_file(
        f"{config['output_dir']}/{config['output_name']}.safetensors",
        file_type=".safetensors",
        db=db
    )
    print("Uploaded main LoRA checkpoint to Eden, file_url:", file_url)
    result["lora_url"] = file_url

    # Extract filenames from URLs
    result["thumbnail_filename"] = get_filename_from_url(thumbnail_url)
    result["lora_filename"] = get_filename_from_url(file_url)

    # Upload checkpoint versions
    checkpoint_versions = []
    # Get all checkpoint files sorted by step number
    checkpoint_pattern = f"{config['output_dir']}/{config['output_name']}-step*.safetensors"
    all_checkpoint_paths = glob.glob(checkpoint_pattern)

    # Sort checkpoints by step number
    def extract_step_number(path):
        filename = os.path.basename(path)
        step_str = filename.split('-step')[1].split('.')[0]
        return int(step_str)

    # Sort in descending order (newest first)
    if all_checkpoint_paths:
        all_checkpoint_paths.sort(key=extract_step_number, reverse=True)
        
        # Take only the 2 most recent checkpoints
        recent_checkpoints = all_checkpoint_paths[:3]
        
        # Upload each checkpoint and add to versions dictionary
        for checkpoint_path in recent_checkpoints:
            checkpoint_step = extract_step_number(checkpoint_path)
            version_url, _ = eden_utils.upload_file(
                checkpoint_path,
                file_type=".safetensors",
                db=db
            )
            version_filename = get_filename_from_url(version_url)
            checkpoint_versions.append({"step": checkpoint_step, "checkpoint": version_filename})
            print(f"Uploaded checkpoint version-{checkpoint_step} to Eden, file_url: {version_url}")
    else:
        print("No checkpoint versions found with pattern:", checkpoint_pattern)
    
    result["checkpoint_versions"] = checkpoint_versions
    return result

def save_model_to_db(
    task_args: Dict[str, Any],
    task_id: str,
    upload_results: Dict[str, Any],
    config: Dict[str, Any],
    task: Dict[str, Any],
    db: str = "STAGE"
) -> str:
    """
    Save model information to the database.
    
    Args:
        task_args: Task arguments
        task_id: Task ID
        upload_results: Results from uploading checkpoints
        config: Training configuration
        task: Task information
        db: Database to use
        
    Returns:
        Model ID
    """
    # Save model to database
    models_collection = get_collection("models3", db=db)
    
    # Prepare model document
    model_doc = {
        "args": task_args,
        "checkpoint": upload_results["lora_filename"],
        "checkpoint_versions": upload_results["checkpoint_versions"],
        "base_model": "flux-dev",
        "name": task_args["name"],
        "public": False,
        "deleted": False,
        "task": ObjectId(task_id) if task_id else task["_id"],
        "thumbnail": upload_results["thumbnail_filename"],
        "lora_trigger_text": config["lora_trigger_text"],
        "lora_mode": config["mode"],
        "user": task["user"],
        "agent": task["agent"],
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }
    
    # Add custom_validation_prompts if present in config and not empty
    if "custom_validation_prompts" in config and config["custom_validation_prompts"] is not None:
        # Only add to DB if the list is not empty
        if isinstance(config["custom_validation_prompts"], list) and len(config["custom_validation_prompts"]) > 0:
            model_doc["custom_validation_prompts"] = config["custom_validation_prompts"]
        
    model_id = models_collection.insert_one(model_doc).inserted_id
    print("saved model_id", model_id)
    return str(model_id)

def update_task_status(
    task_id: str,
    status: str,
    wait_time: float = 0,
    run_time: float = 0,
    result: Optional[Dict] = None,
    error: Optional[str] = None,
    db: str = "STAGE"
) -> None:
    """
    Update the task status in the database.
    
    Args:
        task_id: Task ID
        status: New status (running, completed, failed)
        wait_time: Time spent waiting
        run_time: Time spent running
        result: Task result
        error: Error message if status is 'failed'
        db: Database to use
    """
    tasks_collection = get_collection("tasks3", db=db)
    
    update_data = {
        "status": status,
        "performance": {
            "waitTime": wait_time,
        },
        "updatedAt": datetime.now(timezone.utc),
    }
    
    if run_time > 0:
        update_data["performance"]["runTime"] = run_time
    
    if result is not None:
        update_data["result"] = result
    
    if error is not None:
        update_data["error"] = error
    
    tasks_collection.update_one(
        {"_id": ObjectId(task_id)}, 
        {"$set": update_data}
    )

def run_training_workflow(
    task_id: str = None,
    task_args: Dict[str, Any] = None,
    db: str = "STAGE",
    config_path: str = "template/train_config.json",
    local_test: bool = False,
    skip_db_updates: bool = False
) -> Dict[str, Any]:
    """
    Run the complete training workflow.
    
    Args:
        task_id: Task ID (optional if skip_db_updates is True)
        task_args: Task arguments (required if task_id is None)
        db: Database to use (STAGE or PROD)
        config_path: Path to the base configuration file
        local_test: If True, use test settings
        skip_db_updates: If True, skip database interactions
        
    Returns:
        Dictionary with results
    """
    start_time = datetime.now(timezone.utc)
    results = {}
    task = None
    wait_time = 0
    
    try:
        # Get task if we have a task_id
        if task_id and not skip_db_updates:
            tasks_collection = get_collection("tasks3", db=db)
            task = tasks_collection.find_one({"_id": ObjectId(task_id)})
            
            if not task:
                raise ValueError(f"Task {task_id} not found!")
                
            created_at = task["createdAt"].replace(tzinfo=timezone.utc)
            wait_time = (start_time - created_at).total_seconds()
            
            # Use task args from the task
            task_args = task["args"]
            
            # Mark task as running
            update_task_status(task_id, "running", wait_time=wait_time, db=db)
        else:
            # Create a minimal task object for local testing
            if task_args is None:
                raise ValueError("task_args must be provided when task_id is not provided")
                
            task = {
                "user": "local_user",
                "agent": "local_agent",
                "_id": "local_task_id"
            }
            
        print("task_args: ", task_args)
        
        # Validate custom_validation_prompts if present
        if "custom_validation_prompts" in task_args and task_args["custom_validation_prompts"] is not None:
            custom_prompts = task_args["custom_validation_prompts"]
            if not isinstance(custom_prompts, list):
                raise ValueError("custom_validation_prompts must be a list of strings")
            if len(custom_prompts) > 4:
                raise ValueError("custom_validation_prompts can have at most 4 items")
            for prompt in custom_prompts:
                if not isinstance(prompt, str):
                    raise ValueError("All items in custom_validation_prompts must be strings")
        
        # Prepare the training configuration
        config_json = prepare_training_config(config_path, task_args, local_test)
        
        # Save config to file
        config_path = save_config_to_file(config_json)
        
        # Load the training config
        config = construct_config(config_path)
        
        # Download the dataset
        download_dataset(config["dataset_path"], task_args["lora_training_urls"])
        
        # Preprocess the dataset
        config = prep_dataset(config)
        
        # Construct and run the training command
        cmd = construct_train_command(config)
        run_job(cmd, config)
        
        # Store output information in config for use during cleanup
        results["output_dir"] = config["output_dir"]
        results["output_name"] = config["output_name"]
        
        # Upload checkpoints and create thumbnail if not skipping DB updates
        if not skip_db_updates:
            upload_results = upload_checkpoints(config, db=db)
            results.update(upload_results)
            
            # Save model to database if we have a task_id
            model_id = None
            if task_id:
                model_id = save_model_to_db(task_args, task_id, upload_results, config, task, db=db)
                results["model_id"] = model_id
            
            # Mark task as completed only if we have a task_id
            if task_id:
                finish_time = datetime.now(timezone.utc)
                run_time = (finish_time - start_time).total_seconds()
                
                task_result = [{
                    "output": [{
                        "filename": upload_results["lora_filename"],
                        "metadata": config,
                        "mediaAttributes": {
                            "mimeType": "application/zip"
                        },
                        "thumbnail": upload_results["thumbnail_filename"],
                        "model": model_id
                    }],
                }]
                
                update_task_status(
                    task_id, 
                    "completed", 
                    wait_time=wait_time, 
                    run_time=run_time, 
                    result=task_result,
                    db=db
                )
        
        results["status"] = "success"
        results["config"] = config
        
        # Add output paths to results for local testing
        if skip_db_updates:
            results["output_dir"] = config["output_dir"]
            results["output_name"] = config["output_name"]
        
    except Exception as e:
        logging.error(f"Error: {e}")
        print("Error: ", e)
        
        # Mark task as failed if using DB
        if task_id and not skip_db_updates:
            finish_time = datetime.now(timezone.utc)
            run_time = (finish_time - start_time).total_seconds()
            
            update_task_status(
                task_id, 
                "failed", 
                wait_time=wait_time, 
                run_time=run_time, 
                error=str(e),
                db=db
            )
        
        results["status"] = "error"
        results["error"] = str(e)
    
    # Add timing information
    finish_time = datetime.now(timezone.utc)
    results["run_time"] = (finish_time - start_time).total_seconds()
    
    return results

def main():
    """Entry point for the script when run directly."""
    parser = argparse.ArgumentParser(description='Training script for flux network.')
    parser.add_argument('--task_id', help="Eden task ID")
    parser.add_argument('--db', type=str, default="STAGE", choices=["STAGE", "PROD"], help='Database')
    parser.add_argument('--config', type=str, default="template/train_config.json", help='Path to the training config file (JSON).')
    parser.add_argument('--local_test', action='store_true', help='Run locally for testing')
    args = parser.parse_args()
    
    run_training_workflow(
        task_id=args.task_id,
        db=args.db,
        config_path=args.config,
        local_test=args.local_test,
        skip_db_updates=False
    )

if __name__ == "__main__":
    main()
