from datetime import datetime
start_time = datetime.utcnow()

import logging
import json
import random
import sys
import argparse

import eden_utils
from main import *
from bson import ObjectId
from eden_utils import tasks_collection

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout
)

def main():
    parser = argparse.ArgumentParser(description='Training script for flux network.')
    parser.add_argument('--task_id', help="Eden task ID")
    parser.add_argument('--env', type=str, default="STAGE", choices=["STAGE", "PROD"], help='Environment')
    parser.add_argument('--config', type=str, default="template/train_config.json", help='Path to the training config file (JSON).')
    args = parser.parse_args()

    # Get task
    task = tasks_collection.find_one({"_id": ObjectId(args.task_id)})

    if not task:
        raise ValueError(f"Task {args.task_id} not found!")

    try:
        wait_time = (start_time - task["createdAt"]).total_seconds()

        # Mark task status running
        tasks_collection.update_one(
            {"_id": ObjectId(args.task_id)}, 
            {"$set": {
                "status": "running",
                "performance": {
                    "waitTime": wait_time,
                },
                "updatedAt": datetime.utcnow(),
            }}
        )

        # Get default base args:
        with open(args.config, 'r') as f:
            config_json = json.load(f)

        # Get task args
        task_args = task["args"]
        print("task_args: ", task_args)

        # Update config_json with all values from task_args
        for key, value in task_args.items():
            # Convert all values to strings, handle seed specially
            if key == "seed" and value is None:
                config_json[key] = str(random.randint(0, 2147483648))
            else:
                config_json[key] = str(value)

        #####################################################
        overwrite_dict = {
            "caption_prefix": "",
            "dataset_toml": "template/dataset_template_512_bs2.toml",
            "eval_prompts": "template/eval_prompts_TOK.txt",
            "mode": "face",
            "full_finetune": False,
            "sample_at_first": False,
            "lora_rank": "4",
            "network_alpha": "16",
            "learning_rate": "0.5e-4",
            "max_train_steps": "20",
            "save_every_n_steps": "20",
            "sample_every_n_steps": "2000",
            "gradient_accumulation_steps": "1",
            "noise_offset": "0.1",
            "ip_noise_gamma": "0.1",
            "MODEL_PATH": "models/flux-dev-de-distill-diffusers" }

        # Update config_json with overwrite_dict values
        for key, value in overwrite_dict.items():
            config_json[key] = value
        #####################################################

        print(f"Final training arguments for job:")
        print(config_json)

        with open("tmp_train_config.json", 'w') as f:
            json.dump(config_json, f, indent=2)

        # Load the training config from the provided file
        config = construct_config("tmp_train_config.json")

        # Download the dataset from the URL provided
        download_dataset(config["dataset_path"], task_args["lora_training_urls"])

        # Preprocess the dataset:
        config = prep_dataset(config)

        # Construct and run the LoRA training command
        cmd = construct_train_command(config)
        run_job(cmd, config)

        # make sample_grid thumbnail: 
        thumbnail_url = eden_utils.create_thumbnail(config, env=args.env)

        # upload to eden
        file_url, _ = eden_utils.upload_file(
            f"{config['output_dir']}/{config['output_name']}.safetensors",
            env=args.env
        )
        print("file_url", file_url)

        # make slug
        # slug = eden_utils.make_slug(task)

        # save model
        model_id = eden_utils.models_collection.insert_one({
            "args": task_args,
            "checkpoint": file_url,
            "base_model": "flux-dev",
            "name": task_args["name"],
            "public": False,
            "task": task["_id"],
            "thumbnail": thumbnail_url,
            "lora_trigger_text": config["lora_trigger_text"],
            # "slug": slug,
            "user": task["user"],
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
        }).inserted_id
        print("saved model_id", model_id)

        finish_time = datetime.utcnow()
        run_time = (finish_time - start_time).total_seconds()

        # Mark task status completed
        tasks_collection.update_one(
            {"_id": ObjectId(args.task_id)}, 
            {"$set": {
                "status": "completed",
                "performance": {
                    "waitTime": wait_time,
                    "runTime": run_time,
                },
                "result": [{
                    "filename": file_url.split("/")[-1],
                    "metadata": config,
                    "mediaAttributes": {
                        "mimeType": "application/zip"
                    },
                    "thumbnail": thumbnail_url,
                    "model": model_id
                }],
                "updatedAt": datetime.utcnow(),
            }}
        )
        
    except Exception as e:
        logging.error(f"Error: {e}")
        print("Error: ", e)
        
        finish_time = datetime.utcnow()
        run_time = (finish_time - start_time).total_seconds()
        
        tasks_collection.update_one(
            {"_id": ObjectId(args.task_id)}, 
            {"$set": {
                "status": "failed",
                "error": str(e),
                "performance": {
                    "waitTime": wait_time,
                    "runTime": run_time,
                },
                "updatedAt": datetime.utcnow(),
            }}
        )


if __name__ == "__main__":
    main()

