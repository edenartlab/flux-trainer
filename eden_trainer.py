from datetime import datetime, timezone
import logging
import json
import random
import sys
import argparse

import eden_utils
from main import *
from bson import ObjectId
from eden_utils import get_collection

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout
)

def main():
    start_time = datetime.now(timezone.utc)

    parser = argparse.ArgumentParser(description='Training script for flux network.')
    parser.add_argument('--task_id', help="Eden task ID")
    parser.add_argument('--db', type=str, default="STAGE", choices=["STAGE", "PROD"], help='Database')
    parser.add_argument('--config', type=str, default="template/train_config.json", help='Path to the training config file (JSON).')
    parser.add_argument('--local_test', action='store_true', help='Run locally for testing')
    args = parser.parse_args()

    # Get task
    tasks_collection = get_collection("tasks3", db=args.db)
    task = tasks_collection.find_one({"_id": ObjectId(args.task_id)})

    if not task:
        raise ValueError(f"Task {args.task_id} not found!")

    try:
        created_at = task["createdAt"].replace(tzinfo=timezone.utc)
        wait_time = (start_time - created_at).total_seconds()

        # Mark task status running
        tasks_collection.update_one(
            {"_id": ObjectId(args.task_id)}, 
            {"$set": {
                "status": "running",
                "performance": {
                    "waitTime": wait_time,
                },
                "updatedAt": datetime.now(timezone.utc),
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

        if args.local_test:
            print("=====================================================")
            print(f"WARNING: Running in local test mode, overwriting some training args!")
            print("=====================================================")
            #####################################################
            overwrite_dict = {
                "caption_prefix": "",
                "dataset_toml": "template/dataset_template_512_bs2.toml",
                "eval_prompts": "template/eval_prompts_TOK.txt",
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
                "MODEL_PATH": "models/flux-dev-de-distill-diffusers" }

            # Update config_json with overwrite_dict values
            for key, value in overwrite_dict.items():
                config_json[key] = value
            #####################################################

        # Make sure we're never sampling images before the end of training:
        config_json["sample_every_n_steps"] = 2*config_json["max_train_steps"]

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
        if 1:
            print("Starting thumbnail generation subprocess...")
            thumbnail_url = eden_utils.create_thumbnail(config, db=args.db)
        else:
            grid_path = "EDEN.jpg"
            print("Uploading EDEN default thumbnail to db...")
            thumbnail_url, _ = eden_utils.upload_file(str(grid_path), db=args.db)

        thumbnail_filename = thumbnail_url.split("/")[-1]
        print(f"Thumbnail url: {thumbnail_url}")

        # upload to eden
        file_url, _ = eden_utils.upload_file(
            f"{config['output_dir']}/{config['output_name']}.safetensors",
            file_type=".safetensors",
            db=args.db
        )
        print("Uploaded LoRA to Eden, file_url:", file_url)

        # make slug
        # slug = eden_utils.make_slug(task)

        # save model
        models_collection = get_collection("models3", db=args.db)
        model_id = models_collection.insert_one({
            "args": task_args,
            "checkpoint": file_url,
            "base_model": "flux-dev",
            "name": task_args["name"],
            "public": False,
            "task": task["_id"],
            "thumbnail": thumbnail_filename,
            "lora_trigger_text": config["lora_trigger_text"],
            # "slug": slug,
            "user": task["user"],
            "requester": task["requester"],
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc),
        }).inserted_id
        print("saved model_id", model_id)

        finish_time = datetime.now(timezone.utc)
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
                    "output": [{
                        "filename": file_url.split("/")[-1],
                        "metadata": config,
                        "mediaAttributes": {
                            "mimeType": "application/zip"
                        },
                        "thumbnail": thumbnail_filename,
                        "model": model_id
                    }],
                }],
                "updatedAt": datetime.now(timezone.utc),
            }}
        )
        
    except Exception as e:
        logging.error(f"Error: {e}")
        print("Error: ", e)
        
        finish_time = datetime.now(timezone.utc)
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
                "updatedAt": datetime.now(timezone.utc),
            }}
        )


if __name__ == "__main__":
    main()

