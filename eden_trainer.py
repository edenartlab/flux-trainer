from datetime import datetime
start_time = datetime.utcnow()

import logging
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

        # Get task args
        task_args = task["args"]
        print("task_args", task_args)

        # Load the training config from the provided file
        config = construct_config(args.config)
        config["lora_rank"] = str(task_args["lora_rank"])
        config["learning_rate"] = str(task_args["learning_rate"])
        config["seed"] = str(task_args.get("seed", random.randint(0, 2147483648)))
        config["max_train_steps"] = str(task_args["max_train_steps"])
        config["caption_prefix"] = task_args.get("caption_prefix", config["caption_prefix"])
        
        print(" ========= Config ========== ")
        print(config)
        print(" ========================== ")
        
        # Download the dataset from the URL provided
        lora_training_urls = task_args["lora_training_urls"]
        download_dataset(config["dataset_path"], lora_training_urls)
        
        # Use GPT4v to check if the dataset is a face or style
        if eden_utils.check_if_face(config["dataset_path"]):
            config["mode"] = "face"
        else:
            config["mode"] = "style"

        # Preprocess the dataset if required
        if config.get("prep_dataset"):
            prep_dataset(config["dataset_path"], hard_prep=True)

        # Perform dataset captioning if enabled in the config
        if config.get("caption_mode"):
            florence_caption_dataset(config["dataset_path"], caption_mode=config["caption_mode"])

        # Step 5: Construct and run the training command
        cmd = construct_train_command(config)
        run_job(cmd, config)

        # save the result
        output_dir = config["output_dir"]
        output_name = config["output_name"]

        # upload to eden
        file_url, _ = eden_utils.upload_file(
            f"{output_dir}/{output_name}.safetensors",
            env=args.env
        )
        print("file_url", file_url)

        # make thumbnail and slug
        sample_dir = os.path.join(output_dir, "sample")
        thumbnail_url = eden_utils.create_thumbnail(sample_dir, env=args.env)
        slug = eden_utils.make_slug(task)

        # save model
        model_id = eden_utils.models_collection.insert_one({
            "args": task_args,
            "checkpoint": file_url,
            "base_model": "flux-dev",
            "name": task_args["name"],
            "public": False,
            "task": task["_id"],
            "thumbnail": thumbnail_url,
            "slug": slug,
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

