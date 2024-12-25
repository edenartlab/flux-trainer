import os
import json
import random
import toml
import subprocess
import logging
from typing import Dict, Any, List, Union, Literal
from tqdm import tqdm
from datetime import datetime
import torch
import gc
from PIL import Image
import shutil
from pathlib import Path

def path_to_str(obj: Any) -> Any:
    """Convert Path objects to strings."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: path_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [path_to_str(v) for v in obj]
    return obj

def construct_toml(config: Dict[str, Any]) -> Dict[str, Any]:
    """Construct and update the TOML configuration file."""

    with open(config["dataset_toml"], 'r') as file:
        toml_data = toml.load(file)

    if 'datasets' in toml_data:
        for dataset in toml_data['datasets']:
            if 'subsets' in dataset:
                for subset in dataset['subsets']:
                    subset['image_dir'] = config['dataset_path']
                    if config["caption_prefix"]:
                        subset['caption_prefix'] = config["caption_prefix"]

    logging.info(f"All instances of 'image_dir' in dataset.toml updated to: {config['dataset_path']}")
    
    toml_data['general']['flip_aug'] = config["mode"] != "face"
    if not toml_data['general']['flip_aug']:
        logging.info("Disabling flip augmentation for face mode.")
    
    toml_file_path = Path(config["output_dir"]) / "dataset.toml"
    with open(toml_file_path, 'w') as file:
        toml.dump(toml_data, file)
    
    config["dataset_config"] = str(toml_file_path)

    return config

def construct_config(config_path: str) -> Dict[str, Any]:
    """Construct and update the configuration dictionary."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        config["timestamp"] = timestamp

        dataset_basename = os.path.basename(config["dataset_path"])
        if config.get("output_name"):
            config["output_name"] += f"_{timestamp}_{dataset_basename}"
        else:
            config["output_name"] = f"{timestamp}_{dataset_basename}"
        
        config["output_dir"] = str(Path("results") / config['output_name'])

        # create output directory if it doesn't exist:
        os.makedirs(config["output_dir"], exist_ok=True)
        
        return config
    except Exception as e:
        logging.error(f"Error in construct_config: {str(e)}")
        raise

def run_job(cmd: List[str], config: Dict[str, Any]) -> None:
    """Run the training job and log output."""
    log_file = Path(config["output_dir"]) / f"training_log_{config['timestamp']}.txt"
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()

        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    except Exception as e:
        logging.error(f"Error in run_job: {str(e)}")
        raise

def create_sample_prompts(config):
    """
    Create the sample prompts for evaluation.
    Replace any occurence of "TOK" with the lora_trigger_text from the config.
    """
    new_text_lines = []
    # load the source prompts:
    with open(config['eval_prompts'], 'r') as f:
        text_lines = f.readlines()
        for line in text_lines:
            prompt = line
            if config["caption_prefix"]:
                if "TOK" in prompt:
                    prompt = prompt.replace("TOK", config["caption_prefix"])
                else:
                    prompt = config["caption_prefix"] + " " + prompt
            new_text_lines.append(prompt)
            
    # save the new prompts:
    eval_prompts_path = Path(config["output_dir"]) / "eval_prompts.txt"
    with open(eval_prompts_path, 'w') as f:
        f.writelines(new_text_lines)

    config["eval_prompts"] = str(eval_prompts_path)
    return config

def clean_imgs_and_txt_files(source_data_directory, new_data_dir, error_dir, config, caption_mode):
    """
    Clean root dataset directory and ensure .txt files have matching images
    """
    
    print(f"Preparing dataset from {source_data_directory}...", flush=True)
    total_imgs, resized = 0, 0
    successful_images = set()  # Track successfully processed image base names
    pending_txt_files = {}  # Store txt files to process after image scanning

    for subdir, _, files in os.walk(source_data_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            base_name = os.path.splitext(file)[0]

            if file_path.lower().endswith('.txt'):
                # Store txt file paths for later processing
                pending_txt_files[base_name] = file_path
                continue

            try:  # check if the file can be loaded as an img:
                img = load_image_with_orientation(file_path, mode="RGB")
                
                if max(img.width, img.height) > 2048:
                    # Resize the image with max width/height of 2048
                    img.thumbnail((2048, 2048), Image.LANCZOS)
                    resized += 1
                
                # Save the image as .jpg
                new_filename = base_name + '.jpg'
                new_file_path = os.path.join(new_data_dir, new_filename)
                img.save(new_file_path, 'JPEG', quality=95)
                total_imgs += 1
                successful_images.add(base_name)  # Track successful image

            except Exception as e:
                shutil.copy(file_path, os.path.join(error_dir, file))

    # Process .txt files after all images are handled
    total_txt_files_present = 0
    if caption_mode:
        for base_name, txt_path in pending_txt_files.items():
            if base_name in successful_images:
                # Copy txt files that have matching successful images
                shutil.copy(txt_path, os.path.join(new_data_dir, os.path.basename(txt_path)))
                total_txt_files_present += 1
            else:
                # Move txt files without matching images to error directory
                shutil.copy(txt_path, os.path.join(error_dir, os.path.basename(txt_path)))
    else:
        print(f"Caption mode set to null, removing all .txt files!", flush=True)

    print(f"{total_imgs} imgs from {source_data_directory} converted to .jpg and saved to {new_data_dir}. "
          f"Resized {resized} images. {total_txt_files_present} total .txt files present in data dir.", flush=True)
    config["dataset_path"] = new_data_dir

    return config

def remove_vowels(text):
    # Return None if input is None
    if text is None:
        return None
        
    # Handle empty string
    if text == "":
        return ""
        
    text = text.strip()
    vowels = 'aeiouAEIOU'
    cleaned = ''.join(char for char in text if char not in vowels)

    # remove any extra spaces:
    cleaned = ' '.join(cleaned.split())
    return cleaned

from eden_utils import describe_image_concept, gpt4_v_caption_dataset, florence_caption_dataset, clipseg_mask_generator, load_image_with_orientation, auto_detect_training_mode

def prep_dataset(config, verbose = True):
    new_data_root_dir = os.path.join(config["output_dir"], "dataset")
    new_data_dir = os.path.join(new_data_root_dir, "images")
    error_dir    = os.path.join(new_data_root_dir, "error_files")
    os.makedirs(new_data_root_dir, exist_ok=True)
    os.makedirs(new_data_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # Cleanup and prep dataset:
    config = clean_imgs_and_txt_files(config["dataset_path"], new_data_dir, error_dir, config, config.get("caption_mode"))

    # Use GPT4v to check if the dataset is a face or style
    if not config.get("mode") or config.get("mode") == "auto":
        config["mode"] = auto_detect_training_mode(config["dataset_path"])

    # Generate GPT-4 V caption_prefix and masking_prompt if not provided in the config:
    if (not config.get("caption_prefix") or not config.get("masking_prompt")):
        gpt_description, gpt_masking_prompt = describe_image_concept(new_data_dir, config["caption_mode"])
        if not config.get("caption_prefix"):
            config["caption_prefix"] = gpt_description
        if not config.get("masking_prompt"):
            config["masking_prompt"] = gpt_masking_prompt

    if config["mode"] == "face":  # set masking prompt to "face" for face mode
        config["masking_prompt"] = "face"
    if config["mode"] == "style": # disable masking for style transfer
        config["masking_prompt"] = ""
    
    # custom hack / trick: generate unique trigger text from the descriptions by just removing vowels:
    config["lora_trigger_text"] = remove_vowels(config["caption_prefix"])

    # Perform dataset captioning if enabled in the config:
    trigger_token = "NO_TRIGGER_INJECTED"
    if config.get("caption_mode"):
        if "GPT" in config.get("caption_mode"):
            trigger_token = gpt4_v_caption_dataset(
                config["dataset_path"], 
                caption_mode=config["caption_mode"], 
                traininig_mode=config["mode"])
        else:
            florence_caption_dataset(config["dataset_path"], caption_mode=config["caption_mode"])

    # Inject the final trigger text into the prompts:
    if not config.get("caption_mode"):
        config["caption_prefix"] = config["lora_trigger_text"]
    elif "GPT" in config.get("caption_mode"):
        config["caption_prefix"] = ""
        print("================================================")
        print(f"=================== WARNING ===================")
        print(f"Trigger text replacement not implemented yet!!!")
        print("================================================")
    else: # florence2:
        config["caption_prefix"] = config["lora_trigger_text"]
    
    if config.get("masking_prompt"): # generate masks:
        img_filepaths = sorted([os.path.join(new_data_dir, f) for f in os.listdir(new_data_dir) if f.endswith('.jpg')])
        images = [Image.open(f) for f in img_filepaths]
        masks = clipseg_mask_generator(images, config["masking_prompt"])

        # Now, iterate over the images, add the corresponding mask as alpha channel and save the resulting image as png (overwriting the jpg):
        for i, img in enumerate(images):
            img.putalpha(masks[i])
            img.save(img_filepaths[i].replace('.jpg', '.png'), 'PNG')
            os.remove(img_filepaths[i])

        config["alpha_mask"] = True
    else:
        config["alpha_mask"] = False

    # finally update the config:
    config = create_sample_prompts(config)

    # Convert all Path objects to strings before JSON serialization
    serializable_config = path_to_str(config)
    config = construct_toml(config)
    
    with open(Path(config["output_dir"]) / "config.json", 'w') as f:
        json.dump(serializable_config, f, indent=4)


    if verbose:
        print("==========================================")
        print("==========================================")
        print(" ========== Final Train Settings: ========")
        print(f"Caption Prefix: {config['caption_prefix']}")
        print(f"Masking Prompt: {config['masking_prompt']}")
        print(f"Training Mode: {config['mode']}")
        print(f"Alpha Mask: {config['alpha_mask']}")
        print(f"Dataset TOML: {config['dataset_config']}")
        print(" ========== Final Train Captions: ========")
        for subdir, _, files in os.walk(new_data_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(subdir, file), 'r') as f:
                        print(f"{file}: {f.readline().strip()}")
        print(" ========== Final Train Config: ==========")
        print(config)
        print("==========================================")
        print("==========================================")

    return config