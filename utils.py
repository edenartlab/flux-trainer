import os
import json
import random
from tqdm import tqdm
import torch
import gc
from PIL import Image
import shutil

#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def get_image_and_caption_paths(dataset_dir):
    image_paths = []
    caption_paths = []

    # Walk through all subdirectories and files in dataset_dir
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
            elif file.lower().endswith('.txt'):
                caption_paths.append(os.path.join(root, file))

    # Sort the paths for consistency
    image_paths.sort()
    caption_paths.sort()

    return image_paths, caption_paths

@torch.no_grad()
def florence_caption_dataset(dataset_dir, 
        caption_mode="<CAPTION>",
        florence_model_path="./models",
        batch_size=1):

    os.makedirs(florence_model_path, exist_ok=True)
    image_paths, caption_paths = get_image_and_caption_paths(dataset_dir)

    print(f"Found {len(image_paths)} images and {len(caption_paths)} txt files.")
    if len(caption_paths):
        print(f"WARNING: This script will overwrite the existing txt files!!")
    print(f"Captioning {len(image_paths)} images...")

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", attn_implementation="sdpa", device_map=device, torch_dtype=torch_dtype, trust_remote_code=True, cache_dir=florence_model_path)
            
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True, cache_dir=florence_model_path)

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        
        inputs = processor(text=[caption_mode] * len(batch_images), images=batch_images, return_tensors="pt", padding=True).to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=4
        )

        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
        parsed_answers = [processor.post_process_generation(text, task=caption_mode, image_size=(img.width, img.height)) for text, img in zip(generated_texts, batch_images)]
        
        for path, parsed_answer in zip(batch_paths, parsed_answers):
            caption = parsed_answer[caption_mode].replace("The image shows a ", "A ")
            caption = parsed_answer[caption_mode].replace("<pad>", "")
            basename = os.path.splitext(os.path.basename(path))[0]
            dirname  = os.path.dirname(path)
            with open(f"{os.path.join(dirname, basename)}.txt", "w") as f:
                f.write(caption)

        # Close images to free up memory
        for img in batch_images:
            img.close()

    model.to('cpu')
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return

def prep_dataset(root_directory):
    error_dir = os.path.join(os.path.dirname(root_directory), 'errors')
    os.makedirs(error_dir, exist_ok=True)
    
    print("Preparing dataset folder {root_directory}...")
    total_imgs, resized = 0, 0

    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                # Try loading the file as an image and converting it to RGB
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    
                    if max(img.width, img.height) > 2048:
                        # Resize the image with max width/height of 2048
                        img.thumbnail((2048, 2048), Image.LANCZOS)
                        resized += 1
                    
                    # Save the image as .jpg
                    new_filename = os.path.splitext(file)[0] + '.jpg'
                    new_file_path = os.path.join(subdir, new_filename)
                    img.save(new_file_path, 'JPEG', quality=95)
                    total_imgs += 1
                
                # Delete the original file
                os.remove(file_path)
            except Exception as e:
                # If there was any error, move the file to the errors directory
                print(f"Error processing {file_path}: {e}")
                shutil.move(file_path, os.path.join(error_dir, file))
                print(f"Moved {file} to {error_dir}")

    print(f"{total_imgs} imgs in {root_directory} converted to .jpg Resized {resized} images.")


if __name__ == "__main__":
    folder_path = "/data/xander/Projects/cog/GitHub_repos/flux-trainer/test"
    caption_mode = "<CAPTION>"
    batch_size = 1
    florence_caption_dataset(folder_path, batch_size=batch_size, caption_mode = caption_mode)