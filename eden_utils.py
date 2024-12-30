import io
import os
import random
import base64
import boto3
import hashlib
import mimetypes
import magic
import requests
import warnings
import tempfile
import textwrap
import concurrent.futures
from io import BytesIO
from pydantic import BaseModel
from openai import OpenAI
from typing import Iterator
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv
import gc
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Optional, List, Tuple, Union, Literal, Dict
from transformers import (
    CLIPSegForImageSegmentation,
    CLIPSegProcessor
)

import logging
logger = logging.getLogger(__name__)

load_dotenv()

MONGO_URI=os.getenv("MONGO_URI")
MONGO_DB_NAME_STAGE=os.getenv("MONGO_DB_NAME_STAGE")
MONGO_DB_NAME_PROD=os.getenv("MONGO_DB_NAME_PROD")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_BUCKET_NAME_STAGE = os.getenv("AWS_BUCKET_NAME_STAGE")
AWS_BUCKET_NAME_PROD = os.getenv("AWS_BUCKET_NAME_PROD")


client = MongoClient(MONGO_URI)

s3 = boto3.client(
    's3', 
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION_NAME
)

s3_buckets = {
    "STAGE": AWS_BUCKET_NAME_STAGE,
    "PROD": AWS_BUCKET_NAME_PROD,
}

file_extensions = {
    'audio/mpeg': '.mp3',
    'audio/mp4': '.mp4',
    'audio/flac': '.flac',
    'audio/wav': '.wav',
    'image/jpeg': '.jpg',
    'image/webp': '.webp',
    'image/png': '.png',
    'video/mp4': '.mp4',
    'application/x-tar': '.tar',
    'application/zip': '.zip',
    'application/octet-stream': '.safetensors'
}


def get_collection(collection_name, db):
    db = client[MONGO_DB_NAME_PROD] if db == "PROD" else client[MONGO_DB_NAME_STAGE]
    return db[collection_name]


def PIL_to_bytes(image, ext="JPEG", quality=95):
    """Converts a PIL image to a bytes buffer."""
    if image.mode == 'RGBA' and ext.upper() not in ['PNG', 'WEBP']:
        image = image.convert('RGB')
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=ext, quality=quality)
    return img_byte_arr.getvalue()


def image_to_base64(file_path, max_size):
    """Converts an image to a base64 string."""
    img = Image.open(file_path).convert('RGB')    
    if isinstance(max_size, (int, float)):
        w, h = img.size
        ratio = min(1.0, ((max_size ** 2) / (w * h)) ** 0.5)
        max_size = int(w * ratio), int(h * ratio)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    img_bytes = PIL_to_bytes(img, ext="JPEG", quality=95)
    data = base64.b64encode(img_bytes).decode("utf-8")
    return data

def prep_img_for_gpt_api(image, max_size=(512, 512)):
    """Prepare image for GPT-4 Vision API by resizing and converting to base64."""
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    
    # Resize image while maintaining aspect ratio
    img.thumbnail(max_size)
    
    # Convert to JPEG and then to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def make_slug(task):
    """Makes a slug from a task."""

    task_args = task["args"]
    name = task_args["name"].lower().replace(" ", "-")
    existing_docs = list(models_collection.find({"name": name, "user": task["user"]}))
    versions = [int(doc.get('slug', '').split('/')[-1][1:]) for doc in existing_docs if doc.get('slug')]
    version = max(versions or [0]) + 1
    username = users_collection.find_one({"_id": task["user"]})["username"]
    slug = f"{username}/{name}/v{version}"
    return slug

def get_root_url(db="STAGE"):
    """Returns the root URL for the specified bucket."""
    bucket_name = s3_buckets[db]
    return f"https://{bucket_name}.s3.{AWS_REGION_NAME}.amazonaws.com"
    
    
def upload_file_from_url(url, name=None, file_type=None, db="STAGE"):
    """Uploads a file to an S3 bucket by downloading it to a temporary file and uploading it to S3."""

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile() as tmp_file:
            for chunk in r.iter_content(chunk_size=1024*1024):
                tmp_file.write(chunk)
            tmp_file.flush()
            tmp_file.seek(0)
            return upload_file(tmp_file.name, name, file_type, db)


def upload_file(file_path, name=None, file_type=None, db="STAGE"):
    """Uploads a file to an S3 bucket and returns the file URL."""

    if file_path.startswith('http://') or file_path.startswith('https://'):
        return upload_file_from_url(file_path, name, file_type, db)
    
    with open(file_path, 'rb') as file:
        buffer = file.read()

    return upload_buffer(buffer, name, file_type, db)    


def upload_buffer(buffer, name=None, file_type=None, db="STAGE"):
    """Uploads a buffer to an S3 bucket and returns the file URL."""
    
    assert file_type in [None, '.jpg', '.webp', '.png', '.mp3', 'mp4', '.flac', '.wav', '.safetensors'], \
        "file_type must be one of ['.jpg', '.webp', '.png', '.mp3', 'mp4', '.flac', '.wav', '.safetensors']"

    if isinstance(buffer, Iterator):
        buffer = b"".join(buffer)

    # Get file extension from mimetype
    mime_type = magic.from_buffer(buffer, mime=True)
    originial_file_type = file_extensions.get(mime_type) or mimetypes.guess_extension(mime_type) or f".{mime_type.split('/')[-1]}"
    if not file_type:
        file_type = originial_file_type

    # if it's an image of the wrong type, convert it
    if file_type != originial_file_type and mime_type.startswith('image/'):
        image = Image.open(io.BytesIO(buffer))
        output = io.BytesIO()
        if file_type == '.jpg':
            image.save(output, 'JPEG', quality=95)
            mime_type = 'image/jpeg'
        elif file_type == '.webp':
            image.save(output, 'WEBP', quality=95)
            mime_type = 'image/webp'
        elif file_type == '.png':
            image.save(output, 'PNG', quality=95)
            mime_type = 'image/png'
        buffer = output.getvalue()

    # if no name is provided, use sha256 of content
    if not name:
        hasher = hashlib.sha256()
        hasher.update(buffer)
        name = hasher.hexdigest()
    
    # Upload file to S3
    filename = f"{name}{file_type}"
    file_bytes = io.BytesIO(buffer)
    
    bucket_name = s3_buckets[db]

    s3.upload_fileobj(
        file_bytes, 
        bucket_name, 
        filename, 
        ExtraArgs={'ContentType': mime_type, 'ContentDisposition': 'inline'}
    )

    # Generate and return file URL
    file_url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"
    print(f"==> Uploaded: {file_url}")

    return file_url, name


def make_slug(task, db):
    """Makes a slug from a task."""

    users_collection = get_collection("users3", db)
    models_collection = get_collection("models3", db)
    
    task_args = task["args"]
    name = task_args["name"].lower().replace(" ", "-")
    username = users_collection.find_one({"_id": task["user"]})["username"]
    existing_docs = list(models_collection.find({"slug": {"$regex": f"^{username}/{name}/v"}}))
    versions = [int(doc['slug'].split('/')[-1][1:]) for doc in existing_docs if doc.get('slug')]
    version = max(versions or [0]) + 1
    slug = f"{username}/{name}/v{version}"
    return slug


def print_gpu_memory():
    """
    Print GPU memory usage using only PyTorch's native methods.
    """
    try:
        if not torch.cuda.is_available():
            logger.info("CUDA not available")
            return

        for gpu_id in range(torch.cuda.device_count()):
            # Get memory stats for this GPU
            with torch.cuda.device(gpu_id):
                allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                free = total - allocated
                
                # Get device name
                device_name = torch.cuda.get_device_name(gpu_id)
                
                logger.info(f"\nGPU {gpu_id} ({device_name}):")
                logger.info(f"  Used Memory: {allocated:.2f}GB")
                logger.info(f"  Reserved Memory: {reserved:.2f}GB")
                logger.info(f"  Free Memory: {free:.2f}GB")
                logger.info(f"  Total Memory: {total:.2f}GB")
                logger.info(f"  Memory Utilization: {(allocated/total)*100:.1f}%")

    except Exception as e:
        logger.info(f"Error getting GPU memory usage: {e}")


def create_thumbnail(
    config: dict,
    width: int = 1024,
    height: int = 1024,
    n_steps: int = 35,
    n_imgs: int = 4,
    db: str = "STAGE"
) -> Optional[str]:
    """
    Creates a thumbnail grid with generated samples from the trainer LoRA.
    
    Args:
        config (dict): Configuration dictionary containing mode and output_dir
        width (int): Width of each generated image
        height (int): Height of each generated image
        n_steps (int): Number of steps for generation
        n_imgs (int): Number of images to generate (must be a perfect square)
        env (str): Environment for upload ("STAGE" or "PROD")
    
    Returns:
        Optional[str]: URL of the uploaded thumbnail grid image, or None if creation fails
    
    Raises:
        ValueError: If n_imgs is not a perfect square or if config is invalid
        FileNotFoundError: If required files or directories are missing
    """

    # Print free/used gpu memory using torch cuda:
    print("-----------------------------------------------------")
    print("PRE-thumbnail creation:")
    print_gpu_memory()

    try:
        # Validate inputs
        grid_size = int(n_imgs ** 0.5)
        if grid_size * grid_size != n_imgs:
            raise ValueError(f"n_imgs ({n_imgs}) must be a perfect square")
        
        # Determine prompt file based on mode
        prompt_files = {
            "face": "template/grid_prompts_face.txt",
            "object": "template/grid_prompts_object.txt",
            "style": "template/grid_prompts_style.txt"
        }
        prompt_file = prompt_files.get(config["mode"])

        # Create temporary directories and files with context managers
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            # Sample and clean prompts
            with open(prompt_file, 'r') as f:
                prompt_samples = [p.strip() for p in f.readlines() if p.strip()]
                
            selected_prompts = random.sample(prompt_samples, n_imgs)
            tmp_file.writelines(f"{p}\n" for p in selected_prompts)

        # Create temporary directory for generated images
        with tempfile.TemporaryDirectory() as sample_dir:
            # Find most recent LoRA model
            output_dir = Path(config["output_dir"])
            safetensor_files = list(output_dir.glob("*.safetensors"))
            lora_path = max(safetensor_files, key=lambda p: p.stat().st_mtime)

            print(f"Generating validation grid of {n_imgs} imgs with {lora_path}")

            # Prepare generation command
            cmd = [
                "python", "lora_batch_eval.py",
                "--ckpt_path", "models/flux1-dev.safetensors",
                "--clip_l", "models/clip_l.safetensors",
                "--t5xxl", "models/t5xxl_fp16.safetensors",
                "--ae", "models/ae.safetensors",
                "--prompt_file", tmp_file.name,
                "--output_dir", sample_dir,
                "--lora_path", str(lora_path),
                "--offload",
                "--merge_lora_weights",
                "--steps", str(n_steps),
                "--width", str(width),
                "--height", str(height)
            ]

            # Run generation command (assuming subprocess.run is imported)
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                import subprocess

                process_env = os.environ.copy()
                process_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    env=process_env  # Pass the modified environment
                )
                logging.info(f"Generation command output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Generation command failed: {e.stderr}")
                raise

            # Create image grid
            png_files = list(Path(sample_dir).glob("*.jpg"))
            sampled_files = random.sample(png_files, n_imgs)
            images = [Image.open(f) for f in sampled_files]

            # Calculate grid dimensions
            grid_width = max(img.size[0] for img in images) * grid_size
            grid_height = max(img.size[1] for img in images) * grid_size
            grid_img = Image.new('RGB', (grid_width, grid_height))

            # Place images in grid
            for i, img in enumerate(images):
                x = (i % grid_size) * width
                y = (i // grid_size) * height
                img = img.resize((width, height))
                grid_img.paste(img, (x, y))

            # Save and upload grid
            grid_path = output_dir / "sample_grid.jpg"
            grid_img.save(grid_path, format="JPEG", quality=60)

            try:
                thumbnail_url, _ = upload_file(str(grid_path), db=db)
                print(f"----> Thumbnail URL: {thumbnail_url}")
                return thumbnail_url
            except Exception as e:
                logging.error(f"Failed to upload thumbnail: {e}")
                raise

    except Exception as e:
        logging.error(f"Thumbnail creation failed: {e}")
        return None
    finally:
        if 'tmp_file' in locals():
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


def gpt4_v_caption_dataset(
    dataset_dir,               
    caption_mode="<GPT_CAPTION>",  
    traininig_mode="style",
    lora_trigger_text=None,
    keep_existing_captions=True,  
    batch_size=4
):
    """
    Generate captions for a dataset of images using GPT-4 Vision.
    
    Args:
        dataset_dir (str): Directory containing images to caption
        caption_mode (str): Caption generation mode (maintains API consistency with Florence)
        keep_existing_captions (bool): If True, skip images that already have captions
        batch_size (int): Number of concurrent API requests
        
    Returns:
        None
    """
    # NEW: Get image paths based on keep_existing_captions flag
    if keep_existing_captions:
        image_paths, caption_paths = get_image_paths(dataset_dir, skip_captioned=True)
        print(f"Found {len(image_paths)} images without captions")
        print(f"Found {len(caption_paths)} existing captions to preserve")
    else:
        image_paths = get_image_paths(dataset_dir, skip_captioned=False)
        print(f"Processing all {len(image_paths)} images")

    if not image_paths:
        print("No images to caption.")
        return

    # Load all images and prepare captions list
    images = []
    captions = []
    for path in image_paths:
        images.append(Image.open(path).convert("RGB"))
        caption_path = f"{os.path.splitext(path)[0]}.txt"
        if os.path.exists(caption_path) and keep_existing_captions:
            with open(caption_path, 'r') as f:
                captions.append(f.read().strip())
        else:
            captions.append(None)

    if caption_mode == "<GPT_CAPTION>":
        n_words = 15
    elif caption_mode == "<GPT_DETAILED_CAPTION>":
        n_words = 25
    elif caption_mode == "<GPT_MORE_DETAILED_CAPTION>":
        n_words = 35

    generic_prompt = textwrap.dedent(f"""Concisely describe the content of this image without any assumptions with at most {n_words} words. 
        Don't start with statements like 'The image features...', just describe what you see. 
        The description is a dataset caption that will be used for training a generative AI model (FLUX).""")

    # Adjust prompt based on caption_mode
    if traininig_mode == "style":
        trigger_token = "REF_STYLE"
        base_prompt = textwrap.dedent(f"""
            Ignore the aesthetic and specific style of the image, just focus on the content (objects, characters, composition, actions, ...).
            The style is simply referred to as 'REF_STYLE' and the description should always contain the text '{trigger_token}' as a stylistic hint / modifier.
            """)
    elif traininig_mode == "object":
        trigger_token = "REF_CONCEPT"
        base_prompt = textwrap.dedent(f"""
            Always refer to the main object/concept as 'REF_CONCEPT'. As such the prompt should always contain the trigger token '{trigger_token}' and mainly focus on the style, context and composition instead of on the visual elements that constitute {trigger_token}.
            """)
    elif traininig_mode == "face":
        trigger_token = "REF_CHARACTER"
        base_prompt = textwrap.dedent(f"""
            Always refer to the main subject as 'REF_CHARACTER'. As such the prompt should always contain the trigger token '{trigger_token}' and mainly focus on the style, context and composition instead of on the visual elements that constitute {trigger_token}.
            """)

    base_prompt = generic_prompt + "\n" + base_prompt + "\nReply with just the image description, nothing else!"
    print(f"----- Full chatgpt prompt: -----")
    print(base_prompt)

    client = OpenAI()

    class ImageCaption(BaseModel):
        """Caption for a single image."""
        caption: str
    
    def process_single_image(index, image):
        """Process a single image and return its caption."""
        try:
            base64_image = prep_img_for_gpt_api(image)
            
            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced image captioner that generates concise, descriptive captions for images."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": base_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
                response_format=ImageCaption,
            )
            
            result = ImageCaption.parse_raw(response.choices[0].message.content)
            caption = result.caption.strip().rstrip('.').rstrip(',')

            if lora_trigger_text:
                caption = caption.replace(trigger_token, lora_trigger_text)
            
            caption = caption.strip().rstrip('.').rstrip(',')
            
            # Save caption immediately after generation
            caption_path = f"{os.path.splitext(image_paths[index])[0]}.txt"
            with open(caption_path, "w") as f:
                f.write(caption)
            
            return index, caption
            
        except Exception as e:
            print(f"Error processing image {index}: {str(e)}")
            return index, None
    
    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_index = {
            executor.submit(process_single_image, i, img): i 
            for i, img in enumerate(images) 
            if captions[i] is None
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                idx, caption = future.result()
                captions[idx] = caption
                print(f"Caption {idx + 1}/{len(images)}: {caption}")
            except Exception as exc:
                print(f"Caption generation for image {index + 1} failed with exception: {exc}")
    
    # Cleanup
    del client
    gc.collect()
    
    return trigger_token


def describe_image_concept(images_dir, mode, n=6):
    """Gets both a detailed and concise description of the main visual concept in a set of images."""    
    client = OpenAI()
    
    # Get the list of image files in the directory
    image_files = os.listdir(images_dir)
    image_files = [os.path.join(images_dir, f) for f in image_files 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    
    n = min(n, len(image_files))
    selected_images = random.sample(image_files, n)

    class ImageDescriptions(BaseModel):
        """Detailed and concise descriptions of the main concept in the images."""
        detailed_description: str
        short_description: str

    image_attachments = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_base64(image_path, max_size=512)}",
                "detail": "low"
            },
        }
        for image_path in selected_images
    ]

    if mode == "style":
        gpt_task_description = """Provide two descriptions of the shared style in these images:
1. A detailed visual description of the shared style/aesthetic in these images of maximum 10 words (ignore the specific objects or characters in each image, just focus on the style). Always start with the main category of the images (cartoon, lineart sketch, painting, photograph, ...) followed by the key visual features (like stylistic hints, colors, shapes, ...) of the shared visual aesthetic, avoiding abstract words or interpretations. Your description should help someone generate a specific, representative example of the style.
2. A more concise description (max 4 words) that captures just the essentials of the visual style."""
    else:
        gpt_task_description = """Provide two descriptions of the shared concept in these images:
1. A detailed visual description of the shared concept (object / character / person / ...) in these images of maximum 10 words. Always start with the main category of the thing (man, character, car, ...) followed by the key visual features (like colors, shapes, accessories, expressions, style, ...) of the central subject, avoiding abstract words or interpretations. Your description should help someone generate a specific, representative example of the concept. Use precise, observable terms - for example, describe 'red' instead of 'colorful', 'standing upright' instead of 'positioned', 'wearing a blue hat' instead of 'accessorized'. Avoid describing actions, emotions or contexts. Ignore any aspect of the main concept that varies across examples (therefore never use words like 'or' or 'various' in the description), the goal is to create a clear mental picture of the archetypal instance of what's shown through a single description that captures the visual, common essence of the shared concept in the images.
2. A concise description (max 4 words) that captures just the essential visual concept / base category / class and will be used to generate masks through CLIPSegmentation."""

    response = client.beta.chat.completions.parse(
        model="gpt-4o",  # Using the correct vision model
        messages=[
            {
                "role": "system",
                "content": "You carefully investigate the visual commonalities between presented images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": gpt_task_description
                    },
                    *image_attachments
                ],
            },
        ],
        response_format=ImageDescriptions,
    )
    
    detailed = response.choices[0].message.parsed.detailed_description.lower()
    short = response.choices[0].message.parsed.short_description.lower()
    
    # remove any trailing punctuation from both descriptions
    detailed = detailed.strip().rstrip('.').rstrip(',')
    short = short.strip().rstrip('.').rstrip(',')

    print(f"Detailed description: {detailed}")
    print(f"Short description: {short}")
    
    return detailed, short

from pydantic import BaseModel, field_validator

def auto_detect_training_mode(images_dir, n_img_samples = 6):
    """
    Analyzes sample images to determine the appropriate LoRA training mode.
    Returns one of: "style", "object", or "face"
    """
    client = OpenAI()

    # Get the list of image files in the training directory
    image_files = os.listdir(images_dir)
    image_files = [os.path.join(images_dir, f) for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    n = min(n_img_samples, len(image_files))
    selected_images = random.sample(image_files, n)

    class ImageAnalysis(BaseModel):
        """
        Analyze what the images predominantly depict:
        - If they show primarily one person's face, select 'face'
        - If they show primarily one specific object, character, or thing, select 'object'
        - If they show primarily an artistic style, aesthetic, or diverse subjects in a consistent style, select 'style'
        """
        mode: Literal['face', 'object', 'style']

    image_attachments = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_base64(image_path, max_size=512)}",
                "detail": "low"
            },
        }
        for image_path in selected_images
    ]

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": "Analyze images to determine the appropriate LoRA training mode, which is one of 'face', 'object', or 'style'."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": """Look at the attached images and determine their primary content type (used for LoRA training mode):
                        - Select 'face' if they mostly show pictures of the same person/character's face and the face is the primary thing that remains consistent across images.
                        - Select 'object' if they mostly show a specific object, character, or thing. Character mode should be prefered over 'face' if the entire body of the character should be learned.
                        - Select 'style' if they mostly demonstrate a consistent artistic style or aesthetic across diverse subjects or scenes. This mode disables segmentation masks and learns from all the pixels in the images instead of just the main foreground subject."""
                    },
                    *image_attachments
                ],            
            },
        ],
        response_format=ImageAnalysis,
    )

    mode = response.choices[0].message.parsed.mode
    print(f"----> Auto detected training mode: {mode}")
    
    return mode



#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    try:
        imports.remove("flash_attn")
    except:
        pass
    return imports

def get_image_paths(dataset_dir, skip_captioned=True):
    """
    Get paths to images in dataset_dir. Optionally skip images that already have captions.
    
    Args:
        dataset_dir (str): Directory containing images and captions
        skip_captioned (bool): If True, skip images that already have matching .txt files
        
    Returns:
        tuple: (image_paths, caption_paths) if skip_captioned=True, else just image_paths
    """
    image_paths = []
    caption_paths = []
    existing_captions = set()

    # First collect all caption files if we need to skip them
    if skip_captioned:
        for root, _, files in os.walk(dataset_dir):
            for file in sorted(files):
                if file.lower().endswith('.txt'):
                    caption_paths.append(os.path.join(root, file))
                    existing_captions.add(os.path.splitext(file)[0])

    # Then collect image files
    for root, _, files in os.walk(dataset_dir):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                basename = os.path.splitext(file)[0]
                if not skip_captioned or basename not in existing_captions:
                    image_paths.append(os.path.join(root, file))
    
    return (image_paths, caption_paths) if skip_captioned else image_paths

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@torch.no_grad()
def florence_caption_dataset(dataset_dir, 
        caption_mode="<CAPTION>",
        keep_existing_captions=True,
        florence_model_path="./models",
        batch_size=1):
    """
    Generate captions for images using Florence-2 model.
    
    Args:
        dataset_dir (str): Directory containing images to caption
        caption_mode (str): Caption generation mode for Florence model: <CAPTION> / <DETAILED_CAPTION> / <MORE_DETAILED_CAPTION>
        keep_existing_captions (bool): If True, skip images that already have captions
        florence_model_path (str): Path to store/load Florence model
        batch_size (int): Batch size for processing images
    """
    # Get paths to process
    if keep_existing_captions:
        image_paths, caption_paths = get_image_paths(dataset_dir, skip_captioned=True)
        print(f"Found {len(image_paths)} images without captions")
        print(f"Found {len(caption_paths)} existing captions to preserve")
    else:
        image_paths = get_image_paths(dataset_dir, skip_captioned=False)
        print(f"Processing all {len(image_paths)} images")

    if not image_paths:
        print("No images to caption.")
        return

    # Load model and processor
    os.makedirs(florence_model_path, exist_ok=True)
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            attn_implementation="sdpa",
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            cache_dir=florence_model_path
        )
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        cache_dir=florence_model_path
    )

    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        
        # Generate captions
        inputs = processor(
            text=[caption_mode] * len(batch_images),
            images=batch_images,
            return_tensors="pt",
            padding=True
        ).to(device, torch_dtype)
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=4
        )

        # Process and save captions
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
        parsed_answers = [
            processor.post_process_generation(text, task=caption_mode, image_size=(img.width, img.height))
            for text, img in zip(generated_texts, batch_images)
        ]
        
        for path, parsed_answer in zip(batch_paths, parsed_answers):
            caption = parsed_answer[caption_mode].replace("<pad>", "").replace("The image shows a ", "A ")
            caption_path = f"{os.path.splitext(path)[0]}.txt"
            with open(caption_path, "w") as f:
                f.write(caption)

    # Cleanup
    model.to('cpu')
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return


def load_image_with_orientation(path, mode="RGB"):
    image = Image.open(path)

    # Try to get the Exif orientation tag (0x0112), if it exists
    try:
        exif_data = image._getexif()
        orientation = exif_data.get(0x0112)
    except (AttributeError, KeyError, IndexError):
        orientation = None

    # Apply the orientation, if it's present
    if orientation:
        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            image = image.rotate(90, expand=True)

    if image.mode == 'P':
        image = image.convert('RGBA')
    if image.mode == 'CMYK':
        image = image.convert('RGB')

    # Remove alpha channel if present
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        image = background

    # Convert to the desired mode
    return image.convert(mode)


@torch.no_grad()
@torch.amp.autocast('cuda')
def clipseg_mask_generator(
    images: List[Image.Image],
    target_prompts: Union[List[str], str],
    model_id: Literal[
        "CIDAS/clipseg-rd64-refined", "CIDAS/clipseg-rd16"
    ] = "CIDAS/clipseg-rd64-refined",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bias: float = 0.0,
    temp: float = 0.75,
    cache_dir="./models",
    **kwargs,
) -> List[Image.Image]:
    """
    Returns a greyscale mask for each image based on the target_prompt.
    """
    warnings.filterwarnings("ignore", message=".*not valid for.*ViTImageProcessor.*")
    print(f"Generating CLIPSeg masks for {len(images)} images...", flush=True)

    if isinstance(target_prompts, str):
        print(f'Using "{target_prompts}" as CLIP-segmentation prompt for all images.')
        target_prompts = [target_prompts] * len(images)

    model = None
    masks = []

    if any(target_prompts):
        processor = CLIPSegProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        model = CLIPSegForImageSegmentation.from_pretrained(model_id, cache_dir=cache_dir)
        model = model.to(device)
    else:
        print("No masking prompts provided, returning blank masks.")
        return [Image.new("L", img.size, 255) for img in images]

    for image, prompt in tqdm(zip(images, target_prompts)):
        original_size = image.size

        if prompt != "":
            inputs = processor(
                text=[prompt, ""],
                images=[image] * 2,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits / temp, dim=0)[0]
            probs = (probs + bias).clamp_(0, 1)
            probs = 255 * probs / probs.max()

            # make mask greyscale
            mask = Image.fromarray(probs.cpu().numpy()).convert("L")
            # resize mask to original size
            mask = mask.resize(original_size)
        else:
            mask = Image.new("L", original_size, 255)

        masks.append(mask)

    # cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return masks