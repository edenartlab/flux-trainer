import io
import os
import random
import base64
import boto3
import hashlib
import mimetypes
import magic
import requests
import tempfile
from io import BytesIO
from pydantic import BaseModel
from openai import OpenAI
from typing import Iterator
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

MONGO_URI=os.getenv("MONGO_URI")
MONGO_DB_NAME_STAGE=os.getenv("MONGO_DB_NAME_STAGE")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_BUCKET_NAME_STAGE = os.getenv("AWS_BUCKET_NAME_STAGE")
AWS_BUCKET_NAME_PROD = os.getenv("AWS_BUCKET_NAME_PROD")

client = MongoClient(MONGO_URI)
try:
    db = client[MONGO_DB_NAME_STAGE]
    models_collection = db["models"]
    tasks_collection = db["tasks2"]
    users_collection = db["users"]
except: # allows running the main trainer code without interacting with our mongo db
    db = None
    models_collection = None
    tasks_collection = None
    users_collection = None


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


def get_root_url(env="STAGE"):
    """Returns the root URL for the specified bucket."""
    bucket_name = s3_buckets[env]
    return f"https://{bucket_name}.s3.{AWS_REGION_NAME}.amazonaws.com"
    
    
def upload_file_from_url(url, name=None, file_type=None, env="STAGE"):
    """Uploads a file to an S3 bucket by downloading it to a temporary file and uploading it to S3."""

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile() as tmp_file:
            for chunk in r.iter_content(chunk_size=1024*1024):
                tmp_file.write(chunk)
            tmp_file.flush()
            tmp_file.seek(0)
            return upload_file(tmp_file.name, name, file_type, env)


def upload_file(file_path, name=None, file_type=None, env="STAGE"):
    """Uploads a file to an S3 bucket and returns the file URL."""

    if file_path.startswith('http://') or file_path.startswith('https://'):
        return upload_file_from_url(file_path, name, file_type, env)
    
    with open(file_path, 'rb') as file:
        buffer = file.read()

    return upload_buffer(buffer, name, file_type, env)    


def upload_buffer(buffer, name=None, file_type=None, env="STAGE"):
    """Uploads a buffer to an S3 bucket and returns the file URL."""
    
    assert file_type in [None, '.jpg', '.webp', '.png', '.mp3', 'mp4', '.flac', '.wav'], \
        "file_type must be one of ['.jpg', '.webp', '.png', '.mp3', 'mp4', '.flac', '.wav']"

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
    
    bucket_name = s3_buckets[env]

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


import os
import random
import tempfile
from pathlib import Path
from PIL import Image
from typing import Optional, List, Tuple
import logging

def create_thumbnail(
    config: dict,
    width: int = 1024,
    height: int = 1024,
    n_steps: int = 35,
    n_imgs: int = 4,
    env: str = "STAGE"
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
                import subprocess
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logging.info(f"Generation command output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Generation command failed: {e.stderr}")
                raise

            # Create image grid
            png_files = list(Path(sample_dir).glob("*.png"))
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
            grid_path = f"{sample_dir}_grid.png"
            grid_img.save(grid_path)

            try:
                thumbnail_url, _ = upload_file(grid_path, env=env)
                return thumbnail_url
            except Exception as e:
                logging.error(f"Failed to upload thumbnail: {e}")
                raise

    except Exception as e:
        logging.error(f"Thumbnail creation failed: {e}")
        return None
    finally:
        # Cleanup temporary files
        if 'tmp_file' in locals():
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
        

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

def describe_image_concept(images_dir, mode):
    """Gets both a detailed and concise description of the main visual concept in a set of images."""    
    client = OpenAI()
    
    # Get the list of image files in the directory
    image_files = os.listdir(images_dir)
    image_files = [os.path.join(images_dir, f) for f in image_files 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    
    n = min(6, len(image_files))
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
        gpt_task_description = """Provide two descriptions of the shared concept in these images:

1. A detailed visual description of the shared style/aesthetic in these images of maximum 10 words. Always start with the main category of the images (cartoon, lineart sketch, painting, photograph, ...) followed by the key visual features (like colors, shapes, stylistic hints, ...) of the shared visual aesthetic, avoiding abstract words or interpretations. Your description should help someone generate a specific, representative example of the style.

2. A more concise description (max 5 words) that captures just the essentials of the visual style."""
    else:
        gpt_task_description = """Provide two descriptions of the shared concept in these images:

1. A detailed visual description of the shared concept (object / character / person / ...) in these images of maximum 10 words. Always start with the main category of the thing (man, character, car, ...) followed by the key visual features (like colors, shapes, accessories, expressions, style, ...) of the central subject, avoiding abstract words or interpretations. Your description should help someone generate a specific, representative example of the concept. Use precise, observable terms - for example, describe 'red' instead of 'colorful', 'standing upright' instead of 'positioned', 'wearing a blue hat' instead of 'accessorized'. Avoid describing actions, emotions or contexts. Ignore any aspect of the main concept that varies across examples (therefore never use words like 'or' or 'various' in the description), the goal is to create a clear mental picture of the archetypal instance of what's shown through a single description that captures the visual, common essence of the shared concept in the images.

2. A concise description (max 5 words) that captures just the essential visual concept and will be used to generate masks through CLIPSegmentation."""

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
    print("Analyzing images for training mode:", selected_images)

    class ImageAnalysis(BaseModel):
        """
        Analyze what the images predominantly depict:
        - If they show primarily one person's face, select 'face'
        - If they show primarily one specific object, character, or thing, select 'object'
        - If they show primarily an artistic style, aesthetic, or diverse subjects in a consistent style, select 'style'
        """
        mode: str

        @field_validator('mode')
        def validate_mode(cls, v):
            if v not in ['face', 'object', 'style']:
                raise ValueError('mode must be one of: face, object, style')
            return v

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
    print(f"Detected training mode: {mode}")
    
    return mode