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
db = client[MONGO_DB_NAME_STAGE]
models_collection = db["models"]
tasks_collection = db["tasks2"]
users_collection = db["users"]


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


def create_thumbnail(sample_dir, env="STAGE"):
    """Creates a thumbnail from a sample directory."""

    png_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
    
    if len(png_files) < 4:
        print("Not enough sample images to create a 2x2 grid.")
        return None

    sampled_files = random.sample(png_files, 4)
    images = [Image.open(os.path.join(sample_dir, f)) for f in sampled_files]
    img_size = images[0].size[0]
    grid_img = Image.new('RGB', (img_size * 2, img_size * 2))

    for i, img in enumerate(images):
        img = img.resize((img_size, img_size))
        grid_img.paste(img, ((i % 2) * img_size, (i // 2) * img_size))

    grid_img.save(f"{sample_dir}.png")

    thumbnail_url, _ = upload_file(
        f"{sample_dir}.png",
        env=env
    )

    return thumbnail_url
        

def make_slug(task):
    """Makes a slug from a task."""

    task_args = task["args"]
    name = task_args["name"].lower().replace(" ", "-")
    version = 1 + models_collection.count_documents({"name": name, "user": task["user"]}) 
    username = users_collection.find_one({"_id": task["user"]})["username"]
    slug = f"{username}/{name}/v{version}"
    return slug


def check_if_face(images_dir):
    """Checks if a set of images depict a face."""
    
    client = OpenAI()

    # Get the list of image files in the training directory
    image_files = os.listdir(images_dir)
    image_files = [os.path.join(images_dir, f) for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    n = min(4, len(image_files))
    selected_images = random.sample(image_files, n)
    print("check these images for faces:", selected_images)

    class IsFace(BaseModel):
        """
        Decide whether the attached images are predominantly of a person's face or not. If the images predominantly depict literally anything else other than a person's face, select false.
        """
        is_face: bool

    image_attachments = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_to_base64(image_path, max_size=512)}"
            },
        }
        for image_path in selected_images
    ]

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "Decide whether the attached images depict a person's face or not."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Look at the attached images. If the images predominantly depict the face of a single person, select true. If the images are predominantly of anything else, select false."
                    },
                    *image_attachments
                ],            
            },
        ],
        response_format=IsFace,
    )

    is_face = response.choices[0].message.parsed
    if is_face:
        print("Images are predominantly of a person's face.")
    else:
        print("Images are not of a face, assigning style mode.")

    return is_face
