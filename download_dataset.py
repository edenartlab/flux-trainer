import os
import shutil
import requests
from tqdm import tqdm
import zipfile
from typing import List
import uuid

def download_file(url: str, dataset_path: str):
    """
    Downloads a file from the provided URL and shows a progress bar.
    """
    local_filename = os.path.join(dataset_path, url.split('/')[-1])
    print(f"Downloading {local_filename}...")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kilobyte per block
            
            with tqdm(total=total_size, unit='iB', unit_scale=True) as bar:
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
        return local_filename
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

def unzip_file(file_path: str, extract_to: str):
    """
    Unzips a zip file, placing all files directly in the extract_to directory with unique names.
    """
    print(f"Unzipping {file_path}...")
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename[-1] == '/':  # Skip directories
                    continue
                file_info.filename = f"{uuid.uuid4()}_{os.path.basename(file_info.filename)}"
                zip_ref.extract(file_info, extract_to)
        print(f"Unzipped {file_path} successfully.")
    except Exception as e:
        print(f"Error unzipping {file_path}: {e}")

def download_dataset(dataset_path: str, dataset_urls: List[str]):
    """
    Downloads and extracts datasets from the provided URLs.
    """
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)
    
    for url in dataset_urls:
        local_file = download_file(url, dataset_path)
        if local_file and local_file.endswith(".zip"):
            unzip_file(local_file, dataset_path)
            os.remove(local_file)  # Remove the zip file after extraction
    
    print(f"Dataset downloaded to {dataset_path}:")
    print(os.listdir(dataset_path))
