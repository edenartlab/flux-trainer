import os
import shutil
import requests
from tqdm import tqdm
import zipfile
from typing import List

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
    Unzips a zip file using Python's zipfile module.
    """
    print(f"Unzipping {file_path}...")
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzipped {file_path} successfully.")
    except Exception as e:
        print(f"Error unzipping {file_path}: {e}")

def download_dataset(dataset_urls: List[str]):
    """
    Downloads and extracts datasets from the provided URLs.
    """
    dataset_path = "app/dataset"
    if os.path.exists(dataset_path):
        print("DELETED EXISTING FOLDER")
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)
    
    for url in dataset_urls:
        local_file = download_file(url, dataset_path)
        if local_file and local_file.endswith(".zip"):
            unzip_file(local_file, dataset_path)
            os.remove(local_file)  # Remove the zip file after extraction
    
    print("Dataset downloaded:")
    print(os.listdir(dataset_path))
