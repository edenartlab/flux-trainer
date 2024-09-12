import os
from PIL import Image
import shutil

def prep_dataset(root_directory):
    error_dir = os.path.join(os.path.dirname(root_directory), 'errors')
    os.makedirs(error_dir, exist_ok=True)

    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                # Try loading the file as an image and converting it to RGB
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    
                    # Resize the image with max width/height of 2048
                    img.thumbnail((2048, 2048), Image.LANCZOS)
                    
                    # Save the image as .jpg
                    new_filename = os.path.splitext(file)[0] + '.jpg'
                    new_file_path = os.path.join(subdir, new_filename)
                    img.save(new_file_path, 'JPEG', quality=95)
                
                # Delete the original file
                os.remove(file_path)
            except Exception as e:
                # If there was any error, move the file to the errors directory
                print(f"Error processing {file_path}: {e}")
                shutil.move(file_path, os.path.join(error_dir, file))

if __name__ == "__main__":
    prep_dataset('/data/xander/Projects/cog/GitHub_repos/flux-trainer/test')
