import random
from pathlib import Path
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm
import concurrent.futures

def hue_augmentation(image, hue_change_max=4):
    """
    Apply hue augmentation to the input image.

    :param image: PIL Image object
    :param hue_change_max: Maximum amount to change the hue
    :return: Augmented PIL Image object
    """
    hue_change = random.uniform(1, hue_change_max)
    # Convert the image to HSV color space
    hsv_image = image.convert('HSV')

    # Split into individual channels
    h, s, v = hsv_image.split()

    # Apply the hue change
    h = h.point(lambda i: (i + hue_change) % 256)
    hsv_image = Image.merge('HSV', (h, s, v))
    return hsv_image.convert('RGB')

def color_jitter(image):
    """
    Apply color jittering (brightness, contrast, color) to the image.
    
    :param image: PIL Image object
    :return: Augmented PIL Image object
    """
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
    factor_ranges = [[0.9, 1.1], [0.9, 1.25], [0.9, 1.2]]
    for i, enhancer in enumerate(enhancers):
        low, high = factor_ranges[i]
        factor = random.uniform(low, high)
        image = enhancer(image).enhance(factor)
    return image

def random_crop(image, scale=(0.85, 0.95)):
    """
    Apply random cropping to the image.
    
    :param image: PIL Image object
    :param scale: Tuple of (min_scale, max_scale) determining crop size
    :return: Augmented PIL Image object
    """
    width, height = image.size
    new_width, new_height = width * random.uniform(*scale), height * random.uniform(*scale)

    left = random.uniform(0, width - new_width)
    top = random.uniform(0, height - new_height)
    
    # Ensure crop coordinates are integers
    left, top, right, bottom = int(left), int(top), int(left + new_width), int(top + new_height)
    
    # Resize back to original dimensions
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((width, height), Image.LANCZOS)

def augment_image(image, enable_lr_flips=False):
    """
    Apply a series of augmentations to the image.
    
    :param image: PIL Image object
    :param enable_lr_flips: Whether to enable horizontal flips
    :return: Augmented PIL Image object
    """
    # Add random horizontal flip with 50% probability if enabled
    if enable_lr_flips and random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
    image = hue_augmentation(image)
    image = color_jitter(image)
    image = random_crop(image)
    
    return image

def process_image(img_path, output_dir, n_augmentations, base_name, enable_lr_flips=False):
    """
    Process a single image with augmentations and save the results.
    
    :param img_path: Path to the image
    :param output_dir: Directory to save augmented images
    :param n_augmentations: Number of augmentations to create per image
    :param base_name: Base name for the augmented images
    :param enable_lr_flips: Whether to enable horizontal flips
    :return: None
    """
    try:
        image = Image.open(img_path)
        
        # Create augmentations
        for i in range(n_augmentations):
            aug_image = augment_image(image, enable_lr_flips)
            aug_filename = f"{base_name}_aug{i+1}.jpg"
            aug_path = output_dir / aug_filename
            aug_image.save(aug_path, quality=95)
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def augment_dataset(dataset_path, n_augmentations_per_img=3, enable_lr_flips=False):
    """
    Augment all images in the specified dataset path.
    
    :param dataset_path: Path to the directory containing images
    :param n_augmentations_per_img: Number of augmentations to create per image
    :param enable_lr_flips: Whether to enable horizontal flips
    :return: Total number of images after augmentation
    """
    dataset_path = Path(dataset_path)
    output_dir = dataset_path
    
    # Get all jpg images
    image_paths = list(dataset_path.glob("*.jpg"))
    orig_n_imgs = len(image_paths)
    print(f"Augmenting dataset at {dataset_path}, original size: {orig_n_imgs} images")
    print(f"Horizontal flips: {'Enabled' if enable_lr_flips else 'Disabled'}")
    
    # Process images with progress bar
    with tqdm(total=orig_n_imgs) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            
            for img_path in image_paths:
                base_name = img_path.stem
                future = executor.submit(
                    process_image, 
                    img_path, 
                    output_dir, 
                    n_augmentations_per_img, 
                    base_name,
                    enable_lr_flips
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
    
    # Count the number of augmented images
    augmented_imgs = len(list(output_dir.glob("*.jpg")))
    total_imgs = orig_n_imgs + augmented_imgs
    
    print(f"Augmentation complete!")
    print(f"Original images: {orig_n_imgs}")
    print(f"Augmented images created: {augmented_imgs}")
    print(f"Total dataset size: {total_imgs}")
    
    return total_imgs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment a dataset of images")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--n_augmentations", type=int, default=3, help="Number of augmentations per image")
    parser.add_argument("--enable_lr_flips", action="store_true", help="Enable left-right flips during augmentation")
    
    args = parser.parse_args()
    
    augment_dataset(args.dataset_path, args.n_augmentations, args.enable_lr_flips)