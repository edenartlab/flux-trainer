import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import os
from tqdm import tqdm
import concurrent.futures
import numpy as np
import cv2
import math
import mediapipe as mp
from typing import Optional, Tuple
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

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

def detect_face(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face in the image using MediaPipe face detection.
    
    Args:
        image: PIL Image object
    Returns:
        Tuple of (x, y, width, height) for the detected face or None if no face detected
    """
    try:
        # Convert PIL image to numpy array
        img_np = np.array(image.convert('RGB'))
        
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        
        # Create a face detection object with good defaults
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            # Process the image
            results = face_detection.process(img_np)
            
            # Check if any faces were detected
            if not results.detections:
                return None
            
            # Get the first (most prominent) face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute pixel coordinates
            img_height, img_width = img_np.shape[:2]
            x = int(bbox.xmin * img_width)
            y = int(bbox.ymin * img_height)
            width = int(bbox.width * img_width)
            height = int(bbox.height * img_height)
            
            return (x, y, width, height)
            
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None

def face_aware_crop(image, face_rect, target_ratio=None):
    """
    Apply face-aware cropping to the image.
    
    :param image: PIL Image object
    :param face_rect: Tuple of (x, y, width, height) for the detected face
    :param target_ratio: Target face area to image area ratio (0.1 to 0.6)
    :return: Augmented PIL Image object
    """
    if face_rect is None:
        # Fallback to standard random crop if no face detected
        return random_crop(image)
    
    img_width, img_height = image.size
    img_area = img_width * img_height
    
    face_x, face_y, face_width, face_height = face_rect
    face_area = face_width * face_height
    
    # Calculate current face to image area ratio
    current_ratio = face_area / img_area
    
    # If no target ratio provided, probabilistically select based on current ratio
    if target_ratio is None:
        # Skew towards larger faces for small faces, smaller faces for large faces
        if current_ratio < 0.1:
            # For very small faces, strongly prefer larger target ratios
            target_ratio = random.uniform(0.2, 0.5)
        elif current_ratio > 0.6:
            # For very large faces, prefer smaller target ratios
            target_ratio = random.uniform(0.1, 0.4)
        else:
            # For medium-sized faces, select randomly with more diversity
            target_ratio = random.uniform(0.1, 0.6)
    
    # Determine if we need to crop in or expand (add padding)
    if abs(current_ratio - target_ratio) < 0.05:
        # Current ratio is close enough to target, apply mild random crop
        return random_crop(image, scale=(0.9, 0.98))
    
    elif current_ratio < target_ratio:
        # Face is smaller than target ratio, need to crop in
        face_center_x = face_x + face_width // 2
        face_center_y = face_y + face_height // 2
        
        # Calculate the needed scale factor to achieve target ratio
        scale_factor = math.sqrt(current_ratio / target_ratio)
        
        # Calculate new crop dimensions
        crop_width = int(img_width * scale_factor)
        crop_height = int(img_height * scale_factor)
        
        # Calculate crop coordinates, ensuring face is centered in the crop
        # Add some randomness to crop position, but ensure face remains in view
        max_offset_x = min(face_center_x - crop_width//4, crop_width//4)
        max_offset_y = min(face_center_y - crop_height//4, crop_height//4)
        
        # Random offset from perfect center
        offset_x = random.uniform(-max_offset_x, max_offset_x) 
        offset_y = random.uniform(-max_offset_y, max_offset_y)
        
        crop_left = max(0, int(face_center_x - crop_width//2 + offset_x))
        crop_top = max(0, int(face_center_y - crop_height//2 + offset_y))
        
        # Adjust if crop goes beyond image bounds
        if crop_left + crop_width > img_width:
            crop_left = img_width - crop_width
        if crop_top + crop_height > img_height:
            crop_top = img_height - crop_height
            
        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height
        
        # Apply the crop and resize
        cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        return cropped.resize((img_width, img_height), Image.LANCZOS)
    
    else:
        # Face is larger than target ratio, need to pad/expand
        # Calculate the scale factor
        scale_factor = math.sqrt(current_ratio / target_ratio)
        
        # Calculate new canvas dimensions
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        
        # Create a new canvas (black background)
        new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
        
        # Paste the original image in the center
        paste_x = (new_width - img_width) // 2
        paste_y = (new_height - img_height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        # Resize back to original dimensions
        return new_image.resize((img_width, img_height), Image.LANCZOS)

def augment_image(image, mode=None, enable_lr_flips=False):
    """
    Apply a series of augmentations to the image.
    
    :param image: PIL Image object
    :param mode: Augmentation mode ('face' for face-aware augmentation)
    :param enable_lr_flips: Whether to enable horizontal flips
    :return: Augmented PIL Image object
    """
    # Add random horizontal flip with 50% probability if enabled
    if enable_lr_flips and random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Apply face-aware processing if mode is 'face'
    if mode == 'face':
        # Detect face in the image
        face_rect = detect_face(image)
        
        # Only apply face-aware augmentation for a portion of images
        if face_rect is not None and random.random() < 0.7:
            # Target a range of face/image ratios to create a uniform distribution
            image = face_aware_crop(image, face_rect)
        else:
            # For the rest, apply standard random crop
            image = random_crop(image)
    else:
        # Standard random crop for non-face mode
        image = random_crop(image)
        
    image = hue_augmentation(image)
    image = color_jitter(image)
    
    return image

def process_image(img_path, output_dir, n_augmentations, base_name, mode=None, enable_lr_flips=False):
    """
    Process a single image with augmentations and save the results.
    
    :param img_path: Path to the image
    :param output_dir: Directory to save augmented images
    :param n_augmentations: Number of augmentations to create per image
    :param base_name: Base name for the augmented images
    :param mode: Augmentation mode ('face' for face-aware augmentation)
    :param enable_lr_flips: Whether to enable horizontal flips
    :return: None
    """
    # Handle image paths that might contain special characters or spaces
    img_path_str = str(img_path)
    image = Image.open(img_path_str)
    
    # Clean base_name to avoid path issues
    safe_base_name = str(base_name).replace(" ", "_").replace("(", "").replace(")", "")
    
    # Create augmentations
    for i in range(n_augmentations):
        aug_image = augment_image(image, mode, enable_lr_flips)
        aug_filename = f"{safe_base_name}_aug{i+1}.jpg"
        aug_path = output_dir / aug_filename
        aug_image.save(str(aug_path), quality=95)
        

def augment_dataset(dataset_path, n_augmentations_per_img=3, mode=None, enable_lr_flips=False):
    """
    Augment all images in the specified dataset path.
    
    :param dataset_path: Path to the directory containing images
    :param n_augmentations_per_img: Number of augmentations to create per image
    :param mode: Augmentation mode ('face' for face-aware augmentation)
    :param enable_lr_flips: Whether to enable horizontal flips
    :return: Total number of images after augmentation
    """
    dataset_path = Path(dataset_path)
    output_dir = dataset_path
    
    # Get all jpg images
    image_paths = list(dataset_path.glob("*.jpg"))
    orig_n_imgs = len(image_paths)
    print(f"Augmenting dataset at {dataset_path}, original size: {orig_n_imgs} images")
    print(f"Mode: {mode if mode else 'Standard'}")
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
                    mode,
                    enable_lr_flips
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
    
    # Count the number of augmented images
    augmented_imgs = len(list(output_dir.glob("*.jpg"))) - orig_n_imgs
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
    parser.add_argument("--mode", type=str, choices=[None, "face"], default=None, 
                        help="Augmentation mode. Use 'face' for face-aware augmentation")
    
    args = parser.parse_args()
    
    augment_dataset(args.dataset_path, args.n_augmentations, args.mode, args.enable_lr_flips)