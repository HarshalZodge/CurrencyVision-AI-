"""
Preprocessing Module for CurrencyVision AI.
Provides image loading, resizing, normalization, dynamic dataset scanning,
and image augmentation tools for deep learning model training and inference.
"""

import os
import logging
from typing import Tuple, Dict, List, Optional
import numpy as np
import cv2
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    HAS_TF = True
except ImportError:
    tf = None
    ImageDataGenerator = None
    HAS_TF = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
TARGET_IMAGE_SIZE: Tuple[int, int] = (128, 128)
DEFAULT_CLASSES: List[str] = ["10", "20", "50", "100", "200", "500", "2000"]


def preprocess_image_file(
    image_path: str, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> np.ndarray:
    """
    Reads an image from path, converts to RGB, resizes, and normalizes pixels to [0, 1].

    Args:
        image_path (str): Path to input image file.
        target_size (Tuple[int, int]): Desired (height, width).

    Returns:
        np.ndarray: Preprocessed image tensor of shape (1, height, width, 3).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read image using OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to decode image from path: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Expand batch dimension (1, H, W, C)
    return np.expand_dims(img_normalized, axis=0)


def preprocess_pil_image(
    pil_img: Image.Image, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> np.ndarray:
    """
    Converts a PIL Image object to RGB, resizes, and normalizes for model prediction.

    Args:
        pil_img (PIL.Image.Image): Input PIL Image object.
        target_size (Tuple[int, int]): Target dimensions (width, height).

    Returns:
        np.ndarray: Preprocessed tensor (1, height, width, 3).
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    pil_img_resized = pil_img.resize(target_size, Image.BILINEAR)
    img_arr = np.array(pil_img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_arr, axis=0)


def scan_dataset_classes(dataset_dir: str) -> List[str]:
    """
    Dynamically scans dataset directory for subfolders representing class labels.

    Args:
        dataset_dir (str): Root dataset directory path (e.g. dataset/train).

    Returns:
        List[str]: List of sorted class folder names found in directory.
    """
    if not os.path.exists(dataset_dir):
        logger.warning(f"Dataset directory '{dataset_dir}' does not exist.")
        return DEFAULT_CLASSES

    classes = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ]

    # Sort numerically if possible, else alphabetically
    try:
        classes.sort(key=lambda x: int(x))
    except ValueError:
        classes.sort()

    if not classes:
        logger.warning(f"No class folders found in '{dataset_dir}'. Using default classes.")
        return DEFAULT_CLASSES

    logger.info(f"Discovered {len(classes)} classes dynamically: {classes}")
    return classes


def create_data_generators(
    train_dir: str,
    val_dir: str,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    batch_size: int = 32
) -> Tuple[ImageDataGenerator, ImageDataGenerator, Dict[int, str]]:
    """
    Configures Keras ImageDataGenerators with augmentation for training and validation.

    Args:
        train_dir (str): Path to training dataset directory.
        val_dir (str): Path to validation dataset directory.
        target_size (Tuple[int, int]): Image target dimensions.
        batch_size (int): Mini-batch size.

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator, Dict[int, str]]:
            (train_generator, val_generator, class_indices_map)
    """
    # Training augmentation pipeline
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Validation datagen (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # Reverse class_indices mapping (idx -> class_name)
    class_indices_map = {v: k for k, v in train_generator.class_indices.items()}

    return train_generator, val_generator, class_indices_map


def ensure_dataset_structure(base_dir: str = "dataset") -> None:
    """
    Creates dataset directory tree if missing, and generates sample images
    to allow immediate code execution and training validation.

    Args:
        base_dir (str): Base dataset path.
    """
    train_path = os.path.join(base_dir, "train")
    val_path = os.path.join(base_dir, "validation")

    for denomination in DEFAULT_CLASSES:
        os.makedirs(os.path.join(train_path, denomination), exist_ok=True)
        os.makedirs(os.path.join(val_path, denomination), exist_ok=True)

    # Create dummy images if train folders are empty to make dataset runnable
    for denom in DEFAULT_CLASSES:
        t_folder = os.path.join(train_path, denom)
        v_folder = os.path.join(val_path, denom)

        if len(os.listdir(t_folder)) == 0:
            logger.info(f"Populating sample synthetic training image for ₹{denom}...")
            _create_sample_note_image(os.path.join(t_folder, "sample_1.jpg"), denom)
            _create_sample_note_image(os.path.join(t_folder, "sample_2.jpg"), denom)

        if len(os.listdir(v_folder)) == 0:
            logger.info(f"Populating sample synthetic validation image for ₹{denom}...")
            _create_sample_note_image(os.path.join(v_folder, "val_1.jpg"), denom)


def _create_sample_note_image(output_path: str, label: str) -> None:
    """Helper to generate a visual sample note for dataset initialization."""
    img = np.zeros((256, 512, 3), dtype=np.uint8)

    # Distinguish colors by denomination
    color_map = {
        "10": (30, 80, 160),     # Brownish Red
        "20": (40, 180, 140),    # Greenish Yellow
        "50": (200, 160, 40),    # Cyan / Blue
        "100": (180, 80, 100),   # Lavender / Violet
        "200": (30, 140, 220),   # Bright Orange
        "500": (100, 110, 110),  # Stone Grey
        "2000": (120, 40, 180),  # Magenta
    }
    bg_color = color_map.get(label, (100, 100, 100))
    img[:] = bg_color

    # Add text overlay
    cv2.rectangle(img, (20, 20), (492, 236), (255, 255, 255), 3)
    cv2.putText(
        img,
        f"RESERVE BANK OF INDIA - RS {label}",
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"RS {label}",
        (180, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
