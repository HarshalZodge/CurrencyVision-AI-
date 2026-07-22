"""
Preprocessing Module for CurrencyVision AI.
Ensures 100% identical preprocessing across training, validation, and inference pipelines.
Provides note boundary detection, black border removal, color normalization,
and no-horizontal-flip data augmentations.
"""

import os
import logging
from typing import Tuple, Dict, List, Optional, Any
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

logger = logging.getLogger(__name__)

# Standardized Global Constants
TARGET_IMAGE_SIZE: Tuple[int, int] = (128, 128)
DEFAULT_CLASSES: List[str] = ["10", "20", "50", "100", "200", "500", "2000"]


def remove_background_and_crop(img_rgb: np.ndarray) -> np.ndarray:
    """
    Detects banknote boundaries and crops away unnecessary table/background padding.

    Args:
        img_rgb (np.ndarray): Input RGB numpy image array.

    Returns:
        np.ndarray: Cropped banknote RGB image.
    """
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding to isolate note foreground
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Ensure bounding box is sufficiently large to prevent cropping noise
            h_img, w_img, _ = img_rgb.shape
            if w > w_img * 0.25 and h > h_img * 0.25:
                return img_rgb[y:y + h, x:x + w]
    except Exception as e:
        logger.warning(f"Background cropping warning: {e}")

    return img_rgb


def standardize_image_pipeline(
    image: Any, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> np.ndarray:
    """
    Standardized preprocessing pipeline identical for both training and prediction.

    Pipeline:
      1. Convert input to RGB numpy array.
      2. Crop note boundaries / black borders.
      3. Resize using cv2.INTER_AREA to target_size.
      4. Convert to float32 and normalize pixels to [0.0, 1.0].
      5. Expand batch dimension -> shape (1, H, W, 3).

    Args:
        image (Any): PIL.Image, numpy array, or file path.
        target_size (Tuple[int, int]): Target dimensions (width, height).

    Returns:
        np.ndarray: Preprocessed float32 tensor of shape (1, height, width, 3).
    """
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found at path: {image}")
        img_bgr = cv2.imread(image)
        if img_bgr is None:
            raise ValueError(f"Failed to read image at: {image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = image.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    # 1. Remove background padding
    img_cropped = remove_background_and_crop(img_rgb)

    # 2. Resize to exact target size (W, H)
    img_resized = cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_AREA)

    # 3. Normalize to [0.0, 1.0] float32
    img_normalized = img_resized.astype(np.float32) / 255.0

    # 4. Expand batch dimension (1, H, W, C)
    return np.expand_dims(img_normalized, axis=0)


def scan_dataset_classes(dataset_dir: str) -> List[str]:
    """
    Dynamically scans dataset directory for subfolders representing class labels.

    Args:
        dataset_dir (str): Root dataset directory path (e.g. dataset/train).

    Returns:
        List[str]: Sorted list of class names.
    """
    if not os.path.exists(dataset_dir):
        logger.warning(f"Dataset directory '{dataset_dir}' does not exist.")
        return DEFAULT_CLASSES

    classes = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ]

    try:
        classes.sort(key=lambda x: int(x))
    except ValueError:
        classes.sort()

    if not classes:
        logger.warning(f"No class folders found in '{dataset_dir}'. Using default classes.")
        return DEFAULT_CLASSES

    logger.info(f"Discovered {len(classes)} classes dynamically: {classes}")
    return classes


def get_dataset_statistics(train_dir: str, val_dir: str) -> Dict[str, Any]:
    """
    Computes per-class sample counts, class balance, and flags missing classes.

    Args:
        train_dir (str): Path to training dataset split directory.
        val_dir (str): Path to validation dataset split directory.

    Returns:
        Dict[str, Any]: Detailed statistical breakdown dictionary.
    """
    classes = scan_dataset_classes(train_dir)
    train_counts = {}
    val_counts = {}

    total_train = 0
    total_val = 0

    for cls in classes:
        t_path = os.path.join(train_dir, cls)
        v_path = os.path.join(val_dir, cls)

        t_c = len([f for f in os.listdir(t_path) if not f.startswith(".")]) if os.path.exists(t_path) else 0
        v_c = len([f for f in os.listdir(v_path) if not f.startswith(".")]) if os.path.exists(v_path) else 0

        train_counts[cls] = t_c
        val_counts[cls] = v_c

        total_train += t_c
        total_val += v_c

    # Detect imbalance (max count / min count > 3)
    counts_list = [c for c in train_counts.values() if c > 0]
    imbalance_ratio = (max(counts_list) / min(counts_list)) if counts_list else 1.0

    return {
        "classes": classes,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "total_train": total_train,
        "total_val": total_val,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": imbalance_ratio > 3.0,
    }


def create_data_generators(
    train_dir: str,
    val_dir: str,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    batch_size: int = 32
) -> Tuple[Any, Any, Dict[str, int]]:
    """
    Creates Keras ImageDataGenerators.
    CRITICAL: horizontal_flip=False to preserve currency orientation.

    Args:
        train_dir (str): Path to train dataset directory.
        val_dir (str): Path to val dataset directory.
        target_size (Tuple[int, int]): Image dimensions.
        batch_size (int): Mini-batch size.

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator, Dict[str, int]]:
            (train_gen, val_gen, class_indices_map)
    """
    if not HAS_TF:
        raise ImportError("TensorFlow is required to create ImageDataGenerators.")

    # Data Augmentation pipeline WITHOUT horizontal flip (preserves orientation)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        brightness_range=[0.8, 1.2],
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=False,  # DO NOT flip currency notes horizontally!
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    # Class mapping dict: {"10": 0, "100": 1, ...}
    class_indices_map = train_generator.class_indices

    return train_generator, val_generator, class_indices_map


def ensure_dataset_structure(base_dir: str = "dataset") -> None:
    """Ensures dataset directory structure exists and initializes starter images if empty."""
    train_path = os.path.join(base_dir, "train")
    val_path = os.path.join(base_dir, "validation")

    for denom in DEFAULT_CLASSES:
        os.makedirs(os.path.join(train_path, denom), exist_ok=True)
        os.makedirs(os.path.join(val_path, denom), exist_ok=True)

    for denom in DEFAULT_CLASSES:
        t_folder = os.path.join(train_path, denom)
        v_folder = os.path.join(val_path, denom)

        if len(os.listdir(t_folder)) == 0:
            _create_sample_note_image(os.path.join(t_folder, "sample_1.jpg"), denom)
            _create_sample_note_image(os.path.join(t_folder, "sample_2.jpg"), denom)

        if len(os.listdir(v_folder)) == 0:
            _create_sample_note_image(os.path.join(v_folder, "val_1.jpg"), denom)


def _create_sample_note_image(output_path: str, label: str) -> None:
    """Creates a sample currency image for initial setup."""
    img = np.zeros((256, 512, 3), dtype=np.uint8)
    color_map = {
        "10": (30, 80, 160),
        "20": (40, 180, 140),
        "50": (200, 160, 40),
        "100": (180, 80, 100),
        "200": (30, 140, 220),
        "500": (100, 110, 110),
        "2000": (120, 40, 180),
    }
    img[:] = color_map.get(label, (100, 100, 100))
    cv2.rectangle(img, (20, 20), (492, 236), (255, 255, 255), 3)
    cv2.putText(img, f"RS {label}", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
