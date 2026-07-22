"""
Grad-CAM Module for CurrencyVision AI.
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to generate
visual explainability heatmaps for custom Keras CNN predictions.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    tf = None
    HAS_TF = False

logger = logging.getLogger(__name__)


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "target_conv_layer",
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Computes Grad-CAM activation heatmap for a target convolutional layer.

    Args:
        img_array (np.ndarray): Input preprocessed tensor of shape (1, H, W, 3).
        model (tf.keras.Model): Trained Keras CNN model.
        last_conv_layer_name (str): Name of target Conv2D layer.
        pred_index (Optional[int]): Target class index for Grad-CAM. Defaults to max predicted class.

    Returns:
        np.ndarray: Normalized 2D Grad-CAM heatmap values in range [0, 1].
    """
    # Find layer by name or fallback to last Conv2D
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        logger.warning(f"Layer '{last_conv_layer_name}' not found. Locating last Conv2D layer...")
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No Conv2D layers found in model for Grad-CAM.")
        target_layer = conv_layers[-1]
        last_conv_layer_name = target_layer.name

    # Create sub-model mapping input -> (target_conv_output, model_output)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of target class score w.r.t target layer feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Vector of mean intensity of gradient per feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in feature map by its gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only positive contributions and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    original_pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[Image.Image, Image.Image]:
    """
    Overlays a Grad-CAM heatmap onto the original input image.

    Args:
        original_pil_image (PIL.Image.Image): Original input image object.
        heatmap (np.ndarray): 2D Grad-CAM heatmap array in range [0, 1].
        alpha (float): Transparency blending weight for heatmap (0.0 to 1.0).
        colormap (int): OpenCV colormap (default COLORMAP_JET).

    Returns:
        Tuple[Image.Image, Image.Image]: (Resized Heatmap Image, Superimposed Blended Image).
    """
    orig_np = np.array(original_pil_image.convert("RGB"))
    h, w, _ = orig_np.shape

    # Rescale heatmap to 0-255 uint8 and resize to match original image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply color map to heatmap
    color_heatmap = cv2.applyColorMap(heatmap_resized, colormap)
    color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap on original image
    superimposed = cv2.addWeighted(orig_np, 1.0 - alpha, color_heatmap_rgb, alpha, 0)

    heatmap_pil = Image.fromarray(color_heatmap_rgb)
    superimposed_pil = Image.fromarray(superimposed)

    return heatmap_pil, superimposed_pil
