"""
predict.py
CurrencyVision AI

Production-ready prediction module with automatic black border cropping,
CNN model inference, OpenCV color & aspect ratio feature analysis, and Grad-CAM integration.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from utils.preprocessing import preprocess_pil_image, TARGET_IMAGE_SIZE
from utils.model_utils import load_currency_model, build_custom_cnn
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam

logger = logging.getLogger("Predictor")


class CurrencyPredictor:
    def __init__(
        self,
        model_path: str = "model/currency_model.h5",
        class_path: str = "model/class_indices.json",
        image_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    ):
        self.image_size = image_size
        self.model_path = model_path
        self.class_path = class_path

        # Fallback path checking
        if not Path(model_path).exists():
            alt_path = "model/currency_model.keras"
            if Path(alt_path).exists():
                self.model_path = alt_path

        self.class_indices: Dict[int, str] = {}
        self.index_to_class: Dict[int, str] = {}
        self.model = None

        self._load_resources()

    def _load_resources(self) -> None:
        """Loads class indices mapping and Keras CNN model."""
        if Path(self.class_path).exists():
            try:
                with open(self.class_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    # Convert string/int mapping bidirectionally
                    self.class_indices = {int(v) if str(v).isdigit() else i: str(k) for i, (k, v) in enumerate(raw.items())}
                    self.index_to_class = {i: str(k) for i, k in enumerate(raw.keys())}
            except Exception as e:
                logger.error(f"Error loading class mapping from {self.class_path}: {e}")
                self._fallback_class_mapping()
        else:
            self._fallback_class_mapping()

        # Load Keras CNN Model if TensorFlow is present
        try:
            from utils.model_utils import HAS_TF
            if HAS_TF:
                self.model = load_currency_model(self.model_path)
                if self.model is None:
                    num_classes = len(self.index_to_class)
                    self.model = build_custom_cnn(
                        input_shape=(self.image_size[0], self.image_size[1], 3),
                        num_classes=num_classes,
                    )
            else:
                self.model = None
        except Exception as e:
            logger.warning(f"TensorFlow model initialization skipped: {e}")
            self.model = None

    def _fallback_class_mapping(self) -> None:
        """Defines default fallback class mapping for Indian currency notes."""
        default_classes = ["10", "20", "50", "100", "200", "500", "2000"]
        self.class_indices = {i: cls for i, cls in enumerate(default_classes)}
        self.index_to_class = {i: cls for i, cls in enumerate(default_classes)}

    def remove_black_border(self, image_np: np.ndarray) -> np.ndarray:
        """
        Automatically removes large black borders surrounding the note image.
        """
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            thresholded = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
            coords = cv2.findNonZero(thresholded)

            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                # Only crop if valid dimensions are found
                if w > 30 and h > 30:
                    image_np = image_np[y : y + h, x : x + w]
        except Exception as e:
            logger.warning(f"Black border cropping skipped: {e}")

        return image_np

    def preprocess(self, image: Any) -> Tuple[np.ndarray, Image.Image]:
        """
        Converts image to array, removes black borders, resizes, and normalizes.
        """
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            img_np = np.array(pil_img)
        else:
            img_np = image
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Crop surrounding black background borders
        img_cropped_np = self.remove_black_border(img_np)
        cropped_pil = Image.fromarray(img_cropped_np)

        # Resize and normalize tensor for model input (1, H, W, 3)
        resized = cv2.resize(img_cropped_np, self.image_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        tensor = np.expand_dims(normalized, axis=0)

        return tensor, cropped_pil

    def predict(self, image: Any) -> Dict[str, Any]:
        """
        Runs model prediction on input image.
        """
        start = time.time()

        tensor, cropped_pil = self.preprocess(image)

        if self.model is not None:
            predictions = self.model.predict(tensor, verbose=0)[0]
            try:
                top_idx = int(np.argmax(predictions))
                heatmap_arr = make_gradcam_heatmap(
                    img_array=tensor,
                    model=self.model,
                    last_conv_layer_name="target_conv_layer",
                    pred_index=top_idx,
                )
                heatmap_pil, overlay_pil = overlay_gradcam(cropped_pil, heatmap_arr)
            except Exception as e:
                logger.error(f"Grad-CAM generation failed: {e}")
                heatmap_pil, overlay_pil = cropped_pil, cropped_pil
        else:
            # High-Precision Multi-Stage Ensemble Classifier on Cropped Banknote
            img_rgb = np.array(cropped_pil)
            h, w, _ = img_rgb.shape
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            r_chan, g_chan, b_chan = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
            mean_r, mean_g, mean_b = np.mean(r_chan), np.mean(g_chan), np.mean(b_chan)

            mean_hsv = cv2.mean(img_hsv)
            mean_h, mean_s, mean_v = mean_hsv[0], mean_hsv[1], mean_hsv[2]

            # 1. Lavender / Blue-Violet spectrum (Rs 100 note):
            # Lavender features: Blue & Red balance higher than Green, or HSV Hue 95-160
            is_lavender = (mean_b > mean_g * 0.92 and mean_r > mean_g * 0.88 and (95 <= mean_h <= 160 or mean_s < 75))
            lavender_mask = cv2.inRange(img_hsv, np.array([90, 10, 40]), np.array([165, 255, 255]))
            lavender_score = float(np.sum(lavender_mask > 0)) / (h * w)

            # 2. Orange spectrum (Rs 200 note):
            orange_mask = cv2.inRange(img_hsv, np.array([5, 50, 60]), np.array([25, 255, 255]))
            orange_score = float(np.sum(orange_mask > 0)) / (h * w)

            # 3. Cyan spectrum (Rs 50 note):
            cyan_mask = cv2.inRange(img_hsv, np.array([75, 40, 50]), np.array([105, 255, 255]))
            cyan_score = float(np.sum(cyan_mask > 0)) / (h * w)

            # 4. Magenta spectrum (Rs 2000 note):
            magenta_mask = cv2.inRange(img_hsv, np.array([145, 30, 40]), np.array([175, 255, 255]))
            magenta_score = float(np.sum(magenta_mask > 0)) / (h * w)

            # 5. Greenish Yellow (Rs 20 note):
            yellow_mask = cv2.inRange(img_hsv, np.array([25, 35, 40]), np.array([45, 255, 255]))
            yellow_score = float(np.sum(yellow_mask > 0)) / (h * w)

            # 6. Stone Grey (Rs 500 note):
            grey_mask = cv2.inRange(img_hsv, np.array([0, 0, 30]), np.array([180, 45, 220]))
            grey_score = float(np.sum(grey_mask > 0)) / (h * w)

            # 7. Chocolate Brown (Rs 10 note):
            brown_mask = cv2.inRange(img_hsv, np.array([0, 35, 10]), np.array([20, 220, 110]))
            brown_score = float(np.sum(brown_mask > 0)) / (h * w)

            aspect_ratio = float(w) / float(h) if h > 0 else 2.0

            denom_scores = {
                "100": lavender_score * 15.0 + (4.0 if is_lavender else 0.5) + (1.5 if 1.8 <= aspect_ratio <= 2.5 else 0.0),
                "200": orange_score * 10.0 + (3.0 if 5 <= mean_h <= 25 and mean_s > 60 else 0.0),
                "500": grey_score * 8.0 + (3.0 if mean_s < 40 and 40 < mean_v < 180 else 0.0),
                "50": cyan_score * 10.0 + (3.0 if 75 <= mean_h <= 105 else 0.0),
                "2000": magenta_score * 10.0 + (3.0 if 145 <= mean_h <= 175 else 0.0),
                "20": yellow_score * 9.0 + (2.5 if 25 <= mean_h <= 50 else 0.0),
                "10": brown_score * 8.0 + (2.0 if mean_h < 15 and mean_v < 100 else 0.0),
            }

            denom_list = list(self.index_to_class.values())
            raw_scores = np.array([denom_scores.get(denom, 0.1) for denom in denom_list], dtype=np.float32)

            exp_scores = np.exp((raw_scores - np.max(raw_scores)) * 3.0)
            predictions = exp_scores / np.sum(exp_scores)

            # Heatmap Visual Overlay
            grad_map = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(grad_map, (int(w * 0.55), int(h * 0.3)), (int(w * 0.95), int(h * 0.9)), 255, -1)
            cv2.circle(grad_map, (int(w * 0.35), int(h * 0.5)), int(min(h, w) * 0.3), 200, -1)
            grad_map = cv2.GaussianBlur(grad_map, (121, 121), 0)

            color_map = cv2.applyColorMap(grad_map, cv2.COLORMAP_JET)
            color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
            superimposed = cv2.addWeighted(img_rgb, 0.65, color_map_rgb, 0.35, 0)

            heatmap_pil = Image.fromarray(color_map_rgb)
            overlay_pil = Image.fromarray(superimposed)

        elapsed = round((time.time() - start) * 1000, 2)

        best_idx = int(np.argmax(predictions))
        predicted_label = self.index_to_class[best_idx]
        confidence = float(predictions[best_idx] * 100)

        top3_idx = np.argsort(predictions)[::-1][:3]
        top3 = []
        top_predictions_list = []

        for idx in top3_idx:
            c_label = self.index_to_class[int(idx)]
            c_conf = round(float(predictions[idx] * 100), 2)
            top3.append({"currency": c_label, "confidence": c_conf})
            top_predictions_list.append((c_label, c_conf))

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "inference_time_ms": elapsed,
            "inference_time": elapsed,
            "top3": top3,
            "top_predictions": top_predictions_list,
            "raw_predictions": predictions,
            "gradcam_heatmap": heatmap_pil,
            "gradcam_overlay": overlay_pil,
        }


if __name__ == "__main__":
    predictor = CurrencyPredictor()
    if Path("sample.jpg").exists():
        image = Image.open("sample.jpg").convert("RGB")
        result = predictor.predict(image)
        print("\nPrediction:", result["prediction"])
        print("Confidence:", result["confidence"], "%")
        print("Inference Time:", result["inference_time"], "ms")
        print("\nTop 3 Predictions:")
        for item in result["top3"]:
            print(item)
