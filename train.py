"""
Training Pipeline for CurrencyVision AI.
Scans Indian currency dataset dynamically, sets up Keras data generators with augmentations,
compiles custom CNN with He initialization, trains with callbacks, saves model & class mapping,
and exports evaluation metrics and performance charts.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any

from utils.preprocessing import (
    ensure_dataset_structure,
    scan_dataset_classes,
    create_data_generators,
    TARGET_IMAGE_SIZE,
)
from utils.model_utils import (
    build_custom_cnn,
    get_training_callbacks,
    evaluate_model_performance,
)
from utils.plots import plot_training_history, plot_confusion_matrix_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipeline")


def run_training_pipeline(
    dataset_base: str = "dataset",
    model_output_dir: str = "model",
    epochs: int = 20,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Executes end-to-end dataset verification, model compilation, training, and metrics logging.

    Args:
        dataset_base (str): Path to root dataset directory.
        model_output_dir (str): Directory where currency_model.h5 will be saved.
        epochs (int): Number of max training epochs.
        batch_size (int): Batch size for ImageDataGenerator.

    Returns:
        Dict[str, Any]: Model training and validation summary dictionary.
    """
    logger.info("Starting CurrencyVision AI Training Pipeline...")

    # 1. Ensure dataset folders exist and scan classes dynamically
    train_dir = os.path.join(dataset_base, "train")
    val_dir = os.path.join(dataset_base, "validation")
    ensure_dataset_structure(dataset_base)

    classes = scan_dataset_classes(train_dir)
    num_classes = len(classes)
    logger.info(f"Detected {num_classes} currency classes: {classes}")

    # 2. Build Data Generators
    train_gen, val_gen, class_map = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        target_size=TARGET_IMAGE_SIZE,
        batch_size=batch_size,
    )

    # Save class indices mapping to json for inference consistency
    os.makedirs(model_output_dir, exist_ok=True)
    mapping_path = os.path.join(model_output_dir, "class_indices.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=4)
    logger.info(f"Class mapping saved to '{mapping_path}'")

    # 3. Build Custom CNN Architecture
    input_shape = (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3)
    model = build_custom_cnn(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    # 4. Setup Callbacks
    model_save_path = os.path.join(model_output_dir, "currency_model.h5")
    callbacks = get_training_callbacks(model_save_path=model_save_path)

    # 5. Model Fitting
    logger.info(f"Training CNN for up to {epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Evaluation & Metrics
    logger.info("Evaluating trained model on validation split...")
    eval_metrics = evaluate_model_performance(model, val_gen, classes)

    # 7. Print Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(eval_metrics["classification_report_text"])
    print("=" * 60)

    # Save history stats file
    history_path = os.path.join(model_output_dir, "training_history.json")
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=4)

    logger.info(f"Training pipeline finished successfully! Model saved at '{model_save_path}'")
    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CurrencyVision AI Training CLI")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    run_training_pipeline(
        dataset_base=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
