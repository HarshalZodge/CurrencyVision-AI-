"""
Training Pipeline for CurrencyVision AI.
Prints dataset statistics & imbalance checks, sets up data generators (no horizontal flips),
compiles custom 4-Block CNN with He Normal weights and Label Smoothing,
trains for up to 100 epochs with callbacks, evaluates metrics, and exports reports.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any

from utils.preprocessing import (
    ensure_dataset_structure,
    scan_dataset_classes,
    get_dataset_statistics,
    create_data_generators,
    TARGET_IMAGE_SIZE,
)
from utils.model_utils import (
    build_custom_cnn,
    get_training_callbacks,
    evaluate_model_performance,
    HAS_TF,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipeline")


def print_dataset_summary(stats: Dict[str, Any]) -> None:
    """Prints dataset sample counts, split breakdown, and imbalance warnings."""
    print("\n" + "=" * 65)
    print("📊 CURRENCYVISION AI — DATASET PRE-TRAINING AUDIT SUMMARY")
    print("=" * 65)
    print(f"Discovered Classes ({len(stats['classes'])}): {stats['classes']}")
    print("-" * 65)
    print(f"{'Class (Rs.)':<15} | {'Training Images':<18} | {'Validation Images':<18}")
    print("-" * 65)

    for cls in stats["classes"]:
        t_cnt = stats["train_counts"].get(cls, 0)
        v_cnt = stats["val_counts"].get(cls, 0)
        print(f"Rs. {cls:<11} | {t_cnt:<18} | {v_cnt:<18}")

    print("-" * 65)
    print(f"Total Training Images:   {stats['total_train']}")
    print(f"Total Validation Images: {stats['total_val']}")
    print(f"Class Imbalance Ratio:   {stats['imbalance_ratio']:.2f}x")

    if stats["is_imbalanced"]:
        print("⚠️ WARNING: Class imbalance detected (> 3x ratio). Data augmentation recommended.")
    else:
        print("✅ Dataset balance is acceptable.")
    print("=" * 65 + "\n")


def analyze_overfitting_underfitting(history_dict: Dict[str, Any]) -> None:
    """Analyzes final training epoch metrics to detect overfitting or underfitting."""
    train_acc = history_dict.get("accuracy", [0])[-1] * 100.0
    val_acc = history_dict.get("val_accuracy", [0])[-1] * 100.0
    train_loss = history_dict.get("loss", [0])[-1]
    val_loss = history_dict.get("val_loss", [0])[-1]

    print("\n" + "=" * 65)
    print("🔍 DIAGNOSTIC MODEL PERFORMANCE AUDIT")
    print("=" * 65)
    print(f"Final Train Accuracy: {train_acc:.2f}% | Val Accuracy: {val_acc:.2f}%")
    print(f"Final Train Loss:     {train_loss:.4f}  | Val Loss:     {val_loss:.4f}")
    print("-" * 65)

    gap = train_acc - val_acc

    if gap > 15.0:
        print("⚠️ DIAGNOSIS: High Overfitting Detected!")
        print("💡 Recommendations:")
        print("   1. Increase Dropout probability (e.g., to 0.5 or 0.6).")
        print("   2. Add L2 weight regularization or expand training dataset.")
    elif train_acc < 70.0 and val_acc < 70.0:
        print("⚠️ DIAGNOSIS: Underfitting Detected!")
        print("💡 Recommendations:")
        print("   1. Increase CNN capacity or train for more epochs.")
        print("   2. Adjust initial learning rate.")
    else:
        print("✅ DIAGNOSIS: Model generalization is STABLE and well-balanced!")
    print("=" * 65 + "\n")


def run_training_pipeline(
    dataset_base: str = "dataset",
    model_output_dir: str = "model",
    epochs: int = 100,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Executes full model training and evaluation."""
    if not HAS_TF:
        logger.error("TensorFlow is not installed in the environment. Training cannot proceed.")
        return {}

    logger.info("Starting CurrencyVision AI Training Pipeline...")

    # 1. Dataset Verification & Pre-Training Audit
    train_dir = os.path.join(dataset_base, "train")
    val_dir = os.path.join(dataset_base, "validation")
    ensure_dataset_structure(dataset_base)

    stats = get_dataset_statistics(train_dir, val_dir)
    print_dataset_summary(stats)

    classes = stats["classes"]
    num_classes = len(classes)

    # 2. Build Data Generators (NO horizontal flips)
    train_gen, val_gen, class_map = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        target_size=TARGET_IMAGE_SIZE,
        batch_size=batch_size,
    )

    # Save class indices mapping json: {"10": 0, "100": 1, ...}
    os.makedirs(model_output_dir, exist_ok=True)
    mapping_path = os.path.join(model_output_dir, "class_indices.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=4)
    logger.info(f"Class mapping saved to '{mapping_path}': {class_map}")

    # 3. Build 4-Block Custom CNN
    input_shape = (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3)
    model = build_custom_cnn(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    # 4. Setup Callbacks
    model_save_path = os.path.join(model_output_dir, "currency_model.h5")
    callbacks = get_training_callbacks(model_save_path=model_save_path)

    # 5. Fit Model for up to 100 epochs
    logger.info(f"Fitting CNN model for up to {epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Overfitting/Underfitting Diagnostic Check
    analyze_overfitting_underfitting(history.history)

    # 7. Post-Training Evaluation
    logger.info("Evaluating final model weights on validation split...")
    eval_metrics = evaluate_model_performance(model, val_gen, classes)

    print("\n" + "=" * 65)
    print("📋 FINAL CLASSIFICATION REPORT")
    print("=" * 65)
    print(eval_metrics["classification_report_text"])
    print("=" * 65)

    if eval_metrics["top_misclassified_pairs"]:
        print("\nTop Misclassified Pairs:")
        for pair in eval_metrics["top_misclassified_pairs"]:
            print(f"  • True: Rs. {pair['true_class']:<6} → Predicted: Rs. {pair['predicted_as']:<6} (Count: {pair['count']})")

    # Save history json
    history_path = os.path.join(model_output_dir, "training_history.json")
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=4)

    logger.info(f"Training pipeline finished! Saved best model to '{model_save_path}'")
    return eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CurrencyVision AI Training CLI")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_training_pipeline(
        dataset_base=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
