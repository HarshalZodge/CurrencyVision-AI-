"""
Model Utilities Module for CurrencyVision AI.
Provides 4-Block Custom CNN Architecture with GlobalAveragePooling2D, He Normal initialization,
Label Smoothing Loss, Precision/Recall Metrics, and Training Callbacks.
"""

import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        Conv2D,
        BatchNormalization,
        Activation,
        MaxPooling2D,
        Dropout,
        GlobalAveragePooling2D,
        Dense,
        Input,
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        ModelCheckpoint,
        CSVLogger,
    )
    from tensorflow.keras.metrics import Precision, Recall
    from tensorflow.keras.losses import CategoricalCrossentropy
    HAS_TF = True
except ImportError:
    tf = None
    Sequential = Model = load_model = None
    Conv2D = BatchNormalization = Activation = MaxPooling2D = Dropout = GlobalAveragePooling2D = Dense = Input = None
    EarlyStopping = ReduceLROnPlateau = ModelCheckpoint = CSVLogger = None
    Precision = Recall = CategoricalCrossentropy = None
    HAS_TF = False

logger = logging.getLogger(__name__)


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 7
) -> Any:
    """
    Builds a 4-Block Custom CNN with GlobalAveragePooling2D for Indian Currency Classification.

    Architecture:
      Input (128, 128, 3)
      ↓ Conv2D(32, he_normal) + BatchNorm + ReLU → MaxPool
      ↓ Conv2D(64, he_normal) + BatchNorm + ReLU → MaxPool
      ↓ Conv2D(128, he_normal) + BatchNorm + ReLU → MaxPool
      ↓ Conv2D(256, he_normal, name="target_conv_layer") + BatchNorm + ReLU → MaxPool
      ↓ Dropout(0.4)
      ↓ GlobalAveragePooling2D
      ↓ Dense(256, he_normal) + BatchNorm + ReLU
      ↓ Dropout(0.5)
      ↓ Dense(num_classes, softmax)

    Args:
        input_shape (Tuple[int, int, int]): Shape of input images.
        num_classes (int): Number of currency classes.

    Returns:
        tf.keras.Model: Compiled custom CNN model.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow is required to build CNN model.")

    model = Sequential(name="CurrencyVision_4Block_CNN")

    # Block 1
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", name="conv_block1"))
    model.add(BatchNormalization(name="bn_block1"))
    model.add(Activation("relu", name="relu_block1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block1"))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", name="conv_block2"))
    model.add(BatchNormalization(name="bn_block2"))
    model.add(Activation("relu", name="relu_block2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block2"))

    # Block 3
    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", name="conv_block3"))
    model.add(BatchNormalization(name="bn_block3"))
    model.add(Activation("relu", name="relu_block3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block3"))

    # Block 4 (Grad-CAM Target Layer)
    model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal", name="target_conv_layer"))
    model.add(BatchNormalization(name="bn_block4"))
    model.add(Activation("relu", name="relu_block4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block4"))

    # Dense Classification Head
    model.add(Dropout(0.4, name="dropout_conv"))
    model.add(GlobalAveragePooling2D(name="gap"))
    model.add(Dense(256, kernel_initializer="he_normal", name="dense_256"))
    model.add(BatchNormalization(name="bn_dense"))
    model.add(Activation("relu", name="relu_dense"))
    model.add(Dropout(0.5, name="dropout_dense"))
    model.add(Dense(num_classes, activation="softmax", name="output_layer"))

    # Optimizer with tuned learning rate 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # Categorical Crossentropy with Label Smoothing = 0.1
    loss_fn = CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
        ],
    )

    logger.info(f"Custom 4-Block CNN built with {num_classes} classes and Adam(lr=0.0001).")
    return model


def get_training_callbacks(
    model_save_path: str = "model/currency_model.h5",
    log_csv_path: str = "model/training_log.csv"
) -> List[Any]:
    """
    Constructs callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, and CSVLogger.

    Args:
        model_save_path (str): Filepath to save best model weights.
        log_csv_path (str): Filepath to write CSV training log.

    Returns:
        List[tf.keras.callbacks.Callback]: Configured list of callbacks.
    """
    if not HAS_TF:
        return []

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        verbose=1,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    csv_logger = CSVLogger(filename=log_csv_path, separator=",", append=False)

    return [early_stopping, reduce_lr, checkpoint, csv_logger]


def evaluate_model_performance(
    model: Any,
    val_generator: Any,
    class_labels: List[str]
) -> Dict[str, Any]:
    """
    Evaluates model performance and generates classification metrics & confusion matrix.

    Args:
        model (Any): Keras model.
        val_generator (Any): Validation directory iterator.
        class_labels (List[str]): List of denomination labels.

    Returns:
        Dict[str, Any]: Detailed evaluation metrics dict.
    """
    logger.info("Evaluating model performance on validation dataset...")
    val_generator.reset()
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_labels, zero_division=0
    )

    loss, accuracy, prec_m, rec_m = model.evaluate(val_generator, verbose=0)

    # Find top misclassified class pairs
    misclassified_pairs = []
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            if i != j and cm[i][j] > 0:
                misclassified_pairs.append({
                    "true_class": class_labels[i],
                    "predicted_as": class_labels[j],
                    "count": int(cm[i][j])
                })
    misclassified_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "val_loss": float(loss),
        "val_accuracy": float(accuracy),
        "precision": float(prec_m),
        "recall": float(rec_m),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "top_misclassified_pairs": misclassified_pairs[:5],
    }


def load_currency_model(model_path: str = "model/currency_model.h5") -> Optional[Any]:
    """Loads saved Keras model file safely."""
    if not HAS_TF:
        return None

    if not os.path.exists(model_path):
        alt_path = "model/currency_model.keras"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            return None

    try:
        model = load_model(model_path)
        logger.info(f"Model loaded successfully from '{model_path}'")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from '{model_path}': {e}")
        return None
