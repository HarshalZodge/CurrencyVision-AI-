"""
Model Utilities Module for CurrencyVision AI.
Provides custom CNN model architecture creation, callbacks, evaluation metrics,
and saved model loading functions.
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
        Flatten,
        Dense,
        Input
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    HAS_TF = True
except ImportError:
    tf = None
    Sequential = Model = load_model = None
    Conv2D = BatchNormalization = Activation = MaxPooling2D = Dropout = Flatten = Dense = Input = None
    EarlyStopping = ReduceLROnPlateau = ModelCheckpoint = None
    HAS_TF = False

logger = logging.getLogger(__name__)


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 7
) -> Model:
    """
    Builds a Custom CNN architecture optimized for Indian Currency Note Classification.

    Architecture:
      Input (128, 128, 3)
      ↓ Conv2D(32) + BatchNorm + ReLU
      ↓ MaxPooling2D
      ↓ Conv2D(64) + BatchNorm + ReLU
      ↓ MaxPooling2D
      ↓ Conv2D(128) + BatchNorm + ReLU
      ↓ Dropout(0.3)
      ↓ Conv2D(128)
      ↓ Dropout(0.4)
      ↓ Flatten
      ↓ Dense(256) + ReLU
      ↓ Dropout(0.5)
      ↓ Dense(num_classes) Softmax

    Args:
        input_shape (Tuple[int, int, int]): Shape of input images.
        num_classes (int): Number of output currency categories.

    Returns:
        tf.keras.Model: Compiled Keras sequential CNN model.
    """
    model = Sequential(name="CurrencyVision_Custom_CNN")

    # Block 1
    model.add(Input(shape=input_shape))
    model.add(
        Conv2D(
            32,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_block1",
        )
    )
    model.add(BatchNormalization(name="bn_block1"))
    model.add(Activation("relu", name="relu_block1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block1"))

    # Block 2
    model.add(
        Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_block2",
        )
    )
    model.add(BatchNormalization(name="bn_block2"))
    model.add(Activation("relu", name="relu_block2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pool_block2"))

    # Block 3
    model.add(
        Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_block3",
        )
    )
    model.add(BatchNormalization(name="bn_block3"))
    model.add(Activation("relu", name="relu_block3"))
    model.add(Dropout(0.3, name="dropout_block3"))

    # Block 4 - Target layer for Grad-CAM
    model.add(
        Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="target_conv_layer",
        )
    )
    model.add(BatchNormalization(name="bn_target"))
    model.add(Activation("relu", name="relu_target"))
    model.add(Dropout(0.4, name="dropout_target"))

    # Fully Connected Dense Head
    model.add(Flatten(name="flatten"))
    model.add(Dense(256, kernel_initializer="he_normal", name="dense_256"))
    model.add(Activation("relu", name="relu_dense"))
    model.add(Dropout(0.5, name="dropout_dense"))
    model.add(Dense(num_classes, activation="softmax", name="output_layer"))

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(f"Custom CNN model built successfully with {num_classes} classes.")
    return model


def get_training_callbacks(model_save_path: str = "model/currency_model.h5") -> List[tf.keras.callbacks.Callback]:
    """
    Constructs standard Keras callbacks: EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.

    Args:
        model_save_path (str): Filepath destination to save best model weights.

    Returns:
        List[tf.keras.callbacks.Callback]: List of configured callbacks.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=7,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    return [early_stopping, reduce_lr, checkpoint]


def evaluate_model_performance(
    model: Model,
    val_generator: Any,
    class_labels: List[str]
) -> Dict[str, Any]:
    """
    Evaluates trained model on validation generator and generates classification metrics.

    Args:
        model (tf.keras.Model): Trained Keras model.
        val_generator (DirectoryIterator): Validation data generator.
        class_labels (List[str]): List of class label strings.

    Returns:
        Dict[str, Any]: Metrics dictionary containing confusion matrix, classification report,
                       precision, recall, f1_score, and overall accuracy.
    """
    logger.info("Evaluating model on validation generator...")
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

    loss, accuracy = model.evaluate(val_generator, verbose=0)

    results = {
        "val_loss": float(loss),
        "val_accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }

    logger.info(f"Evaluation complete. Val Accuracy: {accuracy*100:.2f}%, F1-Score: {f1:.4f}")
    return results


def load_currency_model(model_path: str = "model/currency_model.h5") -> Optional[Model]:
    """
    Loads saved Keras model file safely.

    Args:
        model_path (str): Filepath to .h5 model.

    Returns:
        Optional[tf.keras.Model]: Loaded Keras model or None if file missing/corrupted.
    """
    if not os.path.exists(model_path):
        logger.warning(f"Model file at {model_path} does not exist.")
        return None

    try:
        model = load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        return None
