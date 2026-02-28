"""
Training and Evaluation Steps for Training Pipeline

This module contains ZenML steps for model training and evaluation.
"""

import os
import json
from typing import Dict, Any, Tuple
from pathlib import Path
import tempfile

import numpy as np
import mlflow
import mlflow.tensorflow
from typing_extensions import Annotated

from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.model.cnn import create_cnn_model, compile_model

logger = get_logger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@step(experiment_tracker="mlflow_tracker")
def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 5,
) -> Annotated[tf.keras.Model, "trained_model"]:
    """
    Train a CNN model on CIFAR-10 data.

    This step trains the model and logs all metrics, parameters,
    and artifacts to MLflow.

    Args:
        x_train: Preprocessed training images
        y_train: Training labels
        x_val: Preprocessed validation images
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Patience for early stopping

    Returns:
        Trained Keras model
    """
    logger.info("=" * 60)
    logger.info("Step: train")
    logger.info("=" * 60)

    # Set MLflow tracking URI
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    # Enable autologging
    mlflow.tensorflow.autolog(log_models=True)

    logger.info(f"Training data shape: {x_train.shape}")
    logger.info(f"Validation data shape: {x_val.shape}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")

    # Create and compile model
    logger.info("Creating CNN model...")
    model = create_cnn_model(
        input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5, l2_reg=0.001
    )
    model = compile_model(model, learning_rate=learning_rate)

    # Print model summary
    model.summary()

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    # Log parameters to MLflow
    mlflow.log_params(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping_patience": early_stopping_patience,
            "num_train_samples": x_train.shape[0],
            "num_val_samples": x_val.shape[0],
            "model_type": "CNN",
        }
    )

    # Train the model
    logger.info("Starting training...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Log final metrics
    final_train_loss = history.history["loss"][-1]
    final_train_acc = history.history["accuracy"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    logger.info(f"Final Training Loss: {final_train_loss:.4f}")
    logger.info(f"Final Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
    logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")

    # Log to ZenML
    log_artifact_metadata(
        artifact_name="trained_model",
        metadata={
            "final_train_loss": float(final_train_loss),
            "final_train_accuracy": float(final_train_acc),
            "final_val_loss": float(final_val_loss),
            "final_val_accuracy": float(final_val_acc),
            "epochs_trained": len(history.history["loss"]),
        },
    )

    return model


@step(experiment_tracker="mlflow_tracker")
def evaluate(
    model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray
) -> Annotated[Dict[str, Any], "evaluation_metrics"]:
    """
    Evaluate the trained model on test data.

    This step computes various metrics and generates artifacts
    like confusion matrix, classification report, etc.

    Args:
        model: Trained Keras model
        x_test: Preprocessed test images
        y_test: Test labels

    Returns:
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("=" * 60)
    logger.info("Step: evaluate")
    logger.info("=" * 60)

    # Set MLflow tracking URI
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    # Evaluate on test set
    logger.info(f"Evaluating on {x_test.shape[0]} test samples...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

    # Get predictions
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    class_report = classification_report(
        y_test, y_pred, target_names=CIFAR10_CLASSES, output_dict=True
    )

    # Build metrics dictionary
    evaluation_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "per_class_metrics": {
            CIFAR10_CLASSES[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
            for i in range(10)
        },
        "confusion_matrix": cm.tolist(),
    }

    # Log metrics to MLflow
    mlflow.log_metrics(
        {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
        }
    )

    # Create and log confusion matrix plot
    with tempfile.TemporaryDirectory() as tmpdir:
        # Confusion Matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CIFAR10_CLASSES,
            yticklabels=CIFAR10_CLASSES,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        cm_path = Path(tmpdir) / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(cm_path), "plots")

        # Per-class accuracy bar chart
        plt.figure(figsize=(12, 6))
        class_accuracies = [class_report[cls]["precision"] for cls in CIFAR10_CLASSES]
        plt.bar(CIFAR10_CLASSES, class_accuracies)
        plt.title("Per-Class Precision")
        plt.ylabel("Precision")
        plt.xlabel("Class")
        plt.xticks(rotation=45)
        plt.tight_layout()

        accuracy_path = Path(tmpdir) / "per_class_precision.png"
        plt.savefig(accuracy_path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(accuracy_path), "plots")

        # Save classification report as JSON
        report_path = Path(tmpdir) / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(class_report, f, indent=2)
        mlflow.log_artifact(str(report_path), "reports")

    # Log to console
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (weighted): {precision:.4f}")
    logger.info(f"Recall (weighted): {recall:.4f}")
    logger.info(f"F1 Score (weighted): {f1:.4f}")

    # Log metadata to ZenML
    log_artifact_metadata(
        artifact_name="evaluation_metrics",
        metadata={"test_accuracy": float(accuracy), "f1_weighted": float(f1)},
    )

    return evaluation_metrics
