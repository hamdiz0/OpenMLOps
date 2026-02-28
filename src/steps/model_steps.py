"""
Model Registration and Export Steps for Training Pipeline

This module contains ZenML steps for registering models to MLflow
Model Registry and exporting to serving-ready formats.
"""

import os
import tempfile
from typing import Dict, Any
from pathlib import Path
import json

import numpy as np
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from typing_extensions import Annotated

from zenml import step, log_artifact_metadata, get_step_context
from zenml.logger import get_logger

import tensorflow as tf

logger = get_logger(__name__)


@step(experiment_tracker="mlflow_tracker")
def register_model(
    model: tf.keras.Model,
    evaluation_metrics: Dict[str, Any],
    model_name: str = "cifar10-cnn-classifier",
) -> Annotated[str, "model_version"]:
    """
    Register the trained model to MLflow Model Registry.

    This step registers the model with its metadata and metrics,
    making it available for versioning and deployment.

    Args:
        model: Trained Keras model
        evaluation_metrics: Evaluation metrics dictionary
        model_name: Name for the registered model

    Returns:
        Model version string
    """
    logger.info("=" * 60)
    logger.info("Step: register_model")
    logger.info("=" * 60)

    # Set MLflow tracking URI
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    # Get or create an active run
    active_run = mlflow.active_run()
    if active_run is None:
        # Start a new run if none is active
        mlflow.start_run(run_name="model_registration")
        active_run = mlflow.active_run()

    run_id = active_run.info.run_id
    logger.info(f"Active MLflow run ID: {run_id}")

    # Log model with signature
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample input for signature
        sample_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
        sample_output = model.predict(sample_input)

        # Infer signature
        from mlflow.models.signature import infer_signature

        signature = infer_signature(sample_input, sample_output)

        # Log model
        logger.info(f"Logging model to MLflow...")
        model_info = mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
            input_example=sample_input,
        )

        logger.info(f"Model logged to: {model_info.model_uri}")

    # Get the model version
    client = MlflowClient()

    # Get the latest version of the model
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if model_versions:
            latest_version = max([int(mv.version) for mv in model_versions])
            model_version = str(latest_version)

            # Add tags to the model version
            client.set_model_version_tag(
                model_name,
                model_version,
                "test_accuracy",
                str(evaluation_metrics.get("test_accuracy", 0)),
            )
            client.set_model_version_tag(
                model_name,
                model_version,
                "f1_weighted",
                str(evaluation_metrics.get("f1_weighted", 0)),
            )

            logger.info(f"Registered model version: {model_version}")
        else:
            model_version = "1"
            logger.info("First model version registered")
    except Exception as e:
        logger.warning(f"Could not get model version: {e}")
        model_version = "unknown"

    # Log metrics as model properties
    mlflow.log_params(
        {"registered_model_name": model_name, "model_version": model_version}
    )

    # Log to ZenML
    log_artifact_metadata(
        artifact_name="model_version",
        metadata={"model_name": model_name, "version": model_version, "run_id": run_id},
    )

    logger.info(
        f"Model '{model_name}' version {model_version} registered successfully!"
    )

    return model_version


@step
def export_model(
    model: tf.keras.Model, model_version: str, export_path: str = "/app/models"
) -> Annotated[str, "exported_model_path"]:
    """
    Export the model in a serving-ready format.

    This step exports the model in TensorFlow SavedModel format,
    which is compatible with TensorFlow Serving, TFLite, and other
    deployment solutions.

    Args:
        model: Trained Keras model
        model_version: Version string from registration
        export_path: Base path for model export

    Returns:
        Path to the exported model
    """
    logger.info("=" * 60)
    logger.info("Step: export_model")
    logger.info("=" * 60)

    # Create export directory
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Create versioned model path
    model_export_path = export_dir / f"cifar10_cnn_v{model_version}"

    logger.info(f"Exporting model to: {model_export_path}")

    # Export as SavedModel format
    model.save(str(model_export_path), save_format="tf")

    # Also save as H5 for compatibility
    h5_path = export_dir / f"cifar10_cnn_v{model_version}.h5"
    model.save(str(h5_path), save_format="h5")

    # Create model metadata file
    metadata = {
        "model_name": "cifar10-cnn-classifier",
        "version": model_version,
        "format": "tensorflow_savedmodel",
        "input_shape": [None, 32, 32, 3],
        "output_shape": [None, 10],
        "classes": [
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
        ],
        "preprocessing": {"normalization": "divide_by_255", "input_dtype": "float32"},
    }

    metadata_path = model_export_path / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model exported successfully!")
    logger.info(f"  SavedModel: {model_export_path}")
    logger.info(f"  H5 format: {h5_path}")
    logger.info(f"  Metadata: {metadata_path}")

    # Log to MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    if mlflow.active_run():
        mlflow.log_artifact(str(metadata_path), "model_export")

    # Log to ZenML
    log_artifact_metadata(
        artifact_name="exported_model_path",
        metadata={
            "savedmodel_path": str(model_export_path),
            "h5_path": str(h5_path),
            "version": model_version,
        },
    )

    return str(model_export_path)
