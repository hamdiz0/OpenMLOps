"""
Data Processing Steps for Training Pipeline

This module contains ZenML steps for data ingestion, validation,
splitting, and preprocessing.
"""

import os
import subprocess
from typing import Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from typing_extensions import Annotated

from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

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


@step
def ingest_data() -> Tuple[
    Annotated[np.ndarray, "x_train"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "x_test"],
    Annotated[np.ndarray, "y_test"],
]:
    """
    Ingest CIFAR-10 data from DVC-managed storage.

    This step pulls data using DVC from MinIO remote storage (no Git required).
    DVC is configured in --no-scm mode for standalone operation.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    logger.info("=" * 60)
    logger.info("Step: ingest_data")
    logger.info("=" * 60)

    data_dir = Path("/app/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if data exists locally
    required_files = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]
    files_exist = all((data_dir / f).exists() for f in required_files)

    if not files_exist:
        logger.info("Data not found locally. Attempting DVC pull...")
        try:
            # DVC pull from MinIO (no Git required, using --no-scm mode)
            result = subprocess.run(
                ["dvc", "pull"], cwd="/app", capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("DVC pull successful")
                if result.stdout:
                    logger.info(f"DVC output: {result.stdout}")
            else:
                logger.warning(f"DVC pull returned non-zero: {result.stderr}")
                # Check if files exist now despite warning
                files_exist = all((data_dir / f).exists() for f in required_files)
                if not files_exist:
                    raise RuntimeError(
                        "Data not available. Please run 'python scripts/init_data.py' first."
                    )
        except FileNotFoundError:
            raise RuntimeError(
                "DVC not installed. Please ensure DVC is available in the environment."
            )
        except Exception as e:
            logger.error(f"DVC pull failed: {e}")
            raise RuntimeError(
                f"Failed to pull data. Run 'python scripts/init_data.py' first. Error: {e}"
            )

    # Verify files exist before loading
    for f in required_files:
        if not (data_dir / f).exists():
            raise FileNotFoundError(
                f"Required file {f} not found. Run 'python scripts/init_data.py' first."
            )

    # Load data
    logger.info("Loading data from numpy files...")
    x_train = np.load(data_dir / "x_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    x_test = np.load(data_dir / "x_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    # Log metadata
    log_artifact_metadata(
        artifact_name="x_train",
        metadata={
            "shape": list(x_train.shape),
            "dtype": str(x_train.dtype),
            "num_samples": int(x_train.shape[0]),
        },
    )

    logger.info(f"Loaded x_train: {x_train.shape}")
    logger.info(f"Loaded y_train: {y_train.shape}")
    logger.info(f"Loaded x_test: {x_test.shape}")
    logger.info(f"Loaded y_test: {y_test.shape}")

    return x_train, y_train, x_test, y_test


@step
def validate_data(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> Annotated[Dict[str, Any], "validation_report"]:
    """
    Validate the ingested data for quality issues.

    Checks performed:
    - Shape validation
    - Value range validation
    - Label distribution check
    - Missing/NaN value check

    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels

    Returns:
        Validation report dictionary
    """
    logger.info("=" * 60)
    logger.info("Step: validate_data")
    logger.info("=" * 60)

    validation_report = {"is_valid": True, "checks": {}}

    # Check 1: Shape validation
    expected_image_shape = (32, 32, 3)
    train_image_shape = x_train.shape[1:]
    test_image_shape = x_test.shape[1:]

    shape_check = {
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "expected_image_shape": list(expected_image_shape),
        "train_shape_valid": train_image_shape == expected_image_shape,
        "test_shape_valid": test_image_shape == expected_image_shape,
    }
    validation_report["checks"]["shape"] = shape_check

    if not (shape_check["train_shape_valid"] and shape_check["test_shape_valid"]):
        validation_report["is_valid"] = False
        logger.error("Shape validation failed!")
    else:
        logger.info("Shape validation: PASSED")

    # Check 2: Value range (should be 0-255 for raw images)
    value_range_check = {
        "train_min": float(x_train.min()),
        "train_max": float(x_train.max()),
        "test_min": float(x_test.min()),
        "test_max": float(x_test.max()),
        "train_valid": 0 <= x_train.min() and x_train.max() <= 255,
        "test_valid": 0 <= x_test.min() and x_test.max() <= 255,
    }
    validation_report["checks"]["value_range"] = value_range_check

    if not (value_range_check["train_valid"] and value_range_check["test_valid"]):
        validation_report["is_valid"] = False
        logger.error("Value range validation failed!")
    else:
        logger.info("Value range validation: PASSED")

    # Check 3: Label validation (should be 0-9)
    unique_train_labels = np.unique(y_train)
    unique_test_labels = np.unique(y_test)

    label_check = {
        "num_classes": 10,
        "train_unique_labels": unique_train_labels.tolist(),
        "test_unique_labels": unique_test_labels.tolist(),
        "train_valid": len(unique_train_labels) == 10,
        "test_valid": len(unique_test_labels) == 10,
    }
    validation_report["checks"]["labels"] = label_check

    if not (label_check["train_valid"] and label_check["test_valid"]):
        validation_report["is_valid"] = False
        logger.error("Label validation failed!")
    else:
        logger.info("Label validation: PASSED")

    # Check 4: NaN/Inf check
    nan_check = {
        "train_has_nan": bool(np.isnan(x_train).any()),
        "train_has_inf": bool(np.isinf(x_train).any()),
        "test_has_nan": bool(np.isnan(x_test).any()),
        "test_has_inf": bool(np.isinf(x_test).any()),
    }
    nan_check["is_valid"] = not any(
        [
            nan_check["train_has_nan"],
            nan_check["train_has_inf"],
            nan_check["test_has_nan"],
            nan_check["test_has_inf"],
        ]
    )
    validation_report["checks"]["nan_inf"] = nan_check

    if not nan_check["is_valid"]:
        validation_report["is_valid"] = False
        logger.error("NaN/Inf check failed!")
    else:
        logger.info("NaN/Inf check: PASSED")

    # Check 5: Class distribution
    train_label_counts = np.bincount(y_train.flatten(), minlength=10)
    test_label_counts = np.bincount(y_test.flatten(), minlength=10)

    distribution_check = {
        "train_distribution": {
            CIFAR10_CLASSES[i]: int(count) for i, count in enumerate(train_label_counts)
        },
        "test_distribution": {
            CIFAR10_CLASSES[i]: int(count) for i, count in enumerate(test_label_counts)
        },
    }
    validation_report["checks"]["distribution"] = distribution_check
    logger.info("Class distribution computed")

    # Summary
    logger.info(f"Validation complete. Is valid: {validation_report['is_valid']}")

    log_artifact_metadata(
        artifact_name="validation_report",
        metadata={"is_valid": validation_report["is_valid"]},
    )

    return validation_report


@step
def split_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[
    Annotated[np.ndarray, "x_train_split"],
    Annotated[np.ndarray, "y_train_split"],
    Annotated[np.ndarray, "x_val"],
    Annotated[np.ndarray, "y_val"],
]:
    """
    Split training data into train and validation sets.

    Uses stratified splitting to maintain class distribution.

    Args:
        x_train: Training images
        y_train: Training labels
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (x_train_split, y_train_split, x_val, y_val)
    """
    from sklearn.model_selection import train_test_split

    logger.info("=" * 60)
    logger.info("Step: split_data")
    logger.info("=" * 60)

    logger.info(f"Input shape: {x_train.shape}")
    logger.info(f"Validation split: {validation_split}")

    # Stratified split
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train,
        y_train,
        test_size=validation_split,
        stratify=y_train,
        random_state=random_seed,
    )

    logger.info(f"Training set: {x_train_split.shape[0]} samples")
    logger.info(f"Validation set: {x_val.shape[0]} samples")

    # Log metadata
    log_artifact_metadata(
        artifact_name="x_train_split",
        metadata={
            "num_samples": int(x_train_split.shape[0]),
            "validation_split": validation_split,
        },
    )

    return x_train_split, y_train_split, x_val, y_val


@step
def preprocess(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[
    Annotated[np.ndarray, "x_train_processed"],
    Annotated[np.ndarray, "y_train_processed"],
    Annotated[np.ndarray, "x_val_processed"],
    Annotated[np.ndarray, "y_val_processed"],
    Annotated[np.ndarray, "x_test_processed"],
    Annotated[np.ndarray, "y_test_processed"],
]:
    """
    Preprocess the data for training.

    Preprocessing steps:
    - Normalize pixel values to [0, 1]
    - Ensure correct data types
    - Flatten labels if needed

    Args:
        x_train: Training images
        y_train: Training labels
        x_val: Validation images
        y_val: Validation labels
        x_test: Test images
        y_test: Test labels

    Returns:
        Preprocessed data tuple
    """
    logger.info("=" * 60)
    logger.info("Step: preprocess")
    logger.info("=" * 60)

    # Normalize to [0, 1]
    logger.info("Normalizing pixel values to [0, 1]...")
    x_train_processed = x_train.astype("float32") / 255.0
    x_val_processed = x_val.astype("float32") / 255.0
    x_test_processed = x_test.astype("float32") / 255.0

    # Flatten labels if needed (from (n, 1) to (n,))
    y_train_processed = y_train.flatten().astype("int32")
    y_val_processed = y_val.flatten().astype("int32")
    y_test_processed = y_test.flatten().astype("int32")

    # Compute and log statistics
    train_mean = x_train_processed.mean()
    train_std = x_train_processed.std()

    logger.info(f"Training data - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
    logger.info(
        f"Value range: [{x_train_processed.min():.4f}, {x_train_processed.max():.4f}]"
    )

    log_artifact_metadata(
        artifact_name="x_train_processed",
        metadata={
            "mean": float(train_mean),
            "std": float(train_std),
            "min": float(x_train_processed.min()),
            "max": float(x_train_processed.max()),
        },
    )

    logger.info("Preprocessing complete!")
    logger.info(f"  x_train: {x_train_processed.shape}")
    logger.info(f"  x_val: {x_val_processed.shape}")
    logger.info(f"  x_test: {x_test_processed.shape}")

    return (
        x_train_processed,
        y_train_processed,
        x_val_processed,
        y_val_processed,
        x_test_processed,
        y_test_processed,
    )
