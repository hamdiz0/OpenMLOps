"""
Monitoring Steps for Monitoring Pipeline

This module contains ZenML steps for collecting inference data,
running drift detection with Evidently, making trigger decisions,
and storing monitoring artifacts.
"""

import os
import json
import tempfile
from typing import Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
from typing_extensions import Annotated

from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

from evidently import Report
from evidently.presets import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric

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


def images_to_features(images: np.ndarray) -> pd.DataFrame:
    """
    Convert image data to a DataFrame of features for Evidently.

    Since Evidently works with tabular data, we extract statistical
    features from images for drift detection.

    Args:
        images: Array of images (N, H, W, C)

    Returns:
        DataFrame with extracted features
    """
    features = []

    for img in images:
        # Extract per-channel statistics
        img_features = {}

        for c, channel_name in enumerate(["red", "green", "blue"]):
            channel = img[:, :, c]
            img_features[f"{channel_name}_mean"] = channel.mean()
            img_features[f"{channel_name}_std"] = channel.std()
            img_features[f"{channel_name}_min"] = channel.min()
            img_features[f"{channel_name}_max"] = channel.max()
            img_features[f"{channel_name}_median"] = np.median(channel)

        # Overall image statistics
        img_features["brightness"] = img.mean()
        img_features["contrast"] = img.std()

        # Edge statistics (simple gradient)
        gray = img.mean(axis=2)
        dx = np.diff(gray, axis=0)
        dy = np.diff(gray, axis=1)
        img_features["edge_magnitude"] = np.sqrt(
            dx[:-1, :] ** 2 + dy[:, :-1] ** 2
        ).mean()

        features.append(img_features)

    return pd.DataFrame(features)


@step
def collect_inference_data(
    reference_data_path: str = "/app/data",
    sample_size: int = 1000,
    add_drift: bool = True,
    drift_intensity: float = 0.3,
) -> Tuple[
    Annotated[pd.DataFrame, "reference_data"], Annotated[pd.DataFrame, "current_data"]
]:
    """
    Collect inference data and prepare reference data for comparison.

    For demo purposes, this step simulates inference data by:
    1. Loading original test data as reference
    2. Creating synthetic "current" data with optional drift

    In production, this would load actual inference logs.

    Args:
        reference_data_path: Path to reference data
        sample_size: Number of samples to use
        add_drift: Whether to add synthetic drift
        drift_intensity: Intensity of synthetic drift (0-1)

    Returns:
        Tuple of (reference_data, current_data) as DataFrames
    """
    logger.info("=" * 60)
    logger.info("Step: collect_inference_data")
    logger.info("=" * 60)

    # Load reference data (original test set)
    data_dir = Path(reference_data_path)
    x_test = np.load(data_dir / "x_test.npy")

    # Normalize
    x_test = x_test.astype("float32") / 255.0

    # Sample reference data
    np.random.seed(42)
    ref_indices = np.random.choice(
        len(x_test), min(sample_size, len(x_test)), replace=False
    )
    reference_images = x_test[ref_indices]

    # Create current data (simulated inference)
    # In production, this would be actual inference data
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    curr_indices = np.random.choice(
        len(x_test), min(sample_size, len(x_test)), replace=False
    )
    current_images = x_test[curr_indices].copy()

    if add_drift:
        logger.info(f"Adding synthetic drift with intensity {drift_intensity}")
        # Add synthetic drift: brightness shift, noise, etc.

        # Brightness shift
        brightness_shift = drift_intensity * 0.2 * np.random.randn()
        current_images = np.clip(current_images + brightness_shift, 0, 1)

        # Add noise
        noise = np.random.randn(*current_images.shape) * drift_intensity * 0.1
        current_images = np.clip(current_images + noise, 0, 1)

        # Color shift
        color_shift = np.random.randn(3) * drift_intensity * 0.1
        for c in range(3):
            current_images[:, :, :, c] = np.clip(
                current_images[:, :, :, c] + color_shift[c], 0, 1
            )

    # Convert images to feature DataFrames
    logger.info("Extracting features from reference data...")
    reference_df = images_to_features(reference_images)

    logger.info("Extracting features from current data...")
    current_df = images_to_features(current_images)

    logger.info(f"Reference data shape: {reference_df.shape}")
    logger.info(f"Current data shape: {current_df.shape}")

    # Log metadata
    log_artifact_metadata(
        artifact_name="reference_data",
        metadata={
            "num_samples": len(reference_df),
            "num_features": len(reference_df.columns),
        },
    )

    log_artifact_metadata(
        artifact_name="current_data",
        metadata={
            "num_samples": len(current_df),
            "drift_added": add_drift,
            "drift_intensity": drift_intensity if add_drift else 0,
        },
    )

    return reference_df, current_df


@step
def run_evidently_report(
    reference_data: pd.DataFrame, current_data: pd.DataFrame
) -> Tuple[Annotated[Dict[str, Any], "drift_report"], Annotated[str, "report_html"]]:
    """
    Run Evidently drift detection report.

    This step compares reference and current data to detect
    data drift using statistical tests.

    Args:
        reference_data: Reference data (training distribution)
        current_data: Current data (inference distribution)

    Returns:
        Tuple of (drift_report dict, report HTML string)
    """
    logger.info("=" * 60)
    logger.info("Step: run_evidently_report")
    logger.info("=" * 60)

    # Create Evidently report
    logger.info("Creating Evidently Data Drift Report...")

    report = Report(metrics=[DatasetDriftMetric(), DataDriftPreset()])

    # Run the report
    report.run(reference_data=reference_data, current_data=current_data)

    # Get report as dictionary
    report_dict = report.as_dict()

    # Extract key drift information
    drift_results = {
        "timestamp": datetime.now().isoformat(),
        "reference_samples": len(reference_data),
        "current_samples": len(current_data),
        "features_analyzed": list(reference_data.columns),
        "drift_detected": False,
        "drift_share": 0.0,
        "drifted_features": [],
        "feature_drift_scores": {},
    }

    # Parse the report results
    for metric_result in report_dict.get("metrics", []):
        result = metric_result.get("result", {})

        if "drift_share" in result:
            drift_results["drift_share"] = result["drift_share"]
            drift_results["drift_detected"] = result.get("dataset_drift", False)

            # Get per-feature drift
            if "drift_by_columns" in result:
                for col, col_data in result["drift_by_columns"].items():
                    if col_data.get("drift_detected", False):
                        drift_results["drifted_features"].append(col)
                    drift_results["feature_drift_scores"][col] = {
                        "drift_detected": col_data.get("drift_detected", False),
                        "drift_score": col_data.get("drift_score", 0),
                    }

    # Generate HTML report
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        report.save_html(f.name)
        with open(f.name, "r") as html_file:
            report_html = html_file.read()

    logger.info(f"Drift detected: {drift_results['drift_detected']}")
    logger.info(f"Drift share: {drift_results['drift_share']:.2%}")
    logger.info(f"Drifted features: {len(drift_results['drifted_features'])}")

    # Log metadata
    log_artifact_metadata(
        artifact_name="drift_report",
        metadata={
            "drift_detected": drift_results["drift_detected"],
            "drift_share": drift_results["drift_share"],
        },
    )

    return drift_results, report_html


@step
def trigger_decision(
    drift_report: Dict[str, Any], drift_threshold: float = 0.5
) -> Annotated[bool, "should_retrain"]:
    """
    Make a decision on whether to trigger model retraining.

    This step analyzes the drift report and decides if the
    detected drift is significant enough to warrant retraining.

    Args:
        drift_report: Drift detection report from Evidently
        drift_threshold: Threshold for drift share to trigger retrain

    Returns:
        Boolean indicating if retraining should be triggered
    """
    logger.info("=" * 60)
    logger.info("Step: trigger_decision")
    logger.info("=" * 60)

    drift_detected = drift_report.get("drift_detected", False)
    drift_share = drift_report.get("drift_share", 0.0)
    drifted_features = drift_report.get("drifted_features", [])

    logger.info(f"Drift detected: {drift_detected}")
    logger.info(f"Drift share: {drift_share:.2%}")
    logger.info(f"Drift threshold: {drift_threshold:.2%}")
    logger.info(f"Number of drifted features: {len(drifted_features)}")

    # Decision logic
    should_retrain = False
    decision_reason = ""

    if drift_detected and drift_share >= drift_threshold:
        should_retrain = True
        decision_reason = (
            f"Drift share ({drift_share:.2%}) exceeds threshold ({drift_threshold:.2%})"
        )
    elif drift_detected:
        decision_reason = f"Drift detected but share ({drift_share:.2%}) below threshold ({drift_threshold:.2%})"
    else:
        decision_reason = "No significant drift detected"

    logger.info(f"Decision: {'RETRAIN' if should_retrain else 'NO ACTION'}")
    logger.info(f"Reason: {decision_reason}")

    # Log metadata
    log_artifact_metadata(
        artifact_name="should_retrain",
        metadata={
            "decision": should_retrain,
            "reason": decision_reason,
            "drift_share": drift_share,
            "threshold": drift_threshold,
        },
    )

    if should_retrain:
        logger.warning("=" * 60)
        logger.warning("RETRAINING RECOMMENDED!")
        logger.warning("Run: python run_training.py")
        logger.warning("=" * 60)

    return should_retrain


@step
def store_monitoring_artifacts(
    drift_report: Dict[str, Any],
    report_html: str,
    should_retrain: bool,
    output_path: str = "/app/artifacts/monitoring",
) -> Annotated[str, "artifacts_path"]:
    """
    Store monitoring artifacts for persistence and analysis.

    This step saves the drift report, HTML visualization,
    and decision logs to both local storage and MLflow.

    Args:
        drift_report: Drift detection report
        report_html: HTML report string
        should_retrain: Retraining decision
        output_path: Base path for artifact storage

    Returns:
        Path to stored artifacts
    """
    logger.info("=" * 60)
    logger.info("Step: store_monitoring_artifacts")
    logger.info("=" * 60)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path(output_path) / timestamp
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save drift report as JSON
    report_path = artifacts_dir / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2, default=str)
    logger.info(f"Saved drift report: {report_path}")

    # Save HTML report
    html_path = artifacts_dir / "drift_report.html"
    with open(html_path, "w") as f:
        f.write(report_html)
    logger.info(f"Saved HTML report: {html_path}")

    # Save decision summary
    decision_summary = {
        "timestamp": timestamp,
        "drift_detected": drift_report.get("drift_detected", False),
        "drift_share": drift_report.get("drift_share", 0.0),
        "should_retrain": should_retrain,
        "drifted_features": drift_report.get("drifted_features", []),
    }

    decision_path = artifacts_dir / "decision_summary.json"
    with open(decision_path, "w") as f:
        json.dump(decision_summary, f, indent=2)
    logger.info(f"Saved decision summary: {decision_path}")

    # Log to MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    try:
        with mlflow.start_run(run_name=f"monitoring_{timestamp}"):
            # Log metrics
            mlflow.log_metrics(
                {
                    "drift_detected": 1 if drift_report.get("drift_detected") else 0,
                    "drift_share": drift_report.get("drift_share", 0.0),
                    "should_retrain": 1 if should_retrain else 0,
                    "num_drifted_features": len(
                        drift_report.get("drifted_features", [])
                    ),
                }
            )

            # Log artifacts
            mlflow.log_artifact(str(report_path), "monitoring")
            mlflow.log_artifact(str(html_path), "monitoring")
            mlflow.log_artifact(str(decision_path), "monitoring")

            # Log params
            mlflow.log_params(
                {
                    "monitoring_timestamp": timestamp,
                    "reference_samples": drift_report.get("reference_samples", 0),
                    "current_samples": drift_report.get("current_samples", 0),
                }
            )

        logger.info("Artifacts logged to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")

    # Log metadata to ZenML
    log_artifact_metadata(
        artifact_name="artifacts_path",
        metadata={
            "directory": str(artifacts_dir),
            "files": [
                "drift_report.json",
                "drift_report.html",
                "decision_summary.json",
            ],
        },
    )

    logger.info(f"All artifacts stored in: {artifacts_dir}")

    return str(artifacts_dir)
