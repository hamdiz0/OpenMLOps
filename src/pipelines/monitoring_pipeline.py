"""
Monitoring Pipeline for CIFAR-10 CNN Classifier

This pipeline orchestrates drift detection and monitoring:
1. collect_inference_data - Gather reference and current data
2. run_evidently_report - Run drift detection with Evidently
3. trigger_decision - Determine if retraining is needed
4. store_monitoring_artifacts - Save reports to MLflow and local storage
"""

from zenml import pipeline
from zenml.logger import get_logger

from src.steps.monitoring_steps import (
    collect_inference_data,
    run_evidently_report,
    trigger_decision,
    store_monitoring_artifacts,
)

logger = get_logger(__name__)


@pipeline(
    name="monitoring_pipeline",
    enable_cache=False,  # Always run fresh for monitoring
)
def monitoring_pipeline(
    sample_size: int = 1000,
    add_drift: bool = True,
    drift_intensity: float = 0.3,
    drift_threshold: float = 0.5,
):
    """
    Monitoring pipeline for drift detection and retrain triggering.

    This pipeline implements the complete monitoring workflow with:
    - Inference data collection (simulated for demo)
    - Drift detection using Evidently
    - Retrain decision logic
    - Artifact storage in MLflow

    Args:
        sample_size: Number of samples for drift comparison
        add_drift: Whether to add synthetic drift (for demo)
        drift_intensity: Intensity of synthetic drift (0-1)
        drift_threshold: Threshold for drift share to trigger retrain
    """
    logger.info("=" * 60)
    logger.info("Starting Monitoring Pipeline")
    logger.info("=" * 60)

    # Step 1: Collect inference data and reference data
    reference_data, current_data = collect_inference_data(
        sample_size=sample_size,
        add_drift=add_drift,
        drift_intensity=drift_intensity,
    )

    # Step 2: Run Evidently drift detection report
    drift_report, report_html = run_evidently_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    # Step 3: Make trigger decision based on drift
    should_retrain = trigger_decision(
        drift_report=drift_report,
        drift_threshold=drift_threshold,
    )

    # Step 4: Store monitoring artifacts
    artifacts_path = store_monitoring_artifacts(
        drift_report=drift_report,
        report_html=report_html,
        should_retrain=should_retrain,
    )

    logger.info("=" * 60)
    logger.info("Monitoring Pipeline Complete!")
    logger.info("=" * 60)

    return artifacts_path
