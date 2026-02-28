#!/usr/bin/env python3
"""
Run Monitoring Pipeline

Entry point script for executing the CIFAR-10 drift detection pipeline.
This script runs Evidently-based drift detection and outputs retraining recommendations.

Usage:
    python run_monitoring.py [options]

Options:
    --sample-size INT        Number of samples for comparison (default: 1000)
    --add-drift              Add synthetic drift for demo (default: True)
    --no-drift               Run without synthetic drift
    --drift-intensity FLOAT  Intensity of synthetic drift (default: 0.3)
    --drift-threshold FLOAT  Threshold for retrain trigger (default: 0.5)
"""

import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_zenml_stack():
    """Initialize ZenML with the configured stack."""
    from zenml.client import Client

    client = Client()

    # Check if we're connected to the ZenML server
    logger.info(f"ZenML version: {client.zen_store.info.version}")
    logger.info(f"Active stack: {client.active_stack_model.name}")

    return client


def main():
    """Main entry point for monitoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 Drift Detection Pipeline"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples for drift comparison",
    )
    parser.add_argument(
        "--add-drift",
        action="store_true",
        default=True,
        help="Add synthetic drift for demo purposes",
    )
    parser.add_argument(
        "--no-drift", action="store_true", help="Run without synthetic drift"
    )
    parser.add_argument(
        "--drift-intensity",
        type=float,
        default=0.3,
        help="Intensity of synthetic drift (0.0 to 1.0)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.5,
        help="Threshold for drift share to trigger retraining",
    )

    args = parser.parse_args()

    # Handle --no-drift flag
    add_drift = not args.no_drift

    logger.info("=" * 60)
    logger.info("CIFAR-10 Drift Detection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Sample Size: {args.sample_size}")
    logger.info(f"Add Synthetic Drift: {add_drift}")
    if add_drift:
        logger.info(f"Drift Intensity: {args.drift_intensity}")
    logger.info(f"Drift Threshold: {args.drift_threshold}")
    logger.info("=" * 60)

    # Setup ZenML
    logger.info("Initializing ZenML stack...")
    try:
        client = setup_zenml_stack()
    except Exception as e:
        logger.error(f"Failed to initialize ZenML: {e}")
        logger.info("Make sure ZenML server is running and accessible")
        sys.exit(1)

    # Import and run pipeline
    logger.info("Importing monitoring pipeline...")
    from src.pipelines.monitoring_pipeline import monitoring_pipeline

    logger.info("Starting monitoring pipeline...")
    try:
        result = monitoring_pipeline(
            sample_size=args.sample_size,
            add_drift=add_drift,
            drift_intensity=args.drift_intensity,
            drift_threshold=args.drift_threshold,
        )

        logger.info("=" * 60)
        logger.info("Monitoring Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info("Check MLflow UI at http://localhost:5000 for monitoring artifacts")
        logger.info("Check ZenML Dashboard at http://localhost:8080 for pipeline runs")
        logger.info("")
        logger.info("If drift was detected and retraining recommended, run:")
        logger.info("  python run_training.py")

    except Exception as e:
        logger.error(f"Monitoring pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
