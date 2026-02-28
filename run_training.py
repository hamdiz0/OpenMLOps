#!/usr/bin/env python3
"""
Run Training Pipeline

Entry point script for executing the CIFAR-10 CNN training pipeline.
This script initializes the ZenML stack and runs the training workflow.

Usage:
    python run_training.py [options]

Options:
    --epochs INT           Number of training epochs (default: 20)
    --batch-size INT       Training batch size (default: 64)
    --learning-rate FLOAT  Learning rate (default: 0.001)
    --validation-split FLOAT  Validation split ratio (default: 0.2)
    --model-name STR       Model name for registry (default: cifar10-cnn-classifier)
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
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="Run CIFAR-10 CNN Training Pipeline")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cifar10-cnn-classifier",
        help="Model name for MLflow registry",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CIFAR-10 CNN Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Validation Split: {args.validation_split}")
    logger.info(f"Early Stopping Patience: {args.early_stopping_patience}")
    logger.info(f"Model Name: {args.model_name}")
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
    logger.info("Importing training pipeline...")
    from src.pipelines.training_pipeline import training_pipeline

    logger.info("Starting training pipeline...")
    try:
        result = training_pipeline(
            validation_split=args.validation_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            model_name=args.model_name,
        )

        logger.info("=" * 60)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info("Check MLflow UI at http://localhost:5000 for experiment results")
        logger.info("Check ZenML Dashboard at http://localhost:8080 for pipeline runs")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
