"""
Training Pipeline for CIFAR-10 CNN Classifier

This pipeline orchestrates the complete training workflow:
1. ingest_data - Get dataset via DVC
2. validate_data - Check data quality
3. split_data - Split into train/val sets
4. preprocess - Normalize and prepare data
5. train - Train CNN model with MLflow tracking
6. evaluate - Compute metrics and artifacts
7. register_model - Register to MLflow Model Registry
8. export_model - Save in serving-ready format
"""

from zenml import pipeline
from zenml.logger import get_logger

from src.steps.data_steps import ingest_data, validate_data, split_data, preprocess
from src.steps.training_steps import train, evaluate
from src.steps.model_steps import register_model, export_model

logger = get_logger(__name__)


@pipeline(
    name="training_pipeline",
    enable_cache=False,  # Disable caching for demo purposes
)
def training_pipeline(
    validation_split: float = 0.2,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 5,
    model_name: str = "cifar10-cnn-classifier",
):
    """
    Training pipeline for CIFAR-10 CNN image classifier.

    This pipeline implements the complete MLOps training workflow with:
    - Data versioning via DVC
    - Experiment tracking via MLflow
    - Model registration and versioning
    - Artifact management

    Args:
        validation_split: Fraction of training data for validation
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Patience for early stopping
        model_name: Name for model registration
    """
    logger.info("=" * 60)
    logger.info("Starting Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Ingest data via DVC
    x_train, y_train, x_test, y_test = ingest_data()

    # Step 2: Validate data quality
    validation_report = validate_data(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )

    # Step 3: Split data into train/validation sets
    x_train_split, y_train_split, x_val, y_val = split_data(
        x_train=x_train, y_train=y_train, validation_split=validation_split
    )

    # Step 4: Preprocess data (normalization)
    (
        x_train_processed,
        y_train_processed,
        x_val_processed,
        y_val_processed,
        x_test_processed,
        y_test_processed,
    ) = preprocess(
        x_train=x_train_split,
        y_train=y_train_split,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    # Step 5: Train CNN model
    trained_model = train(
        x_train=x_train_processed,
        y_train=y_train_processed,
        x_val=x_val_processed,
        y_val=y_val_processed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
    )

    # Step 6: Evaluate model on test set
    evaluation_metrics = evaluate(
        model=trained_model, x_test=x_test_processed, y_test=y_test_processed
    )

    # Step 7: Register model to MLflow Registry
    model_version = register_model(
        model=trained_model,
        evaluation_metrics=evaluation_metrics,
        model_name=model_name,
    )

    # Step 8: Export model in serving format
    exported_path = export_model(model=trained_model, model_version=model_version)

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete!")
    logger.info("=" * 60)

    return exported_path
