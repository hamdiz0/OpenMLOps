"""Steps module initialization."""

from src.steps.data_steps import ingest_data, validate_data, split_data, preprocess

from src.steps.training_steps import train, evaluate

from src.steps.model_steps import register_model, export_model

from src.steps.monitoring_steps import (
    collect_inference_data,
    run_evidently_report,
    trigger_decision,
    store_monitoring_artifacts,
)

__all__ = [
    # Data steps
    "ingest_data",
    "validate_data",
    "split_data",
    "preprocess",
    # Training steps
    "train",
    "evaluate",
    # Model steps
    "register_model",
    "export_model",
    # Monitoring steps
    "collect_inference_data",
    "run_evidently_report",
    "trigger_decision",
    "store_monitoring_artifacts",
]
