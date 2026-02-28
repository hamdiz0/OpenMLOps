"""
ZenML Pipelines for CIFAR-10 MLOps Project

This module exports both main pipelines:
- training_pipeline: Complete ML training workflow
- monitoring_pipeline: Drift detection and monitoring workflow
"""

from src.pipelines.training_pipeline import training_pipeline
from src.pipelines.monitoring_pipeline import monitoring_pipeline

__all__ = [
    "training_pipeline",
    "monitoring_pipeline",
]
