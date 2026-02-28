#!/bin/bash
set -e

echo "================================================"
echo "MLOps Pipeline Runner - Initialization"
echo "================================================"

# Wait for services to be ready
echo "Waiting for MLflow to be ready..."
until curl -s http://mlflow:5000/health > /dev/null 2>&1; do
    echo "Waiting for MLflow..."
    sleep 2
done
echo "MLflow is ready!"

echo "Waiting for MinIO to be ready..."
until curl -s http://minio:9000/minio/health/live > /dev/null 2>&1; do
    echo "Waiting for MinIO..."
    sleep 2
done
echo "MinIO is ready!"

# Configure DVC with MinIO remote (no Git required)
echo "Configuring DVC..."
cd /app

# Initialize DVC without SCM (Git) integration
if [ ! -d "/app/.dvc" ]; then
    dvc init --no-scm
fi

# Configure DVC remote using environment variables
# Credentials come from .env via docker-compose
dvc remote add -d minio s3://dvc-data --force
dvc remote modify minio endpointurl "${DVC_REMOTE_ENDPOINT_URL:-http://minio:9000}"
dvc remote modify minio access_key_id "${AWS_ACCESS_KEY_ID}"
dvc remote modify minio secret_access_key "${AWS_SECRET_ACCESS_KEY}"

echo "DVC configured with MinIO remote"

# Configure MLflow (environment variables already set via docker-compose)
echo "MLflow tracking URI: ${MLFLOW_TRACKING_URI}"

echo "================================================"
echo "Environment ready!"
echo "MLflow UI: http://localhost:5000"
echo "MinIO Console: http://localhost:9001"
echo "ZenML Dashboard: http://localhost:8080"
echo "================================================"
echo ""
echo "Available commands:"
echo "  python scripts/init_data.py    - Download and prepare CIFAR-10 data"
echo "  python run_training.py         - Run training pipeline"
echo "  python run_monitoring.py       - Run monitoring pipeline"
echo ""

# Execute the command passed to docker run
exec "$@"
