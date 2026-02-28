# CIFAR-10 MLOps Pipeline

A complete MLOps pipeline for training and monitoring a CNN image classifier on the CIFAR-10 dataset.

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Git** | Source code version control |
| **DVC** | Data versioning (remote: MinIO) |
| **MLflow** | Experiment tracking, model registry |
| **ZenML** | Pipeline orchestration |
| **Evidently** | Drift detection & monitoring |
| **Docker** | Containerization |
| **TensorFlow/Keras** | Deep learning framework |

## Architecture

```
                    +------------------+
                    |   MinIO (S3)     |
                    |  Data Storage    |
                    +--------+---------+
                             |
                             | DVC
                             v
+------------+      +------------------+      +------------------+
| PostgreSQL | <--> |  ZenML Server    | <--> | Pipeline Runner  |
|  Database  |      |  Orchestration   |      |  (Training &     |
+------------+      +------------------+      |   Monitoring)    |
                             |               +--------+---------+
                             |                        |
                             v                        | MLflow
                    +------------------+              |
                    |     MLflow       | <------------+
                    |  Tracking Server |
                    +------------------+
```

## Prerequisites

- Docker & Docker Compose
- Git
- 8GB+ RAM recommended

## Quick Start

### 1. Clone and Start Services

```bash
# Start all infrastructure services
docker-compose up -d

# Wait for services to be healthy (about 30-60 seconds)
docker-compose ps
```

### 2. Initialize Data

```bash
# Enter the pipeline runner container
docker-compose exec pipeline-runner bash

# Inside container: Download CIFAR-10 and push to MinIO via DVC
python scripts/init_data.py
```

### 3. Run Training Pipeline

```bash
# Inside pipeline-runner container
python run_training.py

# Or with custom parameters
python run_training.py --epochs 30 --batch-size 128 --learning-rate 0.0001
```

### 4. Run Monitoring Pipeline

```bash
# Inside pipeline-runner container
python run_monitoring.py

# Without synthetic drift (real-world scenario)
python run_monitoring.py --no-drift

# With high drift intensity (to trigger retrain)
python run_monitoring.py --drift-intensity 0.5 --drift-threshold 0.3
```

## Web UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| **MLflow** | http://localhost:5000 | None |
| **ZenML** | http://localhost:8080 | default / (no password) |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |

## Project Structure

```
ml-ops-challenge/
├── docker-compose.yml          # All services configuration
├── requirements.txt            # Python dependencies
├── run_training.py             # Training pipeline entry point
├── run_monitoring.py           # Monitoring pipeline entry point
├── docker/
│   ├── Dockerfile              # Pipeline runner image
│   ├── entrypoint.sh           # Container initialization
│   └── init-db.sql             # PostgreSQL initialization
├── .dvc/
│   └── config                  # DVC remote configuration
├── scripts/
│   └── init_data.py            # CIFAR-10 download script
└── src/
    ├── model/
    │   └── cnn.py              # CNN architecture definition
    ├── steps/
    │   ├── data_steps.py       # Data ingestion, validation, preprocessing
    │   ├── training_steps.py   # Model training and evaluation
    │   ├── model_steps.py      # Model registration and export
    │   └── monitoring_steps.py # Drift detection steps
    └── pipelines/
        ├── training_pipeline.py   # Training workflow
        └── monitoring_pipeline.py # Monitoring workflow
```

## Pipelines

### Training Pipeline

```
ingest_data → validate_data → split_data → preprocess → train → evaluate → register_model → export_model
```

**Steps:**
1. **ingest_data**: Pull data from MinIO via DVC
2. **validate_data**: Check data integrity and statistics
3. **split_data**: Create train/validation split
4. **preprocess**: Normalize pixel values
5. **train**: Train CNN with MLflow tracking
6. **evaluate**: Compute metrics, confusion matrix
7. **register_model**: Register to MLflow Model Registry
8. **export_model**: Save as TensorFlow SavedModel

### Monitoring Pipeline

```
collect_inference_data → run_evidently_report → trigger_decision → store_monitoring_artifacts
```

**Steps:**
1. **collect_inference_data**: Load reference & current data
2. **run_evidently_report**: Detect drift with Evidently
3. **trigger_decision**: Determine if retrain needed
4. **store_monitoring_artifacts**: Save reports to MLflow

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--learning-rate` | 0.001 | Learning rate |
| `--validation-split` | 0.2 | Validation fraction |
| `--early-stopping-patience` | 5 | Early stopping patience |
| `--model-name` | cifar10-cnn-classifier | MLflow model name |

### Monitoring Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sample-size` | 1000 | Samples for comparison |
| `--add-drift` | True | Add synthetic drift |
| `--no-drift` | - | Disable synthetic drift |
| `--drift-intensity` | 0.3 | Drift intensity (0-1) |
| `--drift-threshold` | 0.5 | Retrain threshold |

## Services Configuration

### Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| `minio` | 9000, 9001 | S3-compatible storage |
| `mysql` | 3306 | Database for ZenML & MLflow |
| `mlflow` | 5000 | Experiment tracking |
| `zenml-server` | 8080 | Pipeline orchestration |
| `pipeline-runner` | - | Pipeline execution environment |

### Environment Variables

```bash
# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# ZenML
ZENML_STORE_URL=http://zenml-server:8080

# DVC (for MinIO)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

## Model Architecture

CNN architecture for CIFAR-10 (32x32x3 images, 10 classes):

```
Input (32x32x3)
    │
    ├── Conv2D(32, 3x3) + ReLU + BatchNorm
    ├── Conv2D(32, 3x3) + ReLU + BatchNorm
    ├── MaxPool2D(2x2) + Dropout(0.25)
    │
    ├── Conv2D(64, 3x3) + ReLU + BatchNorm
    ├── Conv2D(64, 3x3) + ReLU + BatchNorm
    ├── MaxPool2D(2x2) + Dropout(0.25)
    │
    ├── Conv2D(128, 3x3) + ReLU + BatchNorm
    ├── Conv2D(128, 3x3) + ReLU + BatchNorm
    ├── MaxPool2D(2x2) + Dropout(0.25)
    │
    ├── Flatten
    ├── Dense(512) + ReLU + BatchNorm + Dropout(0.5)
    └── Dense(10) + Softmax
    
Output (10 classes)
```

## Troubleshooting

### Services not starting

```bash
# Check logs
docker-compose logs -f

# Restart specific service
docker-compose restart zenml-server
```

### DVC push fails

```bash
# Verify MinIO bucket exists
docker-compose exec minio mc ls local/dvc-storage

# Check DVC remote config
dvc remote list -v
```

### ZenML connection issues

```bash
# Re-connect to ZenML server
zenml connect --url http://zenml-server:8080

# Check active stack
zenml stack describe
```

### MLflow not tracking

```bash
# Verify MLflow is accessible
curl http://localhost:5000/health

# Check tracking URI inside container
echo $MLFLOW_TRACKING_URI
```

## Cleanup

```bash
# Stop all services
docker-compose down

# Remove all data (volumes)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## License

MIT
