# MLflow Server with MySQL and S3/MinIO support
FROM ghcr.io/mlflow/mlflow:latest

# Install MySQL driver and AWS SDK for S3/MinIO compatibility
RUN pip install --no-cache-dir \
    pymysql \
    boto3
