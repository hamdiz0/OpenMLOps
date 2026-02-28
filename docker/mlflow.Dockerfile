# MLflow Server with PostgreSQL and S3/MinIO support
# Based on official MLflow image with additional drivers

FROM ghcr.io/mlflow/mlflow:v2.15.1

# Install PostgreSQL driver and AWS SDK for S3/MinIO compatibility
RUN pip install --no-cache-dir \
    psycopg2-binary \
    boto3
