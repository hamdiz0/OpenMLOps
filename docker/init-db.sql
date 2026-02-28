-- MySQL initialization script
-- Creates databases for MLflow and ZenML

-- Create ZenML database (mlflow database is created via MYSQL_DATABASE env var)
CREATE DATABASE IF NOT EXISTS zenml;

-- Grant privileges to the mlops user
GRANT ALL PRIVILEGES ON mlflow.* TO 'mlops'@'%';
GRANT ALL PRIVILEGES ON zenml.* TO 'mlops'@'%';
FLUSH PRIVILEGES;
