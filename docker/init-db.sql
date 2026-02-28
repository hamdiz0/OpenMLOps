-- Create ZenML database
CREATE DATABASE zenml;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE zenml TO mlops;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlops;
