#!/bin/bash
# PostgreSQL initialization script
# This script runs on first container startup to create required databases

set -e

# Use environment variables with defaults
ZENML_DB="${ZENML_DATABASE:-zenml}"
POSTGRES_USER="${POSTGRES_USER:-mlops}"

echo "Creating database: $ZENML_DB"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create ZenML database if it doesn't exist
    SELECT 'CREATE DATABASE $ZENML_DB'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$ZENML_DB')\gexec

    -- Grant privileges
    GRANT ALL PRIVILEGES ON DATABASE $ZENML_DB TO $POSTGRES_USER;
EOSQL

echo "Database initialization complete"
