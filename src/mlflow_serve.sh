#!/usr/bin/env bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go to project root: (script_dir)/..
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Paths equivalent to Python:
# mlflowdb_path = Path.cwd().parent/"projects"/"mlflow.db"
MLFLOW_DB_PATH="$PROJECT_ROOT/projects/mlflow.db"

# artifact_root = Path.cwd().parent / "artifact"
ARTIFACT_ROOT="$PROJECT_ROOT/artifact"

# Export MLflow tracking URI
export MLFLOW_TRACKING_URI="sqlite:///$MLFLOW_DB_PATH"

echo "Using MLflow DB: $MLFLOW_DB_PATH"
echo "Using Artifacts: $ARTIFACT_ROOT"
echo "Tracking URI:    $MLFLOW_TRACKING_URI"

# Start MLflow server
mlflow server \
  --backend-store-uri "sqlite:///$MLFLOW_DB_PATH" \
  --default-artifact-root "file://$ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port 8080
