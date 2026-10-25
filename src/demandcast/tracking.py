"""MLflow experiment tracking and local metadata logging for DemandCast."""

import logging
from pathlib import Path
from typing import Any

import mlflow

from demandcast.config import TrackingConfig, get_project_root, load_config

logger = logging.getLogger(__name__)


def init_mlflow(tracking_config: TrackingConfig | None = None) -> str:
    """Initialize local MLflow tracking server URI and experiment name."""
    config = tracking_config or load_config().tracking
    root = get_project_root()

    # Configure local SQLite database tracking URI
    # On Windows, sqlite:///C:/path/mlflow.db or relative sqlite:///mlflow.db
    db_path = root / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    # Configure artifact storage
    artifact_dir = root / "mlruns"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Set or create experiment
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=config.experiment_name,
            artifact_location=artifact_dir.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(config.experiment_name)
    logger.info(f"MLflow tracking initialized: URI={tracking_uri}, Experiment={config.experiment_name} (ID: {experiment_id})")
    return tracking_uri


def log_experiment_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log parameters, metrics, and artifact files into an MLflow run.

    Returns the unique mlflow run_id.
    """
    init_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log parameters (flatten lists/dicts if needed)
        flat_params = {}
        for k, v in params.items():
            if isinstance(v, (list, dict)):
                flat_params[k] = str(v)
            else:
                flat_params[k] = v
        mlflow.log_params(flat_params)

        # Log numeric metrics
        numeric_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                numeric_metrics[k] = float(v)
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log artifacts
        if artifacts:
            for art_name, art_path in artifacts.items():
                if art_path.exists():
                    mlflow.log_artifact(str(art_path))

        logger.info(f"Logged run '{run_name}' to MLflow. Run ID: {run_id}")
        return run_id
