import shutil
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.utils.common import load_json, save_json
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import ModelPromotionConfig


class ModelPromotion:
    def __init__(self, config: ModelPromotionConfig):
        self.config = config

    def load_metrics(self) -> dict:
        metrics_path = Path(self.config.metrics_file_path)
        if not metrics_path.exists():
            raise FileNotFoundError(f"Evaluation metrics not found: {metrics_path}")

        logger.info(f"Loading evaluation metrics from: {metrics_path}")
        metrics = load_json(path=metrics_path)
        return dict(metrics)

    def should_promote(self, metrics: dict) -> bool:
        promote_metric = self.config.promote_metric
        if promote_metric not in metrics:
            raise KeyError(f"Promotion metric '{promote_metric}' not found in evaluation metrics")

        score = float(metrics[promote_metric])
        logger.info(f"Promotion metric '{promote_metric}': {score}")
        promoted = score <= self.config.promote_threshold
        logger.info(
            f"Promotion threshold: {self.config.promote_threshold} -> {'PASS' if promoted else 'FAIL'}"
        )
        return promoted

    def get_latest_registered_version(self) -> object:
        client = MlflowClient(registry_uri=self.config.mlflow_uri)
        logger.info(f"Searching registered model versions for: {self.config.registered_model_name}")
        versions = client.search_model_versions(f"name='{self.config.registered_model_name}'")
        if not versions:
            raise ValueError(f"No registered versions found for model: {self.config.registered_model_name}")

        versions_sorted = sorted(versions, key=lambda v: int(v.version))
        latest = versions_sorted[-1]
        logger.info(
            f"Latest registered model version: {latest.version}, stage: {latest.current_stage}, run_id: {latest.run_id}"
        )
        return latest

    def transition_to_stage(self, version: str) -> str:
        client = MlflowClient(registry_uri=self.config.mlflow_uri)
        logger.info(
            f"Transitioning model '{self.config.registered_model_name}' version {version} "
            f"to stage '{self.config.target_stage}'"
        )
        client.transition_model_version_stage(
            name=self.config.registered_model_name,
            version=version,
            stage=self.config.target_stage,
            archive_existing_versions=self.config.archive_existing_versions
        )
        logger.info(f"Model version {version} transitioned to {self.config.target_stage}")
        return version

    def save_production_copy(self):
        if not self.config.copy_local_model:
            logger.info("Copying local model to production path is disabled.")
            return None

        source_path = Path(self.config.model_file_path)
        production_path = Path(self.config.production_model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Trained model file not found: {source_path}")

        production_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, production_path)
        logger.info(f"Copied model to production path: {production_path}")
        return str(production_path)

    def promote(self) -> dict:
        metrics = self.load_metrics()
        if not self.should_promote(metrics):
            msg = (
                f"Promotion criteria not met. "
                f"Metric {self.config.promote_metric} must be <= {self.config.promote_threshold}"
            )
            logger.info(msg)
            return {
                "promoted": False,
                "reason": msg,
                "metric": self.config.promote_metric,
                "value": float(metrics.get(self.config.promote_metric, 0.0))
            }

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        latest_version = self.get_latest_registered_version()
        promoted_version = self.transition_to_stage(latest_version.version)
        production_path = self.save_production_copy()

        promotion_summary = {
            "promoted": True,
            "registered_model_name": self.config.registered_model_name,
            "version": promoted_version,
            "stage": self.config.target_stage,
            "metric": self.config.promote_metric,
            "metric_value": float(metrics[self.config.promote_metric]),
            "production_model_path": production_path,
        }

        summary_file = Path(self.config.root_dir) / "promotion_summary.json"
        save_json(path=summary_file, data=promotion_summary)
        logger.info(f"Promotion summary saved to: {summary_file}")

        return promotion_summary