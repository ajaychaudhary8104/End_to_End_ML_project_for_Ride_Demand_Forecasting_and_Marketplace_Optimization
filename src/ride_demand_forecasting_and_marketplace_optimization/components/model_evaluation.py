import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import pickle
from src.ride_demand_forecasting_and_marketplace_optimization.utils.common import save_json
from pathlib import Path
from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import ModelEvaluationConfig



class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _convert_numpy_types(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._convert_numpy_types(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [
                self._convert_numpy_types(item)
                for item in obj
            ]
        return obj

    def load_data(self, data_path: Path) -> pd.DataFrame:
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found: {data_path}"
            )

        logger.info(f"Loading data from {data_path}")
        return pd.read_parquet(data_path)

    def load_model(self, model_path: Path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        logger.info(f"Loading model from {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

    def eval_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Forecasting evaluation metrics.
        """
        y_true = np.asarray(
            y_true,
            dtype=np.float64
        )

        y_pred = np.asarray(
            y_pred,
            dtype=np.float64
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        mse = mean_squared_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_true,
            y_pred
        )

        eps = 1e-10

        mape = np.mean(
            np.abs(
                (y_true - y_pred)
                / np.maximum(
                    np.abs(y_true),
                    eps
                )
            )
        ) * 100

        smape = (
            np.mean(
                2
                * np.abs(y_pred - y_true)
                / (
                    np.abs(y_true)
                    + np.abs(y_pred)
                    + eps
                )
            )
            * 100
        )

        wape = (
            np.sum(
                np.abs(
                    y_true - y_pred
                )
            )
            / (
                np.sum(
                    np.abs(y_true)
                )
                + eps
            )
        ) * 100

        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
            "smape": float(smape),
            "wape": float(wape),
        }
        
        print("MAE :", mae)
        print("MAPE:", mape)
        print("SMAPE:", smape)
        print("WAPE:", wape)

        return metrics

    def create_prediction_artifact(self,y_true,y_pred):
        """
        Create actual vs forecast artifact.
        """

        prediction_df = pd.DataFrame(
            {
                "actual_demand": y_true,
                "predicted_demand": y_pred,
                "error": y_true - y_pred,
                "absolute_error": np.abs(
                    y_true - y_pred
                ),
            }
        )

        return prediction_df
    
    def _prepare_features(self,df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

        target_col = self.config.target_column

        if target_col not in df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found"
            )

        X = df.drop(
            columns=[target_col],
            errors="ignore",
        )

        # Remove target leakage columns
        leakage_columns = [
                "num__completed_rides",
                "num__cancelled_rides",
                "num__revenue",
                "num__supply_demand_ratio",
                "num__marketplace_gap",
                "num__driver_shortage",
                "num__marketplace_pressure",
                "num__surge_x_demand",
                "num__traffic_x_demand",
                "num__borough_avg_demand",
                "num__zone_avg_demand",
                "num__revenue_per_request"
                "num__active_drivers",
                "num__available_drivers",
                "num__busy_drivers",
                "num__acceptance_rate",
                "num__utilization_rate",
                "num__rider_wait_time",
                "num__driver_wait_time"
                    ]
        

        leakage_columns = [
            col
            for col in leakage_columns
            if col in X.columns
        ]

        if leakage_columns:
            logger.warning(
                f"Removing leakage columns: "
                f"{leakage_columns}"
            )

            X = X.drop(
                columns=leakage_columns
            )

        # Convert datetime columns
        datetime_cols = X.select_dtypes(
            include=["datetime64[ns]"]
        ).columns.tolist()

        for col in datetime_cols:
            X[col] = pd.to_datetime(
                X[col]
            ).astype("int64") // 10**9

        y = df[target_col]
        
        logger.info(
            f"Prepared features with shape: {X.shape}, "
            f"target shape: {y.shape}"
        )

        return X, y

    def log_into_mlflow(self):
        """
        Evaluate forecasting model and
        register model in MLflow.
        """

        try:

            test_data = self.load_data(
                Path(
                    self.config.test_data_path
                )
            )

            logger.info(
                f"Test Data Shape: {test_data.shape}"
            )

            model = self.load_model(
                Path(
                    self.config.model_path
                )
            )

            test_x , test_y = self._prepare_features(test_data)
            
            test_y = test_y.values

            predictions = model.predict(
                test_x
            )

            metrics = self.eval_metrics(
                test_y,
                predictions
            )

            prediction_df = (
                self.create_prediction_artifact(
                    test_y,
                    predictions
                )
            )

            save_json(
                path=Path(
                    self.config.metric_file_name
                ),
                data=self._convert_numpy_types(
                    metrics
                ),
            )

            mlflow.set_tracking_uri(
                self.config.mlflow_uri
            )

            mlflow.set_experiment(
                self.config.experiment_name
            )

            tracking_url_type_store = (
                urlparse(
                    mlflow.get_tracking_uri()
                ).scheme
            )

            with mlflow.start_run():

                mlflow.log_params(
                    self.config.all_params
                )

                for (metric_name, metric_value) in metrics.items():

                    mlflow.log_metric(
                        metric_name,
                        metric_value
                    )

                artifact_file = (Path(self.config.root_dir)
                    / "forecast_results.csv"
                )

                prediction_df.to_csv(
                    artifact_file,
                    index=False
                )

                mlflow.log_artifact(
                    str(
                        artifact_file
                    )
                )

                input_example = (
                    test_x.head(5)
                )

                if (tracking_url_type_store!= "file"):
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name="RideForecastingModel",
                        input_example=input_example,
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        input_example=input_example,
                    )

                logger.info(
                    f"MLflow Run Completed: "
                    f"{mlflow.active_run().info.run_id}"
                )

            return metrics

        except Exception as e:
            logger.exception(
                "Error during model evaluation"
            )
            raise e

    def save_evaluation_results(self):

        test_data = self.load_data(
            Path(
                self.config.test_data_path
            )
        )

        model = self.load_model(
            Path(
                self.config.model_path
            )
        )

        test_x , test_y = self._prepare_features(test_data)
            
        test_y = test_y.values

        predictions = model.predict(
            test_x
        )

        metrics = self.eval_metrics(
            test_y,
            predictions
        )

        save_json(
            path=Path(
                self.config.metric_file_name
            ),
            data=self._convert_numpy_types(
                metrics
            ),
        )

        logger.info(
            f"Evaluation Results Saved: "
            f"{self.config.metric_file_name}"
        )

        return metrics

