import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.utils.common import save_bin, save_json
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import ModelTrainingConfig
from pathlib import Path


class ModelTraining:
    """
    Ride Demand Forecasting Model Training Component

    Target:
        ride_requests

    Model:
        XGBoost Regressor

    Metrics:
        RMSE
        MAE
        MAPE
        R2 Score
    """

    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def load_split_data(self, file_path: str) -> pd.DataFrame:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Split data file not found: {file_path}"
            )

        logger.info(
            f"Loading split data from: {file_path}"
        )

        file_path = str(file_path)

        if file_path.endswith(".parquet"):
            data = pd.read_parquet(file_path)
            logger.info(
                f"Loaded data shape: {data.shape}"
            )
            return data

        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
            logger.info(
                f"Loaded data shape: {data.shape}"
            )
            return data

        raise ValueError(
            f"Unsupported file format: {file_path}"
        )

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

    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> XGBRegressor:

        logger.info(
            "Preparing training features"
        )

        X_train, y_train = self._prepare_features(
            train_df
        )

        X_val, y_val = self._prepare_features(
            val_df
        )

        logger.info(
            "Initializing XGBoost Regressor"
        )

        model = XGBRegressor(
            **self.config.model_params
        )

        logger.info(
            "Starting model training"
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_val,
                    y_val,
                )
            ],
            verbose=False,
        )

        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values(
            "importance",
            ascending=False
        )

        print(importance.head(20))
        
        logger.info(
            "Model training completed"
        )

        return model

    def evaluate_model(
        self,
        model: XGBRegressor,
        df: pd.DataFrame,
    ) -> dict:

        logger.info(
            "Evaluating model"
        )

        X, y = self._prepare_features(df)

        y_pred = model.predict(X)

        rmse = np.sqrt(
            mean_squared_error(
                y,
                y_pred,
            )
        )

        mae = mean_absolute_error(
            y,
            y_pred,
        )

        mape = (
            np.mean(
                np.abs(
                    (y - y_pred)
                    / np.maximum(y, 1)
                )
            )
            * 100
        )

        r2 = r2_score(
            y,
            y_pred,
        )

        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2_score": float(r2),
        }

        return metrics

    def save_model(
        self,
        model: XGBRegressor,
    ) -> None:

        model_path = Path(
            self.config.model_file_path
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_bin(
            data=model,
            path=model_path,
        )

        logger.info(
            f"Saved trained model at: "
            f"{model_path}"
        )

    def save_metrics(
        self,
        metrics: dict,
    ) -> None:

        metrics_path = Path(
            self.config.metrics_file_path
        )

        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_json(
            path=metrics_path,
            data=metrics,
        )

        logger.info(
            f"Saved metrics at: "
            f"{metrics_path}"
        )

    def initiate_model_training(
        self,
    ) -> bool:

        try:

            logger.info(
                "Loading train dataset"
            )

            train_df = self.load_split_data(
                self.config.train_file_path
            )

            logger.info(
                "Loading validation dataset"
            )

            val_df = self.load_split_data(
                self.config.validation_file_path
            )

            model = self.train_model(
                train_df=train_df,
                val_df=val_df,
            )

            self.save_model(model)

            validation_metrics = (
                self.evaluate_model(
                    model=model,
                    df=val_df,
                )
            )

            best_iteration = None

            try:
                booster = model.get_booster()

                if hasattr(booster,"best_iteration"):
                    best_iteration = int(
                        booster.best_iteration
                    )

            except Exception:
                pass

            metrics = {
                "validation": validation_metrics,
                #"test": test_metrics,
                "best_iteration": best_iteration,
            }

            self.save_metrics(metrics)

            logger.info(
                "Model training completed successfully"
            )

            return True

        except Exception as e:

            logger.exception(
                f"Error during model training: {e}"
            )

            raise