from pathlib import Path
from typing import List, Dict, Any
import joblib
import pandas as pd
import numpy as np
from feast import FeatureStore
from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.utils.common import create_directories
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import ModelInferenceConfig


class ModelInference:
    """
    Production Grade Ride Demand Forecasting Inference

    Flow
    ----
    1. Receive entity rows
    2. Fetch online features from Feast
    3. Merge request + Feast features
    4. Align with training schema
    5. Apply preprocessing pipeline
    6. Remove leakage features
    7. Predict demand
    8. Generate Marketplace recommendations
    """

    LEAKAGE_COLUMNS = [
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
        "num__available_drivers",
        "num__busy_drivers",
        "num__acceptance_rate",
        "num__utilization_rate",
        "num__rider_wait_time",
        "num__driver_wait_time",
    ]

    def __init__(self, config):

        self.config = config

        create_directories(
            [self.config.root_dir]
        )

        logger.info("Initializing Model Inference")

        self.store = FeatureStore(
            repo_path=str(
                self.config.feature_repo_path
            )
        )

        self.preprocessor = self._load_preprocessor()

        self.model = self._load_model()

        self.training_features = list(
            self.preprocessor.feature_names_in_
        )

        self.model_features = list(
            self.model.feature_names_in_
        )

        logger.info(
            f"Training Features : {len(self.training_features)}"
        )

        logger.info(
            f"Model Features : {len(self.model_features)}"
        )

        logger.info(
            f"Feature Service : "
            f"{self.config.feature_service_name}"
        )

    ####################################################################
    # LOADERS
    ####################################################################

    def _load_preprocessor(self):

        path = Path(
            self.config.preprocessor_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Preprocessor not found : {path}"
            )

        logger.info(
            f"Loading preprocessor : {path}"
        )

        return joblib.load(path)

    def _load_model(self):

        path = Path(
            self.config.model_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Model not found : {path}"
            )

        logger.info(
            f"Loading model : {path}"
        )

        return joblib.load(path)

    ####################################################################
    # FEAST
    ####################################################################

    def fetch_online_features(
        self,
        entity_df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "Fetching online features from Feast"
        )

        feature_service = (
            self.store.get_feature_service(
                self.config.feature_service_name
            )
        )

        feature_df = (
            self.store.get_online_features(
                features=feature_service,
                entity_rows=entity_df.to_dict(
                    orient="records"
                ),
            )
            .to_df()
        )

        logger.info(
            f"Online Feature Shape : "
            f"{feature_df.shape}"
        )

        return feature_df

    ####################################################################
    # FEATURE MERGING
    ####################################################################

    def merge_features(
        self,
        entity_df: pd.DataFrame,
        feast_df: pd.DataFrame
    ) -> pd.DataFrame:

        feast_columns = set(
            feast_df.columns
        )

        entity_only = entity_df[
            [
                col
                for col in entity_df.columns
                if col not in feast_columns
            ]
        ]

        merged = pd.concat(
            [
                entity_only.reset_index(drop=True),
                feast_df.reset_index(drop=True),
            ],
            axis=1,
        )

        merged = merged.loc[
            :,
            ~merged.columns.duplicated()
        ]

        logger.info(
            f"Merged Shape : {merged.shape}"
        )

        return merged

    ####################################################################
    # FEATURE ALIGNMENT
    ####################################################################

    def align_schema(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        df = df.copy()

        datetime_cols = (
            df.select_dtypes(
                include=["datetime64[ns]"]
            )
            .columns
            .tolist()
        )

        for col in datetime_cols:

            df[col] = (
                pd.to_datetime(df[col])
                .astype("int64")
                // 10**9
            )

        missing_features = [

            col

            for col in self.training_features

            if col not in df.columns

        ]

        if missing_features:

            logger.warning(
                f"Missing Features : "
                f"{missing_features}"
            )

            for col in missing_features:

                df[col] = 0

        extra_features = [

            col

            for col in df.columns

            if col not in self.training_features

        ]

        if extra_features:

            logger.info(
                f"Dropping {len(extra_features)} "
                f"extra features"
            )

        df = df[self.training_features]

        return df

    ####################################################################
    # PREPROCESS
    ####################################################################

    def preprocess(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "Running preprocessing pipeline"
        )

        transformed = (
            self.preprocessor.transform(
                df
            )
        )

        transformed = pd.DataFrame(
            transformed,
            columns=self.preprocessor.get_feature_names_out(),
            index=df.index
        )

        leakage_cols = [

            col

            for col in self.LEAKAGE_COLUMNS

            if col in transformed.columns

        ]

        if leakage_cols:

            logger.warning(
                f"Removing leakage columns : "
                f"{leakage_cols}"
            )

            transformed = transformed.drop(
                columns=leakage_cols
            )

        transformed = transformed[self.model_features]

        logger.info(
            f"Final Feature Shape : "
            f"{transformed.shape}"
        )

        return transformed
    
    def generate_marketplace_recommendations(
        self,
        result_df: pd.DataFrame
    ) -> pd.DataFrame:

        df = result_df.copy()

        # ==================================================
        # DEMAND SUPPLY GAP
        # ==================================================

        df["driver_gap"] = (
            df["forecast_demand"]
            - df["available_drivers"]
        )

        # ==================================================
        # SUPPLY DEMAND RATIO
        # ==================================================

        df["forecast_supply_ratio"] = (
            df["available_drivers"]
            /
            np.maximum(
                df["forecast_demand"],
                1
            )
        )

        # ==================================================
        # REQUIRED DRIVERS
        # ==================================================

        df["required_drivers"] = np.ceil(
            df["forecast_demand"]
            /
            np.maximum(
                df["rides_per_driver"],
                1
            )
        )

        # ==================================================
        # SURGE RECOMMENDATION
        # ==================================================

        pressure = (
            df["forecast_demand"]
            /
            np.maximum(
                df["available_drivers"],
                1
            )
        )

        df["recommended_surge"] = np.select(
            [
                pressure < 1.0,
                pressure < 1.2,
                pressure < 1.5,
                pressure < 2.0
            ],
            [
                1.0,
                1.1,
                1.3,
                1.6
            ],
            default=2.0
        )

        # ==================================================
        # MARKETPLACE STATUS
        # ==================================================

        df["marketplace_status"] = np.select(
            [
                pressure < 0.8,
                pressure < 1.2,
                pressure < 1.5
            ],
            [
                "Oversupply",
                "Balanced",
                "High Demand"
            ],
            default="Driver Shortage"
        )

        # ==================================================
        # DRIVER SHORTAGE %
        # ==================================================

        df["shortage_pct"] = np.maximum(
            (
                df["forecast_demand"]
                -
                df["available_drivers"]
            )
            /
            np.maximum(
                df["forecast_demand"],
                1
            ),
            0
        )

        # ==================================================
        # WAIT TIME ESTIMATE
        # ==================================================

        df["predicted_wait_time"] = (
            df["rider_wait_time"]
            *
            (
                1
                +
                df["shortage_pct"]
            )
        )

        # ==================================================
        # REVENUE FORECAST
        # ==================================================

        df["forecast_revenue"] = (
            df["forecast_demand"]
            *
            df["avg_fare"]
            *
            df["recommended_surge"]
        )

        # ==================================================
        # MARKETPLACE RISK
        # ==================================================

        df["risk_score"] = np.clip(
            (
                0.40 * df["shortage_pct"]
                +
                0.30 * (
                    df["traffic_index"] / 100
                )
                +
                0.20 * (
                    df["weather_severity_score"] / 10
                )
                +
                0.10 * (
                    df["event_intensity"] / 5
                )
            ),
            0,
            1
        )

        return df

    ####################################################################
    # PREDICT
    ####################################################################

    def predict(
        self,
        entity_df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            f"Running prediction "
            f"for {len(entity_df)} rows"
        )
        
        if "zone_timestamp_key" not in entity_df.columns:
            entity_df["zone_timestamp_key"] = (entity_df["zone_id"].astype(str) + "_" + entity_df["timestamp" ].dt.strftime("%Y%m%d%H"))

        feast_df = (
            self.fetch_online_features(
                entity_df
            )
        )

        missing_pct = (
            feast_df.isna()
            .mean()
            .sort_values(ascending=False)
        )

        print(missing_pct.head(20))

        print(feast_df[feast_df.isna().all(axis=1)].shape)        
                
        features = self.merge_features(
                    entity_df,
                    feast_df
                )

        features = self.align_schema(
            features
        )

        transformed = self.preprocess(
            features
        )

        predictions = (
            self.model.predict(
                transformed
            )
        )

        result = entity_df.copy()

        result["forecast_demand"] = (
            np.maximum(
                predictions,
                0
            )
            .round(2)
        )

        result = pd.concat(
            [
                result.reset_index(drop=True),
                features.reset_index(drop=True)
            ],
            axis=1
        )
        
        logger.info(f"Before duplicated removal shape : {result.shape}")
        dups = result.columns[
            result.columns.duplicated()
        ]

        logger.info(
            f"Duplicate columns: {list(dups)}"
        )

        logger.info(
            f"Forecast demand count: "
            f"{(result.columns=='forecast_demand').sum()}"
        )

        logger.info(
            f"Available drivers count: "
            f"{(result.columns=='available_drivers').sum()}"
        )

        result = result.loc[:,~result.columns.duplicated(keep="last")]

        logger.info(f"removing duplicated rows : {dups}")

        logger.info(f"after duplicated removal shape : {result.shape}")
        
        result = self.generate_marketplace_recommendations(
            result
        )

        logger.info(
            "Prediction completed"
        )

        return result

    ####################################################################
    # RECORD API
    ####################################################################

    def predict_records(
        self,
        records: List[Dict[str, Any]]
    ) -> pd.DataFrame:

        entity_df = pd.DataFrame(
            records
        )

        return self.predict(
            entity_df
        )

    ####################################################################
    # BATCH INFERENCE
    ####################################################################

    def run_batch_inference(self):

        logger.info(
            "Starting batch inference"
        )

        input_df = pd.read_parquet(
            self.config.input_data_path
        )

        results = self.predict(
            input_df
        )

        output_path = Path(
            self.config.prediction_output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        results.to_csv(
            output_path,
            index=False
        )

        logger.info(
            f"Predictions saved : "
            f"{output_path}"
        )

        return output_path