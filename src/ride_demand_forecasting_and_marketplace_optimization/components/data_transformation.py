import os
import json
import joblib
import numpy as np
import pandas as pd
from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.transformers.feature_manager import FeatureManager
from src.ride_demand_forecasting_and_marketplace_optimization.transformers.feature_selector import RDFeatureSelector
from src.ride_demand_forecasting_and_marketplace_optimization.transformers.preprocessor_builder import *
from src.ride_demand_forecasting_and_marketplace_optimization.transformers.imbalance_handler import *
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import DataTransformationConfig
from feast import FeatureStore


class DataTransformation:
    """
    Production-grade Data Transformation Component

    Flow:
        Feature Engineered Dataset
                ↓
        Feast Historical Features Retrieval
                ↓
        Target Join
                ↓
        Time-Based Train/Val/Test Split
                ↓
        Preprocessing
                ↓
        Optional Feature Selection
                ↓
        Artifact Saving
    """

    def __init__(self, config: DataTransformationConfig):

        self.config = config
        self.preprocessor = None
        self.selector = None

        logger.info(
            "Initializing Data Transformation Component"
        )

        self.data = self.fetch_training_features()

    # ==========================================================
    # FEAST HISTORICAL FEATURES
    # ==========================================================

    def fetch_training_features_from_feast(self) -> pd.DataFrame:

        logger.info(
            "Loading Feature Store"
        )

        store = FeatureStore(
            repo_path=str(
                self.config.feature_repo_path
            )
        )

        source_df = pd.read_parquet(
            self.config.feature_engineered_data_path
        )

        entity_df = (
            source_df[
                [
                    "zone_timestamp_key",
                    "timestamp",
                ]
            ]
            .rename(
                columns={
                    "timestamp": "event_timestamp"
                }
            )
            .drop_duplicates()
        )

        logger.info(f"Entity DataFrame shape: {entity_df.shape}")
        
        logger.info(
            f"Retrieving historical features from "
            f"Feature Service: "
            f"{self.config.feature_service_name}"
        )
        # entity_df = entity_df.sample(10000,random_state=42)
        
        features_df = (
            store.get_historical_features(
                entity_df=entity_df,
                features=store.get_feature_service(
                    self.config.feature_service_name
                ),
            )
            .to_df()
        )

        if features_df.empty:

            raise ValueError(
                "No features returned from Feast."
            )

        logger.info(
            f"Retrieved {features_df.shape[1]} "
            f"columns from Feast"
        )

        target_df = (
            source_df[
                [
                    "zone_timestamp_key",
                    "timestamp",
                    self.config.target_column,
                ]
            ]
            .rename(
                columns={
                    "timestamp": "event_timestamp"
                }
            )
        )
        features_df["event_timestamp"] = pd.to_datetime(
            features_df["event_timestamp"],
            utc=True,
        )

        target_df["event_timestamp"] = pd.to_datetime(
            target_df["event_timestamp"],
            utc=True,
        )

        training_df = (
            features_df.merge(
                target_df,
                on=[
                    "zone_timestamp_key",
                    "event_timestamp",
                ],
                how="inner",
                validate="one_to_one",
            )
            .drop_duplicates()
            .sort_values("event_timestamp")
            .reset_index(drop=True)
        )

        logger.info(
            f"Training dataset shape: "
            f"{training_df.shape}"
        )

        return training_df
    
    # ==========================================================
    # FETCH TRAINING FEATURES
    # ==========================================================
    
    def fetch_training_features(self) -> pd.DataFrame:

        logger.info(
            "Loading feature engineered dataset"
        )

        training_df = pd.read_parquet(
            self.config.feature_engineered_data_path
        )

        if training_df.empty:

            raise ValueError(
                "Feature engineered dataset is empty."
            )

        required_columns = [
            self.config.target_column,
            "timestamp",
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in training_df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing required columns: "
                f"{missing_columns}"
            )

        training_df["event_timestamp"] = pd.to_datetime(
            training_df["timestamp"],
            utc=True,
        )

        training_df = (
            training_df
            .sort_values(
                "event_timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        logger.info(
            f"Training dataset loaded successfully. "
            f"Shape: {training_df.shape}"
        )

        logger.info(
            f"Number of features: "
            f"{training_df.shape[1] - 1}"
        )

        return training_df
    # ==========================================================
    # TIME SERIES SPLIT
    # ==========================================================

    def split_data(
        self,
        data: pd.DataFrame,
    ):

        data = (
            data.sort_values(
                "event_timestamp"
            )
            .reset_index(drop=True)
        )

        n = len(data)

        train_end = int(
            n * self.config.train_size
        )

        val_end = train_end + int(
            n * self.config.validation_size
        )

        train_df = data.iloc[:train_end]

        val_df = data.iloc[
            train_end:val_end
        ]

        test_df = data.iloc[val_end:]

        target_col = (
            self.config.target_column
        )

        X_train = train_df.drop(
            columns=[target_col]
        )

        y_train = train_df[target_col]

        X_val = val_df.drop(
            columns=[target_col]
        )

        y_val = val_df[target_col]

        X_test = test_df.drop(
            columns=[target_col]
        )

        y_test = test_df[target_col]

        logger.info(
            f"Train Shape: {X_train.shape}"
        )

        logger.info(
            f"Validation Shape: {X_val.shape}"
        )

        logger.info(
            f"Test Shape: {X_test.shape}"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

    # ==========================================================
    # PREPROCESSING
    # ==========================================================

    def transform_data(
        self,
        X_train,
        X_val,
        X_test,
    ):

        drop_columns = [
            "event_timestamp",
            "created_timestamp",
        ]

        X_train = X_train.drop(
            columns=drop_columns,
            errors="ignore",
        )

        X_val = X_val.drop(
            columns=drop_columns,
            errors="ignore",
        )

        X_test = X_test.drop(
            columns=drop_columns,
            errors="ignore",
        )

        low_cardinality_columns = [
            col
            for col in FeatureManager.LOW_CARDINALITY_COLUMNS
            if col in X_train.columns
        ]

        high_cardinality_columns = [
            col
            for col in FeatureManager.HIGH_CARDINALITY_COLUMNS
            if col in X_train.columns
        ]

        numerical_columns = [
            col
            for col in X_train.columns
            if (
                col
                not in low_cardinality_columns
                and col
                not in high_cardinality_columns
                and pd.api.types.is_numeric_dtype(
                    X_train[col]
                )
            )
        ]

        logger.info(
            f"Numerical Features: "
            f"{len(numerical_columns)}"
        )

        logger.info(
            f"Low Cardinality Features: "
            f"{len(low_cardinality_columns)}"
        )

        logger.info(
            f"High Cardinality Features: "
            f"{len(high_cardinality_columns)}"
        )

        self.preprocessor = (
            build_preprocessor(
                numerical_columns=numerical_columns,
                low_cardinality_columns=low_cardinality_columns,
                high_cardinality_columns=high_cardinality_columns,
            )
        )

        logger.info(
            "Fitting preprocessing pipeline"
        )

        X_train = (
            self.preprocessor.fit_transform(
                X_train
            )
        )

        X_val = (
            self.preprocessor.transform(
                X_val
            )
        )

        X_test = (
            self.preprocessor.transform(
                X_test
            )
        )

        try:

            feature_names = (
                self.preprocessor.get_feature_names_out()
            )

        except Exception:

            feature_names = [
                f"feature_{i}"
                for i in range(
                    X_train.shape[1]
                )
            ]

        joblib.dump(
            feature_names,
            self.config.feature_names_path,
        )

        X_train = pd.DataFrame(
            X_train,
            columns=feature_names,
        )

        X_val = pd.DataFrame(
            X_val,
            columns=feature_names,
        )

        X_test = pd.DataFrame(
            X_test,
            columns=feature_names,
        )

        return (
            X_train,
            X_val,
            X_test,
        )

    # ==========================================================
    # FEATURE SELECTION
    # ==========================================================

    def select_features(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
    ):

        self.selector = RDFeatureSelector(
            k=self.config.num_selected_features
        )

        logger.info(
            "Running Feature Selection"
        )

        X_train_selected = (
            self.selector.fit_transform(
                X_train,
                y_train,
            )
        )

        X_val_selected = (
            self.selector.transform(
                X_val
            )
        )

        X_test_selected = (
            self.selector.transform(
                X_test
            )
        )

        try:

            selected_columns = (
                X_train.columns[
                    self.selector.selected_features_
                ]
            )

        except Exception:

            selected_columns = [
                f"feature_{i}"
                for i in range(
                    X_train_selected.shape[1]
                )
            ]

        joblib.dump(
            list(selected_columns),
            self.config.selected_features_path,
        )

        X_train_selected = pd.DataFrame(
            X_train_selected,
            columns=selected_columns,
        )

        X_val_selected = pd.DataFrame(
            X_val_selected,
            columns=selected_columns,
        )

        X_test_selected = pd.DataFrame(
            X_test_selected,
            columns=selected_columns,
        )

        return (
            X_train_selected,
            X_val_selected,
            X_test_selected,
        )

    # ==========================================================
    # SAVE PREPROCESSOR
    # ==========================================================

    def save_preprocessor(self):

        joblib.dump(
            self.preprocessor,
            self.config.preprocessor_path,
        )

        logger.info(
            f"Preprocessor saved to: "
            f"{self.config.preprocessor_path}"
        )

    # ==========================================================
    # SAVE DATASETS
    # ==========================================================

    def save_data(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ):

        os.makedirs(
            self.config.split_artifacts_dir,
            exist_ok=True,
        )

        train_df = X_train.copy()
        train_df[
            self.config.target_column
        ] = y_train.values

        val_df = X_val.copy()
        val_df[
            self.config.target_column
        ] = y_val.values

        test_df = X_test.copy()
        test_df[
            self.config.target_column
        ] = y_test.values

        train_df.to_parquet(
            self.config.train_file_path,
            index=False,
        )

        val_df.to_parquet(
            self.config.validation_file_path,
            index=False,
        )

        test_df.to_parquet(
            self.config.test_file_path,
            index=False,
        )

        logger.info(
            "Train/Validation/Test saved"
        )

    # ==========================================================
    # SAVE METADATA
    # ==========================================================

    def save_metadata(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
    ):

        metadata = {
            "feature_service":
                self.config.feature_service_name,
            "target_column":
                self.config.target_column,
            "train_rows":
                int(X_train.shape[0]),
            "validation_rows":
                int(X_val.shape[0]),
            "test_rows":
                int(X_test.shape[0]),
            "selected_features":
                int(X_train.shape[1]),
            "feature_names_path":
                str(
                    self.config.feature_names_path
                ),
            "scale_pos_weight":
                float(
                    get_scale_pos_weight(
                        y_train
                    )
                )
                if y_train.nunique() <= 2
                else 1.0,
        }

        with open(self.config.metadata_path,"w") as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        logger.info(
            f"Metadata saved to: "
            f"{self.config.metadata_path}"
        )

    # ==========================================================
    # MAIN PIPELINE
    # ==========================================================

    def initiate_data_transformation(self):

        logger.info(
            "Starting Data Transformation Pipeline"
        )

        df = self.data.copy()

        df.drop(
            columns=FeatureManager.IDENTIFIER_COLUMNS,
            errors="ignore",
            inplace=True,
        )

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = self.split_data(df)

        (
            X_train,
            X_val,
            X_test,
        ) = self.transform_data(
            X_train,
            X_val,
            X_test,
        )

        if getattr(self.config, "enable_feature_selection", True):

            (
                X_train,
                X_val,
                X_test,
            ) = self.select_features(
                X_train,
                X_val,
                X_test,
                y_train,
            )

            joblib.dump(
                self.selector,
                self.config.feature_selector_path,
            )

        self.save_data(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

        self.save_preprocessor()

        self.save_metadata(
            X_train,
            X_val,
            X_test,
            y_train,
        )

        logger.info(
            "Data Transformation Completed Successfully"
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )