import os
import json
import numpy as np
import pandas as pd
from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    """
    Production-Grade Ride Demand Forecasting &
    Marketplace Optimization Data Preprocessing
    """

    def __init__(self, config: DataPreprocessingConfig):

        self.config = config

        self.data = pd.read_parquet(
            os.path.join(
                self.config.input_data_path,
                "ride_marketplace.parquet"
            )
        )

        self.report = {}

        logger.info(
            f"Dataset Loaded Successfully | "
            f"Shape={self.data.shape}"
        )

    # =====================================================
    # REPORT HELPER
    # =====================================================

    def _update_report(self, key, value):

        self.report[key] = value

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    def remove_duplicates(self):

        before = len(self.data)

        self.data.drop_duplicates(
            inplace=True
        )

        removed = (
            before -
            len(self.data)
        )

        self._update_report(
            "duplicates_removed",
            int(removed)
        )

        logger.info(
            f"Removed {removed} duplicate rows"
        )

    # =====================================================
    # HANDLE INFINITE VALUES
    # =====================================================

    def handle_infinite_values(self):

        numeric_cols = (
            self.data
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        total_inf = 0

        for col in numeric_cols:

            total_inf += int(
                np.isinf(
                    self.data[col]
                ).sum()
            )

        self.data.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        self._update_report(
            "infinite_values_replaced",
            total_inf
        )

        logger.info(
            f"Replaced {total_inf} infinite values"
        )

    # =====================================================
    # HANDLE MISSING NUMERICAL
    # =====================================================

    def handle_missing_numerical(self):

        numeric_cols = (
            self.data
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        report = {}

        for col in numeric_cols:

            missing_count = int(
                self.data[col]
                .isna()
                .sum()
            )

            if missing_count > 0:

                median_value = (
                    self.data[col]
                    .median()
                )

                self.data[col] = (
                    self.data[col]
                    .fillna(
                        median_value
                    )
                )

            report[col] = missing_count

        self._update_report(
            "numerical_missing_values",
            report
        )

        logger.info(
            "Numerical missing values handled"
        )

    # =====================================================
    # HANDLE MISSING CATEGORICAL
    # =====================================================

    def handle_missing_categorical(self):

        cat_cols = (
            self.data
            .select_dtypes(
                include=["object", "category"]
            )
            .columns
        )

        report = {}

        for col in cat_cols:

            missing_count = int(
                self.data[col]
                .isna()
                .sum()
            )

            if missing_count > 0:

                mode_value = (
                    self.data[col]
                    .mode()
                    .iloc[0]
                )

                self.data[col] = (
                    self.data[col]
                    .fillna(
                        mode_value
                    )
                )

            report[col] = missing_count

        self._update_report(
            "categorical_missing_values",
            report
        )

        logger.info(
            "Categorical missing values handled"
        )

    # =====================================================
    # TIMESTAMP PROCESSING
    # =====================================================

    def process_timestamp(self):

        self.data["timestamp"] = pd.to_datetime(
            self.data["timestamp"]
        )

        self.data.sort_values(
            [
                "zone_id",
                "timestamp"
            ],
            inplace=True
        )

        self.data.reset_index(
            drop=True,
            inplace=True
        )

        logger.info(
            "Timestamp processing completed"
        )

    # =====================================================
    # MARKETPLACE RULES
    # =====================================================

    def enforce_marketplace_rules(self):

        before_rows = len(
            self.data
        )

        self.data = self.data[
            self.data["ride_requests"] >= 0
        ]

        self.data = self.data[
            self.data["completed_rides"] >= 0
        ]

        self.data = self.data[
            self.data["cancelled_rides"] >= 0
        ]

        self.data = self.data[
            self.data["active_drivers"] >= 0
        ]

        self.data = self.data[
            self.data["available_drivers"] >= 0
        ]

        self.data = self.data[
            self.data["busy_drivers"] >= 0
        ]

        self.data = self.data[
            self.data["completed_rides"]
            <=
            self.data["ride_requests"]
        ]

        self.data = self.data[
            self.data["cancelled_rides"]
            <=
            self.data["ride_requests"]
        ]

        self.data = self.data[
            self.data["busy_drivers"]
            <=
            self.data["active_drivers"]
        ]

        self.data = self.data[
            self.data["available_drivers"]
            <=
            self.data["active_drivers"]
        ]

        removed_rows = (
            before_rows -
            len(self.data)
        )

        self._update_report(
            "invalid_marketplace_rows_removed",
            int(removed_rows)
        )

        logger.info(
            f"Removed {removed_rows} invalid marketplace rows"
        )

    # =====================================================
    # GEO VALIDATION
    # =====================================================

    def clean_geospatial_data(self):

        before_rows = len(
            self.data
        )

        self.data = self.data[
            self.data["latitude"]
            .between(
                40.40,
                41.10
            )
        ]

        self.data = self.data[
            self.data["longitude"]
            .between(
                -74.40,
                -73.50
            )
        ]

        removed_rows = (
            before_rows -
            len(self.data)
        )

        self._update_report(
            "invalid_geospatial_rows_removed",
            int(removed_rows)
        )

        logger.info(
            f"Removed {removed_rows} invalid geo rows"
        )

    # =====================================================
    # KPI CLEANING
    # =====================================================

    def clean_marketplace_kpis(self):

        self.data["acceptance_rate"] = (
            self.data["acceptance_rate"]
            .clip(
                0.0,
                1.0
            )
        )

        self.data["utilization_rate"] = (
            self.data["utilization_rate"]
            .clip(
                0.0,
                1.0
            )
        )

        self.data["surge_multiplier"] = (
            self.data["surge_multiplier"]
            .clip(
                1.0,
                10.0
            )
        )

        logger.info(
            "Marketplace KPI cleaning completed"
        )

    # =====================================================
    # REVENUE VALIDATION
    # =====================================================

    def validate_revenue_consistency(self):

        calculated_revenue = (
            self.data["completed_rides"]
            *
            self.data["avg_fare"]
        )

        revenue_error = np.abs(
            calculated_revenue -
            self.data["revenue"]
        )

        mismatch_count = int(
            (
                revenue_error > 1.0
            ).sum()
        )

        self._update_report(
            "revenue_mismatch_records",
            mismatch_count
        )

        logger.info(
            f"Revenue mismatches: "
            f"{mismatch_count}"
        )

    # =====================================================
    # NUMERICAL RANGE ENFORCEMENT
    # =====================================================

    def enforce_numerical_ranges(self):

        if (
            not hasattr(
                self.config,
                "numerical_ranges"
            )
            or
            not self.config.numerical_ranges
        ):
            return

        for col, limits in (
            self.config
            .numerical_ranges
            .items()
        ):

            if col not in self.data.columns:
                continue

            self.data[col] = (
                self.data[col]
                .clip(
                    limits.min,
                    limits.max
                )
            )

        logger.info(
            "Numerical ranges enforced"
        )

    # =====================================================
    # OUTLIER HANDLING
    # =====================================================

    def handle_outliers(self):

        excluded_cols = [

            "zone_id",

            "year",

            "quarter",

            "month",

            "week",

            "day",

            "hour",

            "dayofweek",

            "holiday_flag",

            "concert_flag",

            "sports_event_flag",

            "festival_flag",

            "is_weekend"
        ]

        numeric_cols = (
            self.data
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        report = {}

        for col in numeric_cols:

            if col in excluded_cols:
                continue

            q1 = (
                self.data[col]
                .quantile(0.25)
            )

            q3 = (
                self.data[col]
                .quantile(0.75)
            )

            iqr = q3 - q1

            lower = (
                q1 -
                self.config.outlier_iqr_multiplier * iqr
            )

            upper = (
                q3 +
                 self.config.outlier_iqr_multiplier * iqr
            )

            outlier_count = int(
                (
                    (
                        self.data[col]
                        < lower
                    )
                    |
                    (
                        self.data[col]
                        > upper
                    )
                ).sum()
            )

            self.data[col] = (
                self.data[col]
                .clip(
                    lower,
                    upper
                )
            )

            report[col] = outlier_count

        self._update_report(
            "outlier_summary",
            report
        )

        logger.info(
            "Outliers treated using IQR clipping"
        )


    def reduce_mem_usage(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Reduce memory usage of a pandas DataFrame
        by downcasting numerical columns and
        converting low-cardinality object columns
        to category dtype.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        verbose : bool
            Print memory reduction summary

        Returns
        -------
        pd.DataFrame
            Optimized dataframe
        """

        start_mem = (
            df.memory_usage(
                deep=True
            ).sum()
            /
            1024**2
        )

        for col in df.columns:

            col_type = df[col].dtype

            # =================================================
            # NUMERICAL COLUMNS
            # =================================================

            if pd.api.types.is_numeric_dtype(
                col_type
            ):

                c_min = df[col].min()
                c_max = df[col].max()

                # -------------------------
                # Integer Columns
                # -------------------------

                if pd.api.types.is_integer_dtype(
                    col_type
                ):

                    if (
                        c_min >= np.iinfo(np.int8).min
                        and
                        c_max <= np.iinfo(np.int8).max
                    ):

                        df[col] = (
                            df[col]
                            .astype(np.int8)
                        )

                    elif (
                        c_min >= np.iinfo(np.int16).min
                        and
                        c_max <= np.iinfo(np.int16).max
                    ):

                        df[col] = (
                            df[col]
                            .astype(np.int16)
                        )

                    elif (
                        c_min >= np.iinfo(np.int32).min
                        and
                        c_max <= np.iinfo(np.int32).max
                    ):

                        df[col] = (
                            df[col]
                            .astype(np.int32)
                        )

                    else:

                        df[col] = (
                            df[col]
                            .astype(np.int64)
                        )

                # -------------------------
                # Float Columns
                # -------------------------

                else:

                    if (
                        c_min >= np.finfo(np.float16).min
                        and
                        c_max <= np.finfo(np.float16).max
                    ):

                        df[col] = (
                            df[col]
                            .astype(np.float16)
                        )

                    elif (
                        c_min >= np.finfo(np.float32).min
                        and
                        c_max <= np.finfo(np.float32).max
                    ):

                        df[col] = (
                            df[col]
                            .astype(np.float32)
                        )

                    else:

                        df[col] = (
                            df[col]
                            .astype(np.float64)
                        )

            # =================================================
            # OBJECT COLUMNS
            # =================================================

            elif col_type == object:

                num_unique = (
                    df[col]
                    .nunique()
                )

                total_values = (
                    len(df[col])
                )

                if (
                    total_values > 0
                    and
                    num_unique / total_values
                    < 0.50
                ):

                    df[col] = (
                        df[col]
                        .astype("category")
                    )

            # =================================================
            # DATETIME
            # =================================================

            elif pd.api.types.is_datetime64_any_dtype(
                col_type
            ):

                continue

        end_mem = (
            df.memory_usage(
                deep=True
            ).sum()
            /
            1024**2
        )

        reduction = (
            100
            *
            (
                start_mem - end_mem
            )
            /
            start_mem
        )

        if verbose:

            logger.info(
                f"Memory Usage: "
                f"{start_mem:.2f} MB -> "
                f"{end_mem:.2f} MB "
                f"({reduction:.2f}% reduction)"
            )

        return df
    # =====================================================
    # MEMORY OPTIMIZATION
    # =====================================================

    def optimize_memory(self):

        before = (
            self.data
            .memory_usage(
                deep=True
            )
            .sum()
            /
            1024**2
        )

        self.data = self.reduce_mem_usage(
            self.data
        )

        after = (
            self.data
            .memory_usage(
                deep=True
            )
            .sum()
            /
            1024**2
        )

        self._update_report(
            "memory_before_mb",
            round(before, 2)
        )

        self._update_report(
            "memory_after_mb",
            round(after, 2)
        )

        logger.info(
            f"Memory Optimized: "
            f"{before:.2f} MB -> "
            f"{after:.2f} MB"
        )

    # =====================================================
    # SAVE DATA
    # =====================================================

    def save_data(self):

        os.makedirs(
            os.path.dirname(
                self.config.output_data_path
            ),
            exist_ok=True
        )

        self.data.to_parquet(
            self.config.output_data_path,
            index=False
        )

        with open(
            self.config.preprocessing_report_path,
            "w"
        ) as f:

            json.dump(
                self.report,
                f,
                indent=4,
                default=str
            )

        logger.info(
            "Preprocessed data saved successfully"
        )

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def initiate_data_preprocessing(self):

        logger.info(
            "Starting Ride Marketplace Data Preprocessing"
        )

        self.remove_duplicates()

        self.handle_infinite_values()

        self.handle_missing_numerical()

        self.handle_missing_categorical()

        self.process_timestamp()

        self.enforce_marketplace_rules()

        self.clean_geospatial_data()

        self.clean_marketplace_kpis()

        self.validate_revenue_consistency()

        self.enforce_numerical_ranges()

        self.handle_outliers()

        self.optimize_memory()

        self.save_data()

        logger.info(
            "Ride Marketplace Data Preprocessing Completed"
        )

        return self.config.output_data_path, self.config.preprocessing_report_path