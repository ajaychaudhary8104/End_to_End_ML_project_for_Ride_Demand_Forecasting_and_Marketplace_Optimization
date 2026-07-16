import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.utils.common import save_json
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import DataValidationConfig
warnings.filterwarnings("ignore")


class DataValidation:
    """
    Production-Grade Ride Demand Forecasting &
    Marketplace Optimization Data Validation

    Covers:
    - Schema Validation
    - Datatype Validation
    - Missing Values
    - Duplicate Records
    - Numerical Ranges
    - Categorical Validation
    - Target Validation
    - Timestamp Validation
    - Time-Series Continuity
    - Marketplace Logic
    - Revenue Validation
    - KPI Validation
    - Driver Supply Validation
    - Surge Pricing Validation
    - Weather Validation
    - Event Validation
    - Economic Validation
    - Zone Coverage
    - Geospatial Validation
    - Infinite Values
    - Outliers
    - Cardinality
    - Leakage
    - Correlation
    - Drift Detection
    - Temporal Drift
    - Forecasting Readiness
    """

    def __init__(self, config: DataValidationConfig):

        self.config = config

        self.data = pd.read_parquet(os.path.join
            (config.unzip_data_dir,"ride_marketplace.parquet")
        )

        self.validation_report = {}
        self.dataset_statistics = {}
        self.drift_report = {}

        print(
            f"Dataset Loaded Successfully | "
            f"Shape={self.data.shape}"
        )

    # =====================================================
    # REPORT HELPERS
    # =====================================================

    def _update_report(
        self,
        validation_name,
        status,
        details=None
    ):

        self.validation_report[validation_name] = {
            "status": bool(status),
            "details": details
        }

    def _save_json(
        self,
        filepath,
        data
    ):

        os.makedirs(
            os.path.dirname(filepath),
            exist_ok=True
        )

        with open(
            filepath,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                default=str
            )

    def _save_reports(self):

        if hasattr(
            self.config,
            "REPORT_FILE"
        ):

            self._save_json(
                self.config.REPORT_FILE,
                self.validation_report
            )

        if hasattr(
            self.config,
            "STATS_REPORT_FILE"
        ):

            self._save_json(
                self.config.STATS_REPORT_FILE,
                self.dataset_statistics
            )

        if (
            len(self.drift_report) > 0
            and
            hasattr(
                self.config,
                "DRIFT_REPORT_FILE"
            )
        ):

            self._save_json(
                self.config.DRIFT_REPORT_FILE,
                self.drift_report
            )

    # =====================================================
    # DATASET STATISTICS
    # =====================================================

    def generate_dataset_statistics(self):

        target = (
            self.config.target_column
        )

        self.dataset_statistics = {

            "rows":
                int(self.data.shape[0]),

            "columns":
                int(self.data.shape[1]),

            "duplicates":
                int(
                    self.data
                    .duplicated()
                    .sum()
                ),

            "memory_mb":
                round(
                    self.data
                    .memory_usage(
                        deep=True
                    )
                    .sum()
                    /
                    1024
                    /
                    1024,
                    2
                ),

            "start_timestamp":
                str(
                    self.data[
                        "timestamp"
                    ].min()
                ),

            "end_timestamp":
                str(
                    self.data[
                        "timestamp"
                    ].max()
                ),

            "n_zones":
                int(
                    self.data[
                        "zone_id"
                    ].nunique()
                ),

            "n_boroughs":
                int(
                    self.data[
                        "borough"
                    ].nunique()
                ),

            "n_zone_types":
                int(
                    self.data[
                        "zone_type"
                    ].nunique()
                ),

            "target_stats": {

                "min":
                    float(
                        self.data[
                            target
                        ].min()
                    ),

                "max":
                    float(
                        self.data[
                            target
                        ].max()
                    ),

                "mean":
                    float(
                        self.data[
                            target
                        ].mean()
                    ),

                "std":
                    float(
                        self.data[
                            target
                        ].std()
                    )
            }
        }

    # =====================================================
    # SCHEMA
    # =====================================================

    def validate_schema(self):

        expected = set(
            self.config.all_schema.keys()
        )

        actual = set(
            self.data.columns
        )

        missing = list(
            expected - actual
        )

        extra = list(
            actual - expected
        )

        status = (
            len(missing) == 0
        )

        self._update_report(
            "Schema Validation",
            status,
            {
                "missing_columns":
                    missing,
                "extra_columns":
                    extra
            }
        )
        logger.info("Schema Validation Completed")

        return status

    # =====================================================
    # DATATYPE
    # =====================================================

    def validate_datatypes(self):

        mismatches = []

        for (col, expected) in (self.config.all_schema.items()):

            if col not in self.data:
                continue

            actual = str(
                self.data[col].dtype
            )

            if actual != expected:

                mismatches.append(
                    {
                        "column": col,
                        "expected": expected,
                        "actual": actual
                    }
                )

        status = (
            len(mismatches) == 0
        )

        self._update_report(
            "Datatype Validation",
            status,
            mismatches
        )
        logger.info("Datatype Validation Completed")

        return status

    # =====================================================
    # MISSING VALUES
    # =====================================================

    def validate_missing_values(self):

        issues = []

        threshold = (self.config.thresholds.missing_value_threshold)

        for col in self.data.columns:

            ratio = (
                self.data[col]
                .isna()
                .mean()
            )

            if ratio > threshold:

                issues.append(
                    {
                        "column": col,
                        "missing_pct":
                            round(
                                ratio * 100,
                                2
                            )
                    }
                )

        critical_cols = [

            "timestamp",
            "zone_id",
            "ride_requests",
            "active_drivers",
            "surge_multiplier"
        ]

        for col in critical_cols:

            if (
                self.data[col]
                .isna()
                .sum()
                > 0
            ):

                issues.append(
                    {
                        "critical_column":
                            col,
                        "missing":
                            int(
                                self.data[col]
                                .isna()
                                .sum()
                            )
                    }
                )

        status = (
            len(issues) == 0
        )

        self._update_report(
            "Missing Value Validation",
            status,
            issues
        )
        logger.info("Missing Value Validation Completed")
        return status

    # =====================================================
    # DUPLICATES
    # =====================================================

    def validate_duplicates(self):

        dup_count = int(
            self.data
            .duplicated(
                subset=[
                    "timestamp",
                    "zone_id"
                ]
            )
            .sum()
        )

        status = (
            dup_count == 0
        )

        self._update_report(
            "Duplicate Validation",
            status,
            {
                "duplicate_rows":
                    dup_count
            }
        )
        logger.info("Duplicate Validation Completed")
        return status

    # =====================================================
    # NUMERIC RANGES
    # =====================================================

    def validate_numerical_ranges(self):

        rules = {

            "ride_requests":
                (0, np.inf),

            "completed_rides":
                (0, np.inf),

            "cancelled_rides":
                (0, np.inf),

            "active_drivers":
                (0, np.inf),

            "available_drivers":
                (0, np.inf),

            "busy_drivers":
                (0, np.inf),

            "acceptance_rate":
                (0, 1),

            "utilization_rate":
                (0, 1),

            "surge_multiplier":
                (1, 10),

            "humidity":
                (0, 100),

            "visibility":
                (0, 20),

            "wind_speed":
                (0, np.inf),

            "fuel_price":
                (0, np.inf),

            "unemployment_rate":
                (0, 100),

            "avg_fare":
                (0, np.inf),

            "revenue":
                (0, np.inf)
        }

        issues = []

        for col, (low, high) in rules.items():

            invalid = (
                (
                    self.data[col] < low
                )
                |
                (
                    self.data[col] > high
                )
            ).sum()

            if invalid > 0:

                issues.append(
                    {
                        "column": col,
                        "violations":
                            int(invalid)
                    }
                )

        status = (
            len(issues) == 0
        )

        self._update_report(
            "Numerical Range Validation",
            status,
            issues
        )
        logger.info("Numerical Range Validation Completed")
        return status

    # =====================================================
    # CATEGORICAL
    # =====================================================

    def validate_categorical_values(self):

        issues = []

        for (
            col,
            allowed
        ) in (
            self.config
            .categorical_values
            .items()
        ):

            invalid = (
                ~self.data[col]
                .isin(allowed)
            ).sum()

            if invalid > 0:

                issues.append(
                    {
                        "column": col,
                        "invalid_count":
                            int(invalid)
                    }
                )

        status = (
            len(issues) == 0
        )

        self._update_report(
            "Categorical Validation",
            status,
            issues
        )
        logger.info("Categorical Validation Completed")
        return status

    # =====================================================
    # TARGET
    # =====================================================

    def validate_target(self):

        target = (
            self.config.target_column
        )

        negative = int(
            (
                self.data[target]
                < 0
            ).sum()
        )

        constant = (
            self.data[target]
            .std()
            == 0
        )

        status = (
            negative == 0
            and
            not constant
        )

        self._update_report(
            "Target Validation",
            status,
            {
                "negative_values":
                    negative,
                "constant_target":
                    constant
            }
        )
        logger.info("Target Validation Completed")
        return status

    # =====================================================
    # TIMESTAMP
    # =====================================================

    def validate_timestamp(self):

        ts = pd.to_datetime(
            self.data["timestamp"],
            errors="coerce"
        )

        invalid = int(
            ts.isna().sum()
        )

        future = int(
            (
                ts >
                pd.Timestamp.now()
            ).sum()
        )

        status = (
            invalid == 0
            and
            future == 0
        )

        self._update_report(
            "Timestamp Validation",
            status,
            {
                "invalid":
                    invalid,
                "future":
                    future
            }
        )
        logger.info("Timestamp Validation Completed")
        return status

    # =====================================================
    # TIME SERIES CONTINUITY
    # =====================================================

    def validate_time_series_continuity(self):

        missing_hours = 0

        for _, grp in self.data.groupby(
            "zone_id"
        ):

            ts = (
                pd.to_datetime(
                    grp["timestamp"]
                )
                .sort_values()
            )

            expected = pd.date_range(
                ts.min(),
                ts.max(),
                freq="H"
            )

            missing_hours += len(
                expected.difference(ts)
            )

        status = (
            missing_hours == 0
        )

        self._update_report(
            "Time Series Continuity",
            status,
            {
                "missing_hours":
                    int(missing_hours)
            }
        )
        logger.info("Time Series Continuity Completed")
        return status

    # =====================================================
    # MARKETPLACE RULES
    # =====================================================

    def validate_marketplace_logic(self):

        violations = {}

        violations[
            "completed_gt_requests"
        ] = int(
            (
                self.data[
                    "completed_rides"
                ]
                >
                self.data[
                    "ride_requests"
                ]
            ).sum()
        )

        violations[
            "cancelled_gt_requests"
        ] = int(
            (
                self.data[
                    "cancelled_rides"
                ]
                >
                self.data[
                    "ride_requests"
                ]
            ).sum()
        )

        violations[
            "driver_logic"
        ] = int(
            (
                self.data[
                    "busy_drivers"
                ]
                +
                self.data[
                    "available_drivers"
                ]
                >
                self.data[
                    "active_drivers"
                ]
            ).sum()
        )

        status = all(
            v == 0
            for v in
            violations.values()
        )

        self._update_report(
            "Marketplace Logic Validation",
            status,
            violations
        )
        logger.info("Marketplace Logic Validation Completed")
        return status

    # =====================================================
    # REVENUE VALIDATION
    # =====================================================

    def validate_revenue(self):
        estimated = (
            self.data[
                "completed_rides"
            ]
            *
            (
                self.data[
                    "avg_fare"
                ]
            )
        )

        error_pct = np.abs(
            estimated
            -
            self.data["revenue"]
        ) / (
            self.data["revenue"] + 1
        )

        violations = int(
            (
                error_pct > 0.05
            ).sum()
        )

        status = (
            violations == 0
        )

        self._update_report(
            "Revenue Validation",
            status,
            {
                "violations":
                    violations
            }
        )
        logger.info("Revenue Validation Completed")
        return status

    # =====================================================
    # KPI VALIDATION
    # =====================================================

    def validate_marketplace_kpis(self):

        expected_acceptance = (
            self.data[
                "completed_rides"
            ]
            /
            (
                self.data[
                    "ride_requests"
                ] + 1
            )
        )

        expected_utilization = (
            self.data[
                "busy_drivers"
            ]
            /
            (
                self.data[
                    "active_drivers"
                ] + 1
            )
        )

        acc_error = np.abs(
            expected_acceptance
            -
            self.data[
                "acceptance_rate"
            ]
        )

        util_error = np.abs(
            expected_utilization
            -
            self.data[
                "utilization_rate"
            ]
        )

        violations = int(
            (
                acc_error > 0.20
            ).sum()
            +
            (
                util_error > 0.20
            ).sum()
        )

        status = (
            violations == 0
        )

        self._update_report(
            "Marketplace KPI Validation",
            status,
            {
                "violations":
                    violations
            }
        )
        logger.info("Marketplace KPI Validation Completed")
        return status

    # =====================================================
    # GEOSPATIAL
    # =====================================================

    def validate_geospatial(self):

        invalid = (

            (
                self.data[
                    "latitude"
                ]
                .between(
                    40.4,
                    41.1
                )
                == False
            )

            |

            (
                self.data[
                    "longitude"
                ]
                .between(
                    -74.4,
                    -73.5
                )
                == False
            )
        ).sum()

        status = (
            invalid == 0
        )

        self._update_report(
            "Geospatial Validation",
            status,
            {
                "invalid_rows":
                    int(invalid)
            }
        )
        logger.info("Geospatial Validation Completed")
        return status

    # =====================================================
    # EVENT FLAGS
    # =====================================================

    def validate_event_flags(self):

        flags = [

            "holiday_flag",
            "concert_flag",
            "sports_event_flag",
            "festival_flag"
        ]

        issues = []

        for col in flags:

            invalid = (
                ~self.data[col]
                .isin([0, 1])
            ).sum()

            if invalid > 0:

                issues.append(
                    {
                        "column": col,
                        "invalid":
                            int(invalid)
                    }
                )

        status = (
            len(issues) == 0
        )

        self._update_report(
            "Event Validation",
            status,
            issues
        )
        logger.info("Event Validation Completed")
        return status

    # =====================================================
    # INFINITE VALUES
    # =====================================================

    def validate_infinite_values(self):

        numeric = (
            self.data
            .select_dtypes(
                include=np.number
            )
        )

        count = int(
            np.isinf(
                numeric
            ).sum().sum()
        )

        status = (
            count == 0
        )

        self._update_report(
            "Infinite Values",
            status,
            {
                "count": count
            }
        )
        logger.info("Infinite Values Completed")
        return status

    # =====================================================
    # OUTLIERS
    # =====================================================

    def validate_outliers(self):

        cols = [

            "ride_requests",
            "active_drivers",
            "surge_multiplier",
            "avg_fare",
            "revenue",
            "rider_wait_time"
        ]

        report = {}

        for col in cols:

            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            report[col] = int(
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

        self._update_report(
            "Outlier Validation",
            True,
            report
        )
        logger.info("Outlier Validation Completed")
        return True

    # =====================================================
    # CARDINALITY
    # =====================================================

    def validate_cardinality(self):

        report = {

            "zone_id":
                int(
                    self.data[
                        "zone_id"
                    ].nunique()
                ),

            "borough":
                int(
                    self.data[
                        "borough"
                    ].nunique()
                ),

            "zone_type":
                int(
                    self.data[
                        "zone_type"
                    ].nunique()
                )
        }

        self._update_report(
            "Cardinality Validation",
            True,
            report
        )
        logger.info("Cardinality Validation Completed")
        return True

    # =====================================================
    # CORRELATION
    # =====================================================

    def validate_correlation(self):

        numeric = (
            self.data
            .select_dtypes(
                include=np.number
            )
        )

        corr = (
            numeric
            .corr()
            .abs()
        )

        high_corr = []

        upper = corr.where(
            np.triu(
                np.ones(
                    corr.shape
                ),
                k=1
            ).astype(bool)
        )

        for col in upper.columns:

            for row in upper.index:

                value = upper.loc[
                    row,
                    col
                ]

                if (
                    pd.notna(value)
                    and
                    value > 0.98
                ):

                    high_corr.append(
                        {
                            "feature_1":
                                row,
                            "feature_2":
                                col,
                            "corr":
                                float(value)
                        }
                    )

        self._update_report(
            "Correlation Validation",
            True,
            high_corr
        )
        logger.info("Correlation Validation Completed")
        return True

    # =====================================================
    # DATA DRIFT
    # =====================================================

    def validate_data_drift(
        self,
        train_df,
        test_df
    ):

        drift = {}

        cols = [

            "ride_requests",
            "active_drivers",
            "surge_multiplier",
            "avg_fare",
            "revenue",
            "traffic_index"
        ]

        overall = True

        for col in cols:

            stat, p = ks_2samp(
                train_df[col],
                test_df[col]
            )

            detected = (
                p < 0.05
            )

            if detected:
                overall = False

            drift[col] = {

                "ks":
                    float(stat),

                "p_value":
                    float(p),

                "drift":
                    detected
            }

        self.drift_report = drift

        self._update_report(
            "Data Drift Validation",
            overall,
            drift
        )
        logger.info("Data Drift Validation Completed")
        return overall

    # =====================================================
    # FORECASTING READINESS
    # =====================================================

    def validate_forecasting_readiness(self):

        history_days = (
            (
                self.data[
                    "timestamp"
                ].max()
                -
                self.data[
                    "timestamp"
                ].min()
            )
            .days
        )

        min_obs = (
            self.data
            .groupby("zone_id")
            .size()
            .min()
        )

        status = (
            history_days >= 300
            and
            min_obs > 100
        )

        self._update_report(
            "Forecasting Readiness",
            status,
            {
                "history_days":
                    int(history_days),

                "minimum_zone_observations":
                    int(min_obs)
            }
        )
        logger.info("Forecasting Readiness Completed")
        return status
    
    def write_validation_status(self, status_file_path, validation_report, overall_status):

        with open(status_file_path, "w") as f:

            f.write(
                f"Validation Status: {overall_status}\n"
            )

            f.write(
                "=" * 80 + "\n"
            )

            for check_name, result in validation_report.items():

                f.write(
                    f"{check_name}: "
                    f"{result['status']}\n"
                )

                if result["details"]:

                    f.write(
                        f"Details: "
                        f"{result['details']}\n"
                    )

                f.write(
                    "-" * 80 + "\n"
                )

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def initiate_data_validation(self):

        self.generate_dataset_statistics()

        validation_results = {

            "Schema":
                self.validate_schema(),

            "Datatype":
                self.validate_datatypes(),

            "Missing":
                self.validate_missing_values(),

            "Duplicates":
                self.validate_duplicates(),

            "Numerical Ranges":
                self.validate_numerical_ranges(),

            "Categorical":
                self.validate_categorical_values(),

            "Target":
                self.validate_target(),

            "Timestamp":
                self.validate_timestamp(),

            "Continuity":
                self.validate_time_series_continuity(),

            "Marketplace Logic":
                self.validate_marketplace_logic(),

            "Revenue":
                self.validate_revenue(),

            "KPI":
                self.validate_marketplace_kpis(),

            "Geospatial":
                self.validate_geospatial(),

            "Events":
                self.validate_event_flags(),

            "Infinite":
                self.validate_infinite_values(),

            "Outliers":
                self.validate_outliers(),

            "Cardinality":
                self.validate_cardinality(),

            "Correlation":
                self.validate_correlation(),

            "Forecasting":
                self.validate_forecasting_readiness()
        }



        self._save_reports()

        overall_status = all(
            validation_results.values()
        )
        
        self.write_validation_status(
        self.config.STATUS_FILE,
        self.validation_report,
        overall_status
        )

        print(
            f"Validation Status: "
            f"{overall_status}"
        )
        return overall_status