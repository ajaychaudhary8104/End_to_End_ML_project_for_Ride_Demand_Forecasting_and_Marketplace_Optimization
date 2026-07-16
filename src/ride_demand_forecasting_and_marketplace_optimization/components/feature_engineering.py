import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.utils.common import save_json
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import FeatureEngineeringConfig


class FeatureEngineering:
    """
    Production Grade Feature Engineering
    Ride Demand Forecasting &
    Marketplace Optimization
    """

    def __init__(self,config: FeatureEngineeringConfig):

        self.config = config

        self.data = pd.read_parquet(
            self.config.input_data_path
        )

        self.feature_report = {}

        logger.info(
            f"Dataset Loaded "
            f"Shape={self.data.shape}"
        )

    # =====================================================
    # TIME FEATURES
    # =====================================================

    def create_time_features(self):

        self.data["timestamp"] = pd.to_datetime(
            self.data["timestamp"]
        )

        self.data["dayofyear"] = (
            self.data["timestamp"]
            .dt.dayofyear
        )

        self.data["is_month_start"] = (
            self.data["timestamp"]
            .dt.is_month_start
            .astype(int)
        )

        self.data["is_month_end"] = (
            self.data["timestamp"]
            .dt.is_month_end
            .astype(int)
        )

        self.data["is_quarter_end"] = (
            self.data["timestamp"]
            .dt.is_quarter_end
            .astype(int)
        )

        self.data["is_peak_hour"] = (
            self.data["hour"]
            .isin(
                [7, 8, 9, 17, 18, 19]
            )
            .astype(int)
        )

        self.data["is_business_hour"] = (
            self.data["hour"]
            .between(9, 18)
            .astype(int)
        )

        self.data["is_night"] = (
            (
                self.data["hour"] >= 22
            )
            |
            (
                self.data["hour"] <= 5
            )
        ).astype(int)

        logger.info(
            "Time features created"
        )

    # =====================================================
    # CYCLICAL FEATURES
    # =====================================================

    def create_cyclical_features(self):

        self.data["hour_sin"] = np.sin(
            2 *
            np.pi *
            self.data["hour"] /
            24
        )

        self.data["hour_cos"] = np.cos(
            2 *
            np.pi *
            self.data["hour"] /
            24
        )

        self.data["dayofweek_sin"] = np.sin(
            2 *
            np.pi *
            self.data["dayofweek"] /
            7
        )

        self.data["dayofweek_cos"] = np.cos(
            2 *
            np.pi *
            self.data["dayofweek"] /
            7
        )

        self.data["month_sin"] = np.sin(
            2 *
            np.pi *
            self.data["month"] /
            12
        )

        self.data["month_cos"] = np.cos(
            2 *
            np.pi *
            self.data["month"] /
            12
        )

        logger.info(
            "Cyclical features created"
        )

    # =====================================================
    # DEMAND LAGS
    # =====================================================

    def create_demand_lag_features(self):

        self.data.sort_values(
            [
                "zone_id",
                "timestamp"
            ],
            inplace=True
        )

        lag_hours = [
            1,
            2,
            3,
            6,
            12,
            24,
            168
        ]

        for lag in lag_hours:

            self.data[
                f"ride_requests_lag_{lag}"
            ] = (
                self.data
                .groupby("zone_id")
                ["ride_requests"]
                .shift(lag)
            )

        logger.info(
            "Demand lag features created"
        )

    # =====================================================
    # SUPPLY LAGS
    # =====================================================

    def create_supply_lag_features(self):

        for lag in [1, 24]:

            self.data[
                f"active_drivers_lag_{lag}"
            ] = (
                self.data
                .groupby("zone_id")
                ["active_drivers"]
                .shift(lag)
            )

            self.data[
                f"available_drivers_lag_{lag}"
            ] = (
                self.data
                .groupby("zone_id")
                ["available_drivers"]
                .shift(lag)
            )

        logger.info(
            "Supply lag features created"
        )

    # =====================================================
    # ROLLING FEATURES
    # =====================================================

    def create_rolling_features(self):

        grp = (
            self.data
            .groupby("zone_id")
            ["ride_requests"]
        )

        self.data[
            "ride_requests_roll_mean_3"
        ] = (
            grp.transform(
                lambda x:
                x.shift(1)
                .rolling(3)
                .mean()
            )
        )

        self.data[
            "ride_requests_roll_mean_6"
        ] = (
            grp.transform(
                lambda x:
                x.shift(1)
                .rolling(6)
                .mean()
            )
        )

        self.data[
            "ride_requests_roll_mean_24"
        ] = (
            grp.transform(
                lambda x:
                x.shift(1)
                .rolling(24)
                .mean()
            )
        )

        self.data[
            "ride_requests_roll_std_24"
        ] = (
            grp.transform(
                lambda x:
                x.shift(1)
                .rolling(24)
                .std()
            )
        )

        logger.info(
            "Rolling features created"
        )

    # =====================================================
    # DEMAND TREND FEATURES
    # =====================================================

    def create_demand_trend_features(self):

        self.data["demand_growth_24h"] = (
            (
                self.data[
                    "ride_requests_lag_1"
                ]
                -
                self.data[
                    "ride_requests_lag_24"
                ]
            )
            /
            (
                self.data[
                    "ride_requests_lag_24"
                ]
                + 1
            )
        )

        self.data["demand_growth_week"] = (
            (
                self.data[
                    "ride_requests_lag_1"
                ]
                -
                self.data[
                    "ride_requests_lag_168"
                ]
            )
            /
            (
                self.data[
                    "ride_requests_lag_168"
                ]
                + 1
            )
        )

        logger.info(
            "Demand trend features created"
        )

    # =====================================================
    # MARKETPLACE FEATURES
    # =====================================================

    def create_marketplace_features(self):

        self.data[
            "supply_demand_ratio"
        ] = (
            self.data["available_drivers"]
            /
            (
                self.data["ride_requests"]
                + 1
            )
        )

        self.data[
            "marketplace_gap"
        ] = (
            self.data["ride_requests"]
            -
            self.data["available_drivers"]
        )

        self.data[
            "driver_shortage"
        ] = (
            self.data[
                "marketplace_gap"
            ]
            .clip(lower=0)
        )

        self.data[
            "marketplace_pressure"
        ] = (
            self.data["ride_requests"]
            /
            (
                self.data[
                    "active_drivers"
                ]
                + 1
            )
        )

        logger.info(
            "Marketplace features created"
        )

    # =====================================================
    # DRIVER FEATURES
    # =====================================================

    def create_driver_efficiency_features(self):

        self.data[
            "rides_per_driver"
        ] = (
            self.data[
                "completed_rides"
            ]
            /
            (
                self.data[
                    "active_drivers"
                ]
                + 1
            )
        )

        self.data[
            "completion_rate"
        ] = (
            self.data[
                "completed_rides"
            ]
            /
            (
                self.data[
                    "ride_requests"
                ]
                + 1
            )
        )

        self.data[
            "cancellation_rate"
        ] = (
            self.data[
                "cancelled_rides"
            ]
            /
            (
                self.data[
                    "ride_requests"
                ]
                + 1
            )
        )

        logger.info(
            "Driver efficiency features created"
        )

    # =====================================================
    # SURGE FEATURES
    # =====================================================

    def create_surge_features(self):

        self.data[
            "surge_gap"
        ] = (
            self.data[
                "surge_multiplier"
            ]
            - 1
        )

        self.data[
            "surge_x_demand"
        ] = (
            self.data[
                "surge_multiplier"
            ]
            *
            self.data[
                "ride_requests"
            ]
        )

        self.data[
            "surge_x_traffic"
        ] = (
            self.data[
                "surge_multiplier"
            ]
            *
            self.data[
                "traffic_index"
            ]
        )

        logger.info(
            "Surge features created"
        )

    # =====================================================
    # WEATHER FEATURES
    # =====================================================

    def create_weather_features(self):

        self.data[
            "temperature_deviation"
        ] = (
            self.data[
                "temperature"
            ]
            -
            self.data[
                "temperature"
            ].mean()
        )

        self.data[
            "rain_intensity"
        ] = (
            self.data[
                "rainfall"
            ]
            *
            self.data[
                "humidity"
            ]
        )

        self.data[
            "weather_severity_score"
        ] = (
            self.data["rainfall"]
            +
            self.data["snowfall"] * 2
            +
            (
                self.data["wind_speed"]
                / 10
            )
        )

        logger.info(
            "Weather features created"
        )

    # =====================================================
    # TRAFFIC FEATURES
    # =====================================================

    def create_traffic_features(self):

        self.data[
            "traffic_x_demand"
        ] = (
            self.data[
                "traffic_index"
            ]
            *
            self.data[
                "ride_requests"
            ]
        )

        self.data[
            "traffic_pressure"
        ] = (
            self.data[
                "traffic_index"
            ]
            /
            (
                self.data[
                    "available_drivers"
                ]
                + 1
            )
        )

        logger.info(
            "Traffic features created"
        )

    # =====================================================
    # EVENT FEATURES
    # =====================================================

    def create_event_features(self):

        self.data["any_event"] = (

            self.data["holiday_flag"]

            +

            self.data["concert_flag"]

            +

            self.data["sports_event_flag"]

            +

            self.data["festival_flag"]

            > 0

        ).astype(int)

        self.data[
            "event_intensity"
        ] = (

            self.data["holiday_flag"]

            +

            self.data["concert_flag"]

            +

            self.data["sports_event_flag"]

            +

            self.data["festival_flag"]

        )

        logger.info(
            "Event features created"
        )

    # =====================================================
    # ECONOMIC FEATURES
    # =====================================================

    def create_economic_features(self):

        self.data[
            "fuel_cpi_ratio"
        ] = (
            self.data["fuel_price"]
            /
            (
                self.data["cpi"]
                + 1
            )
        )

        self.data[
            "economic_stress"
        ] = (
            self.data[
                "fuel_price"
            ]
            *
            self.data[
                "unemployment_rate"
            ]
        )

        logger.info(
            "Economic features created"
        )

    # =====================================================
    # GEOSPATIAL FEATURES
    # =====================================================

    def create_geospatial_features(self):

        self.data[
            "borough_avg_demand"
        ] = (
            self.data
            .groupby("borough")
            ["ride_requests"]
            .transform("mean")
        )

        self.data[
            "zone_avg_demand"
        ] = (
            self.data
            .groupby("zone_id")
            ["ride_requests"]
            .transform("mean")
        )

        logger.info(
            "Geospatial features created"
        )

    # =====================================================
    # REVENUE FEATURES
    # =====================================================

    def create_revenue_features(self):

        self.data[
            "revenue_per_request"
        ] = (
            self.data["revenue"]
            /
            (
                self.data[
                    "ride_requests"
                ]
                + 1
            )
        )

        self.data[
            "revenue_per_driver"
        ] = (
            self.data["revenue"]
            /
            (
                self.data[
                    "active_drivers"
                ]
                + 1
            )
        )

        self.data[
            "tip_percentage"
        ] = (
            self.data["avg_tip"]
            /
            (
                self.data["avg_fare"]
                + 1
            )
        )

        logger.info(
            "Revenue features created"
        )

    # =====================================================
    # FORECASTING FEATURES
    # =====================================================

    def create_forecasting_features(self):

        self.data[
            "same_hour_yesterday"
        ] = (
            self.data
            .groupby("zone_id")
            ["ride_requests"]
            .shift(24)
        )

        self.data[
            "same_hour_last_week"
        ] = (
            self.data
            .groupby("zone_id")
            ["ride_requests"]
            .shift(168)
        )

        logger.info(
            "Forecasting features created"
        )

    # =====================================================
    # ENTITY KEY
    # =====================================================

    def create_entity_key(self):

        self.data[
            "zone_timestamp_key"
        ] = (

            self.data["zone_id"]
            .astype(str)

            +

            "_"

            +

            self.data[
                "timestamp"
            ]
            .dt.strftime(
                "%Y%m%d%H"
            )
        )

        logger.info(
            "Entity key created"
        )

    # =====================================================
    # HANDLE GENERATED NANS
    # =====================================================

    def handle_generated_nans(self):

        numeric_cols = (
            self.data
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        self.data[
            numeric_cols
        ] = (
            self.data[
                numeric_cols
            ]
            .fillna(0)
        )

        logger.info(
            "Generated NaNs handled"
        )

    # =====================================================
    # SAVE FEATURES
    # =====================================================

    def save_features(self):

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

        self.feature_report[
            "rows"
        ] = int(
            len(self.data)
        )

        self.feature_report[
            "columns"
        ] = int(
            len(self.data.columns)
        )

        self.feature_report[
            "generated_features"
        ] = int(
            len(self.data.columns)
            -
            self.config.original_feature_count
        )

        save_json(
            Path(
                self.config.feature_report_path
            ),
            self.feature_report
        )

        logger.info(
            "Feature engineering completed"
        )

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def initiate_feature_engineering(self):

        logger.info(
            "Starting Feature Engineering"
        )

        self.create_time_features()

        self.create_cyclical_features()

        self.create_demand_lag_features()

        self.create_supply_lag_features()

        self.create_rolling_features()

        self.create_demand_trend_features()

        self.create_marketplace_features()

        self.create_driver_efficiency_features()

        self.create_surge_features()

        self.create_weather_features()

        self.create_traffic_features()

        self.create_event_features()

        self.create_economic_features()

        self.create_geospatial_features()

        self.create_revenue_features()

        self.create_forecasting_features()

        self.create_entity_key()

        self.handle_generated_nans()

        self.save_features()

        logger.info(
            "Feature Engineering Completed"
        )

        return (
            self.config.output_data_path
        )