from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class ValidationThresholds:

    missing_value_threshold: float

    max_duplicate_percentage: float

    correlation_threshold: float

    drift_pvalue_threshold: float


@dataclass(frozen=True)
class ForecastingConfig:

    minimum_history_days: int

    minimum_zone_observations: int


@dataclass(frozen=True)
class MarketplaceRules:

    revenue_tolerance: float

    kpi_tolerance: float

    max_surge_multiplier: float

    min_acceptance_rate: float

    min_utilization_rate: float


@dataclass(frozen=True)
class GeoBounds:

    latitude: Any

    longitude: Any   


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: Path
    REPORT_FILE: Path
    STATS_REPORT_FILE: Path
    DRIFT_REPORT_FILE: Path
    all_schema: Dict[str, str]
    target_column: str
    timestamp_column: str
    categorical_values: Dict[str, List[str]]
    numerical_ranges: Dict[str, Any]
    high_cardinality_columns: List[str]
    leakage_columns: List[str]
    thresholds: ValidationThresholds
    forecasting: ForecastingConfig
    marketplace_rules: MarketplaceRules
    geospatial_bounds: GeoBounds     

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    input_data_path: Path
    output_data_path: Path
    preprocessing_report_path: Path
    target_column: str
    timestamp_column: str
    numerical_ranges: dict
    outlier_method: str 
    outlier_iqr_multiplier: float
    numerical_imputation_method: str
    categorical_imputation_method: str
    optimize_memory: bool
    save_format: str    

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    input_data_path: Path
    output_data_path: Path
    feature_report_path: Path
    original_feature_count: int    

from dataclasses import dataclass
from pathlib import Path

@dataclass
class FeatureStoreConfig:
    root_dir: Path
    feature_columns_path: Path
    feature_repo_path: Path
    offline_feature_path: Path
    registry_path: Path
    online_store_path: Path
    feature_store_yaml_path: Path
    project_name: str
    feature_view_name: str
    feature_service_name: str
    entity_name: str
    join_key: str
    timestamp_column: str
    start_date: str
    end_date: str    