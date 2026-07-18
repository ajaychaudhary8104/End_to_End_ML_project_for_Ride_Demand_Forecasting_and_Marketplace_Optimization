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

@dataclass
class DataTransformationConfig:
    root_dir: Path
    feature_repo_path: Path
    feature_engineered_data_path: Path
    feature_service_name: str
    target_column: str
    train_size: float
    validation_size: float
    test_size: float
    enable_feature_selection: bool
    num_selected_features: int
    split_artifacts_dir: Path
    train_file_path: Path
    validation_file_path: Path
    test_file_path: Path
    preprocessor_path: Path
    feature_selector_path: Path
    feature_names_path: Path
    selected_features_path: Path
    metadata_path: Path    

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_file_path: Path
    validation_file_path: Path
    model_file_path: Path
    metrics_file_path: Path
    model_params: dict
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
    experiment_name: str   

@dataclass(frozen=True)
class ModelPromotionConfig:
    root_dir: Path
    metrics_file_path: Path
    model_file_path: Path
    production_model_path: Path
    registered_model_name: str
    target_stage: str
    mlflow_uri: str
    promote_metric: str
    promote_threshold: float
    archive_existing_versions: bool
    copy_local_model: bool    

@dataclass(frozen=True)
class ModelInferenceConfig:
    root_dir: Path
    feature_repo_path: Path
    feature_service_name: str
    model_path: Path
    input_data_path: Path
    prediction_output_path: Path
    target_column: str
    preprocessor_path: Path             