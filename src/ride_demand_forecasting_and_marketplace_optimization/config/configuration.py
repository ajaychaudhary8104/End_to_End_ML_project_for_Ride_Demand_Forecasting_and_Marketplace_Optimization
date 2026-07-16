from src.ride_demand_forecasting_and_marketplace_optimization.constants import *
from src.ride_demand_forecasting_and_marketplace_optimization.utils.common import read_yaml, create_directories
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import (DataIngestionConfig,
                                                                                           DataValidationConfig,
                                                                                           DataPreprocessingConfig,
                                                                                           FeatureEngineeringConfig,
                                                                                           FeatureStoreConfig,
                                                                                           DataTransformationConfig)


class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH, schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)


        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        schema = self.schema

        create_directories(
            [config.root_dir]
        )

        validation_config = (DataValidationConfig(
                root_dir=Path(config.root_dir),

                unzip_data_dir=Path(config.unzip_data_dir),

                STATUS_FILE=Path(config.STATUS_FILE),

                REPORT_FILE=Path(config.REPORT_FILE),

                DRIFT_REPORT_FILE=Path(config.DRIFT_REPORT_FILE),

                STATS_REPORT_FILE=Path(config.STATS_REPORT_FILE),

                all_schema=schema.COLUMNS,

                numerical_ranges=schema.numerical_ranges,

                categorical_values=schema.categorical_values,

                thresholds=schema.thresholds,

                high_cardinality_columns=
                schema.high_cardinality_columns,

                leakage_columns=
                schema.leakage_columns,

                timestamp_column=
                schema.timestamp_column,

                target_column=
                schema.target_column,

                marketplace_rules= schema.marketplace_rules,

                forecasting= schema.forecasting,

                geospatial_bounds=schema.geospatial_bounds
            )
        )

        return validation_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        config = self.config.data_preprocessing

        schema = self.schema

        create_directories(
            [config.root_dir]
        )

        return DataPreprocessingConfig(

            root_dir=Path(
                config.root_dir
            ),

            input_data_path=Path(
                config.input_data_path
            ),

            output_data_path=Path(
                config.output_data_path
            ),

            preprocessing_report_path=Path(
                config.preprocessing_report_path
            ),

            target_column=
            schema.target_column,

            timestamp_column=
            schema.timestamp_column,

            numerical_ranges= schema.numerical_ranges,

            outlier_method=
            config.outlier_method,

            outlier_iqr_multiplier=
            config.outlier_iqr_multiplier,

            numerical_imputation_method=
            config.numerical_imputation_method,

            categorical_imputation_method=
            config.categorical_imputation_method,

            optimize_memory=
            config.optimize_memory,

            save_format=
            config.save_format
        ) 
    
    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:

        config = self.config.feature_engineering

        create_directories(
            [config.root_dir]
        )

        feature_engineering_config = FeatureEngineeringConfig(

            root_dir=Path(config.root_dir),

            input_data_path=Path(config.input_data_path),

            output_data_path=Path(config.output_data_path),

            feature_report_path=Path(config.feature_report_path),
            original_feature_count=config.original_feature_count
        )    

        return feature_engineering_config
    

    def get_feature_store_config(self):

        config = self.config.feature_store

        create_directories(
            [config.root_dir]
        )


        return FeatureStoreConfig(

            root_dir=Path(config.root_dir),
            
            feature_columns_path=Path(config.feature_columns_path),
            
            feature_repo_path=Path(config.feature_repo_path),

            offline_feature_path=Path(config.offline_feature_path),

            registry_path=Path(config.registry_path),

            online_store_path=Path(config.online_store_path),

            feature_store_yaml_path=Path(config.feature_store_yaml_path),

            project_name=config.project_name,

            feature_view_name=config.feature_view_name,

            feature_service_name=config.feature_service_name,

            entity_name=config.entity_name,

            join_key=config.join_key,

            timestamp_column=config.timestamp_column,

            start_date=config.start_date,

            end_date=config.end_date
        
        ) 
    
    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation

        create_directories(
            [config.root_dir]
        )

        data_transformation_config = DataTransformationConfig(root_dir=Path(config.root_dir),
                                                              feature_repo_path=Path(config.feature_repo_path),
                                                                feature_engineered_data_path=Path(config.feature_engineered_data_path),
                                                              feature_service_name=config.feature_service_name,
                                                                target_column=config.target_column,
                                                                train_size=config.train_size,
                                                                validation_size=config.validation_size,
                                                                test_size=config.test_size,
                                                                enable_feature_selection=config.enable_feature_selection,
                                                                num_selected_features=config.num_selected_features,
                                                                split_artifacts_dir=Path(config.split_artifacts_dir),
                                                                train_file_path=Path(config.train_file_path),
                                                                validation_file_path=Path(config.validation_file_path),
                                                                test_file_path=Path(config.test_file_path),
                                                                preprocessor_path=Path(config.preprocessor_path),
                                                                feature_selector_path=Path(config.feature_selector_path),
                                                                feature_names_path=Path(config.feature_names_path),
                                                                selected_features_path=Path(config.selected_features_path),
                                                                metadata_path=Path(config.metadata_path)
                                                              )
                                                            
        return data_transformation_config
