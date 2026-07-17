from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.ride_demand_forecasting_and_marketplace_optimization.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.ride_demand_forecasting_and_marketplace_optimization.pipeline.stage_03_data_preprocessing import DataPreprocessingTrainingPipeline
from src.ride_demand_forecasting_and_marketplace_optimization.pipeline.stage_04_feature_engineering import FeatureEngineeringTrainingPipeline
from src.ride_demand_forecasting_and_marketplace_optimization.pipeline.stage_05_feature_store import FeatureStoreTrainingPipeline
from src.ride_demand_forecasting_and_marketplace_optimization.pipeline.stage_06_data_transformation import DataTransformationTrainingPipeline
from src.ride_demand_forecasting_and_marketplace_optimization.pipeline.stage_07_model_training import ModelTrainingPipeline
import warnings


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Preprocessing stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_preprocessing = DataPreprocessingTrainingPipeline()
   data_preprocessing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Feature Engineering stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   feature_engineering = FeatureEngineeringTrainingPipeline()
   feature_engineering.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e     

STAGE_NAME = "Feature Store stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
   feature_store = FeatureStoreTrainingPipeline()
   feature_store.main()
   logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e   

STAGE_NAME = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Training stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_training = ModelTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e