from src.ride_demand_forecasting_and_marketplace_optimization.config.configuration import ConfigurationManager
from src.ride_demand_forecasting_and_marketplace_optimization.components.model_training import ModelTraining
from src.ride_demand_forecasting_and_marketplace_optimization import logger

STAGE_NAME = "Model Training stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(config=model_training_config)
        model_training.initiate_model_training()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e