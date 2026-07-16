from src.ride_demand_forecasting_and_marketplace_optimization.config.configuration import ConfigurationManager
from src.ride_demand_forecasting_and_marketplace_optimization.components.data_validation import DataValidation
from src.ride_demand_forecasting_and_marketplace_optimization import logger


STAGE_NAME = "DATA VALIDATION STAGE"

class DataValidationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        validation_config = (
            config.get_data_validation_config()
        )

        validation = DataValidation(
            config=validation_config
        )

        validation_status = (
            validation.initiate_data_validation()
        )

        if not validation_status:

            raise Exception(
                "Data Validation Failed"
            )

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            DataValidationTrainingPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e