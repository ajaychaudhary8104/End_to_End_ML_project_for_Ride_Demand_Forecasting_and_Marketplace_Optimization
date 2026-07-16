from src.ride_demand_forecasting_and_marketplace_optimization.config.configuration import ConfigurationManager
from src.ride_demand_forecasting_and_marketplace_optimization.components.feature_store import FeatureStoreComponent
from src.ride_demand_forecasting_and_marketplace_optimization import logger


STAGE_NAME = "FEATURE STORE STAGE"

class FeatureStoreTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        feature_store_config = (
            config.get_feature_store_config()
        )

        feature_store = FeatureStoreComponent(
            config= feature_store_config
        )

        
        feature_store.initiate_feature_store()
       

if __name__ == "__main__":
    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = (
            FeatureStoreTrainingPipeline()
        )

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<"
        )

    except Exception as e:

        logger.exception(e)

        raise e