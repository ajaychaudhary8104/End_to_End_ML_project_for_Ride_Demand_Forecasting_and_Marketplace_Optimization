from src.ride_demand_forecasting_and_marketplace_optimization.config.configuration import ConfigurationManager
from src.ride_demand_forecasting_and_marketplace_optimization.components.model_promotion import ModelPromotion
from src.ride_demand_forecasting_and_marketplace_optimization import logger


STAGE_NAME = "Model Promotion stage"


class ModelPromotionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_promotion_config = config.get_model_promotion_config()
        model_promotion = ModelPromotion(config=model_promotion_config)
        model_promotion.promote()


if __name__ == '__main__':

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPromotionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        
    except Exception as e:
        logger.exception(e)
        raise e