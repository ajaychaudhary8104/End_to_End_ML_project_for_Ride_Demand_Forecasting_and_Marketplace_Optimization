from src.ride_demand_forecasting_and_marketplace_optimization.config.configuration import ConfigurationManager
from src.ride_demand_forecasting_and_marketplace_optimization.components.inference import ModelInference
from src.ride_demand_forecasting_and_marketplace_optimization import logger


STAGE_NAME = "Model Inference stage"


class ModelInferencePipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        inference_config = config.get_model_inference_config()
        model_inference = ModelInference(config=inference_config)
        output_file = model_inference.run_batch_inference()
        logger.info(f"Batch inference completed, predictions saved to: {output_file}")


if __name__ == '__main__':

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelInferencePipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        
    except Exception as e:
        logger.exception(e)
        raise e