# End_to_End_ML_project_for_Ride_Demand_Forecasting_and_Marketplace_Optimization

# Configuration Workflow

To modify the pipeline:

1. Update `config/config.yaml` - Set paths and parameters
2. Update `params.yaml` - Modify hyperparameters
3. Update `entity/config_entity.py` - Define configuration entities
4. Update `config/configuration.py` - Implement configuration manager
5. Update components in `components/` - Modify pipeline stages
6. Update pipeline stages in `pipeline/` - Update pipeline logic
7. Update `main.py` - Execute the pipeline
8. Update `dvc.yaml` - Define DVC pipeline stages


set MLFLOW_TRACKING_URI=https://dagshub.com/ajaychaudhary8104/End_to_End_ML_project_for_Ride_Demand_Forecasting_and_Marketplace_Optimization.mlflow
set MLFLOW_TRACKING_USERNAME=ajaychaudhary8104
set MLFLOW_TRACKING_PASSWORD=gangapur8955