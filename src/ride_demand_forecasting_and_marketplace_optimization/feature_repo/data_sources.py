from feast import FileSource

forecast_source = FileSource(
    #path=r"D:\End_to_End_ML_project_for_Ride_Demand_Forecasting_and_Marketplace_Optimization\artifacts\feature_engineering\featured_data.parquet",
    path="s3://ride-forecasting-bucket-8104/artifacts/feature_engineering/featured_data.parquet",
    event_timestamp_column="timestamp",
)