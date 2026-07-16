from feast import FeatureView, Field
from feast.types import (
    Int32,
    Int64,
    Float32,
    Float64,
    String,
)

from entities import zone_timestamp
from data_sources import forecast_source

import yaml
from pathlib import Path
from datetime import timedelta


TYPE_MAPPING = {
    "Int32": Int32,
    "Int64": Int64,
    "Float32": Float32,
    "Float64": Float64,
    "String": String,
    "object": String,
}

def load_feature_config():

    config_path = (
        Path(__file__).parent
        / "feature_columns.yaml"
    )

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def build_schema():

    config = load_feature_config()

    feature_cfg = config["feature_columns"]

    feast_features = feature_cfg["feast_features"]

    feature_dtypes = feature_cfg["feature_dtypes"]

    schema = []

    for feature_name in feast_features:

        dtype_name = feature_dtypes[feature_name]

        schema.append(
            Field(
                name=feature_name,
                dtype=TYPE_MAPPING[dtype_name],
            )
        )

    return schema


ride_feature_view = FeatureView(
    name="ride_marketplace_features",
    entities=[zone_timestamp],
    ttl=timedelta(days=30),
    schema=build_schema(),
    source=forecast_source,
)