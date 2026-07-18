import shutil
import subprocess
import sys
from pathlib import Path
import time
from src.ride_demand_forecasting_and_marketplace_optimization import logger
from src.ride_demand_forecasting_and_marketplace_optimization.entity.config_entity import FeatureStoreConfig



class FeatureStoreComponent:

    def __init__(self, config: FeatureStoreConfig):
        self.config = config

    def create_repo(self):

        repo_path = self.config.feature_repo_path

        repo_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # remove old files
        for file in repo_path.glob("*.py"):
            file.unlink()

        (repo_path / "__init__.py").write_text("")
        shutil.copy(self.config.feature_columns_path, repo_path / "feature_columns.yaml")

        # -------------------------------------------------
        # feature_store.yaml
        # -------------------------------------------------

        yaml_content = f"""
project: {self.config.project_name}

registry: registry.db

provider: local
entity_key_serialization_version: 2
online_store:
    type: sqlite
    path: online_store.db
"""

        (repo_path / "feature_store.yaml").write_text(
            yaml_content.strip()
        )

        # -------------------------------------------------
        # entities.py
        # -------------------------------------------------

        entities_content = f"""
from feast import Entity, ValueType

zone_timestamp = Entity(
    name="{self.config.entity_name}",
    join_keys=["{self.config.join_key}"],
    value_type=ValueType.STRING
)
"""

        (repo_path / "entities.py").write_text(
            entities_content.strip()
        )

        # -------------------------------------------------
        # data_sources.py
        # -------------------------------------------------

        datasource_content = f'''
from feast import FileSource

forecast_source = FileSource(
    path=r"{self.config.offline_feature_path.resolve()}",
    event_timestamp_column="{self.config.timestamp_column}",
)
'''

        (repo_path / "data_sources.py").write_text(
            datasource_content.strip()
        )

        # -------------------------------------------------
        # feature_views.py
        # -------------------------------------------------

        feature_view_content = f"""
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


TYPE_MAPPING = {{
    "Int32": Int32,
    "Int64": Int64,
    "Float32": Float32,
    "Float64": Float64,
    "String": String,
    "object": String,
}}

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
    name="{self.config.feature_view_name}",
    entities=[zone_timestamp],
    ttl=timedelta(days=30),
    schema=build_schema(),
    source=forecast_source,
)
"""

        (repo_path / "feature_views.py").write_text(
            feature_view_content.strip()
        )

        # -------------------------------------------------
        # feature_services.py
        # -------------------------------------------------

        service_content = f"""
from feast import FeatureService

from feature_views import ride_feature_view


forecast_service = FeatureService(

    name="{self.config.feature_service_name}",

    features=[
        ride_feature_view
    ]
)
"""

        (repo_path / "feature_services.py").write_text(
            service_content.strip()
        )

    def clean_registry(self):

        registry = self.config.registry_path

        online_store = self.config.online_store_path

        if registry.exists():
            registry.unlink()

        if online_store.exists():
            online_store.unlink()

    def apply(self):
        
        feature_file = Path(self.config.offline_feature_path)

        if not feature_file.exists():
            raise FileNotFoundError(
                f"Offline feature file not found: {feature_file}"
            )

        result = subprocess.run(

            [
                sys.executable,
                "-m",
                "feast.cli",
                "apply"
            ],

            cwd=str(
                self.config.feature_repo_path
            ),

            capture_output=True,

            text=True
        )

        print(result.stdout)

        if result.returncode != 0:

            print(result.stderr)

            raise Exception(
                f"Feast Apply Failed\\n{result.stderr}"
            )
        
    def _materialize_features(self):
        """
        Materialize features from offline store to online store.
        """

        result = subprocess.run(

            [
                sys.executable,
                "-m",
                "feast.cli",
                "materialize",
                str(self.config.start_date),
                str(self.config.end_date)
            ],

            cwd=str(
                self.config.feature_repo_path
            ),

            capture_output=True,

            text=True
        )

        print("=" * 100)
        print("MATERIALIZATION")
        print("=" * 100)

        print("STDOUT:")
        print(result.stdout)

        print("=" * 100)

        print("STDERR:")
        print(result.stderr)

        print("=" * 100)

        if result.returncode != 0:

            raise Exception(
                f"Feast Materialization Failed:\n{result.stderr}"
            )

        logger.info(
            "Features materialized successfully"
        )    

    def initiate_feature_store(self):

        self.create_repo()

        self.clean_registry()

        start = time.time()
        self.apply()
        print(f"Apply Time: {time.time()-start:.2f}s")

        start = time.time()
        self._materialize_features()
        print(f"Materialize Time: {time.time()-start:.2f}s")

        logger.info(
            "Feature Store Component completed successfully"
        )