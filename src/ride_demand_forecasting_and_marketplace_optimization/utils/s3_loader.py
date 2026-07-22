import os
import boto3

from botocore.exceptions import (
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
)

from src.ride_demand_forecasting_and_marketplace_optimization import logger


S3_BUCKET = os.getenv("S3_BUCKET_NAME")

FILES = [
    "artifacts/data_transformation/preprocessor.pkl",
    "artifacts/data_transformation/feature_names.pkl",
    "artifacts/model_promotion/production_model.pkl",
]


def download_artifacts(force_download: bool = False) -> bool:
    """
    Download model artifacts from S3.

    Args:
        force_download (bool):
            If True, download even if file exists.

    Returns:
        bool:
            True if successful.
    """

    if not S3_BUCKET:
        raise ValueError(
            "Environment variable 'S3_BUCKET_NAME' is not set."
        )

    try:
        s3 = boto3.client("s3")

        logger.info(
            f"Starting artifact download from bucket: {S3_BUCKET}"
        )

        for file_key in FILES:

            local_path = file_key

            if (os.path.exists(local_path) and not force_download):
                logger.info(
                    f"Artifact already exists: {local_path}"
                )
                continue

            os.makedirs(os.path.dirname(local_path),exist_ok=True)

            logger.info(
                f"Downloading {file_key}"
            )

            s3.download_file(
                S3_BUCKET,
                file_key,
                local_path,
            )

            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Download failed for {file_key}"
                )

            logger.info(
                f"Downloaded successfully: {file_key}"
            )

        logger.info(
            "All artifacts downloaded successfully."
        )

        return True

    except NoCredentialsError:
        logger.error(
            "AWS credentials not found."
        )
        raise

    except PartialCredentialsError:
        logger.error(
            "Incomplete AWS credentials."
        )
        raise

    except ClientError as e:
        logger.error(
            f"AWS S3 Error: {str(e)}"
        )
        raise

    except Exception as e:
        logger.exception(
            f"Artifact download failed: {str(e)}"
        )
        raise