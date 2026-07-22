import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
import os

S3_BUCKET = "ride-forecasting-bucket-8104"

FILES_TO_UPLOAD = [
    "artifacts/data_transformation/preprocessor.pkl",
    "artifacts/data_transformation/feature_names.pkl",
    "artifacts/model_promotion/production_model.pkl"
]


def upload_to_s3():
    try:
        s3 = boto3.client("s3")

        # Verify AWS credentials
        s3.list_buckets()

        # Verify bucket exists
        s3.head_bucket(Bucket=S3_BUCKET)

        print(f"Connected to bucket: {S3_BUCKET}")

        for file_path in FILES_TO_UPLOAD:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"{file_path} not found")
            
            s3_key = file_path.as_posix()
            
            s3.upload_file(
                str(file_path),
                S3_BUCKET,
                s3_key
            )

            print(f"Uploaded: {file_path}")

        print("\nAll files uploaded successfully.")

    except FileNotFoundError as e:
        print(f"File Error: {e}")

    except NoCredentialsError:
        print("AWS credentials not found.")

    except ClientError as e:
        print(f"AWS Error: {e}")

    except Exception as e:
        print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    upload_to_s3()