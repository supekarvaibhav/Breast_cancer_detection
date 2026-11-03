"""Private image storage backed by Neon Object Storage (S3 compatible)."""
import os
import shutil
from pathlib import Path

from flask import current_app


_REQUIRED_SETTINGS = (
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_ENDPOINT_URL_S3',
    'AWS_REGION',
    'OBJECT_STORAGE_BUCKET',
)


def missing_object_storage_settings() -> list[str]:
    return [key for key in _REQUIRED_SETTINGS if not os.environ.get(key)]


def is_remote_storage_configured() -> bool:
    return not missing_object_storage_settings()


def _s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        's3',
        endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
        region_name=os.environ['AWS_REGION'],
        config=Config(s3={'addressing_style': 'path'}),
    )


def store_image(local_path: str, object_key: str, content_type: str) -> str:
    """Store an image remotely in production or locally during development."""
    if is_remote_storage_configured():
        _s3_client().upload_file(
            local_path,
            os.environ['OBJECT_STORAGE_BUCKET'],
            object_key,
            ExtraArgs={'ContentType': content_type},
        )
        return object_key

    local_name = Path(object_key).name
    destination = os.path.join(current_app.config['UPLOAD_FOLDER'], local_name)
    shutil.copyfile(local_path, destination)
    return local_name


def create_image_url(object_key: str, expires_in: int = 300) -> str:
    """Create a short-lived URL for reading a private image object."""
    return _s3_client().generate_presigned_url(
        'get_object',
        Params={
            'Bucket': os.environ['OBJECT_STORAGE_BUCKET'],
            'Key': object_key,
        },
        ExpiresIn=expires_in,
    )
