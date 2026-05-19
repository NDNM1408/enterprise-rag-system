"""S3 / MinIO client for parse inputs and outputs.

Object key layout:
    <bucket>/<job_id>/input.<ext>
    <bucket>/<job_id>/result.md
    <bucket>/<job_id>/images/<filename>

The bucket is auto-created on first import (best-effort). MinIO accepts the
``ap-southeast-1`` constraint and silently no-ops if the bucket already exists.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Iterator, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from settings import settings

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CLIENT = None


def client():
    """Lazy-init a thread-safe boto3 client."""
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                _CLIENT = boto3.client(
                    "s3",
                    endpoint_url=settings.s3_endpoint_url,
                    region_name=settings.s3_region,
                    aws_access_key_id=settings.s3_access_key,
                    aws_secret_access_key=settings.s3_secret_key,
                    config=Config(
                        signature_version="s3v4",
                        s3={"addressing_style": "path" if settings.s3_force_path_style else "auto"},
                        retries={"max_attempts": 5, "mode": "standard"},
                    ),
                )
                _ensure_bucket(_CLIENT, settings.s3_bucket)
    return _CLIENT


def _ensure_bucket(c, bucket: str) -> None:
    try:
        c.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchBucket"}:
            log.info("creating S3 bucket %s", bucket)
            c.create_bucket(Bucket=bucket)
        else:
            log.warning("head_bucket(%s) failed (%s); assuming bucket exists", bucket, code)


# ---------------------------------------------------------------------------
#  Object operations
# ---------------------------------------------------------------------------

def put_bytes(key: str, payload: bytes, content_type: str = "application/octet-stream") -> None:
    client().put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
    )


def put_file(key: str, path: str, content_type: str = "application/octet-stream") -> None:
    client().upload_file(
        Filename=path,
        Bucket=settings.s3_bucket,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )


def get_bytes(key: str) -> bytes:
    obj = client().get_object(Bucket=settings.s3_bucket, Key=key)
    return obj["Body"].read()


def download_to_file(key: str, path: str) -> None:
    client().download_file(Bucket=settings.s3_bucket, Key=key, Filename=path)


def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse 's3://bucket/key/with/slashes' into (bucket, key).

    Raises ValueError if the URL is not in s3:// form.
    """
    if not url.startswith("s3://"):
        raise ValueError(f"not an s3:// URL: {url!r}")
    rest = url[5:]
    if "/" not in rest:
        raise ValueError(f"s3 URL missing key: {url!r}")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"s3 URL missing bucket or key: {url!r}")
    return bucket, key


def download_url_to_file(url: str, path: str) -> None:
    """Download an object identified by a full ``s3://bucket/key`` URL.

    Allows the worker to read files from buckets other than the configured
    default — used by the orchestrator path where data-api keeps source
    files in its own bucket.
    """
    bucket, key = parse_s3_url(url)
    client().download_file(Bucket=bucket, Key=key, Filename=path)


def stream_upload(key: str, fileobj, content_type: str = "application/octet-stream") -> None:
    """Multipart-aware upload from an open file-like object."""
    client().upload_fileobj(
        Fileobj=fileobj,
        Bucket=settings.s3_bucket,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )


def list_keys(prefix: str) -> Iterator[str]:
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            yield obj["Key"]


def delete_prefix(prefix: str) -> int:
    """Delete every object under ``prefix``. Returns count deleted."""
    keys = list(list_keys(prefix))
    if not keys:
        return 0
    # batch in 1000s — S3 delete_objects limit
    deleted = 0
    c = client()
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i : i + 1000]]
        c.delete_objects(Bucket=settings.s3_bucket, Delete={"Objects": batch})
        deleted += len(batch)
    return deleted


def presign_get(key: str, ttl_seconds: Optional[int] = None) -> str:
    return client().generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=ttl_seconds or settings.s3_presigned_ttl,
    )
