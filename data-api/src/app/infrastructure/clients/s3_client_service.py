import logging
from io import BytesIO
from uuid import uuid4
from fastapi import HTTPException
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError
from app.configurations.configurations import settings


class S3ClientService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.s3_client = self._create_s3_client()

    def _create_s3_client(self) -> BaseClient:
        try:
            import boto3
            return boto3.client(
                "s3",
                region_name=settings.AWS_DEFAULT_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                aws_session_token=settings.AWS_SESSION_TOKEN,
                endpoint_url=settings.S3_ENDPOINT_URL,          # <-- THÊM DÒNG NÀY
            )
        except Exception as e:
            self.logger.error("Failed to create S3 client", exc_info=e)
            raise e

    async def upload_file(self, data_buffer: bytes, bucket: str, kb_id: str, file_name: str):
        target_file = f"{kb_id}/{file_name}"
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=target_file,
                Body=data_buffer
            )
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to upload file to S3: {e}")
            raise HTTPException(status_code=500, detail="S3 upload failed")

    async def delete_file(self, bucket: str, kb_id: str, file_name: str):
        target_file = f"{kb_id}/{file_name}"
        try:
            self.s3_client.delete_object(
                Bucket=bucket,
                Key=target_file
            )
            self.logger.info(f"Deleted file {target_file} from S3 bucket {bucket}.")
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to delete file from S3: {e}")
            raise HTTPException(status_code=500, detail="S3 delete failed")

    async def delete_file_by_key(self, bucket: str, key: str):
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            self.logger.info(f"Deleted {key} from S3 bucket {bucket}.")
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to delete file from S3: {e}")
            raise HTTPException(status_code=500, detail="S3 delete failed")

    async def delete_folder(self, bucket: str, prefix: str):
        """
        Delete all objects with the given prefix (folder) from S3.

        Args:
            bucket: S3 bucket name
            prefix: Folder prefix (e.g., "kb_id/")

        Returns:
            Number of objects deleted
        """
        try:
            # Ensure prefix ends with /
            if not prefix.endswith("/"):
                prefix = prefix + "/"

            deleted_count = 0
            continuation_token = None

            while True:
                # List objects with the prefix
                list_kwargs = {
                    "Bucket": bucket,
                    "Prefix": prefix,
                }
                if continuation_token:
                    list_kwargs["ContinuationToken"] = continuation_token

                response = self.s3_client.list_objects_v2(**list_kwargs)

                if "Contents" not in response:
                    break

                # Delete objects in batches
                objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]

                if objects_to_delete:
                    self.s3_client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": objects_to_delete}
                    )
                    deleted_count += len(objects_to_delete)

                # Check if there are more objects
                if not response.get("IsTruncated"):
                    break

                continuation_token = response.get("NextContinuationToken")

            self.logger.info(f"Deleted {deleted_count} objects from S3 folder {prefix} in bucket {bucket}.")
            return deleted_count

        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to delete folder from S3: {e}")
            raise HTTPException(status_code=500, detail="S3 folder delete failed")

    async def get_file_content(self, bucket: str, url: str) -> str:
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=url)
            return response["Body"].read().decode("utf-8")
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to get file content from S3: {e}")
            raise HTTPException(status_code=500, detail="S3 file retrieval failed")
        
