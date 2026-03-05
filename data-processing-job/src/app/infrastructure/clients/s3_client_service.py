import logging
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError
from app.configurations.configurations import settings


class S3ClientService:
    def __init__(self):
        # Logger must be assigned BEFORE _create_s3_client() so the error
        # handler inside that method can use self.logger.
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
                endpoint_url=settings.S3_ENDPOINT_URL,
            )
        except Exception as e:
            self.logger.error("Failed to create S3 client", exc_info=e)
            raise

    async def upload_file(self, data_buffer: bytes, bucket: str, kb_id: str, file_name: str):
        target_file = f"{kb_id}/{file_name}"
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=target_file,
                Body=data_buffer,
            )
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to upload file to S3: {e}")
            raise RuntimeError(f"S3 upload failed: {e}") from e

    async def delete_file(self, bucket: str, kb_id: str, file_name: str):
        target_file = f"{kb_id}/{file_name}"
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=target_file)
            self.logger.info(f"Deleted file {target_file} from S3 bucket {bucket}.")
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to delete file from S3: {e}")
            raise RuntimeError(f"S3 delete failed: {e}") from e

    async def get_file(self, bucket: str, url: str) -> bytes:
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=url)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to get file content from S3: {e}")
            raise RuntimeError(f"S3 file retrieval failed: {e}") from e

    async def get_txt_file_content(self, bucket: str, url: str) -> str:
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=url)
            return response["Body"].read().decode("utf-8")
        except (BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to get file content from S3: {e}")
            raise RuntimeError(f"S3 file retrieval failed: {e}") from e
