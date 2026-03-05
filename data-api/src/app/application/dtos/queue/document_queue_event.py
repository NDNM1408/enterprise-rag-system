from base_queue_event import BaseQueueEvent
from typing import Optional

class PreProcessDocumentQueueEvent(BaseQueueEvent):
    DocumentId: str
    BucketName: str
    Path: str
    Name: str
    ModelId: Optional[str]
    ModelName: Optional[str]
    TableName: str
    IsURL: bool

class UpsertDocumentQueueEvent(BaseQueueEvent):
    DocumentId: str
    BucketName: str
    Path: str
    Name: str
    ModelId: Optional[str]
    TableName: str
    DataSourceid: str
    parentId: str

class UpsertMetadataQueueEvent(BaseQueueEvent):
    textContent: str
    DocumentId: str
    ModelId: str
    TableName: str
    DatasourceId: str
    parentId: str 

class DeleteDocumentQueueEvent(BaseQueueEvent):
    pass

