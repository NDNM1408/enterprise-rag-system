from pydantic import BaseModel
from datetime import datetime 

class BaseQueueEvent(BaseModel):
    Id: str
    CreatedDate: datetime
