from typing import Optional
from pydantic import BaseModel

class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_path: str 

class DatasetCreate(DatasetBase):
    pass

class DatasetResponse(DatasetBase):
    id: int

    class Config:
        from_attributes = True