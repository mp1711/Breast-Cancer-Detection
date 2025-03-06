from pydantic import BaseModel
from typing import Dict, Optional, List, Any, Union

class ModelBase(BaseModel):
    name: str
    accuracy: float = 0.0
    loss: Optional[float] = None
    auc: Optional[float] = None
    model_path: Optional[str] = None
    dataset_id: Optional[int] = None

class ModelCreate(ModelBase):
    pass

class ModelUpdate(ModelBase):
    pass

class Model(BaseModel):
    id: int
    name: str
    accuracy: float = 0.0
    loss: Optional[float] = None
    auc: Optional[float] = None
    model_path: Optional[str] = None
    dataset_id: Optional[int] = None
    description: Optional[str] = None

    class Config:
        from_attributes = True  

class ModelDetail(Model):
    plots: Dict[str, str] = {}
    metrics: Optional[Dict[str, Any]] = None
    dataset_name: Optional[str] = None 

    class Config:
        from_attributes = True  

class ModelList(BaseModel):
    models: List[Model]

class ModelPerformance(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float