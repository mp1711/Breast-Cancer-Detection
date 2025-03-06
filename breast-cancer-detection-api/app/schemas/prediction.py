from pydantic import BaseModel
from typing import Optional, List

class PredictionResult(BaseModel):
    label: str
    confidence: float
    explanation: Optional[str] = None

class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image string

class PredictionResponse(BaseModel):
    result: PredictionResult
    message: str    
    success: bool
    dataset_name: str

class LIMEExplanation(BaseModel):
    image_path: str
    mask: List[List[int]]  # Coordinates for the mask
    highlighted_image: str  # Base64 encoded image with highlights