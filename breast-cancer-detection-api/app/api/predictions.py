from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Path
from app.ml.predictor import Predictor
from app.schemas.prediction import PredictionResponse, PredictionResult
from app.db.session import get_db
from sqlalchemy.orm import Session
import os
import traceback
from fastapi.responses import FileResponse
from typing import Optional

router = APIRouter()

@router.post("/datasets/{dataset_id}/predict", response_model=PredictionResponse)
async def predict_image_with_dataset(
    dataset_id: int = Path(..., description="The ID of the dataset to use for prediction"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Make a prediction using the best model for a specific dataset."""
    # Define temp_file_path outside the try block to avoid UnboundLocalError
    temp_file_path = None
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, detail="File type not supported. Please upload an image.")

    try:
        # Verify the dataset exists
        from app.db import crud
        dataset = crud.get_dataset(db, dataset_id=dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Try to find the best model using the metadata file
        best_model_path_file = os.path.join(dataset.dataset_path, "best_model_path.txt")
        
        if os.path.exists(best_model_path_file):
            # Read the path from the metadata file
            with open(best_model_path_file, 'r') as f:
                best_model_path = f.read().strip()
        else:
            # If no metadata file, check if the dataset has a best model ID
            if dataset.best_model_id:
                best_model = crud.get_model(db, model_id=dataset.best_model_id)
                if best_model and best_model.model_path:
                    best_model_path = best_model.model_path
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail="Dataset has a best model reference but the model file could not be found."
                    )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="No best model set for this dataset. Please train models and set a best model first."
                )
        
        # Verify that the model file actually exists
        if not os.path.exists(best_model_path):
            raise HTTPException(
                status_code=400, 
                detail=f"Model file not found at {best_model_path}. Please retrain the model."
            )

        # Create upload directories
        os.makedirs("./uploads", exist_ok=True)
        os.makedirs("./predictions", exist_ok=True)

        # Sanitize filename to prevent path traversal attacks
        safe_filename = os.path.basename(file.filename)
        temp_file_path = f"./uploads/{safe_filename}"
        
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Create predictor specifically for this dataset
        predictor = Predictor(model_path=best_model_path, db_session=db)
        result = predictor.predict(temp_file_path)

        # Check for errors in prediction
        if "error" in result:
            print(f"Prediction error: {result['error']}")
        
        # Convert explanation path to URL path if it exists
        explanation_url = None
        if result.get("explanation"):
            filename = os.path.basename(result["explanation"])
            explanation_url = filename
        
        prediction_result = PredictionResult(
            label=result.get("result", "Error"),
            confidence=float(result.get("confidence", 0.0)),
            explanation=explanation_url  # Use URL path instead of file path
        )

        return PredictionResponse(
            result=prediction_result,
            message=f"Prediction completed successfully using dataset {dataset.name}'s best model",
            success=True if result.get("result") not in ["Error", None] else False,
            dataset_name=dataset.name
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up upload file but keep the explanation
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.get("/explanation/{filename}")
async def get_explanation(filename: str):
    """Get a LIME explanation image by filename"""
    # Sanitize filename to prevent path traversal attacks
    safe_filename = os.path.basename(filename)
    file_path = f"./predictions/{safe_filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Explanation image not found")
    
    return FileResponse(
        file_path,
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"}  # Allow caching for 1 hour
    )