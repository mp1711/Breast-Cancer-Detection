from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from typing import List
from sqlalchemy.orm import Session
from datetime import datetime
from app.db import crud
from app.schemas.dataset import DatasetCreate, DatasetResponse
from app.core.security import get_current_active_user
from app.db.session import get_db
import os
import zipfile
import shutil

router = APIRouter()


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    description: str = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user)
):
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only ZIP files are supported"
        )

    datasets_dir = os.path.join("data", "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    dataset_name = os.path.splitext(file.filename)[0]

    dataset_dir = os.path.join(datasets_dir, dataset_name)
    if os.path.exists(dataset_dir):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset '{dataset_name}' already exists"
        )

    os.makedirs(dataset_dir, exist_ok=True)

    temp_file_path = os.path.join(datasets_dir, file.filename)
    try:
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)

        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        benign_dir = os.path.join(dataset_dir, "benign")
        malignant_dir = os.path.join(dataset_dir, "malignant")

        if not (os.path.isdir(benign_dir) and os.path.isdir(malignant_dir)):
            subdirs = [f for f in os.listdir(dataset_dir) if os.path.isdir(
                os.path.join(dataset_dir, f))]
            if len(subdirs) == 1:
                parent_dir = os.path.join(dataset_dir, subdirs[0])
                if (os.path.isdir(os.path.join(parent_dir, "benign")) and
                        os.path.isdir(os.path.join(parent_dir, "malignant"))):
                    for item in os.listdir(parent_dir):
                        shutil.move(
                            os.path.join(parent_dir, item),
                            os.path.join(dataset_dir, item)
                        )
                    shutil.rmtree(parent_dir)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Zip file must contain 'benign' and 'malignant' folders"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Zip file must contain 'benign' and 'malignant' folders"
                )

        benign_count = len([f for f in os.listdir(
            benign_dir) if os.path.isfile(os.path.join(benign_dir, f))])
        malignant_count = len([f for f in os.listdir(
            malignant_dir) if os.path.isfile(os.path.join(malignant_dir, f))])

        # Fix here: Use dataset_path instead of path
        dataset_data = DatasetCreate(
            name=dataset_name,
            description=description or f"Uploaded on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
            f"Contains {benign_count} benign and {malignant_count} malignant images.",
            dataset_path=dataset_dir  # Changed from path to dataset_path
        )

        dataset = crud.create_dataset(db=db, dataset=dataset_data)

        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            dataset_path=dataset.dataset_path  # Changed from file_path to dataset_path
        )

    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ZIP file"
        )
    except Exception as e:
        print(e)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing dataset: {str(e)}"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.get("/", response_model=List[DatasetResponse])
async def get_datasets(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user)
):
    datasets = crud.get_datasets(db, skip=skip, limit=limit)
    return [
        DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            dataset_path=dataset.dataset_path 
        ) for dataset in datasets
    ]

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user)
):
    """Delete a dataset and all associated models and results"""
    dataset = crud.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get all related models to delete their files too
    models = db.query(crud.models.Model).filter(
        crud.models.Model.dataset_id == dataset_id
    ).all()
    
    deleted_resources = {
        "dataset_path": dataset.dataset_path,
        "models": [],
        "results_dir": None
    }
    
    try:
        # 1. Delete model files from disk
        for model in models:
            if model.model_path and os.path.exists(model.model_path):
                try:
                    os.remove(model.model_path)
                    deleted_resources["models"].append(model.model_path)
                except Exception as e:
                    print(f"Warning: Could not delete model file {model.model_path}: {e}")
        
        # 2. Delete models directory for this dataset
        models_dir = os.path.join("models", str(dataset_id))
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            try:
                shutil.rmtree(models_dir)
                deleted_resources["models_dir"] = models_dir
            except Exception as e:
                print(f"Warning: Could not delete models directory {models_dir}: {e}")
        
        # 3. Delete results directory for this dataset
        results_dir = os.path.join("results", str(dataset_id))
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            try:
                shutil.rmtree(results_dir)
                deleted_resources["results_dir"] = results_dir
            except Exception as e:
                print(f"Warning: Could not delete results directory {results_dir}: {e}")
        
        # 4. Delete model records from database
        for model in models:
            db.delete(model)
        
        # 5. Delete the dataset record from database
        db.delete(dataset)
        
        # 6. Delete the dataset directory and contents
        if os.path.exists(dataset.dataset_path) and os.path.isdir(dataset.dataset_path):
            try:
                shutil.rmtree(dataset.dataset_path)
            except Exception as e:
                print(f"Warning: Could not delete dataset directory {dataset.dataset_path}: {e}")
        
        # 7. Commit the database changes
        db.commit()
        
        # Return a detailed success message
        return {
            "message": f"Dataset {dataset_id} deleted successfully",
            "details": {
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "models_deleted": len(models),
                "resources_cleaned": deleted_resources
            }
        }
    
    except Exception as e:
        # Roll back in case of error
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting dataset: {str(e)}"
        )
