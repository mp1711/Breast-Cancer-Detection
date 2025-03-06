from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db import crud
from app.db.session import get_db
import os
import shutil
from app.core.security import get_current_admin_user
from app.schemas import user as user_schemas
from app.schemas import model as model_schemas
from typing import List, Dict, Any
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/datasets/{dataset_id}/models", response_model=List[model_schemas.ModelDetail])
async def get_dataset_models(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """Get all models for a specific dataset"""
    dataset = crud.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    models = db.query(crud.models.Model).filter(
        crud.models.Model.dataset_id == dataset_id
    ).all()

    dataset_name = os.path.basename(dataset.dataset_path)
    results = []

    for model in models:
        # Get available plots for this model
        plots = {}
        valid_plots = ['roc_curve', 'precision_recall',
                       'training_history', 'confusion_matrix']

        for plot_type in valid_plots:
            # Use only the standard path structure
            plot_path = os.path.join("results", str(dataset_id), model.name, f"{plot_type}.png")
            
            if os.path.exists(plot_path):
                plots[plot_type] = f"/api/models/datasets/{dataset_id}/results/{model.name}/{plot_type}"

        # Calculate additional metrics
        metrics = {
            "accuracy": model.accuracy or 0.0,
            "loss": model.loss or 0.0,
            "auc": model.auc or 0.0
        }
        
        # Format description
        description = f"{model.name} model"
        if model.accuracy:
            description = f"{model.name} model with {model.accuracy*100:.2f}% accuracy"
            
        # Create the model detail object with all required fields
        model_detail = model_schemas.ModelDetail(
            id=model.id,
            name=model.name,
            accuracy=model.accuracy or 0.0,
            loss=model.loss,
            auc=model.auc,
            model_path=model.model_path,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            description=description,
            plots=plots,
            metrics=metrics
        )
        results.append(model_detail)

    return results


@router.get("/datasets/{dataset_id}/models/{model_name}", response_model=model_schemas.ModelDetail)
async def get_dataset_model(
    dataset_id: int,
    model_name: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific model for a dataset"""
    dataset = crud.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    model = db.query(crud.models.Model).filter(
        crud.models.Model.name == model_name,
        crud.models.Model.dataset_id == dataset_id
    ).first()

    if not model:
        raise HTTPException(
            status_code=404, detail="Model not found for this dataset")

    # Get available plots for this model using standard path
    plots = {}
    dataset_name = os.path.basename(dataset.dataset_path)
    
    # Check for plots in the standard directory structure
    valid_plots = ['roc_curve', 'precision_recall', 'training_history', 'confusion_matrix']
    
    for plot_type in valid_plots:
        plot_path = os.path.join("results", str(dataset_id), model_name, f"{plot_type}.png")
        
        if os.path.exists(plot_path):
            plots[plot_type] = f"/api/models/datasets/{dataset_id}/results/{model_name}/{plot_type}"

    # Build metrics with safe defaults for nullable fields
    metrics = {
        "accuracy": model.accuracy if model.accuracy is not None else 0.0,
        "loss": model.loss if model.loss is not None else 0.0,
        "auc": model.auc if model.auc is not None else 0.0
    }

    # Create description with properly formatted accuracy
    description = f"{model.name} model"
    if model.accuracy:
        description = f"{model.name} model with {model.accuracy*100:.2f}% accuracy"

    # Explicitly create all fields in the model detail response
    return model_schemas.ModelDetail(
        id=model.id,
        name=model.name,
        accuracy=model.accuracy if model.accuracy is not None else 0.0,
        loss=model.loss,
        auc=model.auc,
        model_path=model.model_path,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        description=description,
        plots=plots,
        metrics=metrics
    )


@router.post("/datasets/{dataset_id}/train")
async def train_dataset_models(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_admin: user_schemas.User = Depends(get_current_admin_user)
):
    """Admin-only endpoint to train models for a dataset. Returns only when training is complete."""
    # Get the dataset
    dataset = crud.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Initialize trainer with dataset-specific directories
    from app.ml.model_trainer import ModelTrainer
    trainer = ModelTrainer(dataset_path=dataset.dataset_path,
                           dataset_id=dataset_id, db_session=db)

    try:
        # Execute training synchronously - will only return when complete
        await trainer.train_all_models()

        # Get all trained models for this dataset after training
        models = db.query(crud.models.Model).filter(
            crud.models.Model.dataset_id == dataset_id
        ).all()

        # Find the best model based on accuracy
        best_model = max(
            models, key=lambda x: x.accuracy or 0) if models else None

        # Just update the database reference to the best model
        if best_model:
            dataset.best_model_id = best_model.id
            db.commit()

            # Optional: Create a symlink for convenience, but don't duplicate the model file
            try:
                # Create a small metadata file that just points to the real model path
                # This avoids duplicating the large model file
                with open(os.path.join(dataset.dataset_path, "best_model_path.txt"), 'w') as f:
                    f.write(best_model.model_path)
            except Exception as e:
                print(f"Note: Could not create best model indicator: {e}")

        return {
            "success": True,
            "message": "Models trained successfully",
            "models": [
                {
                    "name": model.name,
                    "accuracy": model.accuracy
                } for model in models
            ],
            "best_model": {
                "name": best_model.name,
                "accuracy": best_model.accuracy
            } if best_model else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during model training: {str(e)}"
        )


@router.get("/datasets/{dataset_id}/results/{model_name}/{plot_type}")
async def get_dataset_model_plot(
    dataset_id: int,
    model_name: str,
    plot_type: str,
    db: Session = Depends(get_db)
):
    """Get a specific result plot for a model trained on a dataset"""
    dataset = crud.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    valid_models = ['CNN', 'VGG16', 'VGG19', 'ResNet']
    valid_plots = ['roc_curve', 'precision_recall',
                   'training_history', 'confusion_matrix']

    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Must be one of {valid_models}"
        )

    if plot_type not in valid_plots:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plot type. Must be one of {valid_plots}"
        )

    # Use only the standard path
    plot_path = os.path.join("results", str(
        dataset_id), model_name, f"{plot_type}.png")

    if os.path.exists(plot_path):
        return FileResponse(
            plot_path,
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"}
        )

    raise HTTPException(
        status_code=404,
        detail=f"Plot not found. Model may not have been trained yet."
    )


@router.get("/datasets/{dataset_id}/best", response_model=model_schemas.Model)
async def get_dataset_best_model(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """Get the best performing model for a specific dataset"""
    dataset = crud.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if a best model is already selected for this dataset
    if dataset.best_model_id:
        best_model = crud.get_model(db, model_id=dataset.best_model_id)
        if best_model:
            return model_schemas.Model(
                id=best_model.id,
                name=best_model.name,
                accuracy=best_model.accuracy if best_model.accuracy is not None else 0.0,
                loss=best_model.loss,
                auc=best_model.auc,
                model_path=best_model.model_path,
                dataset_id=dataset_id,
                description=f"Best model: {best_model.name} with {best_model.accuracy*100:.2f}% accuracy" if best_model.accuracy else f"Best model: {best_model.name}"
            )

    # Otherwise find the best model by accuracy
    models = db.query(crud.models.Model).filter(
        crud.models.Model.dataset_id == dataset_id
    ).all()

    if not models:
        raise HTTPException(
            status_code=404,
            detail="No models found for this dataset. Train models first."
        )

    best_model = max(models, key=lambda x: x.accuracy or 0)

    return model_schemas.Model(
        id=best_model.id,
        name=best_model.name,
        accuracy=best_model.accuracy if best_model.accuracy is not None else 0.0,
        loss=best_model.loss,
        auc=best_model.auc,
        model_path=best_model.model_path,
        dataset_id=dataset_id,
        description=f"Best model: {best_model.name} with {best_model.accuracy*100:.2f}% accuracy" if best_model.accuracy else f"Best model: {best_model.name}"
    )


@router.post("/datasets/{dataset_id}/models/{model_name}/set-as-best")
async def set_dataset_best_model(
    dataset_id: int,
    model_name: str,
    db: Session = Depends(get_db),
    current_admin: user_schemas.User = Depends(get_current_admin_user)
):
    """Set a specific model as the best model for a dataset"""

    # Validate the model name
    valid_models = ['CNN', 'VGG16', 'VGG19', 'ResNet']
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Must be one of {valid_models}"
        )

    # Get dataset
    dataset = crud.get_dataset(db, dataset_id=dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Find the model by name and dataset_id
    model = db.query(crud.models.Model).filter(
        crud.models.Model.name == model_name,
        crud.models.Model.dataset_id == dataset_id
    ).first()

    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found for dataset {dataset_id}"
        )

    # Validate the model path
    if not model.model_path or not os.path.exists(model.model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Model ({model.name}) has invalid path: {model.model_path}"
        )

    try:
        # Update database reference
        dataset.best_model_id = model.id
        db.commit()

        # Create the metadata file pointing to the model path
        with open(os.path.join(dataset.dataset_path, "best_model_path.txt"), 'w') as f:
            f.write(model.model_path)

        return {
            "success": True,
            "message": f"Model {model.name} set as best model for dataset {dataset.name}",
            "model": {
                "id": model.id,
                "name": model.name,
                "accuracy": model.accuracy
            }
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error setting best model: {str(e)}"
        )
