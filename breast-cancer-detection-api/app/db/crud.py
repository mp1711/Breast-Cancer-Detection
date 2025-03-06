from sqlalchemy.orm import Session
from app.db import models
from app.schemas import user as user_schemas
from app.schemas import dataset as dataset_schemas
from app.schemas import model as model_schemas
from app.schemas import prediction as prediction_schemas


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_name(db: Session, name: str):
    return db.query(models.User).filter(models.User.username == name).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: user_schemas.UserCreate):
    from app.core.security import get_password_hash

    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        is_active=True,
        is_admin=user.is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_dataset(db: Session, dataset_id: int):
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()


def get_datasets(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Dataset).offset(skip).limit(limit).all()


def create_dataset(db: Session, dataset: dataset_schemas.DatasetCreate):
    db_dataset = models.Dataset(
        name=dataset.name,
        description=dataset.description,
        dataset_path=dataset.dataset_path
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def get_model(db: Session, model_id: int):
    return db.query(models.Model).filter(models.Model.id == model_id).first()


def get_models(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Model).offset(skip).limit(limit).all()


def create_model(db: Session, model: model_schemas.ModelCreate):
    db_model = models.Model(**model.model_dump())  # Changed from dict()
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model


def update_model(db: Session, db_model: models.Model, model_data: model_schemas.ModelCreate):
    # Update model fields
    for key, value in model_data.model_dump().items():  # Changed from dict()
        setattr(db_model, key, value)

    # Commit changes
    db.commit()
    db.refresh(db_model)
    return db_model


def get_prediction(db: Session, prediction_id: int):
    return db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()


def get_predictions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Prediction).offset(skip).limit(limit).all()


def create_prediction(db: Session, prediction_data):
    db_prediction = models.Prediction(**prediction_data)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction
