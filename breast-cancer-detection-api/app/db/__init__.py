# File: /skin-cancer-detection-app/skin-cancer-detection-app/backend/app/db/__init__.py

# Import the models to make them available through the package
from app.db.base import Base
from app.db.models import User, Dataset, Model, Prediction