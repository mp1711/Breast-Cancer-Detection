from pydantic.v1 import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Skin Cancer Detection App"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A web application for skin cancer detection using machine learning."
    DATABASE_URL: str = "sqlite:///./app.db"
    SECRET_KEY: str = "your_secret_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    MODEL_PATH: str = "./models/best_model.keras"
    LIME_EXPLANATION_PATH: str = "./predictions/lime_explanation.png"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()