from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, datasets, models, predictions, users
from app.db.session import engine
from app.db.models import Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Skin Cancer Detection API")

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(users.router, prefix="/api/users", tags=["users"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Skin Cancer Detection API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}