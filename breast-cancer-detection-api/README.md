# Breast Cancer Detection App

A web application for automated detection of Breast cancer lesions using deep learning.

## Features

- Upload Breast lesion images for analysis
- Train and compare multiple machine learning models:
  - CNN
  - VGG16
  - VGG19
  - ResNet50
- Interactive visualization of model performance
- LIME explanations for model predictions
- User authentication system
- Admin panel for model management

## Tech Stack

- **Backend**: FastAPI, TensorFlow, SQLAlchemy
- **Frontend**: React, CSS
- **Database**: SQLite
- **ML Models**: CNN, VGG16, VGG19, ResNet50

## Setup and Installation

### Backend

```bash
cd backend
alembic init alembic

```
- Replace code in /alembic/env.py with the below code

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(file)))

# Import your models
from app.db.models import Base

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
fileConfig(config.config_file_name)

# Set SQLAlchemy URL (can be overridden from command line)
from app.core.config import settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

```bash
alembic revision --autogenerate -m "initial_setup"
alembic upgrade head
pip install -r requirements.txt
uvicorn app.main:app --reload
```
