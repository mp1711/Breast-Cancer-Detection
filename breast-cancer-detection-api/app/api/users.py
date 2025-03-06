from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import crud
from app.schemas import user as user_schemas
from app.db.session import get_db
from app.core.security import get_current_active_user
from app.core.security import get_current_admin_user

from typing import List

router = APIRouter()


@router.get("/me", response_model=user_schemas.User)
def read_users_me(current_user=Depends(get_current_active_user)):
    return current_user


@router.get("/admin/all", response_model=List[user_schemas.User])
def read_all_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_admin: user_schemas.User = Depends(get_current_admin_user)
):
    """Admin-only endpoint to get all users"""
    users = crud.get_users(db, skip=skip, limit=limit)
    return users
