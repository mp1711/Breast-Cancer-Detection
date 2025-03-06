from pydantic import BaseModel
from typing import Optional

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str
    is_admin: bool = False  # Default to regular user

class User(UserBase):
    id: int
    is_active: bool
    is_admin: bool = False  # Include admin status in response

    class Config:
        from_attributes = True

class UserInDB(User):
    hashed_password: str