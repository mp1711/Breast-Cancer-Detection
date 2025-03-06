from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    is_admin: bool = False

class UserVerify(BaseModel):
    username:str
    password:str

class UserInDB(UserCreate):
    hashed_password: str

class User(BaseModel):
    username: str
    email: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    email: str
    is_admin: bool

class TokenData(BaseModel):
    username: str | None = None