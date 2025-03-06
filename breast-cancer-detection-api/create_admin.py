import sys
import os
import bcrypt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db import models

def get_password_hash(password):
    # Convert to bytes if passed as string
    if isinstance(password, str):
        password = password.encode('utf-8')
    
    # Generate salt and hash password
    hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
    
    # Return as string for storage
    return hashed_password.decode('utf-8')

def create_admin(email, username, password):
    # Get database session
    db = next(get_db())
    
    try:
        # Check if user exists
        existing_user = db.query(models.User).filter(models.User.email == email).first()
        if existing_user:
            print(f"User with email {email} already exists.")
            return
        
        # Hash password
        hashed_password = get_password_hash(password)
        
        # Create admin user
        admin_user = models.User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            is_active=True,
            is_admin=True
        )
        
        # Add to database
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print(f"Admin user created: {username} (ID: {admin_user.id})")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_admin.py admin@example.com admin_username password")
        sys.exit(1)
    
    create_admin(sys.argv[1], sys.argv[2], sys.argv[3])