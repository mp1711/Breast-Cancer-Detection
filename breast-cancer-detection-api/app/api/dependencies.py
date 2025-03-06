from app.db.session import get_db
from app.core.security import get_current_active_user, get_current_admin_user

# Re-export these dependencies to avoid circular imports
__all__ = ["get_db", "get_current_active_user", "get_current_admin_user"]