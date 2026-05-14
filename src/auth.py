from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from dotenv import load_dotenv
import os

from src.auth_utils import normalize_role

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY is not configured. Add it to .env for local development "
        "or set it in your deployment environment."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

security = HTTPBearer()


fake_users_db = {
    "anuj": {
        "username": "anuj",
        "password": "anuj123",
        "roles": ["super_admin", "admin", "doctor", "nurse"]
    },
    "chintan": {
        "username": "chintan",
        "password": "chintan123",
        "roles": ["super_admin", "admin", "doctor", "nurse"]
    },
    "admin": {
        "username": "admin",
        "password": "admin123",
        "roles": ["admin"]
    },
    "superadmin": {
        "username": "superadmin",
        "password": "superadmin123",
        "roles": ["super_admin"]
    },
    "doctor": {
        "username": "doctor",
        "password": "doctor123",
        "roles": ["doctor"]
    },
    
    "nurse": {
        "username": "nurse",
        "password": "nurse123",
        "roles": ["nurse"]
    }
}


def authenticate_user(username: str, password: str, selected_role: str | None = None):
    username = username.strip().lower()
    user = fake_users_db.get(username)

    # Demo users are checked first so database registration rows cannot shadow
    # the built-in accounts used for classroom/demo access.
    if user:
        if user["password"] != password:
            return False

        allowed_roles = [normalize_role(role) for role in user.get("roles", [])]

        if selected_role:
            selected_role = normalize_role(selected_role)
            if selected_role not in allowed_roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"This user is not allowed to login as {selected_role}."
                )
            return {
                "username": user["username"],
                "role": selected_role,
                "roles": allowed_roles
            }

        default_role = allowed_roles[0] if allowed_roles else user.get("role")

        return {
            "username": user["username"],
            "role": default_role,
            "roles": allowed_roles
        }

    db_user = authenticate_registered_user(username, password, selected_role)
    if db_user:
        return db_user

    return False


def authenticate_registered_user(username: str, password: str, selected_role: str | None = None):
    """Authenticate database-backed registered users before falling back to demo users."""
    try:
        from src.auth_utils import verify_password
        from src.database import SessionLocal
        from src.models import AppUser
    except Exception:
        return False

    db = SessionLocal()
    try:
        user = (
            db.query(AppUser)
            .filter(AppUser.username == username.strip().lower())
            .first()
        )
        if not user:
            return False
        if not verify_password(password, user.password_hash):
            return False

        role = normalize_role(selected_role or user.role)
        if role == "superadmin":
            role = "super_admin"
        if role != user.role:
            raise HTTPException(
                status_code=403,
                detail=f"This user is registered as {user.role}, not {role}."
            )
        status = getattr(user, "account_status", None) or user.status
        if status in {"pending", "pending_approval"}:
            raise HTTPException(
                status_code=403,
                detail="Your account is pending approval."
            )
        if status == "rejected":
            raise HTTPException(
                status_code=403,
                detail="This account request was rejected. Contact an approved administrator."
            )
        if status == "suspended":
            raise HTTPException(
                status_code=403,
                detail="This account is suspended. Contact an administrator."
            )
        if status != "active":
            raise HTTPException(
                status_code=403,
                detail="This account is not active."
            )

        return {
            "username": user.username,
            "role": user.role,
            "roles": [user.role],
            "status": status,
            "full_name": user.full_name,
            "email": user.email,
        }
    finally:
        db.close()


def create_access_token(data: dict):
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return encoded_jwt


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")
        role = payload.get("role")

        if username is None or role is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )

        return {
            "username": username,
            "role": role
        }

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )