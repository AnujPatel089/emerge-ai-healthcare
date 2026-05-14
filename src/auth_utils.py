"""
Shared authentication helpers for registered users.

Passwords are hashed with PBKDF2-SHA256 via passlib. This avoids storing plain
text passwords and does not require platform-specific bcrypt wheels.
"""

from __future__ import annotations

import json
import hashlib
import hmac
import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from passlib.context import CryptContext
except Exception:
    CryptContext = None


pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto") if CryptContext else None
LOCAL_USER_STORE = Path("data/registered_users.json")

ROLE_APPROVAL_MATRIX = {
    "super_admin": {"admin"},
    "admin": {"doctor", "nurse", "patient"},
    "doctor": {"patient"},
    "nurse": set(),
    "patient": set(),
}

ROLE_REGISTRATION_VIEW_MATRIX = {
    "super_admin": {"admin", "doctor", "nurse", "patient"},
    "admin": {"doctor", "nurse", "patient"},
    "doctor": {"patient"},
    "nurse": {"patient"},
    "patient": set(),
}

ROLE_LEVELS = {
    "patient": 1,
    "nurse": 2,
    "doctor": 3,
    "admin": 4,
    "super_admin": 5,
}

ACCOUNT_STATUSES = {"pending", "active", "rejected", "suspended"}


def normalize_role(role: str) -> str:
    normalized = role.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "superadmin":
        return "super_admin"
    return normalized


def can_approve_role(approver_role: str, requested_role: str) -> bool:
    """Hospital-style hierarchy: higher roles approve only permitted lower roles."""
    return normalize_role(requested_role) in ROLE_APPROVAL_MATRIX.get(normalize_role(approver_role), set())


def can_view_registration_role(viewer_role: str, requested_role: str) -> bool:
    """Registration visibility can be broader than approval permission."""
    return normalize_role(requested_role) in ROLE_REGISTRATION_VIEW_MATRIX.get(normalize_role(viewer_role), set())


def approval_level_for_role(role: str) -> str:
    role = normalize_role(role)
    if role == "admin":
        return "super_admin"
    if role == "doctor":
        return "admin"
    if role == "nurse":
        return "admin"
    if role == "patient":
        return "doctor_or_admin"
    if role == "super_admin":
        return "super_admin"
    return "admin"


def hash_password(password: str) -> str:
    if pwd_context:
        return pwd_context.hash(password)
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 260_000).hex()
    return f"pbkdf2_sha256_stdlib${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith("pbkdf2_sha256_stdlib$"):
        _, salt, expected = password_hash.split("$", 2)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 260_000).hex()
        return hmac.compare_digest(digest, expected)
    if pwd_context:
        return pwd_context.verify(password, password_hash)
    return False


def load_local_users() -> list[dict[str, Any]]:
    if not LOCAL_USER_STORE.exists():
        return []
    try:
        return json.loads(LOCAL_USER_STORE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_local_users(users: list[dict[str, Any]]) -> None:
    LOCAL_USER_STORE.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_USER_STORE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def register_local_user(
    full_name: str,
    username: str,
    email: str,
    password: str,
    requested_role: str,
) -> tuple[bool, str]:
    users = load_local_users()
    username_norm = username.strip().lower()
    email_norm = email.strip().lower()

    if any(user["username"].lower() == username_norm for user in users):
        return False, "Username already exists."
    if any(user["email"].lower() == email_norm for user in users):
        return False, "Email already exists."

    role = normalize_role(requested_role)
    status = "pending"
    users.append(
        {
            "id": len(users) + 1,
            "full_name": full_name.strip(),
            "username": username_norm,
            "email": email_norm,
            "password_hash": hash_password(password),
            "requested_role": role,
            "role": role,
            "status": status,
            "account_status": status,
            "approval_level": approval_level_for_role(role),
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            "approved_by": None,
            "approved_at": None,
            "rejected_by": None,
            "rejected_at": None,
            "rejected_reason": None,
        }
    )
    save_local_users(users)
    return True, "Your account is pending approval."


def authenticate_local_user(username: str, password: str, role: str) -> tuple[bool, str, dict[str, Any] | None]:
    username_norm = username.strip().lower()
    role_norm = role.strip().lower()
    for user in load_local_users():
        if user["username"].lower() != username_norm:
            continue
        if not verify_password(password, user["password_hash"]):
            return False, "Invalid username or password.", None
        if user["role"] != role_norm:
            return False, f"This account is registered as {user['role']}, not {role_norm}.", None
        status = user.get("account_status") or user.get("status")
        if status == "pending":
            return False, "Your account is pending approval.", None
        if user["status"] == "rejected":
            return False, "This account request was rejected. Contact an approved administrator.", None
        if status == "suspended":
            return False, "This account is suspended. Contact an administrator.", None
        if status != "active":
            return False, "This account is not active.", None
        return True, "Login successful.", user
    return False, "Invalid username or password.", None


def pending_local_approvals(approver_role: str) -> list[dict[str, Any]]:
    allowed_roles = ROLE_APPROVAL_MATRIX.get(normalize_role(approver_role), set())
    return [
        user for user in load_local_users()
        if user.get("role") in allowed_roles
        and (user.get("account_status") or user.get("status")) == "pending"
    ]


def visible_local_registrations(viewer_role: str) -> list[dict[str, Any]]:
    allowed_roles = ROLE_REGISTRATION_VIEW_MATRIX.get(normalize_role(viewer_role), set())
    return [
        user for user in load_local_users()
        if user.get("role") in allowed_roles
        and (user.get("account_status") or user.get("status")) == "pending"
    ]


def set_local_account_status(
    username: str,
    status: str,
    approved_by: str,
    rejected_reason: str | None = None,
    approver_role: str | None = None,
) -> bool:
    users = load_local_users()
    changed = False
    for user in users:
        if user["username"].lower() == username.lower():
            if approver_role and not can_approve_role(approver_role, user.get("requested_role") or user["role"]):
                return False
            if user["username"].lower() == approved_by.strip().lower():
                return False
            user["status"] = status
            user["account_status"] = status
            now = datetime.utcnow().isoformat(timespec="seconds")
            if status == "active":
                user["approved_by"] = approved_by
                user["approved_at"] = now
                user["rejected_by"] = None
                user["rejected_at"] = None
                user["rejected_reason"] = None
            elif status == "rejected":
                user["rejected_by"] = approved_by
                user["rejected_at"] = now
                user["rejected_reason"] = rejected_reason
            changed = True
            break
    if changed:
        save_local_users(users)
    return changed
