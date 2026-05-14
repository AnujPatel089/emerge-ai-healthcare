"""Compatibility exports for backend authentication helpers.

Approval hierarchy is enforced in src.auth_utils and backend.main.
Nurses are view-only for new patient registrations.
"""

from src.auth import *  # noqa: F401,F403
