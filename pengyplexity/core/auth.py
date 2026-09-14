"""Authentication and admin-gated user management for Pengyplexity.

This module provides:

* Password hashing/verification via ``werkzeug.security`` (scrypt-based,
  salted, never stores plaintext).
* :class:`AuthService` — a thin service layer over the :class:`Store` that
  handles login (password check), session helpers, and admin-only user CRUD.
* A bootstrap helper :func:`create_admin` for the CLI entry point
  (``python -m pengyplexity.cli create-admin <username>``).

Design: this module is **not** a Flask blueprint. It is a pure service that
takes a ``Store`` instance. The Flask layer (``app.py`` / blueprints in later
subtasks) wires it into request handlers with session management. Keeping the
logic here (rather than in route functions) makes it trivially testable
offline with a temp store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash


class AuthError(Exception):
    """Base error for authentication/authorization failures."""


class InvalidCredentialsError(AuthError):
    """Raised when username or password is wrong."""


class UserDisabledError(AuthError):
    """Raised when a user attempts to log in but is disabled."""


class NotAdminError(AuthError):
    """Raised when a non-admin attempts an admin operation."""


class UserExistsError(AuthError):
    """Raised when trying to create a user with an existing username."""


class UserNotFoundError(AuthError):
    """Raised when a user does not exist."""


def hash_password(password: str) -> str:
    """Hash a plaintext password using werkzeug's scrypt-based scheme."""
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a plaintext password against a stored hash."""
    return check_password_hash(password_hash, password)


@dataclass
class AuthService:
    """Service layer for authentication and admin user management.

    Parameters
    ----------
    store:
        A :class:`~pengyplexity.core.store.Store` instance (or any object
        with the same user methods).
    """

    store: Any

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Verify credentials and return the user doc on success.

        Raises
        ------
        InvalidCredentialsError
            If the username is unknown or the password doesn't match.
        UserDisabledError
            If the user exists with a matching password but is disabled.
        """
        user = self.store.get_user_by_username(username)
        if user is None:
            raise InvalidCredentialsError("Invalid username or password")
        if not verify_password(password, user["password_hash"]):
            raise InvalidCredentialsError("Invalid username or password")
        if not user.get("enabled", True):
            raise UserDisabledError("Account is disabled")
        # Touch last_login
        self.store.touch_last_login(user["_id"])
        return user

    # ------------------------------------------------------------------
    # Admin-gated user CRUD
    # ------------------------------------------------------------------

    def _assert_admin(self, admin_user: Dict[str, Any]) -> None:
        """Raise NotAdminError if *admin_user* is not an admin."""
        if not admin_user.get("is_admin"):
            raise NotAdminError("Admin privileges required")

    def create_user(
        self,
        admin_user: Dict[str, Any],
        username: str,
        password: str,
        is_admin: bool = False,
    ) -> Dict[str, Any]:
        """Create a new user (admin only).

        Raises
        ------
        NotAdminError
            If *admin_user* is not an admin.
        UserExistsError
            If *username* is already taken.
        """
        self._assert_admin(admin_user)
        if self.store.get_user_by_username(username) is not None:
            raise UserExistsError(f"Username '{username}' already exists")
        pw_hash = hash_password(password)
        return self.store.create_user(username, pw_hash, is_admin=is_admin)

    def list_users(self, admin_user: Dict[str, Any]) -> List[Dict[str, Any]]:
        """List all users (admin only)."""
        self._assert_admin(admin_user)
        return self.store.list_users()

    def get_user(self, admin_user: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID (admin only)."""
        self._assert_admin(admin_user)
        return self.store.get_user_by_id(user_id)

    def set_enabled(
        self,
        admin_user: Dict[str, Any],
        user_id: str,
        enabled: bool,
    ) -> bool:
        """Enable or disable a user (admin only).

        Raises
        ------
        UserNotFoundError
            If *user_id* doesn't exist.
        """
        self._assert_admin(admin_user)
        user = self.store.get_user_by_id(user_id)
        if user is None:
            raise UserNotFoundError("User not found")
        return self.store.set_user_enabled(user_id, enabled)

    def reset_password(
        self,
        admin_user: Dict[str, Any],
        user_id: str,
        new_password: str,
    ) -> bool:
        """Reset a user's password (admin only).

        Raises
        ------
        UserNotFoundError
            If *user_id* doesn't exist.
        """
        self._assert_admin(admin_user)
        user = self.store.get_user_by_id(user_id)
        if user is None:
            raise UserNotFoundError("User not found")
        pw_hash = hash_password(new_password)
        return self.store.set_user_password(user_id, pw_hash)

    def change_password(
        self,
        user: Dict[str, Any],
        current_password: str,
        new_password: str,
    ) -> bool:
        """Let a logged-in user change their own password (self-service).

        Unlike :meth:`reset_password`, this requires no admin privileges but
        does require proving knowledge of the current password.

        Raises
        ------
        InvalidCredentialsError
            If *current_password* doesn't match the user's stored hash.
        """
        if not verify_password(current_password, user["password_hash"]):
            raise InvalidCredentialsError("Current password is incorrect")
        pw_hash = hash_password(new_password)
        return self.store.set_user_password(user["_id"], pw_hash)

    def delete_user(
        self,
        admin_user: Dict[str, Any],
        user_id: str,
    ) -> bool:
        """Delete a user (admin only). Cannot delete yourself.

        Raises
        ------
        UserNotFoundError
            If *user_id* doesn't exist.
        ValueError
            If attempting to delete yourself.
        """
        self._assert_admin(admin_user)
        if admin_user.get("_id") == user_id:
            raise ValueError("Cannot delete your own account")
        user = self.store.get_user_by_id(user_id)
        if user is None:
            raise UserNotFoundError("User not found")
        return self.store.delete_user(user_id)


# ---------------------------------------------------------------------------
# Bootstrap: create the first admin user (for the CLI)
# ---------------------------------------------------------------------------


def create_admin(store: Any, username: str, password: str) -> Dict[str, Any]:
    """Create the first admin user directly on the store (bypassing AuthService
    since there's no admin yet).

    Raises
    ------
    UserExistsError
        If *username* already exists.
    """
    if store.get_user_by_username(username) is not None:
        raise UserExistsError(f"Username '{username}' already exists")
    pw_hash = hash_password(password)
    return store.create_user(username, pw_hash, is_admin=True)
