"""Flask blueprint for admin user management.

Routes (all admin-gated; non-admin users get 403):

* ``GET /admin`` — list all users.
* ``POST /admin/users`` — create a new user.
* ``POST /admin/users/<user_id>/enable`` — enable a disabled user.
* ``POST /admin/users/<user_id>/disable`` — disable a user.
* ``POST /admin/users/<user_id>/reset-password`` — reset a user's password.
* ``POST /admin/users/<user_id>/delete`` — delete a user.
* ``GET/POST /admin/settings`` — view/edit global settings (system message,
  model connection, agent/tool limits, sandbox execution limits).
* ``POST /admin/settings/reset`` — clear all overrides back to Config/env defaults.

All routes require login first (redirect to /login if unauthenticated), then
check ``user['is_admin']`` (return 403 if not).
"""

from __future__ import annotations

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from .app import get_state
from .core.agent import SYSTEM_PROMPT
from .core.auth import (
    AuthError,
    NotAdminError,
    UserExistsError,
    UserNotFoundError,
)
from .core.settings import (
    SENSITIVE_FIELDS,
    effective_settings,
    raw_overrides,
    save_settings_from_form,
)

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _require_login():
    """Return the current user doc, or None."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    state = get_state(current_app)
    user = state.store.get_user_by_id(user_id)
    if user is None or not user.get("enabled", True):
        session.clear()
        return None
    return user


def _require_admin():
    """Return the current admin user doc, or (response, status)."""
    user = _require_login()
    if user is None:
        return None
    if not user.get("is_admin"):
        return None
    return user


def _guard():
    """Common guard: returns (user, None) on success, (None, (response, status)) on failure."""
    user = _require_login()
    if user is None:
        return None, redirect(url_for("web.login"))
    if not user.get("is_admin"):
        return None, ("403 Forbidden: admin privileges required", 403)
    return user, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@admin_bp.route("")
def list_users():
    """List all users (admin page)."""
    user, err = _guard()
    if err:
        return err

    state = get_state(current_app)
    users = state.auth.list_users(user)
    return render_template(
        "admin_users.html",
        users=users,
        current_user=user,
        error=request.args.get("error", ""),
        success=request.args.get("success", ""),
    )


@admin_bp.route("/users", methods=["POST"])
def create_user():
    """Create a new user."""
    user, err = _guard()
    if err:
        return err

    state = get_state(current_app)
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    is_admin = request.form.get("is_admin") == "on"

    if not username or not password:
        return redirect(url_for("admin.list_users", error="Username and password are required"))

    try:
        state.auth.create_user(user, username, password, is_admin=is_admin)
        return redirect(url_for("admin.list_users", success=f"User '{username}' created"))
    except UserExistsError as e:
        return redirect(url_for("admin.list_users", error=str(e)))


@admin_bp.route("/users/<user_id>/enable", methods=["POST"])
def enable_user(user_id: str):
    """Enable a user."""
    user, err = _guard()
    if err:
        return err

    state = get_state(current_app)
    try:
        state.auth.set_enabled(user, user_id, True)
        return redirect(url_for("admin.list_users", success="User enabled"))
    except UserNotFoundError:
        return redirect(url_for("admin.list_users", error="User not found"))


@admin_bp.route("/users/<user_id>/disable", methods=["POST"])
def disable_user(user_id: str):
    """Disable a user."""
    user, err = _guard()
    if err:
        return err

    state = get_state(current_app)
    try:
        state.auth.set_enabled(user, user_id, False)
        return redirect(url_for("admin.list_users", success="User disabled"))
    except UserNotFoundError:
        return redirect(url_for("admin.list_users", error="User not found"))


@admin_bp.route("/users/<user_id>/reset-password", methods=["POST"])
def reset_password(user_id: str):
    """Reset a user's password."""
    user, err = _guard()
    if err:
        return err

    state = get_state(current_app)
    new_password = request.form.get("new_password", "")

    if not new_password:
        return redirect(url_for("admin.list_users", error="New password is required"))

    try:
        state.auth.reset_password(user, user_id, new_password)
        return redirect(url_for("admin.list_users", success="Password reset"))
    except UserNotFoundError:
        return redirect(url_for("admin.list_users", error="User not found"))


@admin_bp.route("/settings", methods=["GET", "POST"])
def settings():
    """View/edit the global, admin-only settings (system message, model
    connection, agent/tool limits, sandbox execution limits).

    Every field falls back to the env-loaded ``Config`` default when unset —
    see ``core/settings.py``. Saving takes effect on the *next* chat turn
    (``web.py: _apply_effective_settings`` refreshes the shared agent/runner/
    model/tool-context from the store before every turn); no app restart is
    needed.
    """
    user, err = _guard()
    if err:
        return err

    state = get_state(current_app)

    if request.method == "POST":
        save_settings_from_form(state.store, request.form)
        return redirect(url_for("admin.settings", success="Settings saved"))

    settings = effective_settings(state.store, state.config)
    return render_template(
        "admin_settings.html",
        current_user=user,
        settings=settings,
        overrides=raw_overrides(state.store),
        sensitive_fields=SENSITIVE_FIELDS,
        default_system_message=SYSTEM_PROMPT.format(
            max_iterations=settings["max_agent_iterations"]
        ),
        error=request.args.get("error", ""),
        success=request.args.get("success", ""),
    )


@admin_bp.route("/settings/reset", methods=["POST"])
def reset_settings():
    """Clear every admin override, reverting all settings to Config/env defaults."""
    user, err = _guard()
    if err:
        return err

    state = get_state(current_app)
    state.store.reset_settings()
    return redirect(url_for("admin.settings", success="Settings reset to defaults"))


@admin_bp.route("/users/<user_id>/delete", methods=["POST"])
def delete_user(user_id: str):
    """Delete a user."""
    user, err = _guard()
    if err:
        return err

    state = get_state(current_app)
    try:
        state.auth.delete_user(user, user_id)
        return redirect(url_for("admin.list_users", success="User deleted"))
    except UserNotFoundError:
        return redirect(url_for("admin.list_users", error="User not found"))
    except ValueError as e:
        return redirect(url_for("admin.list_users", error=str(e)))
