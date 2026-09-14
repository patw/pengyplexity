"""CLI entry point for Pengyplexity admin bootstrap.

Usage::

    python -m pengyplexity.cli create-admin <username> [--password <pw>]

Creates the first admin user in the store. If ``--password`` is not given,
the password is read from the ``PENGYPLEXITY_ADMIN_PASSWORD`` environment
variable; if that is also missing, a prompt is shown (not available in
headless mode — an error is raised instead).

This is the *only* way to create an admin (no self-sign-up).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from .config import load_config
from .core.auth import UserExistsError, create_admin
from .core.store import Store


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns exit code (0=success, 1=error)."""
    parser = argparse.ArgumentParser(
        prog="pengyplexity-admin",
        description="Pengyplexity admin management CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-admin subcommand
    ca = subparsers.add_parser(
        "create-admin",
        help="Create the first admin user",
    )
    ca.add_argument("username", help="Admin username")
    ca.add_argument(
        "--password",
        default=None,
        help="Admin password (or set PENGYPLEXITY_ADMIN_PASSWORD env var)",
    )
    ca.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: ~/.pengyplexity)",
    )

    # build-sandbox subcommand
    bs = subparsers.add_parser(
        "build-sandbox",
        help="Build the Python environment the code sandbox runs scripts in",
    )
    bs.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the environment is already present",
    )
    bs.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: ~/.pengyplexity)",
    )

    args = parser.parse_args(argv)

    if args.command in {"create_admin", "create-admin"}:
        return _create_admin(args)
    if args.command in {"build_sandbox", "build-sandbox"}:
        return _build_sandbox(args)

    return 1


def _build_sandbox(args: argparse.Namespace) -> int:
    """Handle the build-sandbox subcommand.

    The bwrap sandbox mounts only a read-only ``/usr``, so the interpreter
    reachable inside it is the host's bare system Python — usually without
    matplotlib, numpy or pandas, which makes every chart script fail. This
    builds the separate venv that gets bind-mounted in read-only. The app also
    builds it on first use, but doing it here keeps the first chart fast and
    surfaces a broken host (no ``uv``, no network) at deploy time.
    """
    from .sandbox.pythonenv import SandboxEnvError, SandboxPythonEnv

    config = load_config()
    if args.data_dir:
        config.data_dir = pathlib.Path(args.data_dir)
        config.sandbox_venv = None

    env = SandboxPythonEnv(config.sandbox_venv_path)
    if env.ready and not args.force:
        print(f"Sandbox environment already built at {env.path} (use --force to rebuild).")
        return 0
    print(f"Building sandbox environment at {env.path} ({', '.join(env.packages)})...")
    try:
        env.build(force=args.force)
    except SandboxEnvError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


def _create_admin(args: argparse.Namespace) -> int:
    """Handle the create-admin subcommand."""
    # Resolve data dir
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get(
            "PENGYPLEXITY_DATA_DIR",
            os.path.join(os.path.expanduser("~"), ".pengyplexity"),
        )

    # Resolve password
    password = args.password or os.environ.get("PENGYPLEXITY_ADMIN_PASSWORD")
    if not password:
        print(
            "Error: no password provided. Use --password or set "
            "PENGYPLEXITY_ADMIN_PASSWORD.",
            file=sys.stderr,
        )
        return 1

    # Open the store and create the admin.
    store = Store(data_dir)
    try:
        user = create_admin(store, args.username, password)
        print(f"Admin user '{user['username']}' created (id={user['_id']}).")
        return 0
    except UserExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
