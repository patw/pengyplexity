"""``pengyplexity-discord`` — run the Discord bot.

Usage::

    pengyplexity-discord [--env-file PATH] [--check]

Loads ``.env`` (or ``--env-file``) into the environment without overriding
variables that are already set, checks the settings and the API key, then
connects to Discord and stays connected. ``--check`` stops after the API key
check, without touching Discord. See DISCORD.md.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pengyplexity-discord",
        description="Run the Pengyplexity Discord bot",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Settings file to load (default: .env in the current directory, if present)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check the settings and the API key, then exit without connecting to Discord",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        import discord
        from dotenv import load_dotenv
    except ImportError:
        print("Error: the Discord bot needs its extra installed: uv sync --extra discord", file=sys.stderr)
        return 1

    from .apiclient import ApiError, PengyplexityClient
    from .bot import PengyplexityBot, verify_api
    from .config import ConfigError, load_bot_config
    from .render import describe_api_error

    env_file = Path(args.env_file) if args.env_file else Path(".env")
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    elif args.env_file:
        print(f"Error: {env_file} does not exist.", file=sys.stderr)
        return 1

    try:
        config = load_bot_config()
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    def api_failure(e: ApiError) -> int:
        print(f"Error: {describe_api_error(e)} ({e})", file=sys.stderr)
        return 1

    if args.check:
        async def check() -> None:
            async with PengyplexityClient(config.api_url, config.api_key) as api:
                await verify_api(api)

        try:
            asyncio.run(check())
        except ApiError as e:
            return api_failure(e)
        print("OK: the API key works. Start the bot without --check.")
        return 0

    bot = PengyplexityBot(config)
    try:
        bot.run(config.discord_token, log_handler=None)
    except ApiError as e:
        return api_failure(e)
    except discord.LoginFailure:
        print("Error: Discord rejected DISCORD_BOT_TOKEN. Reset the token in the "
              "Developer Portal (Bot → Reset Token) and update .env.", file=sys.stderr)
        return 1
    except discord.PrivilegedIntentsRequired:
        print("Error: turn on the Message Content Intent for this bot in the Discord "
              "Developer Portal (Bot → Privileged Gateway Intents).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
