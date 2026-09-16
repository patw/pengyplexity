"""Async client for the Pengyplexity JSON API, as the Discord bot uses it.

Only the calls the bot needs: who am I, create a thread, ask a question
(streamed), stop a turn, and download an artifact. Every failure — an HTTP
error, an unreachable server — surfaces as :class:`ApiError` carrying the
API's stable ``code``, so callers switch on codes rather than on exception
types. The API itself is documented in API.md.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

import aiohttp

API_PATH = "/api/v1"
USER_AGENT = "pengyplexity-discord"


def api_root(url: str) -> str:
    """``https://host[/prefix]`` or the same with ``/api/v1`` → the API root."""
    url = url.strip().rstrip("/")
    return url if url.endswith(API_PATH) else url + API_PATH


class ApiError(Exception):
    """A refused or failed API call. ``status`` is 0 when no HTTP response came back."""

    def __init__(
        self, status: int, code: str, message: str, retry_after: Optional[int] = None
    ) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


class ArtifactTooLarge(Exception):
    """An artifact is bigger than the caller is willing to download."""

    def __init__(self, size: int) -> None:
        super().__init__(f"artifact is at least {size} bytes")
        self.size = size


class SSEParser:
    """Incremental Server-Sent Events parser: feed it lines (without the
    newline), get ``(event, data)`` back when an event is complete."""

    def __init__(self) -> None:
        self._event: Optional[str] = None
        self._data: List[str] = []

    def feed(self, line: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        line = line.rstrip("\r")
        if line == "":
            if self._event is None and not self._data:
                return None
            event, raw = self._event or "message", "\n".join(self._data)
            self._event, self._data = None, []
            try:
                data = json.loads(raw) if raw else {}
            except ValueError:
                data = {"raw": raw}
            return event, data if isinstance(data, dict) else {"value": data}
        if line.startswith(":"):  # comment / keep-alive
            return None
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "event":
            self._event = value
        elif name == "data":
            self._data.append(value)
        return None


async def _read_json(resp: aiohttp.ClientResponse) -> Any:
    try:
        return await resp.json(content_type=None)
    except (ValueError, aiohttp.ClientError):
        return None


class PengyplexityClient:
    """One API key's view of one Pengyplexity server."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        request_timeout: float = 60.0,
        connect_timeout: float = 10.0,
        stream_idle_timeout: float = 900.0,
    ) -> None:
        self.api_root = api_root(base_url)
        self.server_root = self.api_root[: -len(API_PATH)]
        self._headers = {"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT}
        self._timeout = aiohttp.ClientTimeout(total=request_timeout, sock_connect=connect_timeout)
        # A turn can run for minutes, with long silences while a model call or
        # a sandboxed script runs, so a streamed answer only has an idle cap —
        # generous enough to outlast the server's own LLM timeout.
        self._stream_timeout = aiohttp.ClientTimeout(
            total=None, sock_connect=connect_timeout, sock_read=stream_idle_timeout
        )
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "PengyplexityClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _http(self) -> aiohttp.ClientSession:
        # Created lazily: a ClientSession must be made inside the running loop.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers)
        return self._session

    def _unreachable(self, error: BaseException) -> ApiError:
        detail = str(error) or type(error).__name__
        return ApiError(0, "unreachable", f"Could not reach {self.server_root}: {detail}")

    @staticmethod
    def _error(status: int, body: Any, headers) -> ApiError:
        err = body.get("error") if isinstance(body, dict) else None
        err = err if isinstance(err, dict) else {}
        retry = (headers.get("Retry-After") or "").strip()
        return ApiError(
            status,
            err.get("code") or f"http_{status}",
            err.get("message") or f"HTTP {status}",
            int(retry) if retry.isdigit() else None,
        )

    async def _request(self, method: str, path: str, payload: Any = None) -> Any:
        try:
            async with self._http().request(
                method, self.api_root + path, json=payload, timeout=self._timeout
            ) as resp:
                body = await _read_json(resp)
                if resp.status >= 400:
                    raise self._error(resp.status, body, resp.headers)
                return body
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise self._unreachable(e) from e

    # -- Calls ----------------------------------------------------------------

    async def me(self) -> Dict[str, Any]:
        return await self._request("GET", "/me")

    async def create_thread(self, title: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/threads", {"title": title} if title else {})

    async def stop(self, thread_id: str) -> bool:
        body = await self._request("POST", f"/threads/{thread_id}/stop")
        return bool((body or {}).get("stopped"))

    async def ask(
        self, thread_id: str, content: str
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Ask a question and yield the stream's ``(event, data)`` pairs, ending
        with the ``message`` event that carries the answer as saved.

        A refusal (409, 429, ...) raises :class:`ApiError` before anything is
        yielded; a failure during the turn arrives as an ``error`` event.
        """
        url = f"{self.api_root}/threads/{thread_id}/messages"
        try:
            async with self._http().post(
                url,
                json={"content": content, "stream": True},
                headers={"Accept": "text/event-stream"},
                timeout=self._stream_timeout,
            ) as resp:
                if resp.status != 200:
                    raise self._error(resp.status, await _read_json(resp), resp.headers)
                parser = SSEParser()
                buffer = bytearray()
                # Split on raw bytes, not with readline(): the final `message`
                # event holds the whole answer on one line and can outgrow
                # aiohttp's line limit. Splitting before decoding also keeps a
                # multi-byte character that straddles two chunks intact.
                async for chunk in resp.content.iter_any():
                    buffer += chunk
                    while (newline := buffer.find(b"\n")) >= 0:
                        line = bytes(buffer[:newline]).decode("utf-8", "replace")
                        del buffer[: newline + 1]
                        if (event := parser.feed(line)) is not None:
                            yield event
                for line in (buffer.decode("utf-8", "replace"), ""):
                    if (event := parser.feed(line)) is not None:
                        yield event
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise self._unreachable(e) from e

    def artifact_url(self, url: str) -> str:
        """Resolve an artifact link from the API against the server, refusing
        any other origin — the download carries the API key."""
        resolved = urljoin(self.server_root + "/", url)
        ours, theirs = urlsplit(self.server_root), urlsplit(resolved)
        if (theirs.scheme, theirs.netloc) != (ours.scheme, ours.netloc):
            raise ApiError(0, "foreign_url", f"Refusing to send the API key to {resolved}")
        return resolved

    async def fetch_artifact(self, url: str, max_bytes: int) -> bytes:
        """Download an artifact, raising :class:`ArtifactTooLarge` past *max_bytes*."""
        target = self.artifact_url(url)
        try:
            async with self._http().get(target, timeout=self._timeout) as resp:
                if resp.status != 200:
                    raise self._error(resp.status, await _read_json(resp), resp.headers)
                if resp.content_length is not None and resp.content_length > max_bytes:
                    raise ArtifactTooLarge(resp.content_length)
                data = bytearray()
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    data += chunk
                    if len(data) > max_bytes:
                        raise ArtifactTooLarge(len(data))
                return bytes(data)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise self._unreachable(e) from e
