"""One question, from Discord message to delivered answer.

:func:`answer_question` owns the logic — which thread to ask in, following
the stream, collecting artifacts, remembering the answer's messages — and
drives the chat platform only through the small :class:`Surface` interface.
The Discord implementation lives in :mod:`.bot`; the tests use a fake one
against a real Pengyplexity app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .apiclient import ApiError, ArtifactTooLarge, PengyplexityClient
from .conversations import ConversationMap, message_key
from .render import (
    describe_api_error,
    megabytes,
    penguin_activity,
    progress_text,
    render_answer,
)

log = logging.getLogger("pengyplexity.discord")

# Discord allows at most 10 attachments per message.
MAX_UPLOADS = 10


@dataclass
class Upload:
    filename: str
    data: bytes


class Surface(Protocol):
    """Where a question's progress and answer are shown."""

    async def progress(self, text: str) -> None:
        """Replace the live progress text. Called often; throttling is the surface's job."""

    async def rename(self, title: str) -> None:
        """The Pengyplexity thread was just named from this question."""

    async def deliver(self, text: str, uploads: List[Upload]) -> List[int]:
        """Show the final answer; return the ids of the messages it used."""


@dataclass
class Outcome:
    thread_id: Optional[str]
    message_ids: List[int] = field(default_factory=list)
    stopped: bool = False
    error: Optional[str] = None


async def answer_question(
    api: PengyplexityClient,
    conversations: ConversationMap,
    key: str,
    question: str,
    surface: Surface,
    *,
    max_upload_bytes: int,
    title: Optional[str] = None,
) -> Outcome:
    """Ask *question* in the thread mapped to *key* (creating one if needed)
    and deliver the answer. API refusals are delivered as a readable message
    rather than raised.

    *title* names a newly created thread. A caller passes it when the question
    text would make a poor title — a channel conversation prefixes the room's
    recent messages, and the server names an unnamed thread from the start of
    its first question.
    """
    try:
        try:
            return await _ask(api, conversations, key, question, surface, max_upload_bytes, title)
        except ApiError as e:
            if e.code != "thread_not_found":
                raise
            # Someone deleted the thread in the web UI. Start the conversation
            # over rather than failing every message from now on.
            log.info("Thread for %s is gone; starting a new one.", key)
            conversations.forget(key)
            return await _ask(api, conversations, key, question, surface, max_upload_bytes, title)
    except ApiError as e:
        log.warning("API refused a question for %s: %s", key, e)
        ids = await surface.deliver(describe_api_error(e), [])
        return Outcome(None, ids, error=e.code)


async def _thread_for(
    api: PengyplexityClient,
    conversations: ConversationMap,
    key: str,
    title: Optional[str] = None,
) -> str:
    thread_id = conversations.thread_for(key)
    if thread_id is None:
        thread_id = (await api.create_thread(title))["id"]
        conversations.link(key, thread_id)
    return thread_id


async def _ask(api, conversations, key, question, surface, max_upload_bytes, title=None) -> Outcome:
    thread_id = await _thread_for(api, conversations, key, title)
    # Counted before the answer, not after: a turn the model failed halfway
    # through still left its question in the thread, and still has to be
    # replayed on the next one.
    conversations.record_turn(key)

    label, partial = penguin_activity(), ""
    streamed_artifacts: List[Dict[str, Any]] = []
    final: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    async for event, data in api.ask(thread_id, question):
        if event == "title":
            await surface.rename(str(data.get("title") or ""))
        elif event == "activity":
            # The server's precise label ("Searching the web…") is dropped on
            # purpose; see render.penguin_activity.
            label = penguin_activity(exclude=label)
            await surface.progress(progress_text(label, partial))
        elif event == "token":
            partial += str(data.get("content") or "")
            await surface.progress(progress_text(label, partial))
        elif event == "artifact":
            streamed_artifacts.append(data)
        elif event == "error":
            error = str(data.get("message") or "The agent failed.")
        elif event == "message":
            final = data

    # The final `message` event is what was saved. It is missing only if the
    # thread was deleted mid-turn; fall back to what streamed.
    message = final.get("message") if final else None
    if final is None and partial:
        message = {"content": partial, "sources": []}
    artifacts = (message or {}).get("artifacts") or streamed_artifacts

    uploads, notes = await _collect_uploads(api, artifacts, max_upload_bytes)
    ids = await surface.deliver(render_answer(message, error=error, notes=notes), uploads)
    for message_id in ids:
        conversations.link(message_key(message_id), thread_id)
    return Outcome(thread_id, ids, stopped=bool(final and final.get("stopped")), error=error)


async def _collect_uploads(
    api: PengyplexityClient, artifacts: List[Dict[str, Any]], max_bytes: int
) -> Tuple[List[Upload], List[str]]:
    """Download the turn's artifacts for re-upload — Discord cannot fetch
    them itself, since they sit behind the API key."""
    uploads: List[Upload] = []
    notes: List[str] = []
    seen = set()
    for artifact in artifacts:
        url = artifact.get("url")
        name = str(artifact.get("filename") or "artifact")
        if not url or url in seen:
            continue
        seen.add(url)
        if len(uploads) >= MAX_UPLOADS:
            notes.append(f"-# {name} not attached (Discord allows {MAX_UPLOADS} files per message).")
            continue
        try:
            uploads.append(Upload(name, await api.fetch_artifact(url, max_bytes)))
        except ArtifactTooLarge as e:
            notes.append(f"-# {name} is too large to attach here ({megabytes(e.size)}+).")
        except ApiError as e:
            log.warning("Could not fetch artifact %s: %s", url, e)
            notes.append(f"-# Couldn't fetch {name}.")
    return uploads, notes
