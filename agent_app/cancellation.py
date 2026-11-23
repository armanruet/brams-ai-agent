"""Cooperative cancellation tokens and interruptible streaming archive downloads."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import threading

import httpx


class OperationCancelled(asyncio.CancelledError):
    """Raised when the user cancels an active LLM, download, or computation task."""


_CANCEL_EVENT: ContextVar[threading.Event | None] = ContextVar("brams_cancel_event", default=None)


def current_cancel_event() -> threading.Event | None:
    return _CANCEL_EVENT.get()


@contextmanager
def cancellation_scope(event: threading.Event | None):
    token = _CANCEL_EVENT.set(event)
    try:
        yield event
    finally:
        _CANCEL_EVENT.reset(token)


def check_cancelled(cancel_event: threading.Event | None = None) -> None:
    event = cancel_event if cancel_event is not None else _CANCEL_EVENT.get()
    if event is not None and event.is_set():
        raise OperationCancelled("Operation cancelled by user.")


def stream_download_bytes(
    url: str,
    *,
    cancel_event: threading.Event | None = None,
    timeout: tuple[float, float] = (5.0, 15.0),
    chunk_size: int = 65536,
    transport: httpx.BaseTransport | None = None,
) -> bytes:
    """Download bytes in chunks, aborting promptly when cancel_event is set."""
    check_cancelled(cancel_event)
    chunks: list[bytes] = []
    with httpx.Client(
        transport=transport,
        trust_env=False,
        follow_redirects=True,
        timeout=httpx.Timeout(timeout[1], connect=timeout[0]),
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                check_cancelled(cancel_event)
                if chunk:
                    chunks.append(chunk)
    check_cancelled(cancel_event)
    return b"".join(chunks)


def stream_download_to_file(
    url: str,
    destination: Path,
    *,
    cancel_event: threading.Event | None = None,
    timeout: tuple[float, float] = (5.0, 15.0),
    chunk_size: int = 65536,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Stream a remote file to disk, removing partial output if cancelled or failed."""
    check_cancelled(cancel_event)
    try:
        with httpx.Client(
            transport=transport,
            trust_env=False,
            follow_redirects=True,
            timeout=httpx.Timeout(timeout[1], connect=timeout[0]),
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=chunk_size):
                        check_cancelled(cancel_event)
                        if chunk:
                            handle.write(chunk)
        check_cancelled(cancel_event)
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
