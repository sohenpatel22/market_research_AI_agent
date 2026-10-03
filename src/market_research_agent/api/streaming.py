"""Server-Sent Events plumbing"""

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator

_DONE = object()


def sse(event: str, data: dict) -> str:
    """Format one SSE message"""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_events(
    make_iterator: Callable[[], Iterator[dict]],
    to_message: Callable[[dict], str],
) -> AsyncIterator[str]:
    """Consume `make_iterator()` entirely inside a single worker thread (tracing contexts are"""
    events: queue.Queue = queue.Queue()
    stop = threading.Event()

    def worker() -> None:
        try:
            for item in make_iterator():
                if stop.is_set():  # client went away; stop between steps
                    break
                events.put(item)
        except Exception as exc:  # noqa: BLE001
            events.put(exc)
        finally:
            events.put(_DONE)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    loop = asyncio.get_running_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, events.get)
            if item is _DONE:
                return
            if isinstance(item, Exception):
                yield sse("error", {"detail": _public_error(item)})
                continue
            yield to_message(item)
    finally:
        stop.set()


def _public_error(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "MissingAPIKeyError":
        return "The language model is not configured on this server."
    return "The request failed. Please try again."


def iter_in_thread(make_iterator: Callable[[], Iterator[dict]]) -> Iterator[dict]:
    """Synchronous counterpart of `stream_events` for callers that iterate on a thread pool (e.g"""
    events: queue.Queue = queue.Queue()
    stop = threading.Event()

    def worker() -> None:
        try:
            for item in make_iterator():
                if stop.is_set():
                    break
                events.put(item)
        except Exception as exc:  # noqa: BLE001
            events.put(exc)
        finally:
            events.put(_DONE)

    threading.Thread(target=worker, daemon=True).start()
    try:
        while True:
            item = events.get()
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
