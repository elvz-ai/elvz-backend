"""Per-request event bus for streaming events from inside graph nodes."""

import asyncio
from typing import Any

from pydantic import BaseModel


class StreamingEvent(BaseModel):
    """Event pushed by nodes during execution."""

    event: str  # "step", "token", "artifact", "error"
    data: dict[str, Any]


class EventBus:
    """Async queue wrapper for streaming events from graph nodes to SSE/WS."""

    def __init__(self):
        self._queue: asyncio.Queue[StreamingEvent | None] = asyncio.Queue()

    def push(self, event: str, **data) -> None:
        """Push an event (non-blocking). Called from inside graph nodes."""
        self._queue.put_nowait(StreamingEvent(event=event, data=data))

    def push_token(self, text: str, node: str = "") -> None:
        """Convenience: push a token chunk."""
        self._queue.put_nowait(
            StreamingEvent(event="token", data={"text": text, "node": node})
        )

    def push_step(self, node: str, status: str, **extra) -> None:
        """Convenience: push a step event."""
        self._queue.put_nowait(
            StreamingEvent(
                event="step", data={"node": node, "status": status, **extra}
            )
        )

    def done(self) -> None:
        """Signal that no more events will be pushed (sentinel)."""
        self._queue.put_nowait(None)

    async def __aiter__(self):
        """Iterate over events until sentinel."""
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
