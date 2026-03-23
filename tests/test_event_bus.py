"""Tests for async EventBus."""
import pytest
import asyncio
from nemori.services.event_bus import EventBus


@pytest.mark.asyncio
async def test_emit_calls_handler():
    bus = EventBus()
    called_with = {}

    async def handler(**kwargs):
        called_with.update(kwargs)

    bus.on("test_event", handler)
    await bus.emit("test_event", data="hello")
    await asyncio.sleep(0.05)
    assert called_with.get("data") == "hello"


@pytest.mark.asyncio
async def test_multiple_handlers():
    bus = EventBus()
    results = []

    async def h1(**kw): results.append("h1")
    async def h2(**kw): results.append("h2")

    bus.on("ev", h1)
    bus.on("ev", h2)
    await bus.emit("ev")
    await asyncio.sleep(0.05)
    assert "h1" in results and "h2" in results


@pytest.mark.asyncio
async def test_no_handler_no_error():
    bus = EventBus()
    await bus.emit("unknown_event")
