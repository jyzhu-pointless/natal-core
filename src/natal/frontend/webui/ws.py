"""WebSocket endpoint: command parsing, subscriptions, frame pump.

One connection = one subscriber with a bounded outbound queue.  Commands are
validated through a Pydantic discriminated union; quick control commands run
inline on the event loop, engine-touching commands run in worker threads so
a long tick never blocks other connections.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from .protocol import (
    Command,
    ErrorFrame,
    HelloFrame,
    PauseCommand,
    PingCommand,
    PlayCommand,
    PongFrame,
    ResetCommand,
    RestoreCommand,
    RunToTickCommand,
    SetIntervalCommand,
    SetRecordEveryCommand,
    StepCommand,
)
from .session import SimulationSession

_command_adapter: TypeAdapter[Command] = TypeAdapter(Command)

#: Bounded per-client queue; overflow drops frames (slow client, not engine).
_OUTBOX_CAPACITY = 512


def session_from_app(app: FastAPI) -> SimulationSession:
    """Return the session stored on *app* at factory time."""
    session = app.state.session
    if not isinstance(session, SimulationSession):
        raise RuntimeError("SimulationSession missing from application state")
    return session


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle one WebSocket connection for the lifetime of the socket."""
    session = session_from_app(websocket.app)
    await websocket.accept()

    hello = HelloFrame(
        type="hello",
        tick=session.tick,
        status=session.status,
        interval_ms=session.interval_ms,
        backend=session.backend,
        dashboard_type=session.dashboard_type,
    )
    await websocket.send_json(hello)
    for frame in session.log_replay():
        await websocket.send_json(frame)

    outbox: asyncio.Queue[object] = asyncio.Queue(  # object: outbound frames are heterogeneous TypedDict payloads
        maxsize=_OUTBOX_CAPACITY
    )
    subscriber_id = session.subscribe(outbox)
    sender = asyncio.create_task(_pump(websocket, outbox))
    try:
        while True:
            raw = await websocket.receive_json()
            await _handle_message(websocket, session, raw)
    except WebSocketDisconnect:
        pass
    finally:
        session.unsubscribe(subscriber_id)
        sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass


async def _pump(
    websocket: WebSocket,
    outbox: asyncio.Queue[object],  # object: outbound frames are heterogeneous TypedDict payloads
) -> None:
    """Drain the outbound queue into the socket until cancelled."""
    while True:
        frame = await outbox.get()
        await websocket.send_json(frame)


async def _handle_message(
    websocket: WebSocket,
    session: SimulationSession,
    raw: object,  # object: arbitrary JSON frame from the browser socket
) -> None:
    """Validate one incoming frame and dispatch its command.

    Engine-touching commands run in worker threads; any exception they raise
    (user hooks are arbitrary Python) is reported back as an error frame so
    a bad command never kills the connection.
    """
    try:
        command = _command_adapter.validate_python(raw)
    except ValidationError as exc:
        await websocket.send_json(ErrorFrame(type="error", message=str(exc)))
        return

    try:
        if isinstance(command, PingCommand):
            await websocket.send_json(
                PongFrame(type="pong", nonce=command.nonce)
            )
        elif isinstance(command, PlayCommand):
            session.play()
        elif isinstance(command, PauseCommand):
            session.pause()
        elif isinstance(command, RunToTickCommand):
            session.run_to_tick(command.tick)
        elif isinstance(command, StepCommand):
            await asyncio.to_thread(session.step_blocking, command.n)
        elif isinstance(command, ResetCommand):
            await asyncio.to_thread(session.reset_blocking)
        elif isinstance(command, RestoreCommand):
            await asyncio.to_thread(session.restore_blocking, command.tick)
        elif isinstance(command, SetIntervalCommand):
            session.set_interval_ms(command.value)
        elif isinstance(command, SetRecordEveryCommand):
            await asyncio.to_thread(session.set_record_every, command.value)
        else:
            # The discriminated union is exhausted; only SetMaxHistoryCommand
            # remains at this point.
            await asyncio.to_thread(session.set_max_history, command.value)
    except Exception as exc:  # noqa: BLE001 -- hook/user code is arbitrary
        session.pause()
        session.log_error(f"{type(exc).__name__}: {exc}")
        await websocket.send_json(ErrorFrame(type="error", message=str(exc)))
