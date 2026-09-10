"""WebSocket protocol definitions for the NATAL Vue web UI.

Client -> server frames are Pydantic models (validated on receipt); server ->
client frames are plain JSON dicts shaped by the TypedDicts below.  Keeping
outbound frames as TypedDicts lets ``json`` serialize them directly while
pyright checks every construction site.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Client -> server commands
# ---------------------------------------------------------------------------


class PingCommand(BaseModel):
    """Liveness probe; answered with a pong frame."""

    type: Literal["ping"]
    nonce: str


class PlayCommand(BaseModel):
    """Start (or resume) the simulation loop."""

    type: Literal["play"]


class PauseCommand(BaseModel):
    """Pause the simulation loop (current tick is kept)."""

    type: Literal["pause"]


class StepCommand(BaseModel):
    """Advance exactly *n* ticks (works while paused or running)."""

    type: Literal["step"]
    n: int = Field(default=1, ge=1, le=10_000)


class RunToTickCommand(BaseModel):
    """Run until the population reaches *tick*, then pause (breakpoint)."""

    type: Literal["run_to_tick"]
    tick: int = Field(ge=0)


class SetIntervalCommand(BaseModel):
    """Set the delay between ticks in milliseconds (0 = turbo)."""

    type: Literal["set_interval_ms"]
    value: int = Field(ge=0, le=60_000)


class SetRecordEveryCommand(BaseModel):
    """Set how often snapshots are recorded into history."""

    type: Literal["set_record_every"]
    value: int = Field(ge=1)


class SetMaxHistoryCommand(BaseModel):
    """Set the rolling history window size."""

    type: Literal["set_max_history"]
    value: int = Field(ge=10)


class ResetCommand(BaseModel):
    """Reset the population to its initial state and clear history."""

    type: Literal["reset"]


class RestoreCommand(BaseModel):
    """Time travel: restore the state recorded at *tick*.

    Restores state from the raw history and truncates all records after it,
    so the simulation can be re-run from that point (what-if branching).
    """

    type: Literal["restore"]
    tick: int = Field(ge=0)


Command = Annotated[
    Union[
        PingCommand,
        PlayCommand,
        PauseCommand,
        StepCommand,
        RunToTickCommand,
        SetIntervalCommand,
        SetRecordEveryCommand,
        SetMaxHistoryCommand,
        ResetCommand,
        RestoreCommand,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Server -> client frames
# ---------------------------------------------------------------------------

SimulationStatus = Literal["ready", "running", "finished", "error"]
LogLevel = Literal["debug", "info", "warning", "error"]


class HelloFrame(TypedDict):
    """Sent once per WebSocket connection after accept."""

    type: Literal["hello"]
    tick: int
    status: SimulationStatus
    interval_ms: int
    backend: str
    dashboard_type: str


class PongFrame(TypedDict):
    type: Literal["pong"]
    nonce: str


class StatusFrame(TypedDict):
    type: Literal["status"]
    status: SimulationStatus
    error: str | None


class TickUpdateFrame(TypedDict):
    """Pushed after every processed tick (or batch in turbo mode)."""

    type: Literal["tick_update"]
    tick: int
    total: float
    female: float
    male: float
    is_finished: bool
    history_len: int


class LogFrame(TypedDict):
    type: Literal["log"]
    ts: float
    level: LogLevel
    source: str
    message: str
    data: str | None


class ErrorFrame(TypedDict):
    type: Literal["error"]
    message: str


class ResetDoneFrame(TypedDict):
    type: Literal["reset_done"]


class RestoredFrame(TypedDict):
    """Acknowledges a successful time-travel restore."""

    type: Literal["restored"]
    tick: int
