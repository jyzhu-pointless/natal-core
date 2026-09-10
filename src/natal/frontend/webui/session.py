"""Simulation session: server-side owner of the population object.

The session is the single authority over the simulation lifecycle.  Unlike
the legacy NiceGUI dashboards (where the tick loop lived inside each browser
page's ``ui.timer``), the run loop lives here so the simulation keeps
running when no browser is connected and every connected tab observes the
same run.

Concurrency model
-----------------

- A daemon *engine thread* executes ticks while ``_running`` is set.  It
  holds ``_engine_lock`` during each batch so engine access is serialized.
- Direct commands (step / reset / restore / setters) run on worker threads
  (``asyncio.to_thread`` from the WebSocket handler) and acquire the same
  lock, so they can never interleave with the batch loop.
- Control flags (play / pause / run-to-tick) are plain flips; the loop
  reacts on its next wake-up.
- Outbound frames fan out to per-connection ``asyncio.Queue``s through
  ``loop.call_soon_threadsafe``; slow clients drop frames instead of
  stalling the engine.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from collections import deque
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.spatial.population import SpatialPopulation

from .protocol import (
    LogFrame,
    LogLevel,
    ResetDoneFrame,
    RestoredFrame,
    SimulationStatus,
    StatusFrame,
    TickUpdateFrame,
)
from .types import DashboardPopulation, uses_rust_backend

DashboardType = Literal["population", "spatial"]
BackendKind = Literal["rust", "python"]

_MAX_LOG_FRAMES = 1000
_TURBO_WINDOW_S = 0.1
_TURBO_MAX_TICKS = 50
_LOOP_IDLE_S = 0.05


class _Subscriber:
    """One connected WebSocket client's outbound pipe."""

    __slots__ = ("loop", "outbox")

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        outbox: asyncio.Queue[object],  # object: outbound frames are heterogeneous TypedDict payloads
    ) -> None:
        self.loop = loop
        self.outbox = outbox


class SimulationSession:
    """Holds the dashboard's population and drives its lifecycle."""

    def __init__(self, population: DashboardPopulation) -> None:
        """Bind the session to a built population.

        Args:
            population: A built (post-``build()``) panmictic or spatial
                population.  The session never rebuilds it.
        """
        self._population = population
        self._status: SimulationStatus = "ready"
        self._error: str | None = None
        self._interval_ms = 50
        self._running = False
        self._run_to_target: int | None = None
        self._record_every = 1
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._engine_lock = threading.Lock()
        self._subscribers: dict[int, _Subscriber] = {}
        self._subscriber_lock = threading.Lock()
        self._next_subscriber_id = 0
        self._logs: deque[LogFrame] = deque(maxlen=_MAX_LOG_FRAMES)
        self._log("info", "session", "session created")

    # -- introspection ----------------------------------------------------

    @property
    def population(self) -> DashboardPopulation:
        """Return the bound population object."""
        return self._population

    @property
    def dashboard_type(self) -> DashboardType:
        """Return which dashboard layout this population requires."""
        if isinstance(self._population, SpatialPopulation):
            return "spatial"
        return "population"

    @property
    def backend(self) -> BackendKind:
        """Return which engine backend currently drives the lifecycle."""
        if uses_rust_backend(self._population):
            return "rust"
        return "python"

    @property
    def tick(self) -> int:
        """Return the population's current tick."""
        return self._population.tick

    @property
    def status(self) -> SimulationStatus:
        """Return the current lifecycle status."""
        return self._status

    @property
    def interval_ms(self) -> int:
        """Return the current per-tick delay in milliseconds."""
        return self._interval_ms

    @property
    def error(self) -> str | None:
        """Return the last engine error message, if any."""
        return self._error

    # -- population narrowing helpers -------------------------------------
    #
    # SpatialPopulation does not share the panmictic base class, so every
    # engine-loop access goes through these guards.  Spatial-specific
    # behaviour (finish conditions, record_every) lands in Phase 3.

    def _finished(self) -> bool:
        """Whether the population reached a stop condition."""
        population = self._population
        if isinstance(population, SpatialPopulation):
            # Phase 3: spatial stop hooks are not surfaced yet.
            return False
        return population.is_finished

    def _counts_summary(self) -> tuple[float, float, float]:
        """Return ``(total, female, male)`` for the current state."""
        population = self._population
        if isinstance(population, SpatialPopulation):
            # aggregate tensor is (sex, age, ztype) after deme summation
            counts: NDArray[np.float64] = population.aggregate_individual_count()
            per_sex = counts.sum(axis=(1, 2))
            return float(per_sex.sum()), float(per_sex[0]), float(per_sex[1])
        counts = population.state.individual_count  # (sex, age, ztype)
        per_sex = counts.sum(axis=(1, 2))
        return float(per_sex.sum()), float(per_sex[0]), float(per_sex[1])

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Start the engine thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopped = False
        self._thread = threading.Thread(
            target=self._engine_loop, name="natal-webui-engine", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the engine thread; the population object is left untouched."""
        self._stopped = True
        self._running = False
        self._wake.set()

    # -- engine thread ----------------------------------------------------

    def _engine_loop(self) -> None:
        while not self._stopped:
            self._wake.wait(timeout=_LOOP_IDLE_S)
            self._wake.clear()
            if self._stopped:
                break
            if not self._running:
                continue
            try:
                self._run_batch()
            except Exception as exc:  # noqa: BLE001 -- engine surface is wide
                self._engine_failed(exc)
                continue
            if self._stopped:
                break
            target = self._run_to_target
            if target is not None and self._population.tick >= target:
                self._run_to_target = None
                self._running = False
                self._set_status("ready")
                self._log(
                    "info",
                    "engine",
                    f"breakpoint reached at tick {self._population.tick}",
                )
            elif self._finished():
                self._running = False
                self._set_status("finished")
            else:
                interval = self._interval_ms / 1000.0
                if interval > 0:
                    time.sleep(interval)

    def _tick_once(self) -> None:
        """Advance one tick with history recording, per population kind.

        ``SpatialPopulation.run_tick()`` performs no history recording at
        all (only ``run(n_steps, record_every)`` records snapshots), so the
        spatial path goes through ``run(1, ...)`` to keep the timeline
        charts populated.  Panmictic ``run_tick`` already records internally.
        """
        population = self._population
        if isinstance(population, SpatialPopulation):
            population.run(1, record_every=self._record_every)
        else:
            population.run_tick()

    def _run_batch(self) -> None:
        """Run one tick (or a turbo batch) under the engine lock."""
        with self._engine_lock:
            if self._interval_ms <= 0:
                start = time.time()
                ticks = 0
                while (
                    time.time() - start < _TURBO_WINDOW_S
                    and ticks < _TURBO_MAX_TICKS
                    and not self._finished()
                ):
                    self._tick_once()
                    ticks += 1
                    target = self._run_to_target
                    if target is not None and self._population.tick >= target:
                        break
            elif not self._finished():
                self._tick_once()
            self._broadcast_tick_update()

    def _engine_failed(self, exc: Exception) -> None:
        self._running = False
        self._error = str(exc)
        self._log("error", "engine", f"{type(exc).__name__}: {exc}", traceback.format_exc())
        self._set_status("error")

    # -- control (called from event-loop / worker threads) ----------------

    def play(self) -> None:
        """Resume (or start) the tick loop."""
        self._run_to_target = None
        self._running = True
        self._set_status("running")
        self._wake.set()

    def pause(self) -> None:
        """Pause the tick loop at the current tick."""
        self._running = False
        self._run_to_target = None
        self._set_status("ready")

    def run_to_tick(self, tick: int) -> None:
        """Run until *tick* is reached, then pause (breakpoint)."""
        if self._finished():
            self._log(
                "warning",
                "engine",
                "run_to_tick ignored: population already finished",
            )
            return
        if tick <= self._population.tick:
            self._log(
                "warning",
                "engine",
                f"run_to_tick ignored: current tick {self._population.tick} "
                f"already >= {tick}",
            )
            return
        self._run_to_target = tick
        self._running = True
        self._set_status("running")
        self._log("info", "engine", f"breakpoint set at tick {tick}")
        self._wake.set()

    def step_blocking(self, n: int) -> None:
        """Advance exactly *n* ticks synchronously (broadcasts each tick)."""
        with self._engine_lock:
            for _ in range(n):
                if self._finished():
                    break
                self._tick_once()
                self._broadcast_tick_update()
            if self._finished():
                self._running = False
                self._set_status("finished")

    def reset_blocking(self) -> None:
        """Reset the population to its initial state and clear history."""
        with self._engine_lock:
            self._population.reset()
        self._running = False
        self._run_to_target = None
        self._error = None
        self._log("info", "engine", "population reset to initial state")
        self._broadcast(ResetDoneFrame(type="reset_done"))
        self._set_status("ready")

    def restore_blocking(self, tick: int) -> int:
        """Restore the raw-history state at *tick* and truncate after it.

        Args:
            tick: Target tick present in the raw history.

        Returns:
            The restored tick.

        Raises:
            ValueError: If the population does not support checkpoint
                restore, raw history is unavailable, or *tick* is not in it.
        """
        population = self._population
        if isinstance(population, SpatialPopulation):
            raise ValueError(
                "Time travel is not available for spatial populations yet"
            )
        with self._engine_lock:
            population.restore_checkpoint(tick)
        self._running = False
        self._run_to_target = None
        self._log(
            "info",
            "engine",
            f"time travel: restored to tick {tick}, later history truncated",
        )
        self._broadcast(RestoredFrame(type="restored", tick=tick))
        self._set_status("ready")
        return tick

    def set_interval_ms(self, value: int) -> None:
        """Set the per-tick delay (0 = turbo batches)."""
        self._interval_ms = int(value)
        self._log("debug", "engine", f"interval set to {value} ms")

    def set_record_every(self, value: int) -> None:
        """Set the history recording interval.

        Panmictic populations expose ``record_every`` on the object; spatial
        populations take it as a ``run()`` argument, so the session stores
        it and forwards it on every spatial tick.
        """
        population = self._population
        if isinstance(population, SpatialPopulation):
            self._record_every = int(value)
        else:
            with self._engine_lock:
                population.record_every = int(value)
        self._log("info", "engine", f"record_every set to {value}")

    def set_max_history(self, value: int) -> None:
        """Set the rolling history window size."""
        with self._engine_lock:
            self._population.max_history = int(value)
        self._log("info", "engine", f"max_history set to {value}")

    # -- status / logging / broadcast -------------------------------------

    @property
    def engine_lock(self) -> threading.Lock:
        """Engine mutex exposed so REST reads serialize with tick batches.

        REST snapshot handlers hold this lock while reading state/history;
        without it a turbo batch can shrink the rolling history between a
        handler's ``ticks`` read and its row indexing (IndexError).
        """
        return self._engine_lock

    def subscribe(self, outbox: asyncio.Queue[object]) -> int:  # object: outbound frames are heterogeneous TypedDict payloads
        """Register a WebSocket client's outbound queue.

        Args:
            outbox: Bounded queue drained by the connection's sender task.

        Returns:
            The subscriber id to pass to :meth:`unsubscribe`.
        """
        with self._subscriber_lock:
            sub_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[sub_id] = _Subscriber(
                asyncio.get_running_loop(), outbox
            )
        return sub_id

    def unsubscribe(self, subscriber_id: int) -> None:
        """Remove a previously registered subscriber."""
        with self._subscriber_lock:
            self._subscribers.pop(subscriber_id, None)

    def _set_status(self, status: SimulationStatus) -> None:
        self._status = status
        if status != "error":
            self._error = None
        self._broadcast(StatusFrame(type="status", status=status, error=self._error))

    def _log(
        self,
        level: LogLevel,
        source: str,
        message: str,
        data: str | None = None,
    ) -> None:
        frame = LogFrame(
            type="log",
            ts=time.time(),
            level=level,
            source=source,
            message=message,
            data=data,
        )
        self._logs.append(frame)
        self._broadcast(frame)

    def log_error(self, message: str) -> None:
        """Record and broadcast an error-level log frame from the WS layer."""
        self._log("error", "command", message)

    def log_replay(self) -> list[LogFrame]:
        """Return copies of the buffered log frames.

        The ring buffer is handed out to every new WebSocket connection, so
        each frame dict is copied: a caller mutating a returned frame must
        not corrupt future replays.
        """
        # dict() copies lose the TypedDict label; the cast re-labels copies
        # of values that are known LogFrames at runtime.
        return [cast("LogFrame", dict(frame)) for frame in self._logs]

    def _broadcast_tick_update(self) -> None:
        total, female, male = self._counts_summary()
        frame = TickUpdateFrame(
            type="tick_update",
            tick=int(self._population.tick),
            total=total,
            female=female,
            male=male,
            is_finished=self._finished(),
            history_len=len(self._population.history),
        )
        self._broadcast(frame)

    def _broadcast(self, frame: object) -> None:  # object: any outbound TypedDict frame
        with self._subscriber_lock:
            subscribers = list(self._subscribers.values())
        for sub in subscribers:
            try:
                sub.loop.call_soon_threadsafe(_offer, sub.outbox, frame)
            except RuntimeError:
                # Event loop already closed; drop the frame.
                pass


def _offer(
    outbox: asyncio.Queue[object],  # object: outbound frames are heterogeneous TypedDict payloads
    frame: object,  # object: any outbound TypedDict frame
) -> None:
    """Deliver *frame* on the loop thread; drop when the client is slow."""
    try:
        outbox.put_nowait(frame)
    except asyncio.QueueFull:
        pass
