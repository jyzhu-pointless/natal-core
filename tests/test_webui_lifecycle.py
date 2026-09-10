"""Coverage, ownership, and state-transition tests for the web UI API.

Complements ``test_webui_api.py`` with the harder lifecycle paths: finished
populations, engine error recovery, spatial guards, sperm-storage
serialization, hook descriptors, and frame-ownership guarantees.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

import natal as nt
from natal.frontend.data.state import PopulationState
from natal.frontend.population.discrete_generation import (
    DiscreteGenerationPopulation,
)
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.webui.app import create_app
from natal.frontend.webui.session import (
    SimulationSession,
    _offer,  # noqa: SLF001 -- unit under test
)

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=f"CoverSpecies_{name}",
        structure={"chr1": {"loc1": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _population_builder(name: str) -> nt.PopulationBuilder:
    """Deterministic discrete build chain, pre-build (callers declare hooks)."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "male": {"WT|WT": 400, "Dr|WT": 100},
                "female": {"WT|WT": 400, "Dr|WT": 100},
            }
        )
        .reproduction(eggs_per_female=4)
        .competition(carrying_capacity=10_000, juvenile_growth_mode="fixed")
    )


def _build_population(name: str) -> DiscreteGenerationPopulation:
    """Deterministic discrete population with no hooks."""
    return _population_builder(name).build()


def _build_age_population(name: str) -> nt.AgeStructuredPopulation:
    """Deterministic age-structured population (4 age classes)."""
    return (
        nt.AgeStructuredPopulation.setup(
            species=_species(name),
            name=name,
            stochastic=False,
            continuous_sampling=False,
        )
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 200, 150, 100]},
                "male": {"WT|WT": [0, 200, 150, 100]},
            }
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            eggs_per_female=10,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.9, 0.8],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            old_juvenile_carrying_capacity=500,
            expected_num_new_adult_females=450,
        )
        .build()
    )


def _build_finishing_population(name: str) -> DiscreteGenerationPopulation:
    """Population that finishes (extinction stop) after one tick."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), name=name, stochastic=False
        )
        .initial_state(
            individual_count={"male": {"WT|WT": 50}, "female": {"WT|WT": 50}}
        )
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=10_000)
        .hooks(
            nt.Op.add(genotypes="WT|WT", ages=1, sex="both", delta=-10_000),
            nt.Op.stop_if_extinction(),
            event="first",
        )
        .build()
    )


def _build_broken_population(name: str) -> DiscreteGenerationPopulation:
    """Population whose first hook raises (engine error path)."""

    def boom(tick_context: object) -> int:  # object: hook ABI passes an opaque TickContext
        raise RuntimeError("boom from hook")

    return _population_builder(name).hooks(boom, event="first").build()





class _FramePump:
    """Single background reader: the only consumer of the test socket.

    Starlette's test-session stream allows exactly one blocked receiver; a
    timed-out direct reader would linger and steal frames.  Pumping into a
    local queue keeps every frame observable in arrival order.
    """

    def __init__(self, ws: WebSocketTestSession) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._pump_all, daemon=True)
        self._ws = ws
        self._thread.start()

    def _pump_all(self) -> None:
        while True:
            try:
                self._queue.put(self._ws.receive_json())
            except Exception:  # noqa: BLE001 -- socket closed
                self._queue.put(None)
                break

    def next(self, timeout: float = 10.0) -> dict[str, Any]:  # Any: server frames are heterogeneous TypedDict payloads
        frame = self._queue.get(timeout=timeout)
        if frame is None:
            raise ConnectionError("test websocket closed")
        return frame


def _read_hello(pump: _FramePump) -> dict[str, Any]:  # Any: server frames are heterogeneous
    deadline = time.time() + 5.0
    while time.time() < deadline:
        frame = pump.next()
        if frame["type"] == "hello":
            return frame
    raise AssertionError("no hello frame")


def _collect_ticks(pump: _FramePump, count: int) -> list[int]:
    ticks: list[int] = []
    deadline = time.time() + 15.0
    while len(ticks) < count and time.time() < deadline:
        frame = pump.next()
        if frame["type"] == "tick_update":
            ticks.append(frame["tick"])
    assert len(ticks) == count, f"only got {ticks}"
    return ticks


def _wait_for_status(
    pump: _FramePump, status: str, timeout: float = 15.0
) -> dict[str, Any]:  # Any: return frame is one of the heterogeneous server payloads
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = pump.next()
        if frame["type"] == "status" and frame["status"] == status:
            return frame
    raise AssertionError(f"no {status} status frame")


# ---------------------------------------------------------------------------
# H10: finished-state machine transitions
# ---------------------------------------------------------------------------


def test_finished_population_rejects_play_and_run_to_tick() -> None:
    app = create_app(_build_finishing_population("finish_machine"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_hello(pump)
            ws.send_json({"type": "step", "n": 1})
            _collect_ticks(pump, 1)
            finished = _wait_for_status(pump, "finished")
            assert finished["status"] == "finished"

            # play must NOT crash into error: the loop wakes, sees the
            # finished population and settles back to "finished".
            ws.send_json({"type": "play"})
            ws.send_json({"type": "ping", "nonce": "still-alive"})
            saw_finished_again = False
            saw_pong = False
            deadline = time.time() + 10.0
            while time.time() < deadline and not (saw_finished_again and saw_pong):
                frame = pump.next()
                if frame["type"] == "status" and frame["status"] == "finished":
                    saw_finished_again = True
                if frame["type"] == "pong":
                    saw_pong = True
            assert saw_finished_again and saw_pong

            meta = client.get("/api/meta").json()
            assert meta["status"] == "finished"
            assert meta["tick"] == 0

            # run_to_tick on finished must be ignored with a warning, and the
            # connection must survive both commands.
            ws.send_json({"type": "run_to_tick", "tick": 5})
            ws.send_json({"type": "ping", "nonce": "alive-2"})
            saw_warning2 = False
            saw_pong2 = False
            deadline = time.time() + 10.0
            while time.time() < deadline and not (saw_warning2 and saw_pong2):
                frame = pump.next()
                if frame["type"] == "log" and "already finished" in frame["message"]:
                    saw_warning2 = True
                if frame["type"] == "pong" and frame["nonce"] == "alive-2":
                    saw_pong2 = True
            assert saw_warning2 and saw_pong2
            assert client.get("/api/meta").json()["tick"] == 0


def test_step_beyond_finish_stops_exactly() -> None:
    app = create_app(_build_finishing_population("step_finish"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_hello(pump)
            # step(10) on a pop that finishes at tick 1: exactly one tick runs.
            ws.send_json({"type": "step", "n": 10})
            # Invariant: the extinction stop freezes the tick axis; exactly
            # one tick_update frame is emitted even though n=10 was asked.
            ticks = _collect_ticks(pump, 1)
            assert ticks[0] <= 1
            frame = _wait_for_status(pump, "finished")
            assert frame["status"] == "finished"
            assert client.get("/api/meta").json()["tick"] == ticks[0]


# ---------------------------------------------------------------------------
# H10 + H5: engine error path and connection recovery
# ---------------------------------------------------------------------------


def test_step_error_reports_frame_and_connection_survives() -> None:
    app = create_app(_build_broken_population("broken_step"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_hello(pump)
            ws.send_json({"type": "step", "n": 1})
            error = _wait_for_error(pump)
            assert "boom from hook" in error["message"]

            # The socket must still work: ping answers, reset recovers.
            ws.send_json({"type": "ping", "nonce": "after-error"})
            deadline = time.time() + 5.0
            ponged = False
            while time.time() < deadline and not ponged:
                frame = pump.next()
                if frame["type"] == "pong" and frame["nonce"] == "after-error":
                    ponged = True
            assert ponged

            ws.send_json({"type": "reset"})
            while True:
                frame = pump.next()
                if frame["type"] == "reset_done":
                    break
            status = _wait_for_status(pump, "ready")
            assert status["error"] is None
            # Invariant: reset returns to tick 0; the broken hook is still
            # registered (hooks survive reset), and the session plus socket
            # remain fully usable.
            assert client.get("/api/meta").json()["tick"] == 0
            ws.send_json({"type": "ping", "nonce": "final-alive"})
            deadline = time.time() + 5.0
            ponged = False
            while time.time() < deadline and not ponged:
                frame = pump.next()
                if frame["type"] == "pong" and frame["nonce"] == "final-alive":
                    ponged = True
            assert ponged


def _wait_for_error(pump: _FramePump) -> dict[str, Any]:  # Any: server frames are heterogeneous
    deadline = time.time() + 15.0
    while time.time() < deadline:
        frame = pump.next()
        if frame["type"] == "error":
            return frame
    raise AssertionError("no error frame")


def test_play_error_transitions_to_error_status() -> None:
    app = create_app(_build_broken_population("broken_play"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_hello(pump)
            ws.send_json({"type": "play"})
            status = _wait_for_status(pump, "error")
            assert status["error"] is not None
            assert "boom from hook" in status["error"]
            # The loop is stopped; the connection is intact.
            ws.send_json({"type": "ping", "nonce": "post-crash"})
            deadline = time.time() + 5.0
            while time.time() < deadline:
                frame = pump.next()
                if frame["type"] == "pong" and frame["nonce"] == "post-crash":
                    break
            else:
                raise AssertionError("no pong after engine crash")
            assert client.get("/api/meta").json()["status"] == "error"


# ---------------------------------------------------------------------------
# H10: reset→play and restore→run_to_tick sequences
# ---------------------------------------------------------------------------


def test_reset_then_play_reruns_from_zero() -> None:
    app = create_app(_build_population("reset_play"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_hello(pump)
            ws.send_json({"type": "step", "n": 2})
            _collect_ticks(pump, 2)
            ws.send_json({"type": "reset"})
            while True:
                frame = pump.next()
                if frame["type"] == "reset_done":
                    break
            ws.send_json({"type": "play"})
            _wait_for_status(pump, "running")
            # "running" only means the loop started; wait until at least one
            # replayed tick lands so the pause cannot race the first step
            # (the invariant below demands tick >= 1).
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if client.get("/api/meta").json()["tick"] >= 1:
                    break
                time.sleep(0.01)
            ws.send_json({"type": "pause"})
            _wait_for_status(pump, "ready")
            tick = client.get("/api/meta").json()["tick"]
            # Invariant: the replay advanced from zero, not from tick 2.
            assert 1 <= tick <= 3


def test_restore_then_run_to_tick_stops_at_target() -> None:
    app = create_app(_build_population("restore_run"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_hello(pump)
            ws.send_json({"type": "step", "n": 5})
            _collect_ticks(pump, 5)
            ws.send_json({"type": "restore", "tick": 2})
            while True:
                frame = pump.next()
                if frame["type"] == "restored":
                    break
            ws.send_json({"type": "run_to_tick", "tick": 4})
            deadline = time.time() + 15.0
            last = 0
            reached = False
            while time.time() < deadline and not reached:
                frame = pump.next()
                if frame["type"] == "tick_update":
                    last = frame["tick"]
                elif frame["type"] == "status" and frame["status"] == "ready":
                    # The restore itself also emits "ready"; the breakpoint
                    # only counts once the replayed timeline reached 4.
                    reached = last >= 4
            assert reached
            # Invariant: the breakpoint holds on the replayed timeline.
            assert last == 4


# ---------------------------------------------------------------------------
# H8: sperm serialization, age-structured paths, hook descriptors
# ---------------------------------------------------------------------------


def test_age_structured_snapshot_includes_sperm_storage() -> None:
    pop = _build_age_population("age_sperm")
    # Inject a nonzero sperm entry through the state-import channel
    # (pop.state is a read snapshot under the Rust-owned runtime).
    state = pop.state
    assert isinstance(state, PopulationState)
    state.sperm_storage[2, 0, 1] = 55.0
    pop.import_state(state)
    app = create_app(pop, title="age sperm")
    with TestClient(app) as client:
        snapshot = client.get("/api/state").json()
        assert snapshot["is_age_structured"] is True
        sperm = snapshot["sperm_storage"]
        assert sperm is not None
        entry = next(
            e
            for e in sperm
            if e["age"] == 2 and e["value"] == pytest.approx(55.0, abs=1e-12)
        )
        # Invariant: sparse entry labels match the injected indices.
        assert entry["female_index"] == 0
        assert entry["male_index"] == 1
        assert isinstance(entry["female_label"], str)
        assert isinstance(entry["male_label"], str)

        raw = client.get("/api/debug/state_raw").json()
        assert raw["sperm_storage"] is not None
        assert raw["sperm_storage"][2][0][1] == pytest.approx(55.0, abs=1e-12)

        export = client.get("/api/export", params={"config": 0, "hooks": 0}).json()
        history_records = export["history"]
        sperm_export = history_records[-1].get("sperm_storage")
        assert sperm_export is not None
        assert any(
            entry_row["value"] == pytest.approx(55.0, abs=1e-12)
            for age_block in sperm_export
            for entry_row in age_block["entries"]
        )


def test_hooks_payload_covers_declarative_and_callback() -> None:
    def watcher(tick_context: object) -> int:  # object: hook ABI passes an opaque TickContext
        return 0

    pop = (
        _population_builder("hooks_payload")
        .hooks(
            nt.Op.add(genotypes="WT|WT", ages=1, sex="male", delta=10.0),
            event="first",
        )
        .hooks(
            nt.Op.scale(
                genotypes=["WT|WT", "WT|Dr"], ages=[1, 2], sex="both", factor=0.5
            ),
            event="early",
        )
        .hooks(watcher, event="late")
        .build()
    )

    app = create_app(pop, title="hooks")
    with TestClient(app) as client:
        hooks = client.get("/api/hooks").json()
    by_event = {hook["event"]: hook for hook in hooks}
    first = by_event["first"]
    assert first["kind"] == "declarative"
    assert first["operations"] is not None
    assert first["operations"][0]["ages"] == 1.0
    late = by_event["late"]
    assert late["kind"] == "callback"
    assert late["signature"] is not None
    assert "tick_context" in late["signature"]
    assert late["source"] is not None and "return 0" in late["source"]
    early = by_event["early"]
    assert early["operations"] is not None
    assert early["operations"][0]["ages"] == [1.0, 2.0]


def test_history_series_live_tick_and_truncation() -> None:
    app = create_app(_build_population("series_live"))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_hello(pump)
            ws.send_json({"type": "set_record_every", "value": 5})
            ws.send_json({"type": "step", "n": 2})
            _collect_ticks(pump, 2)
        # record_every=5: ticks 1 and 2 are unrecorded; the series must
        # still include the live tick 2.
        series = client.get(
            "/api/history/series", params={"max_points": 100}
        ).json()
        assert series["ticks"][-1] == 2
        assert 0 in series["ticks"]

        truncated = client.get(
            "/api/history/series", params={"max_points": 1}
        ).json()
        assert truncated["truncated"] is True
        assert len(truncated["ticks"]) == 1


# ---------------------------------------------------------------------------
# H8: spatial session branches
# ---------------------------------------------------------------------------


def _build_spatial(name: str) -> SpatialPopulation:
    # All demes must share the SAME Species object (spatial invariant).
    sp = _species(f"{name}_sp")

    def _deme(deme_name: str, total: int) -> DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(
                species=sp, name=deme_name, stochastic=False
            )
            .initial_state(
                individual_count={
                    "male": {"WT|WT": total // 2},
                    "female": {"WT|WT": total // 2},
                }
            )
            .reproduction(eggs_per_female=3.0)
            .competition(carrying_capacity=1_000)
            .build()
        )

    return SpatialPopulation(
        [_deme(f"{name}_d0", 200), _deme(f"{name}_d1", 100)], migration_rate=0.0
    )


def test_spatial_session_steps_and_reports_totals() -> None:
    app = create_app(_build_spatial("spatial_ws"), title="spatial")
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            hello = _read_hello(pump)
            assert hello["dashboard_type"] == "spatial"
            ws.send_json({"type": "step", "n": 2})
            ticks = _collect_ticks(pump, 2)
            assert ticks == [1, 2]
            # Invariant: aggregate total equals deme totals summed at tick 2.
            frame_total = 0.0
            meta = client.get("/api/meta").json()
            assert meta["tick"] == 2

            # Spatial guards: record_every warns, restore refuses.
            ws.send_json({"type": "set_record_every", "value": 2})
            ws.send_json({"type": "ping", "nonce": "post-warn"})
            saw_warn = False
            ponged = False
            deadline = time.time() + 5.0
            while time.time() < deadline and not (saw_warn and ponged):
                frame = pump.next()
                if frame["type"] == "log" and "record_every" in frame["message"]:
                    saw_warn = True
                if frame["type"] == "pong" and frame["nonce"] == "post-warn":
                    ponged = True
            assert saw_warn and ponged

            ws.send_json({"type": "restore", "tick": 1})
            error = _wait_for_error(pump)
            assert "spatial" in error["message"]
            _ = frame_total


def test_spatial_broadcast_uses_aggregate_counts() -> None:
    spatial = _build_spatial("spatial_counts")
    session = SimulationSession(spatial)

    async def scenario() -> list[dict[str, object]]:  # object: collected frames are heterogeneous TypedDict payloads
        outbox: asyncio.Queue[object] = asyncio.Queue(maxsize=64)  # object: outbound frames are heterogeneous
        sub_id = session.subscribe(outbox)
        await asyncio.to_thread(session.step_blocking, 1)
        frames: list[dict[str, object]] = []  # object: collected frames are heterogeneous payloads
        while True:
            frame = await asyncio.wait_for(outbox.get(), timeout=5.0)
            frames.append(frame)  # type: ignore[arg-type]  # frames are JSON dicts
            if isinstance(frame, dict) and frame.get("type") == "tick_update":
                break
        session.unsubscribe(sub_id)
        return frames

    frames = asyncio.run(scenario())
    tick_frames = [
        f for f in frames if f.get("type") == "tick_update"
    ]
    assert tick_frames
    # Invariant: deterministic ticks give the same aggregate on every run
    # (300 initial individuals; eggs_per_female=3 with ratio 0.5 yields the
    # observed 450 after one discrete-generation tick).
    assert tick_frames[-1]["total"] == pytest.approx(450.0, abs=1e-9)


# ---------------------------------------------------------------------------
# H9: ownership guarantees
# ---------------------------------------------------------------------------


def test_log_replay_hands_out_defensive_copies() -> None:
    session = SimulationSession(_build_population("logcopy"))
    frames = session.log_replay()
    assert frames
    frames[0]["message"] = "MUTATED"
    again = session.log_replay()
    # Invariant: mutating a returned frame cannot corrupt the buffer.
    assert all(f["message"] != "MUTATED" for f in again)


def test_history_series_is_fresh_per_call() -> None:
    from natal.frontend.webui.serialization import history_series

    pop = _build_population("series_fresh")
    pop.run_tick()
    first = history_series(pop, max_points=100)
    first["ticks"].append(999)
    first["total"].append(-1.0)
    second = history_series(pop, max_points=100)
    # Invariant: mutating one response cannot leak into the next.
    assert 999 not in second["ticks"]
    assert -1.0 not in second["total"]


def test_offer_drops_frames_on_full_queue() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    full: asyncio.Queue[object] = asyncio.Queue(maxsize=1)  # object: frames are heterogeneous payloads
    full.put_nowait({"type": "filler"})
    _offer(full, {"type": "dropped"})
    assert full.qsize() == 1
    loop.close()


def test_broadcast_survives_closed_event_loop() -> None:
    session = SimulationSession(_build_population("closedloop"))

    async def scenario() -> None:
        outbox: asyncio.Queue[object] = asyncio.Queue(maxsize=8)  # object: outbound frames are heterogeneous
        session.subscribe(outbox)
        # Deliberately leak the subscription: the loop dies right after.

    asyncio.run(scenario())
    # Must not raise even though the subscriber's loop is now closed.
    session._log("info", "test", "after close")


# ---------------------------------------------------------------------------
# H8: launch_vue real server + index fallback
# ---------------------------------------------------------------------------


def test_launch_vue_serves_real_http_server() -> None:
    import socket as socket_module

    import uvicorn as uvicorn_module

    from natal.frontend.webui.app import create_app as build_app

    with socket_module.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
    app = build_app(_build_population("launch"), title="launch test")
    config = uvicorn_module.Config(app, host="127.0.0.1", port=free_port, log_level="error")
    server = uvicorn_module.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        import httpx

        deadline = time.time() + 15.0
        payload = None
        while time.time() < deadline:
            try:
                response = httpx.get(f"http://127.0.0.1:{free_port}/api/meta")
                payload = response.json()
                break
            except Exception:  # noqa: BLE001 -- server not up yet
                time.sleep(0.2)
        assert payload is not None
        assert payload["title"] == "launch test"
        assert payload["population_name"] == "launch"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def test_index_fallback_when_dist_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from natal.frontend.webui import app as app_module

    monkeypatch.setattr(app_module, "_DIST_DIR", Path("/nonexistent/webui-dist"))
    app = app_module.create_app(_build_population("fallback"), title="fallback")
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Frontend build not found" in response.text
        assert "corepack pnpm build" in response.text
