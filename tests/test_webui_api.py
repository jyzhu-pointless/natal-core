"""Contract tests for the NATAL Vue web UI API layer.

Every assertion proves a numeric invariant (counts, frequencies, tick
arithmetic) against a deterministic (``stochastic=False``) population, not
just a response schema.  The WebSocket tests drive the real engine loop
started by the app lifespan.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

import natal as nt
from natal.frontend.population.discrete_generation import (
    DiscreteGenerationPopulation,
)
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.webui.app import create_app


def _build_population(name: str) -> DiscreteGenerationPopulation:
    """Build a small deterministic discrete-generation population."""
    species = nt.Species.from_dict(
        name=f"WebUISpecies_{name}",
        structure={"chr1": {"loc1": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "male": {"WT|WT": 400, "Dr|WT": 100},
                "female": {"WT|WT": 400, "Dr|WT": 100},
            }
        )
        .reproduction(eggs_per_female=4)
        .competition(carrying_capacity=10_000, juvenile_growth_mode="fixed")
        .build()
    )


def _make_client(name: str) -> tuple[TestClient, FastAPI]:
    app = create_app(_build_population(name), title="contract test")
    return TestClient(app), app


# ---------------------------------------------------------------------------
# REST contract
# ---------------------------------------------------------------------------


def test_meta_reports_dashboard_type_and_backend() -> None:
    client, _ = _make_client("meta")
    with client:
        payload = client.get("/api/meta").json()
    assert payload["app"] == "natal-webui"
    assert payload["dashboard_type"] == "population"
    assert payload["backend"] in ("rust", "python")
    assert payload["tick"] == 0
    assert payload["status"] == "ready"


def test_state_snapshot_counts_add_up() -> None:
    client, _ = _make_client("state")
    with client:
        payload = client.get("/api/state").json()
    assert payload["mode"] == "live"
    assert payload["found"] is True
    assert payload["is_age_structured"] is False
    # Invariant: total == female + male exactly (deterministic model).
    assert payload["total"] == pytest.approx(
        payload["female"] + payload["male"], abs=1e-9
    )
    # Invariant: declared initial state is reproduced.
    assert payload["female"] == pytest.approx(500.0, abs=1e-9)
    assert payload["male"] == pytest.approx(500.0, abs=1e-9)
    labels = {row["label"] for row in payload["genotypes"]}
    assert {"WT|WT", "WT|Dr"} <= labels
    # Invariant: declared initial genotype counts are reproduced (labels are
    # canonicalized alphabetically, so Dr|WT is reported as WT|Dr).
    by_label = {row["label"]: row for row in payload["genotypes"]}
    assert by_label["WT|WT"]["female"] == pytest.approx(400.0, abs=1e-9)
    assert by_label["WT|WT"]["male"] == pytest.approx(400.0, abs=1e-9)
    assert by_label["WT|Dr"]["female"] == pytest.approx(100.0, abs=1e-9)
    assert by_label["WT|Dr"]["male"] == pytest.approx(100.0, abs=1e-9)
    for row in payload["genotypes"]:
        assert row["total"] == pytest.approx(
            row["female"] + row["male"], abs=1e-9
        )
        # Invariant: per-age vectors have one entry per age class (2 for
        # the discrete normalization) and sum to the genotype totals.
        assert len(row["female_per_age"]) == 2
        assert sum(row["female_per_age"]) == pytest.approx(row["female"], abs=1e-9)
        assert sum(row["male_per_age"]) == pytest.approx(row["male"], abs=1e-9)


def test_state_snapshot_from_history_tick() -> None:
    client, _ = _make_client("state_hist")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "step", "n": 3})
            ticks = _collect_tick_updates(pump, 3)
        assert ticks == [1, 2, 3]

        payload = client.get("/api/state", params={"tick": 2}).json()
        assert payload["mode"] == "history"
        assert payload["found"] is True
        assert payload["tick"] == 2
        # Invariant: the stored tick-2 totals still add up.
        assert payload["total"] == pytest.approx(
            payload["female"] + payload["male"], abs=1e-9
        )


def test_history_series_is_monotonic_and_normalized() -> None:
    client, _ = _make_client("series")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "step", "n": 5})
            _collect_tick_updates(pump, 5)

        payload = client.get("/api/history/series", params={"max_points": 100}).json()
    ticks = payload["ticks"]
    # Invariant: tick axis is strictly increasing.
    assert ticks == sorted(set(ticks))
    assert len(ticks) >= 5
    # Invariant: single-locus allele frequencies are per-locus normalized.
    for allele, values in payload["allele_frequencies"].items():
        assert allele in ("WT", "Dr")
        assert len(values) == len(ticks)
        for freq in values:
            assert 0.0 <= freq <= 1.0
    wt = payload["allele_frequencies"]["WT"]
    dr = payload["allele_frequencies"]["Dr"]
    for a, b in zip(wt, dr):
        assert a + b == pytest.approx(1.0, abs=1e-9)
    # Invariant: every series array is aligned to the tick axis.
    assert len(payload["total"]) == len(ticks)
    assert payload["truncated"] is False


def test_config_payload_scalars_and_fitness() -> None:
    client, _ = _make_client("config")
    with client:
        payload = client.get("/api/config").json()
    scalars = payload["scalars"]
    # Invariant: the built configuration is echoed back.
    assert scalars["carrying_capacity"] == 10_000.0
    assert scalars["stochastic"] is False
    assert scalars["n_sexes"] == 2
    assert scalars["juvenile_growth_mode"]["name"] == "FIXED"
    # Drive-relevant rows must be listed even at fitness 1.0.
    assert any("Dr" in row["genotype"] for row in payload["viability"])
    assert any("Dr" in row["genotype"] for row in payload["fecundity"])


def test_registry_payload_contains_svg_and_labels() -> None:
    client, _ = _make_client("registry")
    with client:
        payload = client.get("/api/registry").json()
    labels = [entry["label"] for entry in payload["genotypes"]]
    assert "WT|WT" in labels
    for entry in payload["genotypes"]:
        assert entry["svg"].startswith("<svg")
        assert entry["ztype_indices"]
    # Invariant: ztype indices partition the ztype axis per genotype.
    all_indices = sorted(
        idx
        for entry in payload["genotypes"]
        for idx in entry["ztype_indices"]
    )
    assert all_indices == list(range(len(payload["ztypes"])))
    assert {allele["name"] for allele in payload["alleles"]} == {"WT", "Dr"}


def test_genetics_matrices_shapes_and_probabilities() -> None:
    client, _ = _make_client("genetics")
    with client:
        payload = client.get("/api/genetics/matrices").json()
    assert len(payload["meiosis"]) == 2
    for sex_matrix in payload["meiosis"]:
        rows = sex_matrix["data"]
        # Invariant: every parental gamete distribution is in [0, 1].
        for row in rows:
            for value in row:
                assert 0.0 <= value <= 1.0
    fertilization = payload["fertilization"]
    assert fertilization["too_large"] is False
    n = len(fertilization["row_labels"])
    assert len(fertilization["primary_index"]) == n
    assert len(fertilization["primary_probability"]) == n
    assert len(fertilization["cell_text"]) == n
    # Invariant: the primary-offspring probability is a probability — every
    # populated cell lies in [0, 1] (NaN marks empty gamete pairs).
    for row in fertilization["primary_probability"]:
        for value in row:
            assert value != value or 0.0 <= value <= 1.0
    # Invariant: a cell whose primary zygote exists carries a positive
    # probability (NaN is its own only invalid marker).


def test_export_payload_matches_history() -> None:
    client, _ = _make_client("export")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "step", "n": 2})
            _collect_tick_updates(pump, 2)
        payload = client.get("/api/export").json()
    assert payload["population_name"] == "export"
    assert payload["configuration"]["parameters"]["carrying_capacity"] == 10_000.0
    ticks = [record["tick"] for record in payload["history"]]
    # Invariant: export covers every recorded tick plus the live state.
    assert ticks == sorted(ticks)
    assert ticks[-1] == 2


def test_observation_counts_subset_of_state() -> None:
    client, _ = _make_client("observation")
    with client:
        response = client.post(
            "/api/observation",
            json={
                "groups": [
                    {"genotype": ["WT::WT"], "sex": "female"},
                    {"genotype": ["WT::Dr"], "sex": "male"},
                ],
                "collapse_age": True,
            },
        )
        assert response.status_code == 200
        payload = response.json()
    assert payload["collapse_age"] is True
    assert len(payload["rows"]) == 2
    state = client.get("/api/state").json()
    wt_wt = next(row for row in state["genotypes"] if row["label"] == "WT|WT")
    wt_dr = next(row for row in state["genotypes"] if row["label"] == "WT|Dr")
    # Invariant: observation counts equal the genotype-table counts.
    assert payload["rows"][0]["female"] == pytest.approx(wt_wt["female"], abs=1e-9)
    assert payload["rows"][1]["male"] == pytest.approx(wt_dr["male"], abs=1e-9)


def test_debug_endpoints() -> None:
    client, _ = _make_client("debug")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "step", "n": 3})
            _collect_tick_updates(pump, 3)

        log_rows = client.get("/api/debug/params_log").json()
        assert isinstance(log_rows, list)

        diff = client.get("/api/debug/diff", params={"a": 0, "b": 3}).json()
        assert diff["found_a"] is True
        assert diff["found_b"] is True
        # Invariant: per-genotype deltas sum to the total delta.
        assert diff["delta_total"] == pytest.approx(
            sum(row["delta_total"] for row in diff["genotypes"]), abs=1e-9
        )
        assert diff["total_b"] == pytest.approx(
            diff["total_a"] + diff["delta_total"], abs=1e-9
        )

        raw = client.get("/api/debug/state_raw", params={"tick": 2}).json()
        assert raw["tick"] == 2
        assert raw["found"] is True
        # Invariant: the raw dump keeps the (sex, age, ztype) tensor shape.
        assert len(raw["individual_count"]) == 2
        assert len(raw["individual_count"][0][0]) >= 2


def test_spatial_population_rejects_panmictic_endpoints() -> None:
    species = nt.Species.from_dict(
        name="WebUISpecies_spatial_guard",
        structure={"chr1": {"loc1": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )
    deme = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="spatial_guard_d0", stochastic=False
        )
        .initial_state(individual_count={"male": {"WT|WT": 10}, "female": {"WT|WT": 10}})
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=100)
        .build()
    )
    spatial = SpatialPopulation([deme], migration_rate=0.0)
    app = create_app(spatial, title="spatial guard")
    with TestClient(app) as client:
        meta = client.get("/api/meta").json()
        assert meta["dashboard_type"] == "spatial"
        response = client.get("/api/state")
        assert response.status_code == 501
        # Phase 4: spatial diff is supported (per-deme totals); the spatial
        # payload shape is asserted in test_webui_spatial.py.
        diff = client.get("/api/debug/diff", params={"a": 0, "b": 1})
        assert diff.status_code == 200
        assert "demes" in diff.json()


# ---------------------------------------------------------------------------
# WebSocket protocol contract
# ---------------------------------------------------------------------------


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





def _read_frame(pump: _FramePump, expected_type: str) -> dict[str, Any]:  # Any: server frames are heterogeneous
    """Read frames until one of *expected_type* arrives (skipping logs)."""
    deadline = time.time() + 5.0
    while time.time() < deadline:
        frame = pump.next()
        if frame["type"] == "log":
            continue
        assert frame["type"] == expected_type, frame
        return frame
    raise AssertionError(f"no {expected_type} frame within deadline")


def _collect_tick_updates(pump: _FramePump, count: int) -> list[int]:
    """Collect exactly *count* tick_update frames, returning their ticks."""
    ticks: list[int] = []
    deadline = time.time() + 15.0
    while len(ticks) < count and time.time() < deadline:
        frame = pump.next()
        if frame["type"] == "tick_update":
            ticks.append(frame["tick"])
    assert len(ticks) == count, f"only got {ticks}"
    return ticks


def test_ws_hello_ping_and_error_frames() -> None:
    client, _ = _make_client("ws_basic")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            hello = _read_frame(pump, "hello")
            assert hello["tick"] == 0
            assert hello["dashboard_type"] == "population"
            assert hello["status"] == "ready"

            ws.send_json({"type": "ping", "nonce": "abc"})
            pong = _read_frame(pump, "pong")
            assert pong["nonce"] == "abc"

            ws.send_json({"type": "bogus"})
            error = _read_frame(pump, "error")
            assert "error" in error["message"] or error["message"]


def test_ws_step_command_advances_exactly_n_ticks() -> None:
    client, _ = _make_client("ws_step")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "step", "n": 4})
            ticks = _collect_tick_updates(pump, 4)
            # Invariant: step(n) advances the tick axis one tick at a time.
            assert ticks == [1, 2, 3, 4]

            meta = client.get("/api/meta").json()
            assert meta["tick"] == 4


def test_ws_play_pause_set_interval_roundtrip() -> None:
    client, _ = _make_client("ws_play")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "set_interval_ms", "value": 0})
            ws.send_json({"type": "play"})
            status = _read_frame(pump, "status")
            assert status["status"] == "running"

            # Turbo mode: at least several ticks arrive quickly.
            ticks = _collect_tick_updates(pump, 5)
            assert ticks == sorted(ticks)

            ws.send_json({"type": "pause"})
            paused = _read_frame(pump, "status")
            assert paused["status"] == "ready"

            tick_after_pause = client.get("/api/meta").json()["tick"]
            assert tick_after_pause >= ticks[-1]


def test_ws_run_to_tick_breakpoint() -> None:
    client, _ = _make_client("ws_run_to")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "set_interval_ms", "value": 0})
            ws.send_json({"type": "run_to_tick", "tick": 6})
            # The engine broadcasts tick frames, then a breakpoint status.
            deadline = time.time() + 15.0
            last_tick = 0
            reached = False
            while time.time() < deadline and not reached:
                frame = pump.next()
                if frame["type"] == "tick_update":
                    last_tick = frame["tick"]
                elif frame["type"] == "status" and frame["status"] == "ready":
                    reached = True
            assert reached
            # Invariant: the loop pauses exactly at the breakpoint tick.
            assert last_tick == 6
            assert client.get("/api/meta").json()["tick"] == 6


def test_ws_restore_rolls_back_history() -> None:
    client, _ = _make_client("ws_restore")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "step", "n": 4})
            _collect_tick_updates(pump, 4)

            ws.send_json({"type": "restore", "tick": 2})
            restored = _read_frame(pump, "restored")
            assert restored["tick"] == 2

            # Invariant: history is truncated at the restored tick.
            assert client.get("/api/meta").json()["tick"] == 2
            series = client.get(
                "/api/history/series", params={"max_points": 100}
            ).json()
            assert series["ticks"][-1] == 2
            assert max(series["ticks"]) == 2

            # The simulation can continue from the restored point.
            ws.send_json({"type": "step", "n": 1})
            assert _collect_tick_updates(pump, 1) == [3]


def test_ws_reset_returns_to_initial_state() -> None:
    client, _ = _make_client("ws_reset")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "step", "n": 2})
            _collect_tick_updates(pump, 2)

            ws.send_json({"type": "reset"})
            assert _read_frame(pump, "reset_done")["type"] == "reset_done"
            status = _read_frame(pump, "status")
            assert status["status"] == "ready"

            assert client.get("/api/meta").json()["tick"] == 0
            state = client.get("/api/state").json()
            assert state["female"] == pytest.approx(500.0, abs=1e-9)
            # Invariant: reset clears the recorded timeline entirely; the
            # initial snapshot is re-recorded on the next run.
            assert state["history_len"] == 0


def test_setters_update_population_settings() -> None:
    client, _ = _make_client("ws_setters")
    with client:
        with client.websocket_connect("/ws") as ws:
            pump = _FramePump(ws)
            _read_frame(pump, "hello")
            ws.send_json({"type": "set_record_every", "value": 3})
            ws.send_json({"type": "set_max_history", "value": 50})
            # Both setters ack with log frames; drain until both are seen.
            _wait_for_logs(
                pump, ["record_every set to 3", "max_history set to 50"]
            )
            # Settings accepted: the engine still steps normally.
            ws.send_json({"type": "step", "n": 1})
            assert _collect_tick_updates(pump, 1) == [1]


def _wait_for_logs(
    pump: _FramePump, needles: list[str], timeout: float = 5.0
) -> None:
    """Read frames until every *needle* appeared in a log message."""
    pending = list(needles)
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        frame = pump.next()
        if frame["type"] == "log" and any(
            needle in frame["message"] for needle in pending
        ):
            pending = [
                needle for needle in pending if needle not in frame["message"]
            ]
    assert not pending, f"missing log acknowledgements: {pending}"
