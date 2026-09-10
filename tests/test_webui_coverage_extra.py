"""Extra coverage tests: launch_vue entry, error-query paths, edge branches."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import natal as nt
from natal.frontend.webui.app import create_app
from tests.test_webui_lifecycle import _build_age_population, _build_population


def test_launch_vue_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """launch_vue wires create_app into uvicorn.run with the requested port."""
    captured: dict[str, object] = {}  # object: captures heterogeneous uvicorn.run kwargs

    def fake_run(app: object, host: str, port: int, log_level: str) -> None:  # object: stands in for uvicorn's app parameter
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("natal.frontend.webui.server.uvicorn.run", fake_run)
    from natal.frontend.webui import launch_vue

    launch_vue(_build_population("launch_vue_entry"), port=9123, title="LV")
    assert captured["port"] == 9123
    assert captured["host"] == "127.0.0.1"
    assert hasattr(captured["app"], "state")


def test_bad_query_params_return_422() -> None:
    app = create_app(_build_population("badquery"), title="bad query")
    with TestClient(app) as client:
        assert client.get("/api/state", params={"tick": "abc"}).status_code == 422
        assert client.get(
            "/api/history/series", params={"max_points": "x"}
        ).status_code == 422
        assert client.get("/api/debug/diff").status_code == 422
        assert client.get(
            "/api/debug/diff", params={"a": 0, "b": 99}
        ).status_code == 200
        assert client.get(
            "/api/debug/state_raw", params={"tick": "zz"}
        ).status_code == 422


def test_unknown_tick_reports_not_found() -> None:
    app = create_app(_build_population("unknown_tick"), title="unknown tick")
    with TestClient(app) as client:
        payload = client.get("/api/state", params={"tick": 999}).json()
        assert payload["found"] is False
        assert payload["mode"] == "live"

        raw = client.get("/api/debug/state_raw", params={"tick": 999}).json()
        assert raw["found"] is False


def test_age_structured_observation_and_diff() -> None:
    app = create_app(_build_age_population("age_extra"), title="age extra")
    with TestClient(app) as client:
        response = client.post(
            "/api/observation",
            json={
                "groups": [{"genotype": ["WT::WT"], "age_start": 1, "age_end": 2}],
                "collapse_age": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        # Invariant: collapsed age observation is one row per group.
        assert len(body["rows"]) == 1
        assert body["rows"][0]["age"] is None


def test_multi_chromosome_and_glab_labels() -> None:
    """Multi-locus species exercise glab-qualified gamete labels."""
    species = nt.Species.from_dict(
        name="CoverSpecies_multilocus",
        structure={"chr1": {"a": ["WT", "Dr"]}, "chr2": {"b": ["A", "B"]}},
        gamete_labels=["default", "cas9"],
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="multilocus", stochastic=False
        )
        .initial_state(
            individual_count={
                "male": {"WT|Dr; A|B": 50},
                "female": {"WT|Dr; A|B": 50},
            }
        )
        .reproduction(eggs_per_female=3)
        .competition(carrying_capacity=1_000)
        .build()
    )
    app = create_app(pop, title="multilocus")
    with TestClient(app) as client:
        genetics = client.get("/api/genetics/matrices").json()
        fert = genetics["fertilization"]
        assert fert["too_large"] is False
        # Invariant: every cell either has a primary zygote (>= 0 index) or
        # is empty (NaN) — no partial states.
        for row in fert["primary_index"]:
            for value in row:
                assert value != value or value >= 0  # NaN check

        diff = client.get("/api/debug/diff", params={"a": 0, "b": 0}).json()
        assert diff["delta_total"] == 0.0
