"""Contract tests for the spatial-dashboard API endpoints."""

from __future__ import annotations

from typing import Any

import natal as nt
import pytest
from fastapi.testclient import TestClient

from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import HexGrid, SquareGrid
from natal.frontend.webui.app import create_app


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=f"SpatialUISpecies_{name}",
        structure={"chr1": {"loc1": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _build_spatial(name: str, topology: HexGrid | SquareGrid) -> SpatialPopulation:
    sp = _species(name)

    def deme(idx: int) -> nt.DiscreteGenerationPopulation:
        return (
            nt.DiscreteGenerationPopulation.setup(
                species=sp, name=f"{name}_d{idx}", stochastic=False
            )
            .initial_state(
                individual_count={
                    "male": {"WT|WT": 100 + 10 * idx},
                    "female": {"WT|WT": 50},
                }
            )
            .reproduction(eggs_per_female=3)
            .competition(carrying_capacity=1_000)
            .build()
        )

    return SpatialPopulation(
        [deme(i) for i in range(topology.n_demes)],
        migration_rate=0.1,
        topology=topology,
    )


def _make_client(
    name: str, topology: HexGrid | SquareGrid
) -> tuple[TestClient, SpatialPopulation]:
    population = _build_spatial(name, topology)
    return TestClient(create_app(population, title=name)), population


def _step(client: TestClient, pump_class: Any, n: int) -> None:  # Any: shared _FramePump class imported lazily to avoid a test-module cycle
    with client.websocket_connect("/ws") as ws:
        pump = pump_class(ws)
        deadline_time = 0.0
        _ = deadline_time
        ws.send_json({"type": "step", "n": n})
        seen = 0
        while seen < n:
            frame = pump.next()
            if frame["type"] == "tick_update":
                seen += 1


def test_spatial_landscape_layout_and_metrics() -> None:
    client, _ = _make_client("landscape", HexGrid(rows=2, cols=2, wrap=False))
    with client:
        payload = client.get("/api/spatial/landscape").json()
    assert payload["topology"]["kind"] == "hex"
    assert payload["n_demes"] == 4
    # Invariant: grid coordinates enumerate every (row, col) exactly once.
    grid = payload["topology"]["grid_ij"]
    assert sorted(map(tuple, grid)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert payload["topology"]["xy"] is not None
    # One [x, y] pair per deme.
    assert len(payload["topology"]["xy"]) == payload["n_demes"]
    # Invariant: per-deme totals are positive and the female/male split adds up.
    for total, female, male in zip(
        payload["totals"], payload["females"], payload["males"]
    ):
        assert total > 0
        assert total == pytest.approx(female + male, abs=1e-9)
    # Invariant: allele frequency rows are aligned per deme and within [0, 1].
    for freq_row in payload["allele_frequencies"]:
        assert len(freq_row) == payload["n_demes"]
        for value in freq_row:
            assert 0.0 <= value <= 1.0
    # Invariant: genotype counts per deme sum to the aggregate totals.
    summed = [sum(col) for col in zip(*payload["genotype_counts"])]
    for summed_deme, total in zip(summed, payload["totals"]):
        assert summed_deme == pytest.approx(total, abs=1e-9)


def test_spatial_landscape_square_and_no_topology() -> None:
    client, _ = _make_client("square", SquareGrid(rows=2, cols=2, wrap=True))
    with client:
        payload = client.get("/api/spatial/landscape").json()
    assert payload["topology"]["kind"] == "square"
    assert payload["topology"]["wrap"] is True


def test_spatial_deme_detail_and_bounds() -> None:
    client, _ = _make_client("deme", HexGrid(rows=2, cols=2, wrap=False))
    with client:
        detail = client.get("/api/spatial/deme/2").json()
        assert detail["index"] == 2
        assert detail["grid_ij"] == [1, 0]
        # Invariant: genotype rows sum back to the deme totals.
        assert detail["total"] == pytest.approx(
            sum(row["total"] for row in detail["genotypes"]), abs=1e-9
        )
        with pytest.raises(IndexError):
            client.get("/api/spatial/deme/9")


def test_spatial_migration_entries() -> None:
    client, population = _make_client("migration", HexGrid(rows=2, cols=2, wrap=False))
    with client:
        detail = client.get("/api/spatial/migration/0").json()
    assert detail["source"] == 0
    assert detail["source_name"] == "migration_d0"
    # Invariant: outbound shares sum to 1 (row-normalized CSR).
    assert sum(entry["share"] for entry in detail["entries"]) == pytest.approx(
        1.0, abs=1e-9
    )
    # Parallelogram HexGrid: the corner (0,0) has 2 in-grid neighbors.
    assert len(detail["entries"]) == 2
    # Discrete demes normalize to two age classes; only the adult class
    # migrates, so the (sex, age) mean is half the configured rate.
    assert detail["rate_mean"] == pytest.approx(0.05, abs=1e-12)
    _ = population


def test_spatial_series_aggregates_demes() -> None:
    from tests.test_webui_lifecycle import _FramePump

    client, population = _make_client("series", HexGrid(rows=2, cols=2, wrap=False))
    with client:
        _step(client, _FramePump, 2)
        payload = client.get("/api/spatial/series").json()
    # Invariant: the series covers ticks 0..2 (two stepped + live).
    assert payload["ticks"] == [0, 1, 2]
    # Invariant: totals increase (deterministic reproduction) and female/male
    # split adds up per point.
    for total, female, male in zip(
        payload["total"], payload["female"], payload["male"]
    ):
        assert total == pytest.approx(female + male, abs=1e-9)
    assert payload["total"][-1] > payload["total"][0]
    _ = population


def test_panmictic_population_gets_404_on_spatial_endpoints() -> None:
    species = _species("pan404")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="pan404_pop", stochastic=False
        )
        .initial_state(individual_count={"male": {"WT|WT": 10}, "female": {"WT|WT": 10}})
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=100)
        .build()
    )
    with TestClient(create_app(pop, title="pan404")) as client:
        assert client.get("/api/spatial/landscape").status_code == 404
        assert client.get("/api/spatial/deme/0").status_code == 404
        assert client.get("/api/spatial/migration/0").status_code == 404
        assert client.get("/api/spatial/series").status_code == 404


def test_spatial_dashboard_genetic_structure_endpoints_serve() -> None:
    """The shared genetic-structure endpoints must serve spatial dashboards.

    Requirement: the spatial dashboard frontend boots through the domain and
    registry stores (``SpatialDashboard.vue`` -> ``domain.initialize()`` /
    ``registry.initialize()``), fetching ``/api/config``, ``/api/hooks``,
    ``/api/genetics/matrices``, and ``/api/registry``.  These endpoints
    serialize the spatial population through deme 0, so whatever access
    route they use must keep working after the DemeSlice surface became
    explicit (unlisted slice members raise ``AttributeError``).  A 500 here
    breaks the whole spatial webui dashboard at boot.
    """
    client, _ = _make_client("genstruct", SquareGrid(2, 2))
    with client:
        config = client.get("/api/config")
        assert config.status_code == 200, config.text
        assert config.json()["scalars"]["carrying_capacity"] == pytest.approx(1000.0)
        assert client.get("/api/hooks").status_code == 200
        assert client.get("/api/genetics/matrices").status_code == 200
        registry = client.get("/api/registry")
        assert registry.status_code == 200, registry.text
        labels = {row["label"] for row in registry.json()["genotypes"]}
        assert {"WT|WT", "WT|Dr", "Dr|Dr"} <= labels


# ---------------------------------------------------------------------------
# Phase 4: spatial debug endpoints
# ---------------------------------------------------------------------------


def test_spatial_debug_endpoints() -> None:
    from tests.test_webui_lifecycle import _FramePump

    client, _ = _make_client("spdebug", HexGrid(rows=2, cols=2, wrap=False))
    with client:
        _step(client, _FramePump, 2)

        # params_log aggregates per-deme journals (empty here, but typed).
        rows = client.get("/api/debug/params_log").json()
        assert isinstance(rows, list)

        # diff: per-deme totals with additive delta invariant.
        diff = client.get("/api/debug/diff", params={"a": 0, "b": 2}).json()
        assert diff["found_a"] is True
        assert diff["found_b"] is True
        assert diff["delta_total"] == pytest.approx(
            sum(entry["delta"] for entry in diff["demes"]), abs=1e-9
        )
        assert len(diff["demes"]) == 4
        for entry in diff["demes"]:
            assert entry["delta"] == pytest.approx(
                entry["total_b"] - entry["total_a"], abs=1e-9
            )

        # raw dump: deme slice keeps the (sex, age, ztype) shape.
        raw = client.get(
            "/api/debug/state_raw", params={"tick": 1, "deme": 2}
        ).json()
        assert raw["found"] is True
        assert raw["mode"] == "history"
        assert raw["deme"] == 2
        assert len(raw["individual_count"]) == 2  # sexes
        # Invariant: the dump's sum equals the deme total in the diff view.
        total = sum(
            value
            for sex in raw["individual_count"]
            for age in sex
            for value in age
        )
        deme_entry = next(e for e in diff["demes"] if e["deme"] == 2)
        assert total <= deme_entry["total_b"] + 1e-9

        # unknown deme → IndexError surfaced as 500 via TestClient raise
        with pytest.raises(IndexError):
            client.get(
                "/api/debug/state_raw", params={"deme": 99}
            )
