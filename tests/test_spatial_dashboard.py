from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from natal.frontend.spatial.topology import HexGrid, SquareGrid
from natal.frontend.ui.spatial_dashboard import SpatialDashboard


class _FakeGenotype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class _FakeRegistry:
    def __init__(self) -> None:
        self.index_to_genotype = [_FakeGenotype("WT|WT"), _FakeGenotype("Dr|WT")]


class _FakeDeme:
    """Deme-slice double exposing the aligned surface the dashboard reads."""

    def __init__(self, config: SimpleNamespace) -> None:
        self.index_registry = _FakeRegistry()
        self.state = SimpleNamespace(individual_count=np.array([[[1, 0], [0, 2], [3, 0]], [[0, 4], [5, 0], [0, 0]]]))
        self._config = config

    @property
    def config(self):
        return self._config

    def export_config(self):
        return self._config


class _FakePop:
    def __init__(self, topology) -> None:
        self.tick = 7
        self.name = "demo"
        self.topology = topology
        self.species = SimpleNamespace()
        config0 = SimpleNamespace(
            new_adult_age=2,
            viability_fitness=np.array(
                [
                    [[1.0, 0.8], [1.0, 0.7], [1.0, 1.0]],
                    [[1.0, 0.6], [1.0, 0.5], [1.0, 1.0]],
                ],
                dtype=float,
            ),
            fecundity_fitness=np.array(
                [
                    [1.0, 0.9],
                    [1.0, 0.4],
                ],
                dtype=float,
            ),
        )

        config1 = SimpleNamespace(
            new_adult_age=2,
            viability_fitness=np.array(
                [
                    [[1.0, 1.0], [1.0, 0.2], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 0.3], [1.0, 1.0]],
                ],
                dtype=float,
            ),
            fecundity_fitness=np.array(
                [
                    [1.0, 1.0],
                    [1.0, 0.1],
                ],
                dtype=float,
            ),
        )

        self.demes = [_FakeDeme(config0), _FakeDeme(config1)]

    @property
    def n_demes(self) -> int:
        return len(self.demes)

    def deme(self, idx: int):
        return self.demes[idx]


def _make_dashboard(topology) -> SpatialDashboard:
    dashboard = SpatialDashboard.__new__(SpatialDashboard)
    dashboard.pop = _FakePop(topology)
    dashboard.selected_deme_idx = 0
    return dashboard


def test_deme_tile_style_uses_geometry_and_selection_state() -> None:
    """Test that color mapping and geometry calculations work for landscape visualization."""
    dashboard = _make_dashboard(HexGrid(rows=2, cols=2))
    dashboard.selected_deme_idx = 1

    # Test color mapping: low value should be different from high value
    color_low = dashboard._get_color_for_value(2.0, 0.0, 20.0)
    color_high = dashboard._get_color_for_value(18.0, 0.0, 20.0)
    assert color_low.startswith("rgb(")
    assert color_high.startswith("rgb(")
    # Colors should be different (viridis goes from dark purple to yellow)
    assert color_low != color_high

    # Test hex geometry: vertices should form a closed polygon
    xs, ys = dashboard._get_hex_vertices(0.0, 0.0, size=1.0)
    assert len(xs) == 7  # 6 vertices + 1 to close
    assert len(ys) == 7
    assert xs[0] == xs[-1]  # First and last should be same (closed)
    assert ys[0] == ys[-1]

    # Test square geometry
    xs_sq, ys_sq = dashboard._get_square_vertices(0.0, 0.0, size=1.0)
    assert len(xs_sq) == 5  # 4 vertices + 1 to close
    assert len(ys_sq) == 5
    assert xs_sq[0] == xs_sq[-1]
    assert ys_sq[0] == ys_sq[-1]



def test_selected_deme_age_rows_and_fitness_helpers() -> None:
    dashboard = _make_dashboard(SquareGrid(rows=2, cols=2))
    state = dashboard.pop.demes[0].state

    age_rows = dashboard._selected_deme_age_rows(state)
    assert age_rows == [
        {"age": 0, "female": 1, "male": 4, "total": 5},
        {"age": 1, "female": 2, "male": 5, "total": 7},
        {"age": 2, "female": 3, "male": 0, "total": 3},
    ]

    genotype_rows = dashboard._selected_genotype_rows(state)
    assert genotype_rows[0]["female"] == 4
    assert genotype_rows[0]["male"] == 5
    assert genotype_rows[0]["fitness"] == {}
    assert genotype_rows[1]["fitness"]["viability"] == "V: 0.7(F) / 0.5(M)"
    assert genotype_rows[1]["fitness"]["fecundity"] == "F: 0.9(F) / 0.4(M)"
    assert genotype_rows[1]["age_rows"] == [{"age": 1, "female": 2, "male": 0, "total": 2}]

    dashboard.selected_deme_idx = 1
    selected_rows = dashboard._selected_genotype_rows(state)
    assert selected_rows[1]["fitness"]["viability"] == "V: 0.2(F) / 0.3(M)"
    assert selected_rows[1]["fitness"]["fecundity"] == "F: 1(F) / 0.1(M)"


def test_large_landscape_mode_threshold_and_click_fallback() -> None:
    dashboard = _make_dashboard(SquareGrid(rows=21, cols=21))
    repeats = (dashboard.pop.topology.n_demes + len(dashboard.pop.demes) - 1) // len(dashboard.pop.demes)
    dashboard.pop.demes = (dashboard.pop.demes * repeats)[: dashboard.pop.topology.n_demes]
    assert dashboard._use_large_landscape_mode() is True

    clicked: list[int] = []
    dashboard._select_deme = clicked.append  # type: ignore[method-assign]  # test double: capturing stub replaces the private click handler
    event = SimpleNamespace(args={"points": [{"x": 3, "y": 4}]})
    dashboard._on_landscape_click(event)
    assert clicked == [dashboard.pop.topology.to_index((4, 3))]


def test_landscape_metric_genotype_and_allele_values_use_slot_queries() -> None:
    """Genotype/allele landscape metrics resolve through the aligned surface.

    The genotype branch reads the registry through the slice's
    ``index_registry``; the allele branch is a container query that reads
    per-deme frequencies through the raw slot (``_deme_object``), because
    the aligned slice carries no ``compute_allele_frequencies``.
    """
    from types import SimpleNamespace

    import natal as nt

    dashboard = SpatialDashboard.__new__(SpatialDashboard)
    species = nt.Species.from_dict(
        name="__dash_metrics__", structure={"chr1": {"loc1": ["WT", "Dr"]}}
    )
    dashboard.pop = (
        nt.SpatialPopulation.builder(
            species, n_demes=2, topology=SquareGrid(1, 2), pop_type="discrete_generation"
        )
        .setup(name="dash_metrics", stochastic=False)
        .initial_state(
            individual_count={"female": {"WT|WT": 30, "Dr|WT": 10}, "male": {"WT|WT": 30}}
        )
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=10000.0)
        .build()
    )
    dashboard.selected_deme_idx = 0

    dashboard.landscape_metric = SimpleNamespace(value="genotype")
    dashboard.landscape_target = SimpleNamespace(value="WT|WT")
    # WT|WT share: (30 F + 30 M) / (30 + 10 + 30) = 6/7 per deme.
    vals = dashboard._get_landscape_values()
    assert vals == [pytest.approx(6.0 / 7.0), pytest.approx(6.0 / 7.0)]

    dashboard.landscape_metric = SimpleNamespace(value="allele")
    dashboard.landscape_target = SimpleNamespace(value="WT")
    # WT allele share: (2*60 + 10) / (2*70) = 13/14 per deme.
    allele_vals = dashboard._get_landscape_values()
    assert allele_vals == [pytest.approx(13.0 / 14.0), pytest.approx(13.0 / 14.0)]
