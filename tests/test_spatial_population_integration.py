#!/usr/bin/env python3

from __future__ import annotations

import natal as nt
import numpy as np
import pytest
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid, SquareGrid, build_adjacency_matrix


def _make_deme(species: nt.Species, name: str, adult_count: float) -> nt.AgeStructuredPopulation:
    return (
        nt.AgeStructuredPopulation
        .setup(species=species, name=name, stochastic=False)
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, adult_count, 0.0, 0.0]},
                "male": {"WT|WT": [0.0, adult_count, 0.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
            male_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
        )
        .reproduction(
            female_age_based_mating_rates=[0.0, 0.0, 0.0, 0.0],
            male_age_based_mating_rates=[0.0, 0.0, 0.0, 0.0],
            eggs_per_female=0.0,
            use_sperm_storage=False,
        )
        .competition(
            juvenile_growth_mode="logistic",
            expected_num_adult_females=100,
            old_juvenile_carrying_capacity=200.0,
        )
        .build()
    )


def test_spatial_population_run_tick_with_real_demes_updates_state(simple_species: nt.Species) -> None:
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=100.0),
        _make_deme(simple_species, "deme_1", adult_count=0.0),
        _make_deme(simple_species, "deme_2", adult_count=0.0),
        _make_deme(simple_species, "deme_3", adult_count=0.0),
    ]

    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config._replace())

    adjacency = build_adjacency_matrix(
        SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
        row_normalize=True,
    )

    spatial = SpatialPopulation(
        demes=demes,
        adjacency=adjacency,
        migration_rate=0.5,
    )

    spatial.run_tick()

    totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    assert spatial.tick == 1
    assert totals[0] < 200.0
    assert totals[1] > 0.0
    assert totals[2] > 0.0
    assert sum(totals) == 200.0


def test_spatial_population_kernel_migration_updates_state(simple_species: nt.Species) -> None:
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=0.0),
        _make_deme(simple_species, "deme_1", adult_count=100.0),
        _make_deme(simple_species, "deme_2", adult_count=0.0),
    ]

    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config._replace())

    spatial = SpatialPopulation(
        demes=demes,
        topology=SquareGrid(rows=1, cols=3, neighborhood="von_neumann", wrap=False),
        migration_kernel=np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        migration_rate=0.5,
    )

    row = spatial.migration_row(1)
    assert np.isclose(row.sum(), 1.0)
    assert np.isclose(row[0], 0.5)
    assert np.isclose(row[2], 0.5)

    spatial.run_tick()

    totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    assert spatial.tick == 1
    assert totals[0] > 0.0
    assert totals[1] < 200.0
    assert totals[2] > 0.0
    assert sum(totals) == 200.0


def test_spatial_population_kernel_migration_row_renormalizes_border_weights(
    simple_species: nt.Species,
) -> None:
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=100.0),
        _make_deme(simple_species, "deme_1", adult_count=0.0),
        _make_deme(simple_species, "deme_2", adult_count=0.0),
    ]

    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config)

    spatial = SpatialPopulation(
        demes=demes,
        topology=SquareGrid(rows=1, cols=3, neighborhood="von_neumann", wrap=False),
        migration_kernel=np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        migration_rate=1.0,
    )

    row = spatial.migration_row(0)
    expected = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    assert np.allclose(row, expected)


def test_spatial_population_hex_kernel_row_matches_valid_border_offsets(
    simple_species: nt.Species,
) -> None:
    demes = [
        _make_deme(simple_species, f"deme_{idx}", adult_count=0.0)
        for idx in range(9)
    ]

    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config)

    spatial = SpatialPopulation(
        demes=demes,
        topology=HexGrid(rows=3, cols=3, wrap=False),
        migration_kernel=np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        migration_rate=1.0,
    )

    row = spatial.migration_row(0)
    expected = np.zeros(9, dtype=np.float64)
    expected[spatial.topology.to_index((0, 1))] = 1.0 / 3.0
    expected[spatial.topology.to_index((1, 0))] = 1.0 / 3.0
    expected[spatial.topology.to_index((1, 1))] = 1.0 / 3.0
    assert np.allclose(row, expected)


def test_spatial_population_hex_kernel_run_tick_matches_border_distribution(
    simple_species: nt.Species,
) -> None:
    demes = [
        _make_deme(simple_species, f"deme_{idx}", adult_count=100.0 if idx == 0 else 0.0)
        for idx in range(9)
    ]

    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config)

    spatial = SpatialPopulation(
        demes=demes,
        topology=HexGrid(rows=3, cols=3, wrap=False),
        migration_kernel=np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        migration_rate=1.0,
    )

    spatial.run_tick()

    totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    assert np.isclose(totals[0], 0.0)
    assert np.isclose(totals[1], 200.0 / 3.0)
    assert np.isclose(totals[3], 200.0 / 3.0)
    assert np.isclose(totals[4], 200.0 / 3.0)
    assert np.isclose(sum(totals), 200.0)


def test_spatial_population_heterogeneous_kernel_bank_routes_per_source(
    simple_species: nt.Species,
) -> None:
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=100.0),
        _make_deme(simple_species, "deme_1", adult_count=100.0),
        _make_deme(simple_species, "deme_2", adult_count=0.0),
    ]

    shared_config = demes[0].export_config()
    for deme in demes[1:]:
        deme.import_config(shared_config)

    right_only = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    left_only = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    spatial = SpatialPopulation(
        demes=demes,
        topology=SquareGrid(rows=1, cols=3, neighborhood="von_neumann", wrap=False),
        migration_strategy="kernel",
        kernel_bank=(right_only, left_only),
        deme_kernel_ids=np.array([0, 1, 0], dtype=np.int64),
        migration_rate=1.0,
    )

    row0 = spatial.migration_row(0)
    row1 = spatial.migration_row(1)
    assert np.allclose(row0, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    assert np.allclose(row1, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    spatial.run_tick()

    totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    # source 0 migrates to 1; source 1 migrates to 0
    assert np.isclose(totals[0], 200.0)
    assert np.isclose(totals[1], 200.0)
    assert np.isclose(totals[2], 0.0)
    assert np.isclose(sum(totals), 400.0)


def test_spatial_population_run_tick_supports_heterogeneous_deme_configs(
    simple_species: nt.Species,
) -> None:
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=100.0),
        _make_deme(simple_species, "deme_1", adult_count=100.0),
    ]

    cfg0 = demes[0].export_config()
    cfg1 = demes[1].export_config()._replace(low_density_growth_rate=1.7)
    demes[1].import_config(cfg1)

    assert cfg0.low_density_growth_rate != demes[1].export_config().low_density_growth_rate

    spatial = SpatialPopulation(
        demes=demes,
        adjacency=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        migration_rate=0.0,
    )

    spatial.run_tick()

    assert spatial.tick == 1
    totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    assert np.isclose(sum(totals), 400.0)


def test_spatial_population_migration_rejects_inconsistent_sampling_modes(
    simple_species: nt.Species,
) -> None:
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=100.0),
        _make_deme(simple_species, "deme_1", adult_count=0.0),
    ]

    cfg1 = demes[1].export_config()._replace(
        is_stochastic=True,
        use_continuous_sampling=True,
    )
    demes[1].import_config(cfg1)

    spatial = SpatialPopulation(
        demes=demes,
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        migration_rate=0.5,
    )

    with pytest.raises(ValueError, match="migration requires consistent"):
        spatial.run_tick()


def test_spatial_population_heterogeneous_configs_use_python_hook_dispatch(
    simple_species: nt.Species,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=100.0),
        _make_deme(simple_species, "deme_1", adult_count=100.0),
    ]

    demes[1].import_config(demes[1].export_config()._replace(low_density_growth_rate=1.9))

    spatial = SpatialPopulation(
        demes=demes,
        adjacency=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        migration_rate=0.0,
    )

    original = SpatialPopulation._run_python_dispatch_tick
    calls = {"count": 0}

    def _wrapped(spatial_population: SpatialPopulation) -> bool:
        calls["count"] += 1
        return bool(original(spatial_population))

    monkeypatch.setattr(SpatialPopulation, "_run_python_dispatch_tick", _wrapped)

    spatial.run_tick()
    assert calls["count"] == 1
    assert spatial.tick == 1


def test_spatial_population_heterogeneous_configs_run_uses_hook_dispatch_each_step(
    simple_species: nt.Species,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=100.0),
        _make_deme(simple_species, "deme_1", adult_count=100.0),
        _make_deme(simple_species, "deme_2", adult_count=100.0),
    ]
    demes[1].import_config(demes[1].export_config()._replace(low_density_growth_rate=1.4))
    demes[2].import_config(demes[2].export_config()._replace(low_density_growth_rate=1.8))

    spatial = SpatialPopulation(
        demes=demes,
        adjacency=np.eye(3, dtype=np.float64),
        migration_rate=0.0,
    )

    original = SpatialPopulation._run_python_dispatch_tick
    calls = {"count": 0}

    def _wrapped(spatial_population: SpatialPopulation) -> bool:
        calls["count"] += 1
        return bool(original(spatial_population))

    monkeypatch.setattr(SpatialPopulation, "_run_python_dispatch_tick", _wrapped)

    spatial.run(n_steps=3)

    assert calls["count"] == 3
    assert spatial.tick == 3


def test_spatial_population_comprehensive_matches_two_deme_migration_theory(
    simple_species: nt.Species,
) -> None:
    """Validate deterministic two-deme migration against closed-form theory.

    Model:
    - Two demes with 100% cross-migration adjacency.
    - Deterministic migration rate ``m`` per tick.
    - No reproduction and no mortality in the tested age buckets.

    For total counts ``x_t`` in deme 0 and ``y_t`` in deme 1:

    ``x_{t+1} = (1 - m) x_t + m y_t``
    ``y_{t+1} = (1 - m) y_t + m x_t``

    Therefore the difference decays as:

    ``x_t - y_t = (x_0 - y_0) (1 - 2m)^t``
    """
    demes = [
        _make_deme(simple_species, "deme_0", adult_count=100.0),
        _make_deme(simple_species, "deme_1", adult_count=0.0),
    ]

    shared_config = demes[0].export_config()
    demes[1].import_config(shared_config._replace())

    migration_rate = 0.25
    spatial = SpatialPopulation(
        demes=demes,
        adjacency=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        migration_rate=migration_rate,
    )

    initial_totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    assert np.isclose(sum(initial_totals), 200.0)

    # Keep the horizon within age buckets that still have survival=1.0 in
    # this fixture so we validate migration theory without mortality effects.
    n_steps = 2
    spatial.run(n_steps=n_steps)

    totals = [float(deme.state.individual_count.sum()) for deme in spatial.demes]
    assert spatial.tick == n_steps

    # Theory: mean is invariant and difference decays by (1 - 2m)^t.
    initial_mean = 0.5 * (initial_totals[0] + initial_totals[1])
    initial_diff = initial_totals[0] - initial_totals[1]
    expected_diff = initial_diff * ((1.0 - 2.0 * migration_rate) ** n_steps)
    expected_deme0 = initial_mean + 0.5 * expected_diff
    expected_deme1 = initial_mean - 0.5 * expected_diff

    assert np.isclose(sum(totals), sum(initial_totals))
    assert np.isclose(totals[0], expected_deme0)
    assert np.isclose(totals[1], expected_deme1)
