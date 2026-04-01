#!/usr/bin/env python3

from __future__ import annotations

import natal as nt
import numpy as np
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid, build_adjacency_matrix


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
        deme.import_config(shared_config)

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
        deme.import_config(shared_config)

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
