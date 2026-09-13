"""Pre-run deme-level ``config`` freshness.

Pins the closed P8-disclosed residual ("session-less deme-level config
stale before the first run"): on both population kinds, in the window
between build (session handoff) and the first run, every deme-level
read surface — ``DemeSlice.config``, ``DemeSlice.export_config()``, the
container ``params`` view, and the underlying declaration draft — must
reflect pre-run writes (``write_ecology`` and the container-level
``tensor_write``) immediately.  The declaration draft is the authority
before the first tick; a read path that routed through a stale
standalone projection instead would resurface the old behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import natal as nt

if TYPE_CHECKING:
    from natal.frontend.spatial.population import SpatialPopulation


@pytest.fixture(scope="module")
def species() -> nt.Species:
    """Two-allele species shared by the freshness tests."""
    return nt.Species.from_dict(
        name="__test_deme_config_freshness__",
        structure={"auto": {"A": ["WT", "Dr"]}},
    )


def _build_discrete(species: nt.Species, name: str) -> SpatialPopulation:
    return (
        nt.SpatialPopulation.builder(
            species, n_demes=4, topology=nt.SquareGrid(2, 2), pop_type="discrete_generation"
        )
        .setup(name=name, stochastic=False)
        .initial_state(
            individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=500.0, low_density_growth_rate=6.0)
        .build()
    )


def _build_age(species: nt.Species, name: str) -> SpatialPopulation:
    return (
        nt.SpatialPopulation.builder(species, n_demes=3, pop_type="age_structured")
        .setup(name=name, stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    {
                        "female": {"WT|WT": [0.0, 100.0, 0.0]},
                        "male": {"WT|WT": [0.0, 100.0, 0.0]},
                    },
                ]
                * 3
            )
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 0.0],
            eggs_per_female=4.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.0],
        )
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )


@pytest.mark.parametrize(
    ("kind", "builder"),
    [("discrete", _build_discrete), ("age", _build_age)],
)
def test_pre_run_write_ecology_is_visible_on_every_config_surface(
    species: nt.Species, kind: str, builder
) -> None:
    """A pre-run ``write_ecology`` shows up on all deme config reads.

    Catches a read path that projects through a stale backend snapshot
    instead of the declaration draft in the pre-first-run window.
    """
    pop = builder(species, f"DemeCfgFresh{kind.title()}")
    deme = pop.deme(0)

    deme.write_ecology("carrying_capacity", 432.0)

    assert deme.config.carrying_capacity == 432.0
    assert pop.deme(0).config.carrying_capacity == 432.0
    assert deme.export_config().carrying_capacity == 432.0
    assert float(pop.params.carrying_capacity[0]) == 432.0


def test_pre_run_container_tensor_write_updates_deme_configs(
    species: nt.Species,
) -> None:
    """A container-level pre-run column write updates every deme read.

    Catches a deme ``config`` that lagged the authoritative ecology
    column after a whole-column ``tensor_write`` before the first run.
    """
    pop = _build_discrete(species, "DemeCfgFreshColumn")

    pop.params.tensor_write("carrying_capacity", np.array([111.0, 222.0, 333.0, 444.0]))

    assert pop.deme(0).config.carrying_capacity == 111.0
    assert pop.deme(3).config.carrying_capacity == 444.0
    assert pop.deme(0).export_config().carrying_capacity == 111.0
