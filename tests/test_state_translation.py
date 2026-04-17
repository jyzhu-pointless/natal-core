"""Tests for human-readable state translation helpers."""

from __future__ import annotations

import json

import numpy as np

import natal as nt
from natal.population_state import DiscretePopulationState, PopulationState
from natal.state_translation import (
    discrete_population_state_to_dict,
    discrete_population_state_to_json,
    output_current_state,
    output_history,
    population_history_to_readable_dict,
    population_history_to_readable_json,
    population_to_observation_dict,
    population_to_observation_json,
    population_state_to_dict,
    population_state_to_json,
    population_to_readable_dict,
    spatial_population_to_observation_dict,
    spatial_population_to_readable_dict,
)
from natal.spatial_population import SpatialPopulation


def test_population_state_to_dict_includes_sperm_storage_and_drops_zeros() -> None:
    individual_count = np.zeros((2, 2, 2), dtype=np.float64)
    individual_count[0, 1, 0] = 12.0
    individual_count[1, 1, 1] = 8.0

    sperm_storage = np.zeros((2, 2, 2), dtype=np.float64)
    sperm_storage[1, 0, 1] = 3.5

    state = PopulationState(
        n_tick=7,
        individual_count=individual_count,
        sperm_storage=sperm_storage,
    )

    result = population_state_to_dict(
        state,
        genotype_labels=["WT|WT", "WT|Drive"],
        sex_labels=["female", "male"],
    )

    assert result["state_type"] == "PopulationState"
    assert result["tick"] == 7
    assert result["individual_count"]["female"]["age_1"]["WT|WT"] == 12.0
    assert "age_0" not in result["individual_count"]["female"]
    assert result["sperm_storage"]["age_1"]["WT|WT"]["WT|Drive"] == 3.5


def test_discrete_population_state_to_json_roundtrip() -> None:
    individual_count = np.zeros((2, 2, 2), dtype=np.float64)
    individual_count[0, 1, 0] = 20.0

    state = DiscretePopulationState(n_tick=4, individual_count=individual_count)
    payload = discrete_population_state_to_json(
        state,
        genotype_labels=["A|A", "A|a"],
        sex_labels=["female", "male"],
    )
    parsed = json.loads(payload)

    assert parsed["state_type"] == "DiscretePopulationState"
    assert parsed["tick"] == 4
    assert parsed["individual_count"]["female"]["age_1"]["A|A"] == 20.0


def test_population_to_readable_dict_uses_registry_genotype_labels_for_discrete() -> None:
    species = nt.Species.from_dict(
        name="StateTranslationDiscrete",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.DiscreteGenerationPopulation
        .setup(species=species, name="DiscReadable", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2)
        .competition(low_density_growth_rate=1.5, carrying_capacity=100)
        .build()
    )

    result = population_to_readable_dict(pop, include_zero_counts=True)

    assert result["state_type"] == "DiscretePopulationState"
    assert "WT|WT" in result["individual_count"]["female"]["age_1"]


def test_population_to_readable_dict_includes_sperm_storage_for_age_structured() -> None:
    species = nt.Species.from_dict(
        name="StateTranslationAge",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.AgeStructuredPopulation
        .setup(species=species, name="AgeReadable", stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 8, 6]},
                "male": {"WT|WT": [0, 8, 6]},
            }
        )
        .survival(
            female_age_based_survival_rates=[1.0, 1.0, 0.8],
            male_age_based_survival_rates=[1.0, 1.0, 0.8],
        )
        .reproduction(eggs_per_female=2, use_sperm_storage=True)
        .competition(old_juvenile_carrying_capacity=100)
        .build()
    )

    result = population_to_readable_dict(pop, include_zero_counts=True)

    assert result["state_type"] == "PopulationState"
    assert "sperm_storage" in result


def test_population_state_to_json_is_valid_json() -> None:
    individual_count = np.zeros((2, 2, 1), dtype=np.float64)
    sperm_storage = np.zeros((2, 1, 1), dtype=np.float64)
    state = PopulationState(n_tick=1, individual_count=individual_count, sperm_storage=sperm_storage)

    payload = population_state_to_json(state, genotype_labels=["WT|WT"])
    parsed = json.loads(payload)
    assert parsed["state_type"] == "PopulationState"


def test_discrete_population_state_to_dict_include_zero_counts() -> None:
    individual_count = np.zeros((2, 2, 1), dtype=np.float64)
    state = DiscretePopulationState(n_tick=0, individual_count=individual_count)

    result = discrete_population_state_to_dict(
        state,
        genotype_labels=["WT|WT"],
        include_zero_counts=True,
    )
    assert result["individual_count"]["female"]["age_0"]["WT|WT"] == 0.0


def test_population_history_to_readable_dict_for_age_structured() -> None:
    species = nt.Species.from_dict(
        name="HistReadableAge",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.AgeStructuredPopulation
        .setup(species=species, name="HistAge", stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 6, 4]},
                "male": {"WT|WT": [0, 6, 4]},
            }
        )
        .survival(
            female_age_based_survival_rates=[1.0, 1.0, 0.9],
            male_age_based_survival_rates=[1.0, 1.0, 0.9],
        )
        .reproduction(eggs_per_female=2, use_sperm_storage=True)
        .competition(old_juvenile_carrying_capacity=100)
        .build()
    )

    pop.run(n_steps=2, record_every=1, clear_history_on_start=True)
    payload = population_history_to_readable_dict(pop, include_zero_counts=True)

    assert payload["state_type"] == "PopulationState"
    assert payload["n_snapshots"] == 3
    assert payload["snapshots"][0]["tick"] == 0
    assert payload["snapshots"][-1]["tick"] == 2
    assert "sperm_storage" in payload["snapshots"][-1]


def test_population_history_to_readable_json_for_discrete() -> None:
    species = nt.Species.from_dict(
        name="HistReadableDisc",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.DiscreteGenerationPopulation
        .setup(species=species, name="HistDisc", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 8]},
                "male": {"WT|WT": [0, 8]},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2)
        .competition(low_density_growth_rate=1.2, carrying_capacity=100)
        .build()
    )

    pop.run(n_steps=2, record_every=1, clear_history_on_start=True)
    _, history = pop.export_state()
    assert history is not None

    payload = population_history_to_readable_json(pop, history=history, include_zero_counts=True)
    parsed = json.loads(payload)

    assert parsed["state_type"] == "DiscretePopulationState"
    assert parsed["n_snapshots"] == 3
    assert parsed["snapshots"][0]["tick"] == 0
    assert parsed["snapshots"][-1]["tick"] == 2


def test_population_to_observation_dict_with_groups_for_discrete() -> None:
    species = nt.Species.from_dict(
        name="ObsReadableDiscrete",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.DiscreteGenerationPopulation
        .setup(species=species, name="ObsDisc", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 9]},
                "male": {"WT|WT": [0, 7]},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2)
        .competition(low_density_growth_rate=1.2, carrying_capacity=100)
        .build()
    )

    result = population_to_observation_dict(
        pop,
        groups={
            "adult_wt_female": {
                "genotype": ["WT|WT"],
                "sex": "female",
                "age": [1],
            }
        },
        collapse_age=False,
    )

    assert result["labels"] == ["adult_wt_female"]
    assert result["observed"]["adult_wt_female"]["female"]["age_1"] == 9.0


def test_output_current_state_can_write_json_file(tmp_path) -> None:
    species = nt.Species.from_dict(
        name="ObsOutputCurrent",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.DiscreteGenerationPopulation
        .setup(species=species, name="ObsCurrent", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 9]},
                "male": {"WT|WT": [0, 7]},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2)
        .competition(low_density_growth_rate=1.2, carrying_capacity=100)
        .build()
    )

    output_file = tmp_path / "current_state.json"
    payload = output_current_state(
        pop,
        groups={"adult_wt": {"genotype": ["WT|WT"], "age": [1]}},
        collapse_age=False,
        include_zero_counts=True,
        output_path=output_file,
    )

    parsed = json.loads(output_file.read_text(encoding="utf-8"))
    assert parsed == payload
    assert parsed["labels"] == ["adult_wt"]


def test_output_history_can_write_json_file(tmp_path) -> None:
    species = nt.Species.from_dict(
        name="ObsOutputHistory",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.DiscreteGenerationPopulation
        .setup(species=species, name="ObsHistory", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 8]},
                "male": {"WT|WT": [0, 8]},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2)
        .competition(low_density_growth_rate=1.2, carrying_capacity=100)
        .build()
    )

    pop.run(n_steps=2, record_every=1, clear_history_on_start=True)

    output_file = tmp_path / "history.json"
    payload = output_history(
        pop,
        groups={"adult_wt": {"genotype": ["WT|WT"], "age": [1]}},
        collapse_age=False,
        include_zero_counts=True,
        output_path=output_file,
    )

    parsed = json.loads(output_file.read_text(encoding="utf-8"))
    assert parsed == payload
    assert parsed["n_snapshots"] == 3
    assert parsed["snapshots"][0]["labels"] == ["adult_wt"]


def test_population_create_observation_is_reusable_for_current_state() -> None:
    species = nt.Species.from_dict(
        name="ObsCreateFromPopulation",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.DiscreteGenerationPopulation
        .setup(species=species, name="ObsReusable", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 9]},
                "male": {"WT|WT": [0, 7]},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2)
        .competition(low_density_growth_rate=1.2, carrying_capacity=100)
        .build()
    )

    observation = pop.create_observation(
        groups={"adult_wt": {"genotype": ["WT|WT"], "age": [1]}},
        collapse_age=False,
    )
    payload = output_current_state(
        pop,
        observation=observation,
        include_zero_counts=True,
    )

    assert payload["collapse_age"] is False
    assert payload["labels"] == ["adult_wt"]
    assert payload["observed"]["adult_wt"]["female"]["age_1"] == 9.0


def test_output_history_accepts_prebuilt_observation() -> None:
    species = nt.Species.from_dict(
        name="ObsHistoryWithObject",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.DiscreteGenerationPopulation
        .setup(species=species, name="ObsHistoryObj", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 8]},
                "male": {"WT|WT": [0, 8]},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2)
        .competition(low_density_growth_rate=1.2, carrying_capacity=100)
        .build()
    )

    pop.run(n_steps=2, record_every=1, clear_history_on_start=True)

    observation = pop.create_observation(
        groups={"adult_wt": {"genotype": ["WT|WT"], "age": [1]}},
        collapse_age=False,
    )
    payload = output_history(
        pop,
        observation=observation,
        include_zero_counts=True,
    )

    assert payload["labels"] == ["adult_wt"]
    assert payload["n_snapshots"] == 3
    assert payload["snapshots"][0]["labels"] == ["adult_wt"]


def test_population_to_observation_json_is_valid_and_collapsed() -> None:
    species = nt.Species.from_dict(
        name="ObsReadableAge",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    pop = (
        nt.AgeStructuredPopulation
        .setup(species=species, name="ObsAge", stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0, 5, 4]},
                "male": {"WT|WT": [0, 6, 4]},
            }
        )
        .survival(
            female_age_based_survival_rates=[1.0, 1.0, 0.9],
            male_age_based_survival_rates=[1.0, 1.0, 0.9],
        )
        .reproduction(eggs_per_female=2, use_sperm_storage=True)
        .competition(old_juvenile_carrying_capacity=100)
        .build()
    )

    payload = population_to_observation_json(
        pop,
        groups={"all_wt": {"genotype": ["WT|WT"]}},
        collapse_age=True,
        include_zero_counts=True,
    )
    parsed = json.loads(payload)

    assert parsed["collapse_age"] is True
    assert parsed["labels"] == ["all_wt"]
    assert "female" in parsed["observed"]["all_wt"]


def test_spatial_population_to_readable_dict_contains_demes_and_aggregate() -> None:
    species = nt.Species.from_dict(
        name="SpatialReadable",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    demes = [
        (
            nt.AgeStructuredPopulation
            .setup(species=species, name="deme_0", stochastic=False)
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0, 5, 0]},
                    "male": {"WT|WT": [0, 5, 0]},
                }
            )
            .survival(
                female_age_based_survival_rates=[1.0, 1.0, 0.9],
                male_age_based_survival_rates=[1.0, 1.0, 0.9],
            )
            .reproduction(eggs_per_female=2, use_sperm_storage=False)
            .competition(old_juvenile_carrying_capacity=100)
            .build()
        ),
        (
            nt.AgeStructuredPopulation
            .setup(species=species, name="deme_1", stochastic=False)
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0, 3, 0]},
                    "male": {"WT|WT": [0, 4, 0]},
                }
            )
            .survival(
                female_age_based_survival_rates=[1.0, 1.0, 0.9],
                male_age_based_survival_rates=[1.0, 1.0, 0.9],
            )
            .reproduction(eggs_per_female=2, use_sperm_storage=False)
            .competition(old_juvenile_carrying_capacity=100)
            .build()
        ),
    ]

    spatial = SpatialPopulation(demes=demes, migration_rate=0.0)
    result = spatial_population_to_readable_dict(spatial, include_zero_counts=True)

    assert result["state_type"] == "SpatialPopulation"
    assert result["n_demes"] == 2
    assert "deme_0" in result["demes"]
    assert result["aggregate"]["state_type"] == "PopulationState"


def test_spatial_population_to_observation_dict_aggregate_matches_sum() -> None:
    species = nt.Species.from_dict(
        name="SpatialObs",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )

    demes = [
        (
            nt.AgeStructuredPopulation
            .setup(species=species, name="deme_a", stochastic=False)
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0, 2, 0]},
                    "male": {"WT|WT": [0, 1, 0]},
                }
            )
            .survival(
                female_age_based_survival_rates=[1.0, 1.0, 0.9],
                male_age_based_survival_rates=[1.0, 1.0, 0.9],
            )
            .reproduction(eggs_per_female=2, use_sperm_storage=False)
            .competition(old_juvenile_carrying_capacity=100)
            .build()
        ),
        (
            nt.AgeStructuredPopulation
            .setup(species=species, name="deme_b", stochastic=False)
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [0, 4, 0]},
                    "male": {"WT|WT": [0, 3, 0]},
                }
            )
            .survival(
                female_age_based_survival_rates=[1.0, 1.0, 0.9],
                male_age_based_survival_rates=[1.0, 1.0, 0.9],
            )
            .reproduction(eggs_per_female=2, use_sperm_storage=False)
            .competition(old_juvenile_carrying_capacity=100)
            .build()
        ),
    ]

    spatial = SpatialPopulation(demes=demes, migration_rate=0.0)
    observed = spatial_population_to_observation_dict(
        spatial,
        groups={"wt_adults": {"genotype": ["WT|WT"], "age": [1]}},
        collapse_age=False,
        include_zero_counts=True,
    )

    agg_female = observed["aggregate"]["observed"]["wt_adults"]["female"]["age_1"]
    agg_male = observed["aggregate"]["observed"]["wt_adults"]["male"]["age_1"]
    assert agg_female == 6.0
    assert agg_male == 4.0
