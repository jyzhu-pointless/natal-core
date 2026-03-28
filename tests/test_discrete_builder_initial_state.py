from __future__ import annotations

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from natal.discrete_generation_population import DiscreteGenerationPopulation
from natal.genetic_structures import Species
from natal.population_builder import DiscreteGenerationPopulationBuilder
from natal.type_def import Sex


def _make_species() -> Species:
    return Species.from_dict(
        f"DiscreteInitSpecies_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT", "Drive"],
            }
        },
    )


class TestDiscreteBuilderInitialState(unittest.TestCase):
    def setUp(self) -> None:
        self.species = _make_species()
        self.gt_wt_wt = self.species.get_genotype_from_str("WT|WT")
        self.gt_drive_wt = self.species.get_genotype_from_str("Drive|WT")

    def test_build_allows_none_carrying_capacity(self) -> None:
        builder = DiscreteGenerationPopulationBuilder(self.species)
        builder.competition(carrying_capacity=None)
        builder.initial_state(
            {
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )

        fake_config = object()

        class _FakePopulation:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        with patch("natal.population_builder.PopulationConfigBuilder.build", return_value=fake_config) as build_mock:
            with patch("natal.discrete_generation_population.DiscreteGenerationPopulation", _FakePopulation):
                pop = builder.build()

        self.assertIsNotNone(pop)
        self.assertIs(build_mock.call_args.kwargs["carrying_capacity"], None)
        self.assertIs(build_mock.call_args.kwargs["expected_num_adult_females"], None)

    def _make_population_shell(self) -> DiscreteGenerationPopulation:
        pop = DiscreteGenerationPopulation.__new__(DiscreteGenerationPopulation)
        pop._species = self.species

        all_genotypes = self.species.get_all_genotypes()
        genotype_to_index = {gt: i for i, gt in enumerate(all_genotypes)}
        pop._registry = SimpleNamespace(
            get_genotype_index=lambda gt: genotype_to_index[gt],
            genotype_to_index=genotype_to_index,
        ) # type: ignore

        pop._state = SimpleNamespace(
            individual_count=np.zeros((2, 2, len(all_genotypes)), dtype=np.float64)
        ) # type: ignore
        return pop

    def test_initial_state_supports_age0_and_age1(self) -> None:
        pop = self._make_population_shell()
        pop._distribute_initial_population(
            {
                "female": {
                    "WT|WT": {0: 3, 1: 7},
                },
                "male": {
                    "WT|WT": [2, 5],
                    "Drive|WT": 4,
                },
            }
        )

        female_idx = int(Sex.FEMALE.value)
        male_idx = int(Sex.MALE.value)
        wt_idx = pop._registry.get_genotype_index(self.gt_wt_wt) # type: ignore
        drive_wt_idx = pop._registry.get_genotype_index(self.gt_drive_wt) # type: ignore

        self.assertEqual(pop._state.individual_count[female_idx, 0, wt_idx], 3.0)
        self.assertEqual(pop._state.individual_count[female_idx, 1, wt_idx], 7.0)

        self.assertEqual(pop._state.individual_count[male_idx, 0, wt_idx], 2.0)
        self.assertEqual(pop._state.individual_count[male_idx, 1, wt_idx], 5.0)

        self.assertEqual(pop._state.individual_count[male_idx, 0, drive_wt_idx], 0.0)
        self.assertEqual(pop._state.individual_count[male_idx, 1, drive_wt_idx], 4.0)

    def test_initial_state_rejects_invalid_age_keys(self) -> None:
        pop = self._make_population_shell()

        with self.assertRaises(ValueError):
            pop._distribute_initial_population(
                {
                    "female": {"WT|WT": {2: 1}},
                }
            )


if __name__ == "__main__":
    unittest.main()
