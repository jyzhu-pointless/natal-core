from __future__ import annotations

import unittest
import uuid
from unittest.mock import patch

import numpy as np

from natal.genetic_structures import Species
from natal.population_builder import (
    AgeStructuredPopulationBuilder,
    DiscreteGenerationPopulationBuilder,
    PopulationConfigBuilder,
)


def _make_species(prefix: str = "BuilderInjectionSpecies") -> Species:
    return Species.from_dict(
        f"{prefix}_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT", "Drive"],
            }
        },
    )


class TestPopulationBuilderInitialInjection(unittest.TestCase):
    def setUp(self) -> None:
        self.species = _make_species()

    def test_survival_parser_supports_legacy_formats(self) -> None:
        seq_none = PopulationConfigBuilder._resolve_survival_param([1.0, 0.5, None], 5, [0.0])
        self.assertTrue(np.allclose(seq_none, np.array([1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)))

        from_dict = PopulationConfigBuilder._resolve_survival_param({1: 0.7, 3: 0.2}, 4, [0.0])
        self.assertTrue(np.allclose(from_dict, np.array([1.0, 0.7, 1.0, 0.2], dtype=np.float64)))

        from_callable = PopulationConfigBuilder._resolve_survival_param(lambda age: 1.0 - 0.1 * age, 3, [0.0])
        self.assertTrue(np.allclose(from_callable, np.array([1.0, 0.9, 0.8], dtype=np.float64)))

        from_scalar = PopulationConfigBuilder._resolve_survival_param(0.3, 4, [0.0])
        self.assertTrue(np.allclose(from_scalar, np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float64)))

    def test_age_builder_injects_initial_count_into_config(self) -> None:
        class _FakePopulation:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.species = kwargs["species"]
                self._config = kwargs["population_config"]
                self._index_core = type("I", (), {"genotype_to_index": {}})()

            def apply_recipe(self, recipe) -> None:
                return None

        fake_config = object()
        builder = (
            AgeStructuredPopulationBuilder(self.species)
            .age_structure(n_ages=4, new_adult_age=2)
            .initial_state(
                {
                    "female": {"WT|WT": [1, 2, 3, 4]},
                    "male": {"Drive|WT": {2: 5}},
                }
            )
        )

        with patch("natal.population_builder.PopulationConfigBuilder.build", return_value=fake_config) as build_mock:
            with patch("natal.age_structured_population.AgeStructuredPopulation", _FakePopulation):
                pop = builder.build()

        self.assertIsNotNone(pop)
        init_arr = build_mock.call_args.kwargs["initial_individual_count"]
        self.assertIsInstance(init_arr, np.ndarray)
        self.assertEqual(init_arr.shape, (2, 4, len(self.species.get_all_genotypes())))
        self.assertGreater(init_arr.sum(), 0)

    def test_discrete_builder_injects_initial_count_into_config(self) -> None:
        class _FakePopulation:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.species = kwargs["species"]
                self._config = kwargs["population_config"]
                self._index_core = type("I", (), {"genotype_to_index": {}})()

            def apply_recipe(self, recipe) -> None:
                return None

        fake_config = object()
        builder = (
            DiscreteGenerationPopulationBuilder(self.species)
            .initial_state(
                {
                    "female": {"WT|WT": 10},
                    "male": {"Drive|WT": [2, 3]},
                }
            )
        )

        with patch("natal.population_builder.PopulationConfigBuilder.build", return_value=fake_config) as build_mock:
            with patch("natal.discrete_generation_population.DiscreteGenerationPopulation", _FakePopulation):
                pop = builder.build()

        self.assertIsNotNone(pop)
        init_arr = build_mock.call_args.kwargs["initial_individual_count"]
        self.assertIsInstance(init_arr, np.ndarray)
        self.assertEqual(init_arr.shape, (2, 2, len(self.species.get_all_genotypes())))
        self.assertGreater(init_arr.sum(), 0)


if __name__ == "__main__":
    unittest.main()
