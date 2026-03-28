from __future__ import annotations

import unittest
import uuid
from unittest.mock import patch

from natal.genetic_structures import Species
from natal.population_builder import AgeStructuredPopulationBuilder
from natal.helpers import resolve_sex_label


def _make_simple_species() -> Species:
    # Single chromosome, single locus, two alleles -> four diploid ordered genotypes
    return Species.from_dict(
        f"TestSpecies_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["A", "a"],
            }
        },
    )


class TestPopulationBuilderFitnessPatterns(unittest.TestCase):
    def setUp(self) -> None:
        self.simple_species = _make_simple_species()
        self.all_genotypes = self.simple_species.get_all_genotypes()

    def test_resolve_genotype_selector_exact_string(self) -> None:
        self.simple_species.resolve_genotype_selectors(
            selector="A|a",
            all_genotypes=self.all_genotypes,
            context="viability",
        )

        matched = self.simple_species.resolve_genotype_selectors(
            selector="A|a",
            all_genotypes=self.all_genotypes,
            context="viability",
        )

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0], self.simple_species.get_genotype_from_str("A|a"))

    def test_resolve_genotype_selector_pattern_string(self) -> None:
        matched = self.simple_species.resolve_genotype_selectors(
            selector="A|*",
            all_genotypes=self.all_genotypes,
            context="fecundity",
        )

        # Maternal haplotype fixed to A, paternal can be A or a
        self.assertEqual(len(matched), 2)

    def test_resolve_genotype_selector_tuple_union(self) -> None:
        matched = self.simple_species.resolve_genotype_selectors(
            selector=("A|a", "a|A"),
            all_genotypes=self.all_genotypes,
            context="viability",
        )

        self.assertEqual(len(matched), 2)

    def test_iter_sexual_selection_entries_nested_and_flat(self) -> None:
        nested = {
            "A|*": {
                "a|*": 0.8,
            }
        }
        nested_entries = list(AgeStructuredPopulationBuilder._iter_sexual_selection_entries(nested))
        self.assertEqual(nested_entries, [("A|*", "a|*", 0.8)])

        flat = {
            "a|*": 0.7,
        }
        flat_entries = list(AgeStructuredPopulationBuilder._iter_sexual_selection_entries(flat))
        self.assertEqual(flat_entries, [("*", "a|*", 0.7)])

    def test_sex_label_to_index_mapping_for_viability(self) -> None:
        self.assertEqual(resolve_sex_label("female"), 0)
        self.assertEqual(resolve_sex_label("f"), 0)
        self.assertEqual(resolve_sex_label("male"), 1)
        self.assertEqual(resolve_sex_label("m"), 1)

        with self.assertRaises(ValueError):
            resolve_sex_label("unknown")

    def test_resolve_genotype_selector_invalid_or_empty_match_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.simple_species.resolve_genotype_selectors(
                selector="NotAValidPattern(",
                all_genotypes=self.all_genotypes,
                context="sexual_selection",
            )

        with self.assertRaises(ValueError):
            self.simple_species.resolve_genotype_selectors(
                selector="B|*",
                all_genotypes=self.all_genotypes,
                context="sexual_selection",
            )

    def test_fitness_is_applied_to_config_setters_during_build(self) -> None:
        class _FakeConfig:
            def __init__(self) -> None:
                self.viability_calls = []
                self.fecundity_calls = []
                self.sexual_selection_calls = []

            def set_viability_fitness(self, sex_idx: int, genotype_idx: int, value: float) -> None:
                self.viability_calls.append((sex_idx, genotype_idx, float(value)))

            def set_fecundity_fitness(self, sex_idx: int, genotype_idx: int, value: float) -> None:
                self.fecundity_calls.append((sex_idx, genotype_idx, float(value)))

            def set_sexual_selection_fitness(self, female_idx: int, male_idx: int, value: float) -> None:
                self.sexual_selection_calls.append((female_idx, male_idx, float(value)))

        class _FakeIndexCore:
            def __init__(self, genotypes) -> None:
                self.genotype_to_index = {gt: i for i, gt in enumerate(genotypes)}

        class _FakePopulation:
            def __init__(self, species, population_config, name=None, initial_individual_count=None, initial_sperm_storage=None, hooks=None) -> None:
                self.species = species
                self._config = population_config
                self._registry = _FakeIndexCore(species.get_all_genotypes())
                self._index_registry = self._registry

            def apply_recipe(self, recipe) -> None:
                return None
            
            @property
            def new_adult_age(self):
                return 2

        fake_config = _FakeConfig()

        builder = AgeStructuredPopulationBuilder(self.simple_species)
        builder.initial_state(individual_count={})
        builder.fitness(
            viability={"A|a": {"female": 0.25, "male": 0.75}},
            fecundity={"A|a": 0.5},
            sexual_selection={"A|a": {"a|A": 0.33}},
        )

        with patch("natal.population_builder.PopulationConfigBuilder.build", return_value=fake_config):
            with patch("natal.age_structured_population.AgeStructuredPopulation", _FakePopulation):
                pop = builder.build()

        self.assertIsNotNone(pop)

        genotype_a_bar_a = self.simple_species.get_genotype_from_str("A|a")
        genotype_a_lower_bar_a_upper = self.simple_species.get_genotype_from_str("a|A")
        idx_a_bar_a = pop._registry.genotype_to_index[genotype_a_bar_a]
        idx_a_lower_bar_a_upper = pop._registry.genotype_to_index[genotype_a_lower_bar_a_upper]

        self.assertIn((0, idx_a_bar_a, 0.25), fake_config.viability_calls)
        self.assertIn((1, idx_a_bar_a, 0.75), fake_config.viability_calls)

        self.assertIn((0, idx_a_bar_a, 0.5), fake_config.fecundity_calls)
        self.assertIn((1, idx_a_bar_a, 0.5), fake_config.fecundity_calls)

        self.assertIn((idx_a_bar_a, idx_a_lower_bar_a_upper, 0.33), fake_config.sexual_selection_calls)


if __name__ == "__main__":
    unittest.main()
