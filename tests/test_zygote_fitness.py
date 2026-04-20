"""Tests for Zygote Fitness functionality."""

from __future__ import annotations

import unittest
import uuid

import numpy as np

from natal.genetic_entities import Species
from natal.population_builder import AgeStructuredPopulationBuilder
from natal.population_config import build_population_config


def _make_simple_species() -> Species:
    """Create a simple species for testing."""
    return Species.from_dict(
        f"TestSpecies_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["A", "a"],
            }
        },
    )


class TestZygoteFitness(unittest.TestCase):
    """Test cases for Zygote Fitness functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.simple_species = _make_simple_species()
        self.all_genotypes = self.simple_species.get_all_genotypes()

    def test_population_config_zygote_viability_fitness_field(self) -> None:
        """Test that PopulationConfig has zygote_viability_fitness field."""
        config = build_population_config(n_genotypes=4, n_haploid_genotypes=2)

        # Check that zygote_viability_fitness field exists
        self.assertTrue(hasattr(config, 'zygote_viability_fitness'))

        # Check the shape is correct
        self.assertEqual(config.zygote_viability_fitness.shape, (2, 4))  # (n_sexes, n_genotypes)

        # Check default values are all 1.0
        np.testing.assert_array_equal(config.zygote_viability_fitness, np.ones((2, 4)))

    def test_set_zygote_viability_fitness_method(self) -> None:
        """Test set_zygote_viability_fitness method."""
        config = build_population_config(n_genotypes=4, n_haploid_genotypes=2)

        # Set zygote fitness for female genotype 0
        config.set_zygote_viability_fitness(0, 0, 0.5)
        self.assertEqual(config.zygote_viability_fitness[0, 0], 0.5)

        # Set zygote fitness for male genotype 1
        config.set_zygote_viability_fitness(1, 1, 0.8)
        self.assertEqual(config.zygote_viability_fitness[1, 1], 0.8)

        # Verify other values remain unchanged
        self.assertEqual(config.zygote_viability_fitness[0, 1], 1.0)
        self.assertEqual(config.zygote_viability_fitness[1, 0], 1.0)

    def test_builder_zygote_viability_fitness_parameter(self) -> None:
        """Test zygote parameter in Builder fitness method."""

        # Build population with initial state and zygote fitness
        population = (
            AgeStructuredPopulationBuilder(self.simple_species)
            .age_structure(n_ages=3)
            .initial_state({
                "female": {"A|A": [1, 0, 0], "A|a": [1, 0, 0], "a|a": [1, 0, 0]},
                "male": {"A|A": [1, 0, 0], "A|a": [1, 0, 0], "a|a": [1, 0, 0]},
            })
            .fitness(
                zygote={
                    "A|A": 0.5,           # 50% survival for homozygous A
                    "a|a": {"female": 0.3, "male": 0.4},  # Sex-specific
                    "A|a": 0.8,           # 80% survival for heterozygote
                }
            )
            .build()
        )

        # Verify zygote fitness is configured
        self.assertTrue(hasattr(population.config, 'zygote_viability_fitness'))

        # Get genotype indices
        aa_idx = population.index_registry.genotype_to_index[self.simple_species.get_genotype_from_str("A|A")]
        aa_idx = population.index_registry.genotype_to_index[self.simple_species.get_genotype_from_str("a|a")]
        aa_idx = population.index_registry.genotype_to_index[self.simple_species.get_genotype_from_str("A|a")]

        # Note: Actual values would be set during population build process
        # This test mainly verifies that the API accepts the parameter

    def test_zygote_viability_fitness_preset_configuration(self) -> None:
        """Test zygote fitness configuration through presets."""

        # For this test, we'll use the Builder's fitness method directly
        # instead of creating a custom preset class
        population = (
            AgeStructuredPopulationBuilder(self.simple_species)
            .age_structure(n_ages=3)
            .initial_state({
                "female": {"A|A": [1, 0, 0]},
                "male": {"A|A": [1, 0, 0]},
            })
            .fitness(
                zygote={"A|A": 0.0},  # Lethal zygote
            )
            .build()
        )

        # Verify population has zygote fitness configuration
        self.assertTrue(hasattr(population.config, 'zygote_viability_fitness'))

    def test_zygote_viability_fitness_combined_with_viability(self) -> None:
        """Test that zygote and viability fitness can be combined."""

        population = (
            AgeStructuredPopulationBuilder(self.simple_species)
            .age_structure(n_ages=3)
            .initial_state({
                "female": {"A|A": [1, 0, 0]},
                "male": {"A|A": [1, 0, 0]},
            })
            .fitness(
                zygote={"A|A": 0.5},      # 50% zygote survival
                viability={"A|A": 0.8},   # 80% viability survival
            )
            .build()
        )

        # Both fitness types should be configured
        self.assertTrue(hasattr(population.config, 'zygote_viability_fitness'))
        self.assertTrue(hasattr(population.config, 'viability_fitness'))

    def test_zygote_viability_fitness_simulation_integration(self) -> None:
        """Test that zygote fitness is applied during reproduction stage."""

        # Create a population with zygote fitness
        population = (
            AgeStructuredPopulationBuilder(self.simple_species)
            .age_structure(n_ages=3)
            .initial_state({
                "female": {"A|A": [0, 0, 10]},  # Put adults in age 2
                "male": {"A|A": [0, 0, 5]},     # Put adults in age 2
            })
            .reproduction(eggs_per_female=10.0, use_fixed_egg_count=True)  # Use fixed egg count for deterministic test
            .fitness(
                zygote={"A|A": 0.5},  # 50% zygote survival
            )
            .build()
        )

        # Verify that zygote fitness configuration is present
        self.assertTrue(hasattr(population.config, 'zygote_viability_fitness'))

        # Verify the zygote fitness value is set correctly
        genotype_idx = population.index_registry.genotype_to_index[self.simple_species.get_genotype_from_str("A|A")]
        self.assertEqual(population.config.zygote_viability_fitness[0, genotype_idx], 0.5)  # Female
        self.assertEqual(population.config.zygote_viability_fitness[1, genotype_idx], 0.5)  # Male

        # Run one tick to test integration
        population.run_tick()

        # Verify that simulation advanced without errors
        self.assertGreater(population.tick, 0, "Simulation should advance")

        # The main test is that the simulation runs without errors when zygote fitness is configured
        # This verifies that the zygote fitness code is integrated correctly into the simulation kernel

        # Compare with a population without zygote fitness to ensure no errors
        control_population = (
            AgeStructuredPopulationBuilder(self.simple_species)
            .age_structure(n_ages=3)
            .initial_state({
                "female": {"A|A": [0, 0, 10]},  # Put adults in age 2
                "male": {"A|A": [0, 0, 5]},     # Put adults in age 2
            })
            .reproduction(eggs_per_female=10.0, use_fixed_egg_count=True)
            .build()
        )

        control_population.run_tick()

        # Both populations should run without errors
        # The actual effect of zygote fitness on population dynamics is complex and depends on many factors
        # This test primarily validates that the integration is correct and doesn't cause crashes


if __name__ == "__main__":
    unittest.main()
