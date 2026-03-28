from __future__ import annotations

import unittest
import uuid
import numpy as np

from natal.genetic_structures import Species
from natal.genetic_presets import _apply_preset_fitness_patch


class _FakeConfig:
    def __init__(self, n_genotypes: int) -> None:
        self.new_adult_age = 1
        self.viability_fitness = np.ones((2, 1, n_genotypes), dtype=np.float64)
        self.fecundity_fitness = np.ones((2, n_genotypes), dtype=np.float64)
        self.sexual_selection_fitness = np.ones((n_genotypes, n_genotypes), dtype=np.float64)

    def set_viability_fitness(self, sex_idx: int, genotype_idx: int, value: float, age: int = -1) -> None:
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex_idx][age][genotype_idx] = float(value)

    def set_fecundity_fitness(self, sex_idx: int, genotype_idx: int, value: float) -> None:
        self.fecundity_fitness[sex_idx][genotype_idx] = float(value)

    def set_sexual_selection_fitness(self, female_idx: int, male_idx: int, value: float) -> None:
        self.sexual_selection_fitness[female_idx][male_idx] = float(value)


class _FakeIndexCore:
    def __init__(self, genotypes) -> None:
        self.genotype_to_index = {gt: i for i, gt in enumerate(genotypes)}


class _FakePopulation:
    def __init__(self, species: Species) -> None:
        self.species = species
        all_genotypes = species.get_all_genotypes()
        self._index_registry = _FakeIndexCore(all_genotypes)
        self._config = _FakeConfig(len(all_genotypes))


def _make_species() -> Species:
    return Species.from_dict(
        f"PresetPatchSpecies_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT", "Drive"],
            }
        },
    )


class TestPresetFitnessPatch(unittest.TestCase):
    def setUp(self) -> None:
        self.species = _make_species()
        self.pop = _FakePopulation(self.species)
        self.gt_wt_wt = self.species.get_genotype_from_str("WT|WT")
        self.gt_drive_wt = self.species.get_genotype_from_str("Drive|WT")
        self.gt_drive_drive = self.species.get_genotype_from_str("Drive|Drive")

    def test_viability_allele_scaling_is_multiplicative_by_copy_number(self) -> None:
        patch = {
            "viability_allele": {
                "Drive": 0.8,
            }
        }
        _apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_wt], 0.8)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_drive], 0.64)

    def test_fecundity_allele_scaling_is_multiplicative_by_copy_number(self) -> None:
        patch = {
            "fecundity_allele": {
                "Drive": 0.5,
            }
        }
        _apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_drive_wt], 0.5)
        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_drive_drive], 0.25)


if __name__ == "__main__":
    unittest.main()
