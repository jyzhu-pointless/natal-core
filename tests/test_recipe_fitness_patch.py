from __future__ import annotations

import unittest
import uuid

import numpy as np

from natal.frontend.presets import apply_preset_fitness_patch
from natal.frontend.genetics import Species


class _FakeConfig:
    def __init__(self, n_genotypes: int) -> None:
        self.new_adult_age = 1
        self.viability_fitness = np.ones((2, 1, n_genotypes), dtype=np.float64)
        self.fecundity_fitness = np.ones((2, n_genotypes), dtype=np.float64)
        self.zygote_viability_fitness = np.ones((2, n_genotypes), dtype=np.float64)
        self.sexual_selection_fitness = np.ones((n_genotypes, n_genotypes), dtype=np.float64)

    def set_viability_fitness(self, sex_idx: int, genotype_idx: int, value: float, age: int = -1) -> None:
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex_idx][age][genotype_idx] = float(value)

    def set_fecundity_fitness(self, sex_idx: int, genotype_idx: int, value: float) -> None:
        self.fecundity_fitness[sex_idx][genotype_idx] = float(value)

    def set_zygote_viability_fitness(self, sex_idx: int, genotype_idx: int, value: float) -> None:
        self.zygote_viability_fitness[sex_idx][genotype_idx] = float(value)

    def set_sexual_selection_fitness(self, female_idx: int, male_idx: int, value: float) -> None:
        self.sexual_selection_fitness[female_idx][male_idx] = float(value)


class _FakeIndexCore:
    def __init__(self, genotypes) -> None:
        self._genotypes = list(genotypes)
        self.genotype_to_index = {gt: i for i, gt in enumerate(self._genotypes)}

    @property
    def index_to_genotype(self) -> list:
        # Match real IndexRegistry: unique genotypes in registration order
        return list(self.genotype_to_index.keys())

    def ztype_indices_for(self, genotype) -> list[int]:
        # In the single-slab case, genotype index = ZType index
        idx = self.genotype_to_index.get(genotype)
        return [idx] if idx is not None else []


class _FakePopulation:
    def __init__(self, species: Species) -> None:
        self.species = species
        all_genotypes = species.get_all_genotypes()
        self._index_registry = _FakeIndexCore(all_genotypes)
        self._config = _FakeConfig(len(all_genotypes))

    @property
    def index_registry(self):
        return self._index_registry

    @property
    def config(self):
        return self._config



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

    def test_viability_per_allele_scaling_is_multiplicative_by_copy_number(self) -> None:
        patch = {
            "viability_per_allele": {
                "Drive": 0.8,
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_wt], 0.8)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_drive], 0.64)

    def test_viability_per_allele_dominant_mode(self) -> None:
        patch = {
            "viability_per_allele": {
                "Drive": (0.8, "dominant"),
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_wt], 0.8)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_drive], 0.8)

    def test_fecundity_per_allele_scaling_is_multiplicative_by_copy_number(self) -> None:
        patch = {
            "fecundity_per_allele": {
                "Drive": 0.5,
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_drive_wt], 0.5)
        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_drive_drive], 0.25)

    def test_fecundity_per_allele_custom_mode(self) -> None:
        patch = {
            "fecundity_per_allele": {
                "Drive": ((0.6, 0.3), "custom"),
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_drive_wt], 0.6)
        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][idx_drive_drive], 0.3)

    def test_sexual_selection_per_allele_recessive_mode(self) -> None:
        patch = {
            "sexual_selection_per_allele": {
                "Drive": (0.4, "recessive"),
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        f_idx = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        m_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        m_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        m_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.sexual_selection_fitness[f_idx][m_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.sexual_selection_fitness[f_idx][m_drive_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.sexual_selection_fitness[f_idx][m_drive_drive], 0.4)

    def test_zygote_per_allele_scaling_is_multiplicative_by_copy_number(self) -> None:
        patch = {
            "zygote_per_allele": {
                "Drive": 0.5,
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_wt], 0.5)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_drive], 0.25)

    def test_zygote_per_allele_dominant_mode(self) -> None:
        patch = {
            "zygote_per_allele": {
                "Drive": (0.7, "dominant"),
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_wt], 0.7)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_drive], 0.7)

    def test_zygote_per_allele_recessive_mode(self) -> None:
        patch = {
            "zygote_per_allele": {
                "Drive": (0.3, "recessive"),
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_drive], 0.3)

    def test_zygote_per_allele_custom_mode(self) -> None:
        patch = {
            "zygote_per_allele": {
                "Drive": ((0.6, 0.2), "custom"),
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_wt], 0.6)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_drive], 0.2)

    def test_zygote_per_allele_sex_specific_scaling(self) -> None:
        patch = {
            "zygote_per_allele": {
                "Drive": ({"female": 0.8, "male": 0.5}, "multiplicative"),
            }
        }
        apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_wt], 0.8)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[0][idx_drive_drive], 0.64)

        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[1][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[1][idx_drive_wt], 0.5)
        self.assertAlmostEqual(self.pop._config.zygote_viability_fitness[1][idx_drive_drive], 0.25)


    def test_import_from_canonical_path(self) -> None:
        """apply_preset_fitness_patch is importable from natal.frontend.fitness._patch."""
        from natal.frontend.fitness._patch import apply_preset_fitness_patch as _patch_fn

        patch = {
            "viability_per_allele": {
                "Drive": 0.8,
            }
        }
        _patch_fn(self.pop, patch)

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_wt], 0.8)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_drive], 0.64)

    def test_with_minimal_fitness_population_view(self) -> None:
        """Works with a minimal object satisfying the RecipeHost protocol.

        The protocol requires only three attributes (config, species,
        index_registry) — no Population or live session needed.
        """
        from natal.frontend.fitness._patch import apply_preset_fitness_patch as _patch_fn

        class _MinimalView:
            __slots__ = ('config', 'species', 'index_registry')
            def __init__(self, config, species, index_registry):
                self.config = config
                self.species = species
                self.index_registry = index_registry

        view = _MinimalView(self.pop._config, self.pop.species, self.pop._index_registry)
        patch = {
            "viability_per_allele": {
                "Drive": 0.5,
            }
        }
        _patch_fn(view, patch)

        idx_wt_wt = self.pop._index_registry.genotype_to_index[self.gt_wt_wt]
        idx_drive_wt = self.pop._index_registry.genotype_to_index[self.gt_drive_wt]
        idx_drive_drive = self.pop._index_registry.genotype_to_index[self.gt_drive_drive]

        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_wt_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_wt], 0.5)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][idx_drive_drive], 0.25)


class TestPresetFitnessPatchKeyValidation(unittest.TestCase):
    """Unknown top-level patch keys are rejected before any tensor is written."""

    def setUp(self) -> None:
        self.species = _make_species()
        self.pop = _FakePopulation(self.species)
        self.idx_drive_wt = self.pop._index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive|WT")
        ]

    def test_unknown_key_only_is_rejected_and_reports_allowed_keys(self) -> None:
        """A patch made only of unknown keys raises and names the supported set."""
        # "viability_allele" is the historical misspelling that once shipped in a
        # docstring example; it used to be skipped silently.
        with self.assertRaises(ValueError) as ctx:
            apply_preset_fitness_patch(self.pop, {"viability_allele": {"Drive": 0.8}})  # type: ignore

        message = str(ctx.exception)
        self.assertIn("'viability_allele'", message)
        self.assertIn("viability_per_allele", message)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][self.idx_drive_wt], 1.0)

    def test_unknown_key_rejects_the_whole_patch_before_any_write(self) -> None:
        """A mixed patch writes nothing: key validation precedes every setter."""
        patch = {
            "viability_per_allele": {"Drive": 0.5},
            "not_a_patch_key": {"Drive": 0.1},
        }
        with self.assertRaises(ValueError) as ctx:
            apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        self.assertIn("'not_a_patch_key'", str(ctx.exception))
        # The legal entry in the same patch must not have been applied.
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][self.idx_drive_wt], 1.0)

    def test_uncomparable_unknown_keys_still_report_a_value_error(self) -> None:
        """Reporting the bad keys must not depend on them being sortable.

        An int and a tuple cannot be ordered against each other, so a plain
        ``sorted`` over the unknown keys would raise a sort ``TypeError``
        instead of the promised ``ValueError``.
        """
        with self.assertRaises(ValueError) as ctx:
            apply_preset_fitness_patch(self.pop, {1: {}, (2,): {}})  # type: ignore

        message = str(ctx.exception)
        self.assertIn("1", message)
        self.assertIn("(2,)", message)
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][self.idx_drive_wt], 1.0)
        self.assertAlmostEqual(self.pop._config.fecundity_fitness[0][self.idx_drive_wt], 1.0)

    def test_unknown_key_leaves_every_tensor_family_unwritten(self) -> None:
        """The key check precedes every setter, not only the first tensor family.

        The patch carries one legal entry for each of the four tensor
        families plus an unknown key; a per-branch key check (or a check
        placed after the first writer) would already have rewritten the
        tensors whose entries come first.
        """
        fields = (
            "viability_fitness",
            "fecundity_fitness",
            "sexual_selection_fitness",
            "zygote_viability_fitness",
        )
        before = {name: getattr(self.pop._config, name).copy() for name in fields}
        patch = {
            "viability": {"WT|WT": 0.5},
            "fecundity": {"WT|WT": 0.5},
            "sexual_selection": {"WT|WT": 0.5},
            "zygote": {"WT|WT": 0.5},
            "viability_per_allele": {"Drive": 0.5},
            "fecundity_per_allele": {"Drive": 0.5},
            "sexual_selection_per_allele": {"Drive": 0.5},
            "zygote_per_allele": {"Drive": 0.5},
            "unknown_top_level_key": {},
        }

        with self.assertRaises(ValueError):
            apply_preset_fitness_patch(self.pop, patch)  # type: ignore

        # Every tensor (and every dtype/shape) is byte-identical: nothing was
        # written before the rejection.
        for name in fields:
            np.testing.assert_array_equal(
                getattr(self.pop._config, name),
                before[name],
                err_msg=f"{name} was modified before the key check",
            )

    def test_empty_patch_is_accepted_without_writes(self) -> None:
        """The legal empty patch stays a no-op."""
        apply_preset_fitness_patch(self.pop, {})  # type: ignore
        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][self.idx_drive_wt], 1.0)

    def test_every_supported_key_is_accepted(self) -> None:
        """Each advertised key is recognised (empty value = no-op).

        Guards against the key set drifting away from what the function
        actually tolerates.
        """
        from natal.frontend.fitness._patch import SUPPORTED_PRESET_FITNESS_PATCH_KEYS

        for key in sorted(SUPPORTED_PRESET_FITNESS_PATCH_KEYS):
            with self.subTest(key=key):
                apply_preset_fitness_patch(self.pop, {key: {}})  # type: ignore

        self.assertAlmostEqual(self.pop._config.viability_fitness[0][0][self.idx_drive_wt], 1.0)


if __name__ == "__main__":
    unittest.main()
