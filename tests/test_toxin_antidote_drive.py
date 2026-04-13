from __future__ import annotations

import unittest
import uuid

from natal.genetic_presets import ToxinAntidoteDrive
from natal.genetic_structures import Species
from natal.index_registry import compress_hg_glab
from natal.population_builder import DiscreteGenerationPopulationBuilder


class TestToxinAntidoteDriveFitnessPatch(unittest.TestCase):
    def test_default_patch_has_no_sexual_selection_effect(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_Default",
            drive_allele="Drive",
            target_allele="WT",
            disrupted_allele="Disrupted",
        )

        patch = preset.fitness_patch()

        self.assertEqual(
            patch["viability_per_allele"],
            {"Disrupted": (0.0, "recessive")},
        )
        self.assertEqual(
            patch["fecundity_per_allele"],
            {"Disrupted": (1.0, "recessive")},
        )
        self.assertNotIn("sexual_selection_per_allele", patch)

    def test_tuple_sexual_selection_is_exported_in_patch(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_TupleSexSel",
            drive_allele="Drive",
            target_allele="WT",
            disrupted_allele="Disrupted",
            sexual_selection_scaling=(1.0, 0.7),
            sexual_selection_mode="dominant",
        )

        patch = preset.fitness_patch()

        self.assertEqual(
            patch["sexual_selection_per_allele"],
            {"Disrupted": ((1.0, 0.7), "dominant")},
        )

    def test_scalar_sexual_selection_mode_is_exported_in_patch(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_ScalarSexSel",
            drive_allele="Drive",
            target_allele="WT",
            disrupted_allele="Disrupted",
            sexual_selection_scaling=0.85,
            sexual_selection_mode="dominant",
        )

        patch = preset.fitness_patch()

        self.assertEqual(
            patch["sexual_selection_per_allele"],
            {"Disrupted": (0.85, "dominant")},
        )


def _make_species(prefix: str = "TADriveSpecies") -> Species:
    return Species.from_dict(
        f"{prefix}_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT", "Drive", "Disrupted"],
            }
        },
    )


def _make_species_cross_locus(prefix: str = "TADriveCrossLocusSpecies") -> Species:
    return Species.from_dict(
        f"{prefix}_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT1", "Drive"],
                "L2": ["WT2", "Disrupted"],
            }
        },
    )


def _make_species_cross_chromosome(prefix: str = "TADriveCrossChromSpecies") -> Species:
    return Species.from_dict(
        f"{prefix}_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT1", "Drive"],
            },
            "Chr2": {
                "L2": ["WT2", "Disrupted"],
            },
        },
    )


def _build_population(species: Species):
    return (
        DiscreteGenerationPopulationBuilder(species)
        .setup(name="TADrivePop", stochastic=False)
        .initial_state(
            {
                "female": {"Drive|WT": 20},
                "male": {"Drive|WT": 20},
            }
        )
        .build()
    )


def _build_population_cross_locus(species: Species):
    return (
        DiscreteGenerationPopulationBuilder(species)
        .setup(name="TADriveCrossLocusPop", stochastic=False)
        .initial_state(
            {
                "female": {"Drive/WT2|WT1/WT2": 20},
                "male": {"Drive/WT2|WT1/WT2": 20},
            }
        )
        .build()
    )


def _build_population_cross_chromosome(species: Species):
    return (
        DiscreteGenerationPopulationBuilder(species)
        .setup(name="TADriveCrossChromPop", stochastic=False)
        .initial_state(
            {
                "female": {"Drive|WT1;WT2|WT2": 20},
                "male": {"Drive|WT1;WT2|WT2": 20},
            }
        )
        .build()
    )


class TestToxinAntidoteDriveConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.species = _make_species()
        self.population = _build_population(self.species)
        self.hg_wt = self.species.get_haploid_genotype_from_str("WT")
        self.hg_drive = self.species.get_haploid_genotype_from_str("Drive")
        self.hg_disrupted = self.species.get_haploid_genotype_from_str("Disrupted")

    def test_gamete_conversion_female_rate_one_replaces_target_allele(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_Gamete",
            drive_allele="Drive",
            target_allele="WT",
            disrupted_allele="Disrupted",
            conversion_rate={"female": 1.0, "male": 0.0},
            embryo_disruption_rate=0.0,
        )
        preset.bind_species(self.species)

        modifier = preset.gamete_modifier(self.population)
        self.assertIsNotNone(modifier)
        if modifier is None:
            self.fail("Expected non-empty gamete modifier")

        updates = modifier()

        gt_idx = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive|WT")
        ]
        self.assertIn((0, gt_idx), updates)

        converted = updates[(0, gt_idx)]
        n_glabs = int(self.population.config.n_glabs)
        hg_freq = {}
        for cidx, freq in converted.items():
            hg_idx = cidx // n_glabs
            hg = self.population.registry.index_to_haplo[hg_idx]
            hg_freq[hg] = hg_freq.get(hg, 0.0) + float(freq)

        self.assertAlmostEqual(hg_freq.get(self.hg_drive, 0.0), 0.5)
        self.assertAlmostEqual(hg_freq.get(self.hg_disrupted, 0.0), 0.5)
        self.assertAlmostEqual(hg_freq.get(self.hg_wt, 0.0), 0.0)

    def test_zygote_conversion_embryo_rate_one_converts_drive_wt_to_drive_disrupted(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_Zygote",
            drive_allele="Drive",
            target_allele="WT",
            disrupted_allele="Disrupted",
            conversion_rate=0.0,
            embryo_disruption_rate={"female": 1.0, "male": 0.0},
        )
        preset.bind_species(self.species)

        modifier = preset.zygote_modifier(self.population)
        self.assertIsNotNone(modifier)
        if modifier is None:
            self.fail("Expected non-empty zygote modifier")

        updates = modifier()

        n_glabs = int(self.population.config.n_glabs)
        drive_idx = self.population.index_registry.haplo_to_index[self.hg_drive]
        wt_idx = self.population.index_registry.haplo_to_index[self.hg_wt]
        c_drive = compress_hg_glab(drive_idx, 0, n_glabs)
        c_wt = compress_hg_glab(wt_idx, 0, n_glabs)

        self.assertIn((c_drive, c_wt), updates)

        dist = updates[(c_drive, c_wt)]
        drive_disrupted_idx = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive|Disrupted")
        ]

        self.assertEqual(len(dist), 1)
        self.assertAlmostEqual(dist.get(drive_disrupted_idx, 0.0), 1.0)


class TestToxinAntidoteDriveCrossLocusConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.species = _make_species_cross_locus()
        self.population = _build_population_cross_locus(self.species)
        self.hg_drive_wt2 = self.species.get_haploid_genotype_from_str("Drive/WT2")
        self.hg_wt1_wt2 = self.species.get_haploid_genotype_from_str("WT1/WT2")
        self.hg_drive_disrupted = self.species.get_haploid_genotype_from_str("Drive/Disrupted")
        self.hg_wt1_disrupted = self.species.get_haploid_genotype_from_str("WT1/Disrupted")

    def test_gamete_conversion_cross_locus_drive_and_target_loci_are_supported(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_CrossLocus_Gamete",
            drive_allele="Drive",
            target_allele="WT2",
            disrupted_allele="Disrupted",
            conversion_rate={"female": 1.0, "male": 0.0},
            embryo_disruption_rate=0.0,
        )
        preset.bind_species(self.species)

        modifier = preset.gamete_modifier(self.population)
        self.assertIsNotNone(modifier)
        if modifier is None:
            self.fail("Expected non-empty gamete modifier")

        updates = modifier()
        gt_idx = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive/WT2|WT1/WT2")
        ]
        self.assertIn((0, gt_idx), updates)

        converted = updates[(0, gt_idx)]
        n_glabs = int(self.population.config.n_glabs)
        hg_freq = {}
        for cidx, freq in converted.items():
            hg_idx = cidx // n_glabs
            hg = self.population.registry.index_to_haplo[hg_idx]
            hg_freq[hg] = hg_freq.get(hg, 0.0) + float(freq)

        # Drive is on L1, target/disrupted are on L2: conversion should still work.
        self.assertAlmostEqual(hg_freq.get(self.hg_drive_wt2, 0.0), 0.0)
        self.assertAlmostEqual(hg_freq.get(self.hg_wt1_wt2, 0.0), 0.0)
        self.assertAlmostEqual(hg_freq.get(self.hg_drive_disrupted, 0.0), 0.5)
        self.assertAlmostEqual(hg_freq.get(self.hg_wt1_disrupted, 0.0), 0.5)

    def test_zygote_conversion_cross_locus_produces_expected_three_outcomes(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_CrossLocus_Zygote",
            drive_allele="Drive",
            target_allele="WT2",
            disrupted_allele="Disrupted",
            conversion_rate=0.0,
            embryo_disruption_rate={"female": 0.5, "male": 0.0},
        )
        preset.bind_species(self.species)

        modifier = preset.zygote_modifier(self.population)
        self.assertIsNotNone(modifier)
        if modifier is None:
            self.fail("Expected non-empty zygote modifier")

        updates = modifier()

        n_glabs = int(self.population.config.n_glabs)
        drive_wt2_idx = self.population.index_registry.haplo_to_index[self.hg_drive_wt2]
        wt1_wt2_idx = self.population.index_registry.haplo_to_index[self.hg_wt1_wt2]
        c_drive_wt2 = compress_hg_glab(drive_wt2_idx, 0, n_glabs)
        c_wt1_wt2 = compress_hg_glab(wt1_wt2_idx, 0, n_glabs)

        self.assertIn((c_drive_wt2, c_wt1_wt2), updates)

        dist = updates[(c_drive_wt2, c_wt1_wt2)]
        idx_maternal_only = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive/Disrupted|WT1/WT2")
        ]
        idx_paternal_only = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive/WT2|WT1/Disrupted")
        ]
        idx_both = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive/Disrupted|WT1/Disrupted")
        ]

        self.assertGreater(dist.get(idx_maternal_only, 0.0), 0.0)
        self.assertGreater(dist.get(idx_paternal_only, 0.0), 0.0)
        self.assertGreater(dist.get(idx_both, 0.0), 0.0)


class TestToxinAntidoteDriveCrossChromosomeConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.species = _make_species_cross_chromosome()
        self.population = _build_population_cross_chromosome(self.species)
        self.hg_drive_wt2 = self.species.get_haploid_genotype_from_str("Drive;WT2")
        self.hg_wt1_wt2 = self.species.get_haploid_genotype_from_str("WT1;WT2")
        self.hg_drive_disrupted = self.species.get_haploid_genotype_from_str("Drive;Disrupted")
        self.hg_wt1_disrupted = self.species.get_haploid_genotype_from_str("WT1;Disrupted")

    def test_gamete_conversion_cross_chromosome_drive_and_target_are_supported(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_CrossChrom_Gamete",
            drive_allele="Drive",
            target_allele="WT2",
            disrupted_allele="Disrupted",
            conversion_rate={"female": 1.0, "male": 0.0},
            embryo_disruption_rate=0.0,
        )
        preset.bind_species(self.species)

        modifier = preset.gamete_modifier(self.population)
        self.assertIsNotNone(modifier)
        if modifier is None:
            self.fail("Expected non-empty gamete modifier")

        updates = modifier()
        gt_idx = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive|WT1;WT2|WT2")
        ]
        self.assertIn((0, gt_idx), updates)

        converted = updates[(0, gt_idx)]
        n_glabs = int(self.population.config.n_glabs)
        hg_freq = {}
        for cidx, freq in converted.items():
            hg_idx = cidx // n_glabs
            hg = self.population.registry.index_to_haplo[hg_idx]
            hg_freq[hg] = hg_freq.get(hg, 0.0) + float(freq)

        self.assertAlmostEqual(hg_freq.get(self.hg_drive_wt2, 0.0), 0.0)
        self.assertAlmostEqual(hg_freq.get(self.hg_wt1_wt2, 0.0), 0.0)
        self.assertAlmostEqual(hg_freq.get(self.hg_drive_disrupted, 0.0), 0.5)
        self.assertAlmostEqual(hg_freq.get(self.hg_wt1_disrupted, 0.0), 0.5)

    def test_zygote_conversion_cross_chromosome_produces_expected_three_outcomes(self) -> None:
        preset = ToxinAntidoteDrive(
            name="TA_CrossChrom_Zygote",
            drive_allele="Drive",
            target_allele="WT2",
            disrupted_allele="Disrupted",
            conversion_rate=0.0,
            embryo_disruption_rate={"female": 0.5, "male": 0.0},
        )
        preset.bind_species(self.species)

        modifier = preset.zygote_modifier(self.population)
        self.assertIsNotNone(modifier)
        if modifier is None:
            self.fail("Expected non-empty zygote modifier")

        updates = modifier()

        n_glabs = int(self.population.config.n_glabs)
        drive_wt2_idx = self.population.index_registry.haplo_to_index[self.hg_drive_wt2]
        wt1_wt2_idx = self.population.index_registry.haplo_to_index[self.hg_wt1_wt2]
        c_drive_wt2 = compress_hg_glab(drive_wt2_idx, 0, n_glabs)
        c_wt1_wt2 = compress_hg_glab(wt1_wt2_idx, 0, n_glabs)

        self.assertIn((c_drive_wt2, c_wt1_wt2), updates)

        dist = updates[(c_drive_wt2, c_wt1_wt2)]
        idx_maternal_only = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive|WT1;Disrupted|WT2")
        ]
        idx_paternal_only = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive|WT1;WT2|Disrupted")
        ]
        idx_both = self.population.index_registry.genotype_to_index[
            self.species.get_genotype_from_str("Drive|WT1;Disrupted|Disrupted")
        ]

        self.assertGreater(dist.get(idx_maternal_only, 0.0), 0.0)
        self.assertGreater(dist.get(idx_paternal_only, 0.0), 0.0)
        self.assertGreater(dist.get(idx_both, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
