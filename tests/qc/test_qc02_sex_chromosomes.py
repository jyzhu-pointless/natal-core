"""QC spot-checks 4-6: sex-chromosome (XY/ZW) transmission numerics.

Re-verifies the repaired CR-0 public-builder chain on the current HEAD
and extends coverage to the age-structured engine path, which the
regular suite pins only for the discrete-generation path.
"""

from __future__ import annotations

import pytest

import natal as nt


def _xy_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1", "X2"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
    )


def _zw_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrZ": {"sex_type": "Z", "loci": {"sz": ["Z1", "Z2"]}},
            "chrW": {"sex_type": "W", "loci": {"sw": ["W1"]}},
        },
        unordered=False,
    )


def _carries(genotype: nt.Genotype, chromosome_name: str) -> bool:
    chrom = genotype.species.get_chromosome(chromosome_name)
    for side in (genotype.maternal, genotype.paternal):
        try:
            side.get_haplotype_for_chromosome(chrom)
            return True
        except ValueError:
            continue
    return False


class TestXyDiscreteRegression:
    def test_public_builder_xy_exact(self) -> None:
        """Claim: XY offspring sex follows genotype, not a coin flip.

        Reference: 1000 XX females + 1000 XY males, eggs=1 -> 1000 zygotes
        split 500/500; per sex the autosomal cross is 1:2:1.  Y-bearing
        ztypes must be male-only, X-only ztypes female-only.  Rejects a
        regression of the CR-0 chain (empty genotype strings, missing sex
        masks, first-ztype fallback).
        """
        species = _xy_species("qc_xy_disc")
        female = species.get_genotype_from_str("A|a;X1|X2")
        male = species.get_genotype_from_str("A|a;X1|Y1")
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=species, name="qc_xy_disc_pop", stochastic=False
            )
            .initial_state(
                individual_count={"female": {female: 1000}, "male": {male: 1000}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=1, sex_ratio=0.5)
            .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                         growth_mode="no_competition")
            .build()
        )
        assert pop.config.has_sex_chromosomes is True
        pop.run(1)

        counts = pop.state.individual_count
        assert float(counts.sum()) == 1000.0
        assert float(counts[0].sum()) == 500.0
        assert float(counts[1].sum()) == 500.0

        registry = pop.registry
        per_sex_autosomal: dict[tuple[int, str], float] = {}
        for genotype, slab in registry.index_to_ztype:
            idx = registry.ztype_index(genotype, slab)
            total = float(counts[:, :, idx].sum())
            is_male = _carries(genotype, "chrY")
            female_mass = float(counts[0, :, idx].sum())
            male_mass = float(counts[1, :, idx].sum())
            if is_male:
                assert female_mass == 0.0, genotype.to_string()
            else:
                assert male_mass == 0.0, genotype.to_string()
            key = (1 if is_male else 0, genotype.to_string().split(";")[0])
            per_sex_autosomal[key] = per_sex_autosomal.get(key, 0.0) + total
        for sex in (0, 1):
            assert per_sex_autosomal[(sex, "A|A")] == 125.0
            assert per_sex_autosomal[(sex, "A|a")] == 125.0
            assert per_sex_autosomal[(sex, "a|A")] == 125.0
            assert per_sex_autosomal[(sex, "a|a")] == 125.0


class TestXyAgeStructured:
    def test_generational_turnover_respects_genetic_sex(self) -> None:
        """Claim: age-structured XY offspring sex follows the genotype —
        Y-bearing ztypes are male-only and sit in the male axis at full
        mass; X-only ztypes are female-only.

        One reproducing cohort, eggs=2, deterministic, growth_mode=fixed
        -> newborn cohort is exactly 1200: 600 female (X1|X1, X2|X1) +
        600 male (X1|Y1, X2|Y1), 300 per ztype.

        FIXED (was: confirmed defect).  Root cause was the PYTHON builder,
        not the kernel: PopulationBuilder.age_structure() rebuilt the
        population config without forwarding female_only_by_sex_chrom /
        male_only_by_sex_chrom, so the draft fell into the compatibility
        heuristic that cannot distinguish homo- from heterogametic ztypes
        and emitted all-False masks; the kernel then assigned sex from the
        compatibility ratio (~150 males inside the female ztype X1|X1).
        age_structure() now forwards the blueprint masks, and assembly()
        rejects the mask-less combination outright.  The discrete path was
        never affected, which is why the regular CR-0 suite missed it.

        Only the four zygote types reachable from these homozygous parents
        are asserted at 300; unreachable catalog entries must stay empty.
        """
        species = _xy_species("qc_xy_age")
        female = species.get_genotype_from_str("A|A;X1|X2")
        male = species.get_genotype_from_str("A|A;X1|Y1")
        pop = (
            nt.AgeStructuredPopulation.setup(
                species=species, name="qc_xy_age_pop", stochastic=False
            )
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {female: {1: 600}},
                    "male": {male: {1: 600}},
                }
            )
            .survival(
                female_age_based_survival=[1.0, 0.0],
                male_age_based_survival=[1.0, 0.0],
            )
            .reproduction(
                female_age_based_mating_rate=[0.0, 1.0],
                male_age_based_mating_rate=[0.0, 1.0],
                eggs_per_female=2,
                sex_ratio=0.5,
            )
            .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                         growth_mode="fixed")
            .build()
        )
        assert pop.config.has_sex_chromosomes is True
        pop.run(1)

        counts = pop.state.individual_count
        assert float(counts.sum()) == 1200.0
        assert float(counts[0].sum()) == 600.0
        assert float(counts[1].sum()) == 600.0
        registry = pop.registry
        wrong_axis_mass = 0.0
        reachable = 0
        for genotype, slab in registry.index_to_ztype:
            idx = registry.ztype_index(genotype, slab)
            is_male = _carries(genotype, "chrY")
            female_mass = float(counts[0, :, idx].sum())
            male_mass = float(counts[1, :, idx].sum())
            wrong_axis_mass += female_mass if is_male else male_mass
            assert (female_mass if is_male else male_mass) == 0.0, (
                f"{genotype.to_string()}: sex-fixed ztype gained mass on the "
                f"wrong sex axis (female={female_mass} male={male_mass})"
            )
            if female_mass + male_mass > 0.0:
                # Only the four zygote types reachable from these homozygous
                # parents carry mass; the rest of the catalog stays empty.
                reachable += 1
                assert (male_mass if is_male else female_mass) == 300.0
        assert wrong_axis_mass == 0.0
        assert reachable == 4


class TestZwAgeStructured:
    def test_generational_turnover_respects_genetic_sex(self) -> None:
        """Claim: ZW mirrors XY — W-bearing ztypes are female-only.

        FIXED: the age-structure builder used to drop the sex-chromosome
        masks (see the XY docstring), so the male ZZ ztype held ~150
        females.  It now forwards the blueprint masks.
        """
        species = _zw_species("qc_zw_age")
        female = species.get_genotype_from_str("A|A;W1|Z1")
        male = species.get_genotype_from_str("A|A;Z1|Z2")
        pop = (
            nt.AgeStructuredPopulation.setup(
                species=species, name="qc_zw_age_pop", stochastic=False
            )
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {female: {1: 600}},
                    "male": {male: {1: 600}},
                }
            )
            .survival(
                female_age_based_survival=[1.0, 0.0],
                male_age_based_survival=[1.0, 0.0],
            )
            .reproduction(
                female_age_based_mating_rate=[0.0, 1.0],
                male_age_based_mating_rate=[0.0, 1.0],
                eggs_per_female=2,
                sex_ratio=0.5,
            )
            .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                         growth_mode="fixed")
            .build()
        )
        pop.run(1)
        counts = pop.state.individual_count
        assert float(counts.sum()) == 1200.0
        assert float(counts[0].sum()) == 600.0
        assert float(counts[1].sum()) == 600.0
        for genotype, slab in pop.registry.index_to_ztype:
            idx = pop.registry.ztype_index(genotype, slab)
            is_female = _carries(genotype, "chrW")
            female_mass = float(counts[0, :, idx].sum())
            male_mass = float(counts[1, :, idx].sum())
            assert (male_mass if is_female else female_mass) == 0.0, (
                f"{genotype.to_string()}: sex-fixed ztype gained mass on the "
                f"wrong sex axis (female={female_mass} male={male_mass})"
            )
            if female_mass + male_mass > 0.0:
                # Four reachable zygote types, every one of them 300 on its own
                # sex axis; the unreachable catalog entries stay at zero.
                assert (female_mass if is_female else male_mass) == 300.0
