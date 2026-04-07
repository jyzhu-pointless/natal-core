import numpy as np

import natal as nt
from natal.discrete_generation_population import DiscreteGenerationPopulation
from natal.population_config import (
    NO_COMPETITION,
    build_population_config,
    initialize_gamete_map,
    initialize_zygote_map,
)


def _has_chromosome(haploid: object, chromosome: object) -> bool:
    try:
        haploid.get_haplotype_for_chromosome(chromosome)
        return True
    except ValueError:
        return False


def _pick_xy_parents(species: nt.Species) -> tuple[object, object, object]:
    chrom_a = species.get_chromosome("chrA")
    chrom_x = species.get_chromosome("chrX")
    chrom_y = species.get_chromosome("chrY")
    assert chrom_a is not None
    assert chrom_x is not None
    assert chrom_y is not None

    locus_a = chrom_a.loci[0]

    female_parent = None
    male_parent = None
    for genotype in species.get_all_genotypes():
        maternal = genotype.maternal
        paternal = genotype.paternal

        maternal_has_x = _has_chromosome(maternal, chrom_x)
        maternal_has_y = _has_chromosome(maternal, chrom_y)
        paternal_has_x = _has_chromosome(paternal, chrom_x)
        paternal_has_y = _has_chromosome(paternal, chrom_y)

        allele_m = maternal.get_gene_at_locus(locus_a).name
        allele_p = paternal.get_gene_at_locus(locus_a).name
        is_heterozygous_autosome = {allele_m, allele_p} == {"A", "a"}

        if (
            is_heterozygous_autosome
            and maternal_has_x
            and not maternal_has_y
            and paternal_has_x
            and not paternal_has_y
        ):
            female_parent = genotype

        if (
            is_heterozygous_autosome
            and maternal_has_x
            and not maternal_has_y
            and not paternal_has_x
            and paternal_has_y
        ):
            male_parent = genotype

    assert female_parent is not None
    assert male_parent is not None
    return female_parent, male_parent, locus_a


def test_discrete_generation_xy_offspring_genotype_distribution_matches_mendelian() -> None:
    species = nt.Species.from_dict(
        name="DiscreteXYMendelian",
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y"]}},
        },
    )

    female_parent, male_parent, locus_a = _pick_xy_parents(species)

    diploid_genotypes = species.get_all_genotypes()
    haploid_genotypes = species.get_all_haploid_genotypes()

    gamete_map = initialize_gamete_map(
        haploid_genotypes=haploid_genotypes,
        diploid_genotypes=diploid_genotypes,
        n_glabs=1,
    )
    zygote_map = initialize_zygote_map(
        haploid_genotypes=haploid_genotypes,
        diploid_genotypes=diploid_genotypes,
        n_glabs=1,
    )

    config = build_population_config(
        n_genotypes=len(diploid_genotypes),
        n_haploid_genotypes=len(haploid_genotypes),
        n_sexes=2,
        n_ages=2,
        n_glabs=1,
        is_stochastic=False,
        use_continuous_sampling=False,
        age_based_survival_rates=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        age_based_mating_rates=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        female_age_based_relative_fertility=np.array([0.0, 1.0], dtype=np.float64),
        viability_fitness=np.ones((2, 2, len(diploid_genotypes)), dtype=np.float64),
        fecundity_fitness=np.ones((2, len(diploid_genotypes)), dtype=np.float64),
        sexual_selection_fitness=np.ones((len(diploid_genotypes), len(diploid_genotypes)), dtype=np.float64),
        age_based_relative_competition_strength=np.array([1.0, 1.0], dtype=np.float64),
        new_adult_age=1,
        expected_eggs_per_female=1.0,
        use_fixed_egg_count=True,
        carrying_capacity=1.0e12,
        sex_ratio=0.5,
        low_density_growth_rate=1.0,
        juvenile_growth_mode=NO_COMPETITION,
        has_sex_chromosomes=True,
        genotype_to_gametes_map=gamete_map,
        gametes_to_zygote_map=zygote_map,
    )

    parent_count = 1000.0
    pop = DiscreteGenerationPopulation(
        species=species,
        population_config=config,
        initial_individual_count={
            "female": {female_parent: parent_count},
            "male": {male_parent: parent_count},
        },
    )

    pop.run(1)
    state = pop._state.individual_count

    female_age1 = state[0, 1, :]
    male_age1 = state[1, 1, :]

    female_by_autosome: dict[str, float] = {"AA": 0.0, "Aa": 0.0, "aa": 0.0}
    male_by_autosome: dict[str, float] = {"AA": 0.0, "Aa": 0.0, "aa": 0.0}
    female_by_phase: dict[str, float] = {"A|A": 0.0, "A|a": 0.0, "a|A": 0.0, "a|a": 0.0}
    male_by_phase: dict[str, float] = {"A|A": 0.0, "A|a": 0.0, "a|A": 0.0, "a|a": 0.0}

    for idx, genotype in enumerate(pop._registry.index_to_genotype):
        allele_m = genotype.maternal.get_gene_at_locus(locus_a).name
        allele_p = genotype.paternal.get_gene_at_locus(locus_a).name
        autosome_key = "".join(sorted([allele_m, allele_p]))
        autosome_key = "Aa" if autosome_key == "Aa" else autosome_key
        phase_key = f"{allele_m}|{allele_p}"

        female_by_autosome[autosome_key] += float(female_age1[idx])
        male_by_autosome[autosome_key] += float(male_age1[idx])
        female_by_phase[phase_key] += float(female_age1[idx])
        male_by_phase[phase_key] += float(male_age1[idx])

    female_counts = np.array(
        [female_by_autosome["AA"], female_by_autosome["Aa"], female_by_autosome["aa"]],
        dtype=np.float64,
    )
    male_counts = np.array(
        [male_by_autosome["AA"], male_by_autosome["Aa"], male_by_autosome["aa"]],
        dtype=np.float64,
    )

    # In deterministic mode with fixed eggs and no recruitment loss,
    # offspring mass should be conserved at 1000 total (500 per sex).
    np.testing.assert_allclose(female_counts.sum(), 500.0)
    np.testing.assert_allclose(male_counts.sum(), 500.0)

    # Mendelian autosomal segregation: AA:Aa:aa = 1:2:1 for each sex.
    np.testing.assert_allclose(
        female_counts,
        np.array([125.0, 250.0, 125.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        male_counts,
        np.array([125.0, 250.0, 125.0], dtype=np.float64),
    )

    # Keep phased heterozygotes explicit: A|a and a|A are distinct genotypes,
    # each receiving half of the heterozygous mass under this symmetric cross.
    np.testing.assert_allclose(female_by_phase["A|a"], 125.0)
    np.testing.assert_allclose(female_by_phase["a|A"], 125.0)
    np.testing.assert_allclose(male_by_phase["A|a"], 125.0)
    np.testing.assert_allclose(male_by_phase["a|A"], 125.0)


def test_discrete_generation_x_linked_two_alleles_from_heterozygous_female() -> None:
    species = nt.Species.from_dict(
        name="DiscreteXLinkedTwoAlleles",
        structure={
            "chrA": {"loci": {"A": ["A"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1", "X2"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y"]}},
        },
    )

    chrom_x = species.get_chromosome("chrX")
    chrom_y = species.get_chromosome("chrY")
    assert chrom_x is not None
    assert chrom_y is not None
    locus_x = chrom_x.loci[0]

    female_parent = None
    male_parent = None
    for genotype in species.get_all_genotypes():
        maternal = genotype.maternal
        paternal = genotype.paternal

        maternal_has_x = _has_chromosome(maternal, chrom_x)
        maternal_has_y = _has_chromosome(maternal, chrom_y)
        paternal_has_x = _has_chromosome(paternal, chrom_x)
        paternal_has_y = _has_chromosome(paternal, chrom_y)

        x_m_gene = maternal.get_gene_at_locus(locus_x)
        x_p_gene = paternal.get_gene_at_locus(locus_x)
        x_m = "" if x_m_gene is None else x_m_gene.name
        x_p = "" if x_p_gene is None else x_p_gene.name

        if (
            maternal_has_x
            and not maternal_has_y
            and paternal_has_x
            and not paternal_has_y
            and {x_m, x_p} == {"X1", "X2"}
        ):
            female_parent = genotype

        if (
            maternal_has_x
            and not maternal_has_y
            and not paternal_has_x
            and paternal_has_y
            and x_m == "X1"
        ):
            male_parent = genotype

    assert female_parent is not None
    assert male_parent is not None

    diploid_genotypes = species.get_all_genotypes()
    haploid_genotypes = species.get_all_haploid_genotypes()

    gamete_map = initialize_gamete_map(
        haploid_genotypes=haploid_genotypes,
        diploid_genotypes=diploid_genotypes,
        n_glabs=1,
    )
    zygote_map = initialize_zygote_map(
        haploid_genotypes=haploid_genotypes,
        diploid_genotypes=diploid_genotypes,
        n_glabs=1,
    )

    config = build_population_config(
        n_genotypes=len(diploid_genotypes),
        n_haploid_genotypes=len(haploid_genotypes),
        n_sexes=2,
        n_ages=2,
        n_glabs=1,
        is_stochastic=False,
        use_continuous_sampling=False,
        age_based_survival_rates=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        age_based_mating_rates=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        female_age_based_relative_fertility=np.array([0.0, 1.0], dtype=np.float64),
        viability_fitness=np.ones((2, 2, len(diploid_genotypes)), dtype=np.float64),
        fecundity_fitness=np.ones((2, len(diploid_genotypes)), dtype=np.float64),
        sexual_selection_fitness=np.ones((len(diploid_genotypes), len(diploid_genotypes)), dtype=np.float64),
        age_based_relative_competition_strength=np.array([1.0, 1.0], dtype=np.float64),
        new_adult_age=1,
        expected_eggs_per_female=1.0,
        use_fixed_egg_count=True,
        carrying_capacity=1.0e12,
        sex_ratio=0.5,
        low_density_growth_rate=1.0,
        juvenile_growth_mode=NO_COMPETITION,
        has_sex_chromosomes=True,
        genotype_to_gametes_map=gamete_map,
        gametes_to_zygote_map=zygote_map,
    )

    pop = DiscreteGenerationPopulation(
        species=species,
        population_config=config,
        initial_individual_count={
            "female": {female_parent: 1000.0},
            "male": {male_parent: 1000.0},
        },
    )

    pop.run(1)
    state = pop._state.individual_count

    female_age1 = state[0, 1, :]
    male_age1 = state[1, 1, :]

    female_by_maternal_x: dict[str, float] = {"X1": 0.0, "X2": 0.0}
    male_by_maternal_x: dict[str, float] = {"X1": 0.0, "X2": 0.0}

    for idx, genotype in enumerate(pop._registry.index_to_genotype):
        x_m = genotype.maternal.get_gene_at_locus(locus_x).name
        if x_m in female_by_maternal_x:
            female_by_maternal_x[x_m] += float(female_age1[idx])
            male_by_maternal_x[x_m] += float(male_age1[idx])

    np.testing.assert_allclose(female_age1.sum(), 500.0)
    np.testing.assert_allclose(male_age1.sum(), 500.0)

    # Heterozygous mother transmits X1/X2 at 1:1 into both sex cohorts.
    np.testing.assert_allclose(female_by_maternal_x["X1"], 250.0)
    np.testing.assert_allclose(female_by_maternal_x["X2"], 250.0)
    np.testing.assert_allclose(male_by_maternal_x["X1"], 250.0)
    np.testing.assert_allclose(male_by_maternal_x["X2"], 250.0)


def test_discrete_generation_runs_when_y_chromosome_has_no_locus() -> None:
    species = nt.Species.from_dict(
        name="DiscreteYWithoutLocus",
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X"]}},
            "chrY": {"sex_type": "Y", "loci": {}},
        },
    )

    diploid_genotypes = species.get_all_genotypes()
    female_parent = diploid_genotypes[0]
    male_parent = diploid_genotypes[0]

    haploid_genotypes = species.get_all_haploid_genotypes()

    gamete_map = initialize_gamete_map(
        haploid_genotypes=haploid_genotypes,
        diploid_genotypes=diploid_genotypes,
        n_glabs=1,
    )
    zygote_map = initialize_zygote_map(
        haploid_genotypes=haploid_genotypes,
        diploid_genotypes=diploid_genotypes,
        n_glabs=1,
    )

    config = build_population_config(
        n_genotypes=len(diploid_genotypes),
        n_haploid_genotypes=len(haploid_genotypes),
        n_sexes=2,
        n_ages=2,
        n_glabs=1,
        is_stochastic=False,
        use_continuous_sampling=False,
        age_based_survival_rates=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        age_based_mating_rates=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        female_age_based_relative_fertility=np.array([0.0, 1.0], dtype=np.float64),
        viability_fitness=np.ones((2, 2, len(diploid_genotypes)), dtype=np.float64),
        fecundity_fitness=np.ones((2, len(diploid_genotypes)), dtype=np.float64),
        sexual_selection_fitness=np.ones((len(diploid_genotypes), len(diploid_genotypes)), dtype=np.float64),
        age_based_relative_competition_strength=np.array([1.0, 1.0], dtype=np.float64),
        new_adult_age=1,
        expected_eggs_per_female=1.0,
        use_fixed_egg_count=True,
        carrying_capacity=1.0e12,
        sex_ratio=0.5,
        low_density_growth_rate=1.0,
        juvenile_growth_mode=NO_COMPETITION,
        has_sex_chromosomes=True,
        genotype_to_gametes_map=gamete_map,
        gametes_to_zygote_map=zygote_map,
    )

    pop = DiscreteGenerationPopulation(
        species=species,
        population_config=config,
        initial_individual_count={
            "female": {female_parent: 1000.0},
            "male": {male_parent: 1000.0},
        },
    )

    pop.run(1)
    state = pop._state.individual_count
    age1_total = float(state[:, 1, :].sum())

    assert np.isfinite(age1_total)
    assert age1_total >= 0.0
