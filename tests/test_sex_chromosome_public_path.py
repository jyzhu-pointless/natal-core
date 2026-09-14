"""Sex-chromosome public-path acceptance (CR-0 / CR-13 regression).

The public builder chain must handle XY and ZW species end to end:
genotype strings round-trip (including the sex-chromosome pair), precise
initialization places counts on the exact (Genotype, slab) ztype instead
of a same-autosome opposite-sex first match, sex constraints come from
the genetic structure (XX/XY, ZW/ZZ) rather than gamete row sums, and
every engine path — staged and Wright-Fisher alike — conserves the
offspring total under one sex-allocation probability definition.

All expected values are exact: deterministic mode with viability 1 and
no competition, so offspring numbers follow directly from Mendelian
segregation.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _xy_species(name: str, somatic_labels: list[str] | None = None) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1", "X2"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
        somatic_labels=somatic_labels,
    )


def _build_xy(
    species: nt.Species,
    female_key: object,
    male_key: object,
    *,
    sex_ratio: float = 0.5,
    compress: bool = False,
) -> nt.DiscreteGenerationPopulation:
    builder = nt.DiscreteGenerationPopulation.setup(
        species=species, name="xy_public", stochastic=False, compress=compress
    )
    return (
        builder.initial_state(
            individual_count={"female": {female_key: 1000}, "male": {male_key: 1000}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=sex_ratio)
        # Explicit no-competition: these tests pin offspring sex/type
        # conservation, not the regulation curve.
        .competition(
            growth_mode="no_competition",
            carrying_capacity=1e12,
            low_density_growth_rate=2.0,
        )
        .build()
    )


def test_xy_public_builder_offspring_sex_and_types_are_exact() -> None:
    """XX offspring are female, XY male, totals conserved, 1:2:1 autosomal."""
    species = _xy_species("cr0_xy_pub")
    female = species.get_genotype_from_str("A|a;X1|X2")
    male = species.get_genotype_from_str("A|a;X1|Y1")
    pop = _build_xy(species, female, male)
    pop.run(1)

    counts = pop.state.individual_count
    assert float(counts.sum()) == 1000.0
    assert float(counts[0].sum()) == 500.0
    assert float(counts[1].sum()) == 500.0

    registry = pop.registry

    # Mask oracle: an offspring ztype is male-constrained exactly when its
    # paternal haploid carries the Y chromosome, female-constrained
    # otherwise (father transmits X or Y; mother only X).
    chr_y = species.get_chromosome("chrY")

    def _is_male_ztype(genotype: nt.Genotype) -> bool:
        try:
            genotype.paternal.get_haplotype_for_chromosome(chr_y)
            return True
        except ValueError:
            return False

    # Per-sex exact 1:2:1 autosomal split, retaining the maternal/paternal
    # ordering that this unordered=False species explicitly requests.
    per_sex: dict[tuple[int, str], float] = {}
    for genotype, slab in registry.index_to_ztype:
        idx = registry.ztype_index(genotype, slab)
        total = float(counts[:, :, idx].sum())
        sex = 1 if _is_male_ztype(genotype) else 0
        assert (counts[1 - sex, :, idx] == 0).all(), genotype.to_string()
        per_sex[(sex, genotype.to_string().split(";")[0])] = (
            per_sex.get((sex, genotype.to_string().split(";")[0]), 0.0) + total
        )
    for sex in (0, 1):
        assert per_sex[(sex, "A|A")] == 125.0
        assert per_sex[(sex, "A|a")] == 125.0
        assert per_sex[(sex, "a|A")] == 125.0
        assert per_sex[(sex, "A|a")] + per_sex[(sex, "a|A")] == 250.0
        assert per_sex[(sex, "a|a")] == 125.0

    female_only = pop.config.female_only_by_sex_chrom
    male_only = pop.config.male_only_by_sex_chrom
    for genotype, slab in registry.index_to_ztype:
        idx = registry.ztype_index(genotype, slab)
        is_male = _is_male_ztype(genotype)
        assert male_only[idx] == is_male, genotype.to_string()
        assert female_only[idx] == (not is_male), genotype.to_string()


def test_xy_masks_hold_for_every_sex_ratio() -> None:
    """Genetic sex wins: sex_ratio never moves the 50/50 XX/XY split."""
    species = _xy_species("cr0_xy_ratio")
    female = species.get_genotype_from_str("A|a;X1|X2")
    male = species.get_genotype_from_str("A|a;X1|Y1")
    pop = _build_xy(species, female, male, sex_ratio=0.9)
    pop.run(1)
    counts = pop.state.individual_count
    assert float(counts[0].sum()) == 500.0
    assert float(counts[1].sum()) == 500.0


def test_xy_string_and_label_inputs_land_on_exact_ztypes() -> None:
    """String keys and (genotype, slab) tuples place counts identically."""
    species = _xy_species("cr0_xy_keys")
    female = species.get_genotype_from_str("A|a;X1|X2")
    male = species.get_genotype_from_str("A|a;X1|Y1")

    by_object = _build_xy(species, female, male)
    by_string = _build_xy(species, "A|a;X1|X2", "A|a;X1|Y1")
    np.testing.assert_array_equal(
        by_object.state.individual_count, by_string.state.individual_count
    )


def test_xy_slab_expanded_initial_state_and_masks() -> None:
    """Slab-qualified keys land on the exact slab; masks expand per slab."""
    species = _xy_species("cr0_xy_slab", somatic_labels=["default", "t"])
    female = species.get_genotype_from_str("A|a;X1|X2")
    male = species.get_genotype_from_str("A|a;X1|Y1")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="xy_slab_pub", stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {(female, "t"): 500},
                "male": {"A|a;X1|Y1@default": 700},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
        .build()
    )
    registry = pop.registry
    counts = pop.state.individual_count
    assert float(counts[0, :, registry.ztype_index(female, "t")].sum()) == 500.0
    assert float(counts[1, :, registry.ztype_index(male, "default")].sum()) == 700.0
    assert float(counts.sum()) == 1200.0

    # Structure-derived masks expand across the slab axis.
    chr_y = species.get_chromosome("chrY")
    for genotype, slab in registry.index_to_ztype:
        idx = registry.ztype_index(genotype, slab)
        try:
            genotype.paternal.get_haplotype_for_chromosome(chr_y)
            is_male = True
        except ValueError:
            is_male = False
        assert pop.config.male_only_by_sex_chrom[idx] == is_male
        assert pop.config.female_only_by_sex_chrom[idx] == (not is_male)


def test_xy_compressed_axis_conserves_and_masks() -> None:
    """Compression prunes the axis but sex masks and totals stay exact."""
    species = _xy_species("cr0_xy_compress")
    # Seed only A so that a-bearing genotypes are genuinely unreachable;
    # both heterozygous parents would make the complete ordered axis reachable.
    female = species.get_genotype_from_str("A|A;X1|X2")
    male = species.get_genotype_from_str("A|A;X1|Y1")
    pop = _build_xy(species, female, male, compress=True)
    assert len(pop.registry.index_to_ztype) < len(species.get_all_genotypes(unordered=False)) * 1
    pop.run(1)
    counts = pop.state.individual_count
    assert float(counts.sum()) == 1000.0
    assert float(counts[0].sum()) == 500.0
    assert float(counts[1].sum()) == 500.0


def test_zw_public_builder_offspring_sex_and_types_are_exact() -> None:
    """ZW species: WZ females, ZZ males, conserved totals."""
    species = nt.Species.from_dict(
        name="cr0_zw_pub",
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrZ": {"sex_type": "Z", "loci": {"sz": ["Z1", "Z2"]}},
            "chrW": {"sex_type": "W", "loci": {"sw": ["W1"]}},
        },
        unordered=False,
    )
    female = species.get_genotype_from_str("A|a;W1|Z1")
    male = species.get_genotype_from_str("A|a;Z1|Z1")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="zw_public", stochastic=False
        )
        .initial_state(
            individual_count={"female": {female: 1000}, "male": {male: 1000}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        # Explicit no-competition: this test pins offspring sex/type
        # conservation, not the regulation curve.
        .competition(
            growth_mode="no_competition",
            carrying_capacity=1e12,
            low_density_growth_rate=2.0,
        )
        .build()
    )
    pop.run(1)
    counts = pop.state.individual_count
    assert float(counts.sum()) == 1000.0
    assert float(counts[0].sum()) == 500.0
    assert float(counts[1].sum()) == 500.0

    registry = pop.registry
    # Mask oracle: a ztype is female-constrained exactly when its maternal
    # haploid carries the W chromosome (W is maternal-only).
    chr_w = species.get_chromosome("chrW")
    for genotype, slab in registry.index_to_ztype:
        idx = registry.ztype_index(genotype, slab)
        try:
            genotype.maternal.get_haplotype_for_chromosome(chr_w)
            is_female = True
        except ValueError:
            is_female = False
        assert pop.config.female_only_by_sex_chrom[idx] == is_female
        assert pop.config.male_only_by_sex_chrom[idx] == (not is_female)


def test_wright_fisher_path_matches_staged_sex_allocation() -> None:
    """The WF fused path conserves totals exactly like the staged path.

    Regression for CR-13: the WF path used to multiply each sex's gamete
    row sum independently (both 1.0 with baseline maps), doubling the
    offspring total to 2000 under ``extreme_speed_mode=3``.
    """
    species = _xy_species("cr13_xy_wf")
    female = species.get_genotype_from_str("A|a;X1|X2")
    male = species.get_genotype_from_str("A|a;X1|Y1")

    totals = []
    for mode in (0, 3):
        built = _build_xy(species, female, male)
        # The public builder already materializes the native blueprint.
        # Set the mode before materializing the comparison population;
        # replacing only built._config would leave the native mode unchanged.
        pop = nt.DiscreteGenerationPopulation(
            species=species,
            population_config=built.config._replace(extreme_speed_mode=mode),
            initial_individual_count={
                "female": {female: 1000}, "male": {male: 1000},
            },
        )
        pop.run(1)
        state = pop.state.individual_count
        totals.append(
            (float(state[0].sum()), float(state[1].sum()), float(state.sum()))
        )
    assert totals[0] == (500.0, 500.0, 1000.0)
    assert totals[1] == (500.0, 500.0, 1000.0)


def test_every_enumerated_genotype_round_trips_with_two_autosomes() -> None:
    """Strings round-trip for a multi-autosome XY species, diploid and haploid.

    The genetics documentation promises that every enumerated genotype's
    string parses back to the same object; multiple autosome segments in
    front of the sex-group segment exercise the segment ordering of both
    stringification and parsing.
    """
    species = nt.Species.from_dict(
        name="cr0_xy_roundtrip2",
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrB": {"loci": {"B": ["B", "b"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1", "X2"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
    )
    for genotype in species.get_all_genotypes(unordered=False):
        assert species.get_genotype_from_str(genotype.to_string()) is genotype, (
            genotype.to_string()
        )
    for haploid in species.get_all_haploid_genotypes():
        assert species.get_haploid_genome_from_str(haploid.to_string()) is haploid, (
            haploid.to_string()
        )
    male = species.get_genotype_from_str("A|a;B|b;X1|Y1")
    assert male.to_string() == "A|a;B|b;X1|Y1"


def test_classify_genotype_sex_without_sex_chromosomes_is_none() -> None:
    """classify_genotype_sex returns None when the species has no sex groups."""
    species = nt.Species.from_dict(
        name="cr0_classify_plain", structure={"chr1": {"loc": ["A", "B"]}}
    )
    genotype = species.get_all_genotypes(unordered=False)[0]
    assert species.classify_genotype_sex(genotype) is None


def test_initial_state_resolver_rejects_unknown_keys() -> None:
    """Exact resolution raises TypeError/KeyError instead of first-match fallback.

    The KeyError cases need a compressed axis so a species-valid genotype
    is genuinely absent from the active catalog.
    """
    from natal.frontend.model.initial_state import resolve_genotype_key_ztype_index

    species = _xy_species("cr0_resolver_errors")
    pop = _build_xy(species, "A|A;X1|X2", "A|A;X1|Y1", compress=True)
    registry = pop.registry

    with pytest.raises(TypeError, match="must be Genotype, str, or tuple"):
        resolve_genotype_key_ztype_index(42, species, registry)
    with pytest.raises(TypeError, match="Tuple first element must be Genotype or str"):
        resolve_genotype_key_ztype_index((42, "default"), species, registry)

    inactive = species.get_genotype_from_str("a|a;X1|X2")
    assert registry.ztype_indices_for(inactive) == []  # off the active axis
    with pytest.raises(KeyError, match="not in the active ztype catalog"):
        resolve_genotype_key_ztype_index("a|a;X1|X2", species, registry)
    with pytest.raises(KeyError, match="not in the active ztype catalog"):
        resolve_genotype_key_ztype_index(inactive, species, registry)
    # The tuple path goes through registry.ztype_index, which raises its
    # own KeyError for the missing (genotype, slab) key — still an
    # explicit failure, not a silent fallback.
    with pytest.raises(KeyError):
        resolve_genotype_key_ztype_index(("a|a;X1|X2", "default"), species, registry)


def test_tuple_string_slab_key_lands_on_exact_ztype() -> None:
    """A (genotype-string, slab) tuple resolves by identity like an object key."""
    species = _xy_species("cr0_tuple_str_key")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="xy_tuple_str", stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {("A|A;X1|X2", "default"): 400},
                "male": {"A|A;X1|Y1@default": 600},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0)
        .build()
    )
    registry = pop.registry
    female = species.get_genotype_from_str("A|A;X1|X2")
    male = species.get_genotype_from_str("A|A;X1|Y1")
    counts = pop.state.individual_count
    assert float(counts[0, :, registry.ztype_index(female, "default")].sum()) == 400.0
    assert float(counts[1, :, registry.ztype_index(male, "default")].sum()) == 600.0
    assert float(counts.sum()) == 1000.0
