"""Age-structured sex-chromosome masking (regression).

``PopulationBuilder.age_structure()`` rebuilds the population config from the
species blueprint.  That rebuild used to omit the structure-derived sex masks,
so every sex-fixed genotype had its sex decided by the compatibility ratio:
with the usual baseline maps (both sexes summing to 1) that sends about half of
each sex-fixed genotype's mass to the wrong sex axis, where it is then handled
with the other sex's survival and mating rates.  ``from_species()`` forwarded
the masks correctly, so only the age-structured rebuild regressed — which is
why the regular CR-0 suite (discrete-only) never caught it.

The contract asserted here is the one the kernel already implements: a genotype
that can only be one sex must never gain mass on the other sex's axis.  The
reachable ztypes of the crosses below are pinned at their exact Mendelian
values (600 females and 600 males split evenly over the zygote types); the
remaining catalog entries are unreachable for these homozygous parents and must
stay empty.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _xy_species(name: str) -> nt.Species:
    """Return a minimal XY species: one autosome plus X/Y."""
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
    """Return a minimal ZW species: one autosome plus Z/W."""
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
    """Return whether *genotype* carries *chromosome_name* on either side."""
    chrom = genotype.species.get_chromosome(chromosome_name)
    for side in (genotype.maternal, genotype.paternal):
        try:
            side.get_haplotype_for_chromosome(chrom)
            return True
        except ValueError:
            continue
    return False


def _build(
    species: nt.Species,
    name: str,
    female_key: str,
    male_key: str,
    *,
    use_age_structure: bool,
) -> nt.AgeStructuredPopulation:
    """Build the two-parent cohort, optionally through ``age_structure()``."""
    builder = nt.AgeStructuredPopulation.setup(
        species=species, name=name, stochastic=False
    )
    if use_age_structure:
        builder = builder.age_structure(n_ages=2, new_adult_age=1)
    female = species.get_genotype_from_str(female_key)
    male = species.get_genotype_from_str(male_key)
    return (
        builder.initial_state(
            individual_count={"female": {female: {1: 600}}, "male": {male: {1: 600}}}
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
        .competition(
            carrying_capacity=1e12, low_density_growth_rate=2.0, growth_mode="fixed"
        )
        .build()
    )


_XY_CASE = ("xy", _xy_species, "A|A;X1|X2", "A|A;X1|Y1", "chrY")
_ZW_CASE = ("zw", _zw_species, "A|A;W1|Z1", "A|A;Z1|Z2", "chrW")
_CASES = (_XY_CASE, _ZW_CASE)


def _sex_of(genotype: nt.Genotype, marker: str) -> int:
    """Return the sex axis index a genotype is fixed to (0 female, 1 male).

    ``marker`` is the chromosome whose presence identifies heterogamety: a
    Y-bearing genotype is male, a W-bearing genotype is female, and the
    homogametic remainder takes the opposite sex.
    """
    heterogametic_is_male = marker == "chrY"
    carries = _carries(genotype, marker)
    return (1 if carries else 0) if heterogametic_is_male else (0 if carries else 1)


@pytest.mark.parametrize("use_age_structure", [False, True], ids=["from_species", "age_structure"])
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_sex_fixed_genotypes_never_cross_the_sex_axis(
    case: tuple[str, object, str, str, str],
    use_age_structure: bool,
) -> None:
    """Sex-fixed genotypes must hold all their mass on their own sex axis."""
    label, species_fn, female_key, male_key, marker = case
    species = species_fn(f"{label}_mask_{use_age_structure}")  # type: ignore[operator]
    pop = _build(
        species,
        f"{label}_mask_{use_age_structure}",
        female_key,
        male_key,
        use_age_structure=use_age_structure,
    )
    pop.run(1)

    counts = pop.state.individual_count
    assert float(counts.sum()) == 1200.0
    assert float(counts[0].sum()) == 600.0
    assert float(counts[1].sum()) == 600.0

    wrong_axis_mass = 0.0
    reachable = 0
    for genotype, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype, slab)
        female_mass = float(counts[0, :, idx].sum())
        male_mass = float(counts[1, :, idx].sum())
        sex = _sex_of(genotype, marker)
        wrong_axis_mass += female_mass if sex == 1 else male_mass
        if female_mass + male_mass > 0.0:
            reachable += 1
            # Each reachable zygote type carries exactly one quarter of the
            # 1200 newborns, all of it on its own sex axis.
            assert (male_mass if sex == 1 else female_mass) == 300.0, (
                f"{genotype.to_string()}: expected 300.0 on sex axis {sex}, "
                f"got female={female_mass} male={male_mass}"
            )
            assert (female_mass if sex == 1 else male_mass) == 0.0, (
                f"{genotype.to_string()}: sex-fixed genotype gained mass on the "
                f"wrong sex axis (female={female_mass} male={male_mass})"
            )

    assert wrong_axis_mass == 0.0
    assert reachable == 4, f"expected 4 reachable zygote types, got {reachable}"


def test_missing_sex_masks_are_rejected() -> None:
    """Declaring sex chromosomes without the structure masks must fail loudly.

    The compatibility heuristic cannot tell homogametic from heterogametic
    pairs when every baseline map row sums to 1, so it would silently produce
    empty masks and let the compatibility ratio assign sex.  ``assembly``
    refuses that combination instead.
    """
    from natal.frontend.model import build_population_config

    species = _xy_species("xy_missing_masks")
    bp = species.get_config_blueprint()

    with pytest.raises(ValueError, match="no genotype is sex-fixed"):
        build_population_config(
            n_genotypes=bp["n_genotypes"],
            n_gtypes=bp["n_gtypes"],
            n_glabs=1,
            n_slabs=1,
            gamete_labels=species.gamete_labels or ["default"],
            somatic_labels=species.somatic_labels or ["default"],
            zygotes_to_gametes_map=bp["zygotes_to_gametes_map"],
            gametes_to_zygotes_map=bp["gametes_to_zygotes_map"],
            n_ages=2,
            new_adult_age=1,
            carrying_capacity=1000.0,
            has_sex_chromosomes=True,
        )

    # Passing the blueprint masks is the supported spelling and must succeed.
    config = build_population_config(
        n_genotypes=bp["n_genotypes"],
        n_gtypes=bp["n_gtypes"],
        n_glabs=1,
        n_slabs=1,
        gamete_labels=species.gamete_labels or ["default"],
        somatic_labels=species.somatic_labels or ["default"],
        zygotes_to_gametes_map=bp["zygotes_to_gametes_map"],
        gametes_to_zygotes_map=bp["gametes_to_zygotes_map"],
        n_ages=2,
        new_adult_age=1,
        carrying_capacity=1000.0,
        has_sex_chromosomes=True,
        female_only_by_sex_chrom=bp["female_only_by_sex_chrom"],
        male_only_by_sex_chrom=bp["male_only_by_sex_chrom"],
    )
    assert np.count_nonzero(config.female_only_by_sex_chrom) > 0
    assert np.count_nonzero(config.male_only_by_sex_chrom) > 0
