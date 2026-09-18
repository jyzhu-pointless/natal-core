"""Independent ownership, dependency, and sex-group contract coverage."""

import itertools

import numpy as np
import pytest

import natal as nt


def _linked_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name, structure={"c": {"l1": ["A", "a"], "l2": ["B", "b"]}},
        unordered=False,
    )


@pytest.mark.parametrize("write_style", ["setter", "slice", "asarray", "copyto", "ufunc"])
def test_real_rate_mutation_updates_baseline_probabilities(write_style: str) -> None:
    """Each public/shared write changes each recombinant from r/2=.05 to .2."""
    species = _linked_species(f"independent_rate_{write_style}")
    rates = species.chromosomes[0].recombination_map
    rates[:] = 0.1
    genotypes = species.get_all_genotypes(unordered=False)
    haploids = species.get_all_haploid_genotypes()
    gi = genotypes.index(species.get_genotype_from_str("A/B|a/b"))
    hi = haploids.index(species.get_haploid_genome_from_str("A/b"))
    before = species.get_config_blueprint()
    assert before["zygotes_to_gametes_map"][0, gi, hi] == pytest.approx(0.05)
    if write_style == "setter":
        rates[0] = 0.4
    elif write_style == "slice":
        rates[:][0] = 0.4
    elif write_style == "asarray":
        np.asarray(rates)[0] = 0.4
    elif write_style == "copyto":
        np.copyto(np.asarray(rates), np.array([0.4]))
    else:
        view = np.asarray(rates)
        np.multiply(view, 4.0, out=view)
    after = species.get_config_blueprint()
    assert after["zygotes_to_gametes_map"][0, gi, hi] == pytest.approx(0.2)
    assert before["zygotes_to_gametes_map"][0, gi, hi] == pytest.approx(0.05)


@pytest.mark.parametrize("invalid", [-0.1, 0.6, np.nan, np.inf])
def test_invalid_rate_view_fails_then_recovers(invalid: float) -> None:
    """Invalid dependencies raise ValueError; fixing them permits fresh computation."""
    species = _linked_species("independent_invalid_rate")
    species.get_config_blueprint()
    view = np.asarray(species.chromosomes[0].recombination_map)
    view[:] = invalid
    with pytest.raises(ValueError):
        species.get_config_blueprint()
    view[:] = 0.2
    assert np.isfinite(species.get_config_blueprint()["offspring_tensor"]).all()


def test_rate_storage_shape_change_is_rejected_before_reuse() -> None:
    """A shared ndarray resize cannot silently change the number of intervals."""
    species = _linked_species("independent_rate_shape")
    species.get_config_blueprint()
    view = np.asarray(species.chromosomes[0].recombination_map)
    view.resize((2,), refcheck=False)
    with pytest.raises(ValueError, match="shape"):
        species.get_config_blueprint()


@pytest.mark.parametrize("field", [
    "zygotes_to_gametes_map", "gametes_to_zygotes_map", "offspring_tensor",
    "female_ztype_compatibility", "male_ztype_compatibility",
])
def test_returned_baseline_array_mutation_cannot_change_future_values(field: str) -> None:
    """Range-valid writes, even after re-enabling writing, cannot poison the baseline."""
    species = _linked_species(f"independent_ownership_{field}")
    baseline = species.get_config_blueprint()
    expected = baseline[field].copy()
    array = baseline[field]
    try:
        array.flags.writeable = True
        array.fill(0.0)
    except ValueError:
        pass
    np.testing.assert_array_equal(species.get_config_blueprint()[field], expected)


@pytest.mark.parametrize("unordered", [False, True])
@pytest.mark.parametrize("system", ["XY", "ZW"])
@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_sex_group_roundtrip_and_pattern_independent_of_chromosome_order(
    unordered: bool, system: str, order: tuple[int, int, int],
) -> None:
    """Autosome/sex declaration order cannot change roundtrip or exact matching."""
    parts = [
        ("autosome", {"loci": {"L": ["A", "a"]}}),
        (system[0], {"sex_type": system[0], "loci": {"L1": [system[0]]}}),
        (system[1], {"sex_type": system[1], "loci": {"L2": [system[1]]}}),
    ]
    species = nt.Species.from_dict(
        name=f"independent_order_{system}_{unordered}_{''.join(map(str, order))}",
        structure=dict(parts[index] for index in order), unordered=unordered,
    )
    for genotype in species.get_all_genotypes(unordered=unordered):
        text = genotype.to_string()
        assert species.get_genotype_from_str(text) is genotype
        assert nt.parse_selector(text, species=species, kind="genotype").matches(genotype), text
    for haploid in species.get_all_haploid_genotypes():
        assert species.get_haploid_genome_from_str(haploid.to_string()) is haploid


def test_group_resolution_rejects_genome_from_another_species() -> None:
    """A chromosome name match cannot substitute for the correct species identity."""
    from natal.frontend.patterns._groups import group_haplotype

    source = _linked_species("independent_group_source")
    other = _linked_species("independent_group_other")
    genome = source.get_all_haploid_genotypes()[0]
    with pytest.raises(ValueError, match="exactly one chromosome"):
        group_haplotype(genome, [other.chromosomes[0]])
