"""Independent contracts for live frequency queries and chromosome mutation.

Frequency references come from Mendelian proportions or isolated age survival.
These deterministic checks reject stale snapshots, not sampling distributions.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

import natal as nt


def _chromosome() -> nt.Chromosome:
    species = nt.Species(f"DisclosureChromosome_{uuid4().hex}")
    chromosome = species.add("c")
    assert isinstance(chromosome, nt.Chromosome)
    return chromosome


def _three_loci() -> nt.Chromosome:
    chromosome = _chromosome()
    for name, position in (("A", 0), ("B", 10), ("C", 20)):
        chromosome.add(name, position=position)
    chromosome.set_recombination("A", "B", 0.1)
    chromosome.set_recombination("B", "C", 0.2)
    return chromosome


def test_discrete_frequency_reads_current_generation_without_refresh() -> None:
    """Viability removes WT homozygotes; surviving allele shares are 3/7, 4/7."""
    species = nt.Species.from_dict(
        f"DisclosureFrequencyDiscrete_{uuid4().hex}", {"c": {"L": ["WT", "Drive"]}}
    )
    population = (
        nt.DiscreteGenerationPopulation.setup(species, stochastic=False)
        .initial_state(individual_count={
            "female": {"WT|WT": 50, "WT|Drive": 50},
            "male": {"WT|WT": 50, "WT|Drive": 50},
        })
        .competition(juvenile_growth_mode=nt.NO_COMPETITION)
        .fitness(viability={"WT|WT": 0.0})
        .build()
    )
    assert population.compute_allele_frequencies() == {"WT": 0.75, "Drive": 0.25}
    population.run(1)
    # Random mating gives 9:6:1 zygotes; removal of the first class leaves
    # 6 heterozygotes and 1 Drive homozygote: 6 WT and 8 Drive gene copies.
    assert population.compute_allele_frequencies() == pytest.approx(
        {"WT": 3 / 7, "Drive": 4 / 7}, rel=1e-14, abs=1e-15
    )
    population.run(1)
    # At p=3/7, removing WT/WT gives p' = p/(1+p) = 3/10.
    assert population.compute_allele_frequencies() == pytest.approx(
        {"WT": 0.3, "Drive": 0.7}, rel=1e-14, abs=1e-15
    )


def test_age_frequency_reads_survivors_then_extinction_without_refresh() -> None:
    """With no births, only younger Drive homozygotes survive the first tick."""
    species = nt.Species.from_dict(
        f"DisclosureFrequencyAge_{uuid4().hex}", {"c": {"L": ["WT", "Drive"]}}
    )
    population = (
        nt.AgeStructuredPopulation.setup(species, stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": [0, 0, 50], "Drive|Drive": [0, 50, 0]},
            "male": {"WT|WT": [0, 0, 50], "Drive|Drive": [0, 50, 0]},
        })
        .survival(female_age_based_survival=[1, 1, 0], male_age_based_survival=[1, 1, 0])
        .reproduction(eggs_per_female=0)
        .build()
    )
    assert population.compute_allele_frequencies() == {"WT": 0.5, "Drive": 0.5}
    population.run(1)
    assert population.compute_allele_frequencies() == {"WT": 0.0, "Drive": 1.0}
    population.run(1)
    assert population.compute_allele_frequencies() == {"WT": 0.0, "Drive": 0.0}


def test_chained_add_after_empty_and_single_locus_queries() -> None:
    """Prior cache reads cannot hide subsequent chained children."""
    chromosome = _chromosome()
    assert chromosome.loci == []
    with pytest.raises(ValueError, match="fewer than 2 loci"):
        _ = chromosome.recombination_map
    chromosome.add("A").add_alleles(["A1", "A2"])
    assert [locus.name for locus in chromosome.loci] == ["A"]
    with pytest.raises(ValueError, match="fewer than 2 loci"):
        _ = chromosome.recombination_map
    chromosome.add("B").add_alleles(["B1", "B2"])
    chromosome.set_recombination("A", "B", 0.125)
    chromosome.add("C").add_alleles(["C1", "C2"])
    assert [locus.name for locus in chromosome.loci] == ["A", "B", "C"]
    np.testing.assert_array_equal(chromosome.recombination_map[:], [0.125, 0.0])


def test_batch_add_preserves_insertion_rates_and_keyword_precedence() -> None:
    """Insertion inherits the old following interval; item kwargs override defaults."""
    chromosome = _chromosome()
    chromosome.add(["A", "C"])
    chromosome.set_recombination("A", "C", 0.2)
    added = chromosome.add(
        [("B", {"position": 0.5, "recombination_rate_with_previous": 0.1})],
        position=99,
    )
    assert isinstance(added, list) and len(added) == 1
    assert [locus.name for locus in chromosome.loci] == ["A", "B", "C"]
    np.testing.assert_array_equal(chromosome.recombination_map[:], [0.1, 0.2])


def test_add_locus_instance_preserves_existing_rates_and_is_idempotent() -> None:
    """Instance and name additions share the same insertion and identity contract."""
    chromosome = _three_loci()
    locus = nt.Locus(f"D_{uuid4().hex}", position=30)
    assert chromosome.add_locus(locus, recombination_rate_with_previous=0.125) is locus
    assert chromosome.get_locus(locus.name) is locus
    np.testing.assert_array_equal(chromosome.recombination_map[:], [0.1, 0.2, 0.125])
    assert chromosome.add_locus(locus) is locus
    np.testing.assert_array_equal(chromosome.recombination_map[:], [0.1, 0.2, 0.125])


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_duplicate_chained_add_is_idempotent(name: str) -> None:
    """The registry's cached-instance add contract preserves existing linkage."""
    chromosome = _three_loci()
    existing = chromosome.get_locus(name)
    assert chromosome.add(name) is existing
    np.testing.assert_array_equal(chromosome.recombination_map[:], [0.1, 0.2])


def test_position_change_without_reordering_preserves_rates() -> None:
    """Moving within an interval changes coordinates, not manually specified rates."""
    chromosome = _three_loci()
    locus = chromosome.get_locus("C")
    assert locus is not None
    locus.position = 21
    assert [child.name for child in chromosome.loci] == ["A", "B", "C"]
    np.testing.assert_array_equal(chromosome.recombination_map[:], [0.1, 0.2])


@pytest.mark.parametrize(
    ("name", "position", "order", "rates"),
    [
        ("A", 5, "ABCD", [0.1, 0.2, 0.25]),
        ("A", 15, "BACD", [0.0, 0.2, 0.25]),
        ("A", 25, "BCAD", [0.2, 0.0, 0.25]),
        ("A", 40, "BCDA", [0.2, 0.25, 0.0]),
        ("B", -5, "BACD", [0.0, 0.3, 0.25]),
        ("B", 5, "ABCD", [0.1, 0.2, 0.25]),
        ("B", 25, "ACBD", [0.3, 0.0, 0.25]),
        ("B", 40, "ACDB", [0.3, 0.25, 0.0]),
        ("C", -5, "CABD", [0.0, 0.1, 0.45]),
        ("C", 5, "ACBD", [0.0, 0.1, 0.45]),
        ("C", 15, "ABCD", [0.1, 0.2, 0.25]),
        ("C", 40, "ABDC", [0.1, 0.45, 0.0]),
        ("D", -5, "DABC", [0.0, 0.1, 0.2]),
        ("D", 5, "ADBC", [0.0, 0.1, 0.2]),
        ("D", 15, "ABDC", [0.1, 0.0, 0.2]),
        ("D", 25, "ABCD", [0.1, 0.2, 0.25]),
    ],
)
def test_position_reordering_uses_existing_removal_and_insertion_rules(
    name: str, position: int, order: str, rates: list[float]
) -> None:
    """Exhaust all old/new index pairs for four loci under remove/reinsert rules."""
    chromosome = _three_loci()
    chromosome.add("D", position=30, recombination_rate_with_previous=0.25)
    locus = chromosome.get_locus(name)
    assert locus is not None
    locus.position = position
    assert [child.name for child in chromosome.loci] == list(order)
    assert chromosome.recombination_map.loci_names == list(order)
    # Literal references apply the documented distance sum on removal and
    # zero previous interval on reinsertion; 1e-15 covers roundoff in the sum.
    np.testing.assert_allclose(
        chromosome.recombination_map[:], rates, rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("batch", [False, True])
def test_remove_after_query_updates_loci_and_merges_adjacent_rates(batch: bool) -> None:
    """Removing the middle locus joins the two distances under the existing rule."""
    chromosome = _three_loci()
    locus = chromosome.get_locus("B")
    assert locus is not None
    chromosome.remove([locus] if batch else locus)
    assert [child.name for child in chromosome.loci] == ["A", "C"]
    assert chromosome.recombination_map["A", "C"] == pytest.approx(0.3, abs=1e-15)
    chromosome.remove(["A", "missing"] if batch else "A")
    assert [child.name for child in chromosome.loci] == ["C"]
    with pytest.raises(ValueError, match="fewer than 2 loci"):
        _ = chromosome.recombination_map


@pytest.mark.parametrize("batch", [False, True])
def test_remove_rejects_non_locus_children(batch: bool) -> None:
    """Invalid child types produce a clear type error without deleting valid loci."""
    chromosome = _three_loci()
    invalid = nt.Species(f"DisclosureInvalidChild_{uuid4().hex}")
    with pytest.raises(TypeError):
        chromosome.remove([invalid] if batch else invalid)
    assert [child.name for child in chromosome.loci] == ["A", "B", "C"]


def test_add_rejects_malformed_spec_list() -> None:
    """A malformed tuple cannot silently become a child."""
    chromosome = _three_loci()
    with pytest.raises(TypeError, match="Invalid item"):
        chromosome.add([("D", {}, "extra")])
    np.testing.assert_array_equal(chromosome.recombination_map[:], [0.1, 0.2])


def test_add_rejects_non_list_batch_as_before() -> None:
    """The established add contract accepts a string or list, not a tuple batch."""
    chromosome = _three_loci()
    with pytest.raises(AssertionError, match="Expected str"):
        chromosome.add(("D", "E"))
    assert [child.name for child in chromosome.loci] == ["A", "B", "C"]
