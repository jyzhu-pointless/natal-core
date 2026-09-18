"""The unified selector/target entries and the single promotion rule.

FRONTEND_REFACTOR_PLAN.md §5.1 gives matching and keep-or-replace
conversion one entry each (``parse_selector`` / ``parse_target``), and
§5.6 requires one selector spelling to match identically through fitness,
presets, rules, observation and hooks.  The promotion rule finalized with
the entries: on an unordered species every ``|`` in a *selector* is
promoted to ``::`` (any ``::`` already written is preserved); content-only
parsing (the ``Species`` ``parse_*`` helpers) keeps ``|`` strictly ordered;
targets are never promoted.

The two-chromosome case is the regression that motivated the rule: the
fitness writer used to promote only the first ``|`` while
``IndividualSelector`` promoted all of them, so ``*|A; *|B@infected``
selected four ZTypes in one and wrote two in the other (external plan
review, 2026-09-18).
"""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.patterns import (
    PatternParseError,
    ZygoteTypePattern,
    parse_selector,
    parse_target,
)
from natal.frontend.patterns.parser import GenotypePatternParser


@pytest.fixture(scope="module")
def species() -> nt.Species:
    """Two unordered chromosome groups and two somatic labels."""
    return nt.Species.from_dict(
        "selector_entries_two_chr",
        {"c1": {"l1": ["A", "a"]}, "c2": {"l2": ["B", "b"]}},
        somatic_labels=["default", "infected"],
    )


@pytest.fixture(scope="module")
def population(species: nt.Species) -> nt.Population:
    return (
        nt.PopulationBuilder.from_species(species)
        .setup(stochastic=False)
        .build()
    )


def _selected_ztypes(population: nt.Population, query: str) -> set[int]:
    coordinates = nt.IndividualSelector(ztype=query).compile_coordinates(
        population.index_registry, n_sexes=2, n_ages=population.config.n_ages
    )
    return {ztype for _, _, ztype in coordinates}


def _written_ztypes(species: nt.Species, query: str) -> set[int]:
    builder = nt.PopulationBuilder.from_species(species).setup(stochastic=False)
    builder.fitness(viability={query: 0.5})
    arr = builder._config.viability_fitness
    return {i for i, value in enumerate(arr[0, 0]) if value == 0.5}


def test_fitness_and_individual_selector_agree_for_two_chromosomes(
    species: nt.Species, population: nt.Population
) -> None:
    """The regression from the plan review: one spelling, one match set."""
    query = "*|A; *|B@infected"
    assert _selected_ztypes(population, query) == _written_ztypes(species, query)
    assert _selected_ztypes(population, query) == {1, 3, 5, 7}


def test_every_selector_caller_agrees_on_one_spelling(
    species: nt.Species, population: nt.Population
) -> None:
    """fitness, IndividualSelector, resolve_zygote_type and the entry itself."""
    query = "*|A; *|B@infected"
    expected = _selected_ztypes(population, query)

    pattern = parse_selector(query, species=species, kind="ztype")
    from_entries = set(population.index_registry.resolve_ztype_indices(pattern))

    from natal.frontend.patterns import resolve_zygote_type

    from_helper = set(resolve_zygote_type(query, species, population.index_registry))

    assert from_entries == expected
    assert from_helper == expected
    assert _written_ztypes(species, query) == expected


def test_unordered_promotion_covers_every_separator(
    species: nt.Species, population: nt.Population
) -> None:
    """``A|B; C|D`` selects exactly what ``A::B; C::D`` selects."""
    ordered = _selected_ztypes(population, "A|a; B|b")
    unordered = _selected_ztypes(population, "A::a; B::b")
    assert ordered == unordered


def test_mixed_separators_promote_the_ordered_ones(
    species: nt.Species, population: nt.Population
) -> None:
    """An explicit ``::`` never blocks promoting a remaining ``|``."""
    mixed = _selected_ztypes(population, "A::a; B|b")
    assert mixed == _selected_ztypes(population, "A::a; B::b")


def test_ordered_species_never_promotes() -> None:
    ordered = nt.Species.from_dict(
        "selector_entries_ordered",
        {"c1": {"l1": ["A", "a"]}, "c2": {"l2": ["B", "b"]}},
        unordered=False,
        somatic_labels=["default"],
    )
    population = nt.PopulationBuilder.from_species(ordered).setup(stochastic=False).build()
    # ``|`` stays strictly ordered, so the two parental spellings select
    # two distinct genotypes instead of collapsing onto one canonical form.
    forward = _selected_ztypes(population, "A|a; B|b")
    reverse = _selected_ztypes(population, "a|A; b|B")
    assert len(forward) == 1 and len(reverse) == 1
    assert forward != reverse


def test_content_parsing_stays_strictly_ordered(species: nt.Species) -> None:
    """The grammar helpers do not promote; only the selector entry does."""
    strict_filter = species.parse_genotype_pattern("a|A; b|B")
    promoted_filter = species.parse_genotype_pattern("a::A; b::B")
    genotypes = species.get_all_genotypes(unordered=True)
    assert [g for g in genotypes if strict_filter(g)] == []
    assert len([g for g in genotypes if promoted_filter(g)]) == 1


def test_genotype_level_species_selector_promotes(species: nt.Species) -> None:
    """``resolve_genotype_selectors`` funnels through the entry as well."""
    strict = species.resolve_genotype_selectors(
        selector="a|A; b|B",
        all_genotypes=species.get_all_genotypes(unordered=True),
        context="probe",
    )
    unordered = species.resolve_genotype_selectors(
        selector="a::A; b::B",
        all_genotypes=species.get_all_genotypes(unordered=True),
        context="probe",
    )
    assert [gt.to_string() for gt in strict] == [gt.to_string() for gt in unordered]


class TestParseSelectorKinds:
    def test_ztype_is_the_default_kind(self, species: nt.Species) -> None:
        pattern = parse_selector("A|a@infected", species=species)
        assert isinstance(pattern, ZygoteTypePattern)
        assert pattern.slab is not None and pattern.slab.matches("infected")

    def test_genotype_kind_rejects_a_label(self, species: nt.Species) -> None:
        with pytest.raises(PatternParseError, match="does not take an '@label' suffix"):
            parse_selector("A|a@infected", species=species, kind="genotype")

    def test_haploid_kind_rejects_a_label(self, species: nt.Species) -> None:
        with pytest.raises(PatternParseError, match="does not take an '@label' suffix"):
            parse_selector("A@deposited", species=species, kind="haploid")

    def test_gtype_kind_carries_the_glab(self, species: nt.Species) -> None:
        gamete = parse_selector("A; B@deposited", species=species, kind="gtype")
        assert gamete.glab is not None and gamete.glab.matches("deposited")

    def test_non_string_input_raises_type_error(self, species: nt.Species) -> None:
        with pytest.raises(TypeError, match="selector must be a string"):
            parse_selector(123, species=species)  # type: ignore[arg-type]

    def test_unknown_kind_raises_value_error(self, species: nt.Species) -> None:
        with pytest.raises(ValueError, match="unknown"):
            parse_selector("A|a", species=species, kind="allele")  # type: ignore[arg-type]


class TestParseTarget:
    def test_syntax_and_validate_stages(self, species: nt.Species) -> None:
        parsed = parse_target("A|A; B::b@marked", species=species)
        with pytest.raises(ValueError, match="ambiguous"):
            parsed.validate(species)
        clean = parse_target("A|A; B|b@marked", species=species)
        clean.validate(species)

    def test_validate_flag_runs_the_check_eagerly(self, species: nt.Species) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            parse_target("A::a@marked", species=species, validate=True)

    def test_unordered_target_is_never_promoted(self, species: nt.Species) -> None:
        """A replacement must say which side it replaces."""
        with pytest.raises(ValueError, match="unordered"):
            parse_target("A::a; B|b@marked", species=species, validate=True)

    def test_require_label_kept_for_the_legacy_form(self, species: nt.Species) -> None:
        with pytest.raises(PatternParseError, match="must be"):
            parse_target("A|A; B|b", species=species, require_label=True)

    def test_non_string_target_raises_type_error(self, species: nt.Species) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            parse_target(("A", "a"), species=species)  # type: ignore[arg-type]


def test_parser_entries_survive_as_delegating_spellings(
    species: nt.Species,
) -> None:
    """The old spellings still parse; they delegate to the same entry."""
    parser = GenotypePatternParser(species)
    from_type = ZygoteTypePattern.parse("A|a@infected", species)
    from_entry = parse_selector("A|a@infected", species=species, kind="ztype")
    assert repr(from_type) == repr(from_entry)  # LabPattern has no __eq__
    assert parser.parse_haplotype_pattern("A@x").glab is not None
