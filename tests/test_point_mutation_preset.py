"""PointMutation preset contract tests.

Confirmed requirement (TODO.legacy.md ARCH-021): a point mutation declares one
source allele and one or more target alleles, and the targets *compete* —
each keeps its declared germline rate instead of losing mass to the targets
declared before it.  ``GameteConversionRuleSet`` cascades rules in
declaration order, so the preset compensates internally with
``r'_k = r_k / (1 - sum_{i<k} r_i)``.

Every realized distribution below is the closed form of that contract:

- one target, homo/heterozygous parent: ``{source: 1-r, target: r}`` on the
  source-carrying gametes;
- several targets: ``{source: 1 - sum(r_k), target_k: r_k}``.

The preset is germline-only: the embryonic channel is deferred, so
``zygote_modifier`` always returns ``None`` and no ``zygotic_mutation_rate``
parameter exists.

The tests also pin the declared-error paths (mixed forms, mismatched rate
counts, rates summing above 1 in strict mode) and the public export.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.genetics.compile import project_mendelian_maps
from natal.frontend.model import build_discrete_engine_config


def _species(name: str = "_point_mutation") -> nt.Species:
    """Return a single-locus species with the mutation alleles."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["A", "B", "C", "D"]}},
        gamete_labels=["default"],
    )


def _two_locus_species(name: str) -> nt.Species:
    """Return a species whose second chromosome carries target allele ``E``."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["A", "B"]}, "chr2": {"far": ["E", "F"]}},
        gamete_labels=["default"],
    )


def _host(species: nt.Species) -> SimpleNamespace:
    """Build a minimal RecipeHost over *species*."""
    registry = build_registry(species)
    z2g, g2z = project_mendelian_maps(species, registry)
    config = build_discrete_engine_config(
        n_genotypes=len(registry.index_to_genotype),
        n_gtypes=len(registry.index_to_gtype),
        n_glabs=len(registry.glab_labels),
        n_slabs=len(registry.slab_labels),
        zygotes_to_gametes_map=z2g,
        gametes_to_zygotes_map=g2z,
        has_sex_chromosomes=False,
    )
    return SimpleNamespace(
        species=species, registry=registry, index_registry=registry, config=config
    )


def _gamete_row(
    host: SimpleNamespace, genotype: str, sex: nt.Sex, preset: nt.PointMutation
) -> dict[str, float]:
    """Return one parent genotype's post-conversion gamete distribution by name."""
    modifier = preset.gamete_modifier(host)
    assert modifier is not None, "the configured rates must produce a modifier"
    ztype_idx = host.registry.ztype_index(
        host.species.get_genotype_from_str(genotype), "default"
    )
    row = modifier()[(sex, ztype_idx)]
    return {
        host.registry.index_to_gtype[gidx][0].to_string(): float(prob)
        for gidx, prob in row.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# Germline conversion
# ══════════════════════════════════════════════════════════════════════════════


def test_single_target_converts_source_gametes_at_declared_rate() -> None:
    """A point mutation splits source gametes ``1-r``/``r``; the parent genotype
    only sets the Mendelian baseline, since the rules carry no parent filter."""
    host = _host(_species())
    preset = nt.PointMutation(
        "SingleMut", source_allele="A", target_allele="B", mutation_rate=0.2
    )

    # Homozygote: every gamete carries the source allele.
    assert _gamete_row(host, "A|A", nt.Sex.FEMALE, preset) == {
        "A": pytest.approx(0.8),
        "B": pytest.approx(0.2),
    }
    # Heterozygote: the same 0.2 applies to the 0.5 source share only.
    assert _gamete_row(host, "A|B", nt.Sex.FEMALE, preset) == {
        "A": pytest.approx(0.5 * 0.8),
        "B": pytest.approx(0.5 + 0.5 * 0.2),
    }


def test_multi_target_shares_match_declared_rates_not_cascade_order() -> None:
    """Competing targets each keep their declared rate (TODO's worked example)."""
    host = _host(_species())
    preset = nt.PointMutation(
        "MultiMut",
        source_allele="A",
        target_alleles=["B", "C", "D"],
        mutation_rates=[0.3, 0.5, 0.1],
    )

    # Compensation table r'_k = r_k / (1 - sum_{i<k} r_i).
    np.testing.assert_allclose(
        preset.effective_rates(),
        [(0.3, 0.3), (0.5 / 0.7, 0.5 / 0.7), (0.1 / 0.2, 0.1 / 0.2)],
    )
    # Realized shares equal the declared rates; the residual source mass is
    # 1 - sum(r_k) = 0.1, not the cascade's 0.7 * 0.5 * 0.8.
    row = _gamete_row(host, "A|A", nt.Sex.FEMALE, preset)
    assert row == {
        "A": pytest.approx(0.1),
        "B": pytest.approx(0.3),
        "C": pytest.approx(0.5),
        "D": pytest.approx(0.1),
    }
    assert sum(row.values()) == pytest.approx(1.0)


def test_sex_specific_rates_compete_within_each_sex() -> None:
    """Each sex's rates are compensated independently; a zero sex adds no rule."""
    host = _host(_species())
    preset = nt.PointMutation(
        "SexedMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[{"female": 0.2, "male": 0.4}, (0.3, 0.1)],
    )

    # Female: 0.2 for B, then 0.3 of the remaining 0.8 for C.
    assert _gamete_row(host, "A|A", nt.Sex.FEMALE, preset) == {
        "A": pytest.approx(0.5),
        "B": pytest.approx(0.2),
        "C": pytest.approx(0.3),
    }
    # Male: 0.4 for B, then 0.1 of the remaining 0.6 for C.
    assert _gamete_row(host, "A|A", nt.Sex.MALE, preset) == {
        "A": pytest.approx(0.5),
        "B": pytest.approx(0.4),
        "C": pytest.approx(0.1),
    }
    np.testing.assert_allclose(
        preset.effective_rates(), [(0.2, 0.4), (0.3 / 0.8, 0.1 / 0.6)]
    )


def test_rate_pair_accepts_a_list() -> None:
    """A two-element list is read as the ``(female, male)`` pair, like a tuple."""
    preset = nt.PointMutation(
        "ListPairMut", source_allele="A", target_allele="B", mutation_rate=[0.1, 0.2]
    )

    assert preset.effective_rates() == ((0.1, 0.2),)


def test_zero_rate_sex_keeps_the_unmutated_baseline() -> None:
    """A sex with no declared rate gets no rule and no conversion at all."""
    host = _host(_species())
    preset = nt.PointMutation(
        "FemaleOnlyMut",
        source_allele="A",
        target_allele="B",
        mutation_rate=(0.0, 0.25),
    )

    assert _gamete_row(host, "A|A", nt.Sex.FEMALE, preset) == {"A": pytest.approx(1.0)}
    assert _gamete_row(host, "A|A", nt.Sex.MALE, preset) == {
        "A": pytest.approx(0.75),
        "B": pytest.approx(0.25),
    }


def test_all_zero_rates_produce_no_modifiers() -> None:
    """A preset that converts nothing registers neither modifier."""
    host = _host(_species())
    preset = nt.PointMutation(
        "SilentMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[0.0, 0.0],
        rate_mode="proportional",
    )

    assert preset.gamete_modifier(host) is None
    assert preset.zygote_modifier(host) is None


def test_proportional_rate_mode_normalizes_ratios() -> None:
    """``rate_mode="proportional"`` reads rates as proportions summing to 1."""
    host = _host(_species())
    preset = nt.PointMutation(
        "RatioMut",
        source_allele="A",
        target_alleles=["B", "C", "D"],
        mutation_rates=[2.0, 3.0, 5.0],
        rate_mode="proportional",
    )

    np.testing.assert_allclose(
        preset.effective_rates(),
        [(0.2, 0.2), (0.3 / 0.8, 0.3 / 0.8), (0.5 / 0.5, 0.5 / 0.5)],
    )
    # Declared proportions sum to 1, so the source allele is fully converted.
    assert _gamete_row(host, "A|A", nt.Sex.MALE, preset) == {
        "B": pytest.approx(0.2),
        "C": pytest.approx(0.3),
        "D": pytest.approx(0.5),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Germline-only scope (the embryonic channel is deferred)
# ══════════════════════════════════════════════════════════════════════════════


def test_zygote_stage_channel_is_not_registered() -> None:
    """The preset is germline-only: no zygote modifier is ever returned.

    The embryonic channel (`zygotic_mutation_rate`) was implemented and then
    deliberately withdrawn (TODO.legacy.md ARCH-021), so `zygote_modifier` must stay None
    even for a fully configured germline preset — this catches a reappearing
    zygote-stage effect that no documented parameter could switch off.
    """
    host = _host(_species())
    preset = nt.PointMutation(
        "GermlineOnly", source_allele="A", target_alleles=["B", "C"],
        mutation_rates=[0.3, 0.5],
    )

    assert preset.zygote_modifier(host) is None
    assert preset.gamete_modifier(host) is not None


def test_zygotic_mutation_rate_parameter_is_rejected() -> None:
    """The withdrawn embryonic parameter fails loudly instead of being ignored."""
    with pytest.raises(TypeError, match="zygotic_mutation_rate"):
        nt.PointMutation(
            "WithdrawnZygotic",
            source_allele="A",
            target_allele="B",
            mutation_rate=0.1,
            zygotic_mutation_rate=0.3,  # type: ignore[call-arg]  # withdrawn parameter under test
        )


# ══════════════════════════════════════════════════════════════════════════════
# Fitness patch and public integration
# ══════════════════════════════════════════════════════════════════════════════


def test_fitness_patch_covers_every_target_allele() -> None:
    """Fitness scaling is declared once for the whole target group."""
    preset = nt.PointMutation(
        "PatchMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[1e-5, 1e-6],
        viability_scaling=0.9,
        fecundity_scaling=0.8,
        fecundity_mode="recessive",
    )

    assert preset.fitness_patch() == {
        "viability_per_allele": {("B", "C"): (0.9, "multiplicative")},
        "fecundity_per_allele": {("B", "C"): (0.8, "recessive")},
        "sexual_selection_per_allele": {("B", "C"): (1.0, "multiplicative")},
        "zygote_per_allele": {("B", "C"): (1.0, "multiplicative")},
    }


def test_allele_inputs_accept_gene_objects() -> None:
    """``AlleleSpecifier`` allows Gene objects as well as names."""
    species = _species()
    preset = nt.PointMutation(
        "GeneInputMut",
        source_allele=species.get_gene("A"),
        target_alleles=[species.get_gene("B"), "C"],
        mutation_rates=[0.1, 0.1],
        species=species,
    )

    assert preset.effective_rates() == ((0.1, 0.1), (0.1 / 0.9, 0.1 / 0.9))
    assert tuple(gene.name for gene in preset.target_alleles) == ("B", "C")


def test_bound_allele_properties_resolve_genes() -> None:
    """``source_allele``/``target_alleles`` resolve names through the species."""
    species = _species()
    preset = nt.PointMutation(
        "BoundMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[1e-5, 1e-6],
        species=species,
    )

    assert preset.source_allele is species.get_gene("A")
    assert tuple(gene.name for gene in preset.target_alleles) == ("B", "C")


def test_preset_applies_through_the_public_builder() -> None:
    """A builder-applied preset writes the compensated table into the model."""
    species = _species("_point_mutation_e2e")
    preset = nt.PointMutation(
        "E2EMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[0.25, 0.5],
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(species=species, name="E2EPop", stochastic=False)
        .initial_state(
            individual_count={"female": {"A|A": 10}, "male": {"A|A": 10}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .presets(preset)
        .build()
    )

    registry = pop.index_registry
    parent = registry.ztype_index(species.get_genotype_from_str("A|A"), "default")
    row = {
        registry.index_to_gtype[gidx][0].to_string(): pytest.approx(prob)
        for gidx, prob in enumerate(pop.config.zygotes_to_gametes_map[0, parent, :])
        if prob > 0
    }
    assert row == {
        "A": pytest.approx(0.25),
        "B": pytest.approx(0.25),
        "C": pytest.approx(0.5),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Error paths
# ══════════════════════════════════════════════════════════════════════════════


def test_competing_rates_above_one_are_rejected_in_strict_mode() -> None:
    """Strict mode refuses rates that cannot all happen."""
    with pytest.raises(ValueError, match="exceeds 1"):
        nt.PointMutation(
            "OverfullMut",
            source_allele="A",
            target_alleles=["B", "C"],
            mutation_rates=[0.6, 0.5],
        )


def test_float_dust_sum_leaves_no_source_mass() -> None:
    """Rates that round to a sum of exactly 1 but keep a positive tail fail.

    ``0.5 + 0.5 + 1e-300`` rounds to 1.0 in floating point, so the sum check
    passes; the compensation then finds no unconverted mass for the tail.
    """
    with pytest.raises(ValueError, match="no unconverted source mass"):
        nt.PointMutation(
            "DustMut",
            source_allele="A",
            target_alleles=["B", "C", "D"],
            mutation_rates=[0.5, 0.5, 1e-300],
        )


def test_unknown_rate_mode_is_rejected() -> None:
    """An unknown ``rate_mode`` fails at construction, not at build."""
    with pytest.raises(ValueError, match="rate_mode must be one of"):
        nt.PointMutation(
            "BadModeMut",
            source_allele="A",
            target_allele="B",
            mutation_rate=0.1,
            rate_mode="scaled",  # type: ignore[arg-type]  # runtime boundary under test
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"source_allele": "A", "target_allele": "B"},
            "declare mutation_rate",
        ),
        (
            {"source_allele": "A", "mutation_rate": 0.1},
            "declare target_allele",
        ),
        (
            {
                "source_allele": "A",
                "target_allele": "B",
                "mutation_rate": 0.1,
                "mutation_rates": [0.1],
            },
            "not both",
        ),
        (
            {
                "source_allele": "A",
                "target_allele": "B",
                "target_alleles": ["C"],
                "mutation_rate": 0.1,
            },
            "declare either target_allele or target_alleles",
        ),
        (
            {"source_allele": "A", "target_allele": "A", "mutation_rate": 0.1},
            "equals the source allele",
        ),
        (
            {
                "source_allele": "A",
                "target_alleles": ["B", "B"],
                "mutation_rates": [0.1, 0.1],
            },
            "must be distinct",
        ),
        (
            {"source_allele": "A", "target_alleles": [], "mutation_rates": []},
            "at least one target allele",
        ),
        (
            {
                "source_allele": "A",
                "target_alleles": ["B", "C"],
                "mutation_rates": [0.1],
            },
            "expected 2 rate declaration",
        ),
        (
            {
                "source_allele": "A",
                "target_allele": "B",
                "mutation_rate": (0.1, 0.2, 0.3),
            },
            "must be a number, a \\(female, male\\) pair",
        ),
        (
            {
                "source_allele": "A",
                "target_allele": "B",
                "mutation_rate": {"female": "high"},
            },
            "a per-sex rate must be a number",
        ),
        (
            {
                "source_allele": "A",
                "target_allele": "B",
                "mutation_rate": {"woman": 0.1},
            },
            "a per-sex rate key must name female or male",
        ),
        (
            {"source_allele": "A", "target_allele": "B", "mutation_rate": {2: 0.1}},
            "a per-sex rate key must name female or male",
        ),
        (
            {"source_allele": "A", "target_allele": "B", "mutation_rate": "high"},
            "must be a number, a \\(female, male\\) pair, or a per-sex mapping",
        ),
        (
            {"source_allele": "A", "target_allele": "B", "mutation_rate": ("a", 0.1)},
            "must be a number, a \\(female, male\\) pair, or a per-sex mapping",
        ),
    ],
)
def test_invalid_declarations_raise_value_error(
    kwargs: dict[str, object], match: str
) -> None:
    """Each malformed declaration form is rejected with a specific message."""
    with pytest.raises(ValueError, match=match):
        nt.PointMutation("BadMut", **kwargs)  # type: ignore[arg-type]  # runtime boundary under test


def test_non_string_allele_is_rejected() -> None:
    """An allele input that is neither a Gene nor a string is a TypeError."""
    with pytest.raises(TypeError, match="source_allele must be a Gene"):
        nt.PointMutation(
            "BadAlleleMut",
            source_allele=7,  # type: ignore[arg-type]  # runtime boundary under test
            target_allele="B",
            mutation_rate=0.1,
        )


def test_unknown_allele_fails_at_build_time() -> None:
    """Allele existence is checked by the rule compiler against the species."""
    host = _host(_species("_point_mutation_unknown"))
    preset = nt.PointMutation(
        "GhostMut", source_allele="A", target_allele="Ghost", mutation_rate=0.1
    )

    with pytest.raises(ValueError, match="Ghost"):
        preset.gamete_modifier(host)


def test_cross_locus_target_fails_at_build_time() -> None:
    """A target on another chromosome is not a point mutation of the source."""
    host = _host(_two_locus_species("_point_mutation_cross_locus"))
    preset = nt.PointMutation(
        "CrossLocusMut", source_allele="A", target_allele="E", mutation_rate=0.1
    )

    with pytest.raises(ValueError, match="same locus"):
        preset.gamete_modifier(host)


# ══════════════════════════════════════════════════════════════════════════════
# Reconfiguration
# ══════════════════════════════════════════════════════════════════════════════


def test_reconfigure_preset_rebuilds_the_conversion_table() -> None:
    """``reconfigure_preset`` re-reads the written attribute and rebuilds.

    The generic updater writes the raw value through ``setattr``, so the
    preset must normalize whatever shape lands there before compiling.
    """
    species = _species("_point_mutation_reconfigure")
    preset = nt.PointMutation(
        "ReconfMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[0.2, 0.2],
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(species=species, name="ReconfPop", stochastic=False)
        .initial_state(
            individual_count={"female": {"A|A": 10}, "male": {"A|A": 10}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .presets(preset)
        .build()
    )

    pop.update().reconfigure_preset(preset, mutation_rates=[0.1, 0.3])

    registry = pop.index_registry
    parent = registry.ztype_index(species.get_genotype_from_str("A|A"), "default")
    row = {
        registry.index_to_gtype[gidx][0].to_string(): pytest.approx(prob)
        for gidx, prob in enumerate(pop.config.zygotes_to_gametes_map[0, parent, :])
        if prob > 0
    }
    assert row == {
        "A": pytest.approx(0.6),
        "B": pytest.approx(0.1),
        "C": pytest.approx(0.3),
    }


def test_reconfigure_single_target_with_a_bare_rate() -> None:
    """The single-target form tolerates a bare rate written by reconfigure."""
    preset = nt.PointMutation(
        "BareReconfMut", source_allele="A", target_allele="B", mutation_rate=0.1
    )

    preset.mutation_rates = 0.4  # type: ignore[assignment]  # runtime boundary under test

    assert preset.effective_rates() == ((0.4, 0.4),)


def test_reconfigure_with_a_malformed_rate_shape_raises() -> None:
    """A reconfigured rate value that cannot be normalized fails loudly."""
    host = _host(_species("_point_mutation_reconfigure_bad"))
    preset = nt.PointMutation(
        "BadReconfMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[0.1, 0.1],
    )
    preset.mutation_rates = "half"  # type: ignore[assignment]  # runtime boundary under test

    with pytest.raises(ValueError, match="must be a sequence with one rate declaration"):
        preset.gamete_modifier(host)

    preset.mutation_rates = (0.1, 0.1, 0.1)
    with pytest.raises(ValueError, match="expected 2 rate declaration"):
        preset.effective_rates()


def test_reconfigure_to_an_invalid_sum_fails_at_build() -> None:
    """Rates written past the strict bound are caught when the preset compiles."""
    host = _host(_species("_point_mutation_reconfigure_sum"))
    preset = nt.PointMutation(
        "SumReconfMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[0.1, 0.1],
    )
    preset.mutation_rates = [0.6, 0.5]

    with pytest.raises(ValueError, match="exceeds 1"):
        preset.gamete_modifier(host)


# ══════════════════════════════════════════════════════════════════════════════
# Public export
# ══════════════════════════════════════════════════════════════════════════════


def test_point_mutation_is_exported_from_the_public_surfaces() -> None:
    """``PointMutation`` is reachable from ``natal`` and the presets package."""
    from natal.frontend.presets import PointMutation as FrontendPointMutation

    assert nt.PointMutation is FrontendPointMutation
    assert "PointMutation" in nt.frontend.presets.__all__
