"""Compile-time error contracts of the CR-1 conversion engines (evaluator).

TODO CR-1: "未知键、拼写错误、阶段不支持的键、非法模式均在编译时显式报错，
不解释为匹配失败。"  These tests pin the error branches of the two rule-set
compilers: invalid filter patterns and sexes, cross-locus allele targets,
and conversion targets that fall outside a *compressed* population's active
axis (surfaced when the compiled modifier cascades).

Reviewer finding (2026-09-13): the underlying haploid genome pattern parser
accepts essentially any token (unknown alleles, empty strings, unbalanced
brackets) and lets it match nothing, so the "invalid pattern" rejection on
the gamete ``current`` / zygote ``maternal`` / ``paternal`` filters is only
reachable through an empty ``@``-lab suffix (``"@@@"``).  Patterns naming
alleles that do not exist in the species compile and become silent no-ops
— see :class:`TestPatternUnknownAlleles` for the failing repair targets.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.genetics.compile import project_mendelian_maps
from natal.frontend.model import build_discrete_engine_config
from natal.frontend.modifiers.gamete_conversion import GameteConversionRuleSet
from natal.frontend.modifiers.zygote_conversion import ZygoteConversionRuleSet


def species_fixture() -> nt.Species:
    """Fresh single-locus species shared by the fixtures and direct calls."""
    return nt.Species.from_dict(
        name="_compile_errors",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default", "tagged"],
    )


@pytest.fixture
def species() -> Iterator[nt.Species]:
    yield species_fixture()


def _host(sp: nt.Species) -> SimpleNamespace:
    """A minimal RecipeHost over the species' full registry."""
    registry = build_registry(sp)
    z2g, g2z = project_mendelian_maps(sp, registry)
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
        species=sp, registry=registry, index_registry=registry, config=config
    )


def _compressed_population(base: nt.Species) -> nt.DiscreteGenerationPopulation:
    """Build a compressed population whose axis only contains WT states."""
    sp = nt.Species.from_dict(
        name=base.name + "_compressed",
        structure={"chr1": {"A": ["WT", "Dr"]}},
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=sp, name="_compile_errors_pop", stochastic=False,
            compress=True,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 100},
                "male": {"WT|WT": 100},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
        .build()
    )
    # Guard: the helper is only meaningful if compression actually pruned
    # the Dr states, so the target-outside-axis errors below are exercised.
    haplo_names = [hg.to_string() for hg in pop.index_registry.index_to_haplo]
    assert "Dr" not in haplo_names, "compression did not prune Dr; test is vacuous"
    return pop


class TestGameteCompileErrors:
    """Invalid gamete-rule declarations fail the compile/apply boundary."""

    def test_invalid_parent_sex_raises_at_compile(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_compile_errors_sex",
                structure={"chr1": {"A": ["WT", "Dr"]}},
            )
        )
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=1.0,
            filters={"parent_sex": "bogus"},
        )
        with pytest.raises(ValueError, match="invalid parent_sex"):
            rs.to_gamete_modifier(host)

    def test_parent_sex_both_is_accepted(self) -> None:
        """``parent_sex: both`` is the explicit no-op filter."""
        host = _host(
            nt.Species.from_dict(
                name="_compile_errors_both",
                structure={"chr1": {"A": ["WT", "Dr"]}},
            )
        )
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=1.0,
            filters={"parent_sex": "both"},
        )
        rows = rs.to_gamete_modifier(host)()
        assert rows  # compiled and cascaded, not rejected

    def test_syntactically_invalid_parent_pattern_raises(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_compile_errors_parent",
                structure={"chr1": {"A": ["WT", "Dr"]}},
            )
        )
        rs = GameteConversionRuleSet()
        # "WT|" is an empty allele pattern on the zygote (producer) side.
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=1.0,
            filters={"parent": "WT|"},
        )
        with pytest.raises(ValueError, match="invalid parent filter"):
            rs.to_gamete_modifier(host)

    def test_syntactically_invalid_current_label_raises(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_compile_errors_cur",
                structure={"chr1": {"A": ["WT", "Dr"]}},
            )
        )
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=1.0,
            filters={"current": "WT@@"},
        )
        with pytest.raises(ValueError, match="at most one @ separator"):
            rs.to_gamete_modifier(host)

    def test_target_outside_published_axis_rejects_runtime_update(
        self, species
    ) -> None:
        """Full compilation succeeds, but runtime publication may not expand axes."""
        pop = _compressed_population(species)
        rs = GameteConversionRuleSet()
        rs.add_gtype_convert(to="Dr@*", rate=1.0)
        modifier = rs.to_gamete_modifier(pop)
        with pytest.raises(ValueError, match="closed|external|inheritance"):
            pop.add_gamete_modifier(modifier, refresh=True)


class TestZygoteCompileErrors:
    """Invalid zygote-rule declarations fail the compile/apply boundary."""

    def test_invalid_current_filter_raises_at_compile(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_z_cur_err", structure={"chr1": {"A": ["WT", "Dr"]}}
            )
        )
        rs = ZygoteConversionRuleSet()
        # A bare token is not a valid zygote pattern (needs '|' or '::').
        rs.add_ztype_convert(to="*@*", rate=0.5, filters={"current": "Nope"})
        with pytest.raises(ValueError, match="invalid current filter"):
            rs.to_zygote_modifier(host)

    def test_syntactically_invalid_maternal_label_raises(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_z_mat_err", structure={"chr1": {"A": ["WT", "Dr"]}}
            )
        )
        rs = ZygoteConversionRuleSet()
        rs.add_ztype_convert(to="*@*", rate=0.5, filters={"maternal": "WT@@"})
        with pytest.raises(ValueError, match="at most one @ separator"):
            rs.to_zygote_modifier(host)

    def test_syntactically_invalid_paternal_label_raises(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_z_pat_err", structure={"chr1": {"A": ["WT", "Dr"]}}
            )
        )
        rs = ZygoteConversionRuleSet()
        rs.add_ztype_convert(to="*@*", rate=0.5, filters={"paternal": "WT@@"})
        with pytest.raises(ValueError, match="at most one @ separator"):
            rs.to_zygote_modifier(host)

    def test_cross_locus_allele_target_raises_on_zygote_stage(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_z_cross",
                structure={"chr1": {"A": ["WT", "Dr"], "B": ["B1", "B2"]}},
            )
        )
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="B1", rate=1.0)
        with pytest.raises(ValueError, match="same locus"):
            rs.to_zygote_modifier(host)

    def test_target_outside_published_axis_rejects_runtime_update(
        self, species
    ) -> None:
        pop = _compressed_population(species)
        rs = ZygoteConversionRuleSet()
        rs.add_ztype_convert(to="Dr|Dr@*", rate=1.0)
        modifier = rs.to_zygote_modifier(pop)
        with pytest.raises(ValueError, match="closed|external|inheritance"):
            pop.add_zygote_modifier(modifier, refresh=True)

    def test_allele_rule_leaves_unselected_copy_untouched(self) -> None:
        """A copy without the source allele stays; the other converts."""
        sp = nt.Species.from_dict(
            name="_z_het", structure={"chr1": {"A": ["WT", "Dr"]}}
        )
        host = _host(sp)
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=0.5)
        rows = rs.to_zygote_modifier(host)()
        reg = host.registry
        wt = reg.gtype_index(sp.get_haploid_genotype_from_str("WT"), "default")
        dr = reg.gtype_index(sp.get_haploid_genotype_from_str("Dr"), "default")
        het = reg.ztype_index(sp.get_genotype_from_str("WT|Dr"), "default")
        homo = reg.ztype_index(sp.get_genotype_from_str("Dr|Dr"), "default")
        row = rows[(min(wt, dr), max(wt, dr))]
        # The heterozygote has one WT copy: it converts independently at
        # rate 0.5 while the copy without the source allele stays.
        assert row[het] == pytest.approx(0.5)
        assert row[homo] == pytest.approx(0.5)


class TestPatternUnknownAlleles:
    """REPAIR TARGETS (evaluator, 2026-09-13) — compile-time pattern validity.

    TODO CR-1: "未知键、拼写错误、阶段不支持的键、非法模式均在编译时显式
    报错，不解释为匹配失败。"  A filter pattern naming an allele that does
    not exist in the species (e.g. a typo like ``"Dr|Wt"``) currently
    compiles, matches nothing, and the rule silently never fires — exactly
    the "interpreted as match failure" outcome the contract forbids.
    ``ZygoteTypePattern.parse("WT|Nope", species)`` and
    ``GenotypePatternParser.parse_haploid_genome_pattern("Nope")`` both
    succeed, and the compiled matcher simply returns False forever.

    Expected after repair: each compile below raises ``ValueError`` naming
    the unknown allele.  All four tests fail on the reviewed tree.
    """

    def test_gamete_current_pattern_unknown_allele_raises(self) -> None:
        host = _host(species_fixture())
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=1.0,
            filters={"current": "Nope"},
        )
        with pytest.raises(ValueError, match="Nope"):
            rs.to_gamete_modifier(host)

    def test_gamete_current_pattern_unknown_allele_in_pair_raises(self) -> None:
        host = _host(species_fixture())
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=1.0,
            filters={"current": "WT|Nope"},
        )
        with pytest.raises(ValueError, match="Nope"):
            rs.to_gamete_modifier(host)

    def test_zygote_current_pattern_unknown_allele_raises(self) -> None:
        host = _host(species_fixture())
        rs = ZygoteConversionRuleSet()
        rs.add_ztype_convert(to="*@*", rate=0.5, filters={"current": "WT|Nope"})
        with pytest.raises(ValueError, match="Nope"):
            rs.to_zygote_modifier(host)

    def test_zygote_maternal_pattern_unknown_allele_raises(self) -> None:
        host = _host(species_fixture())
        rs = ZygoteConversionRuleSet()
        rs.add_ztype_convert(to="*@*", rate=0.5, filters={"maternal": "Nope"})
        with pytest.raises(ValueError, match="Nope"):
            rs.to_zygote_modifier(host)


class TestRemainingCompileBranches:
    """Remaining compile-time validation branches (coverage completion)."""

    def test_zygote_invalid_target_genotype_raises(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_z_bad_tgt", structure={"chr1": {"A": ["WT", "Dr"]}}
            )
        )
        rs = ZygoteConversionRuleSet()
        rs.add_ztype_convert(to="Nope|Nope@*", rate=1.0)
        with pytest.raises(ValueError, match="not a valid diploid genotype"):
            rs.to_zygote_modifier(host)

    def test_zygote_unknown_source_allele_raises(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_z_bad_src", structure={"chr1": {"A": ["WT", "Dr"]}}
            )
        )
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="Nope", to_allele="Dr", rate=1.0)
        with pytest.raises(ValueError, match="not registered"):
            rs.to_zygote_modifier(host)

    def test_zygote_maternal_filter_nonmatching_rows_unchanged(self) -> None:
        """Rows whose maternal gamete fails the filter pass through untouched."""
        sp = nt.Species.from_dict(
            name="_z_nofilter",
            structure={"chr1": {"A": ["WT", "Dr"]}},
            gamete_labels=["default", "tagged"],
        )
        host = _host(sp)
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=1.0,
            filters={"maternal": "*@tagged"},
        )
        rows = rs.to_zygote_modifier(host)()
        reg = host.registry
        wt = reg.gtype_index(sp.get_haploid_genotype_from_str("WT"), "default")
        dr = reg.gtype_index(sp.get_haploid_genotype_from_str("Dr"), "default")
        z = reg.ztype_index(sp.get_genotype_from_str("WT|WT"), "default")
        # Every maternal gamete here is default-labeled: no branch converts.
        row = rows[(wt, wt)]
        assert row[z] == pytest.approx(1.0)
        assert row.get(dr, 0.0) == pytest.approx(0.0)

    def test_zygote_maternal_invalid_label_suffix_raises(self) -> None:
        host = _host(
            nt.Species.from_dict(
                name="_z_badlab", structure={"chr1": {"A": ["WT", "Dr"]}}
            )
        )
        rs = ZygoteConversionRuleSet()
        # The filter validator owns the single '@' scan, so "(" reaches
        # LabPattern.parse as the label part and is rejected there.
        rs.add_ztype_convert(to="*@*", rate=0.5, filters={"maternal": "WT@("})
        with pytest.raises(ValueError, match="invalid filter label"):
            rs.to_zygote_modifier(host)

    def test_zygote_allele_conversion_outside_axis_rejects_runtime_update(
        self, species
    ) -> None:
        """An allele event cannot open a new type in the runtime layout."""
        pop = _compressed_population(species)
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=1.0)
        modifier = rs.to_zygote_modifier(pop)
        with pytest.raises(ValueError, match="closed|external|inheritance"):
            pop.add_zygote_modifier(modifier, refresh=True)


class TestDeclarationBranchCompletion:
    """Declaration-time validation branches not hit by the contract tests."""

    def test_non_string_filter_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="filter keys must be strings"):
            GameteGtypeRule(to="*@*", rate=0.5, filters={1: "A"})

    def test_zygote_allele_names_must_be_strings(self) -> None:
        with pytest.raises(TypeError, match="to_allele"):
            ZygoteAlleleRule(
                from_allele="WT", to_allele=object(), rate=0.5,
            )
        with pytest.raises(TypeError, match="from_allele"):
            ZygoteAlleleRule(
                from_allele=object(), to_allele="Dr", rate=0.5,
            )

    def test_replace_allele_ignores_other_locus_target(self) -> None:
        """A target gene absent from the source's locus leaves the copy as-is."""
        sp = nt.Species.from_dict(
            name="_ra_other_locus",
            structure={"chr1": {"A": ["WT", "Dr"], "B": ["B1", "B2"]}},
        )
        hg = sp.get_haploid_genotype_from_str("WT/B1")
        assert replace_allele_in_haploid(hg, "WT", "B2") is None


from natal.frontend.modifiers.conversion_rules import (  # noqa: E402
    GameteGtypeConversionRule as GameteGtypeRule,
)
from natal.frontend.modifiers.conversion_rules import (  # noqa: E402
    ZygoteAlleleConversionRule as ZygoteAlleleRule,
)
from natal.frontend.modifiers.conversion_rules import (  # noqa: E402
    replace_allele_in_haploid,
)


class TestPatternTokenizerEdgeCases:
    """REPAIR TARGET (evaluator revalidation, 2026-09-13) — token coverage.

    ``validate_name`` (natal.frontend.utils.helpers) allows gene names
    matching ``[A-Za-z0-9_]+`` — including a leading digit.  The
    validator's token regex ``[A-Za-z_][A-Za-z0-9_]*`` mis-tokenizes such
    a name ("9L" becomes "L") and falsely rejects a legal pattern.
    Expected after repair: the filter compiles and cascades.  Fails on
    the current tree with "filter pattern names unknown allele 'L'".
    """

    def test_leading_digit_allele_name_compiles(self) -> None:
        sp = nt.Species.from_dict(
            name="_digit_gene", structure={"chr1": {"A": ["WT", "9L"]}}
        )
        host = _host(sp)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(
            from_allele="WT", to_allele="9L", rate=1.0,
            filters={"current": "9L"},
        )
        rows = rs.to_gamete_modifier(host)()
        assert rows  # a legal pattern must compile, not be rejected


class TestSharedLabelScan:
    """Both stages share one ``@`` scan, so malformed labels report alike.

    The conversion modules used to split the ``@label`` suffix with their own
    ``rsplit`` and parse the suffix a second time; the filter validator then
    scanned the same string again with a different rule.  The scan now lives
    in one function, which is what these tests pin.
    """

    @pytest.mark.parametrize("stage", ["gamete", "zygote"])
    @pytest.mark.parametrize(
        "pattern,message",
        [
            ("WT@x@y", "at most one @ separator"),
            ("WT@", "empty genotype or label"),
            ("@default", "empty genotype or label"),
            ("WT@(", "invalid filter label"),
            ("WT@nope", "unknown filter labels"),
        ],
    )
    def test_malformed_label_reports_the_shared_message(
        self, stage: str, pattern: str, message: str
    ) -> None:
        sp = nt.Species.from_dict(
            name=f"_shared_label_{stage}", structure={"chr1": {"A": ["WT", "Dr"]}}
        )
        host = _host(sp)
        if stage == "gamete":
            rules = GameteConversionRuleSet().add_allele_convert(
                from_allele="WT", to_allele="Dr", rate=1.0,
                filters={"current": pattern},
            )
            compile_it = rules.to_gamete_modifier
        else:
            rules = ZygoteConversionRuleSet().add_ztype_convert(
                to="*@*", rate=0.5, filters={"maternal": pattern}
            )
            compile_it = rules.to_zygote_modifier

        with pytest.raises(ValueError, match=message):
            compile_it(host)

    @pytest.mark.parametrize("stage", ["gamete", "zygote"])
    def test_legal_label_forms_still_compile(self, stage: str) -> None:
        """The shared scan must not narrow the accepted label forms."""
        sp = nt.Species.from_dict(
            name=f"_shared_label_ok_{stage}",
            structure={"chr1": {"A": ["WT", "Dr"]}},
            gamete_labels=["default", "cas9"],
        )
        host = _host(sp)
        for pattern in ("WT", "WT@cas9", "WT@*"):
            if stage == "gamete":
                rules = GameteConversionRuleSet().add_allele_convert(
                    from_allele="WT", to_allele="Dr", rate=1.0,
                    filters={"current": pattern},
                )
                modifier = rules.to_gamete_modifier(host)
            else:
                rules = ZygoteConversionRuleSet().add_ztype_convert(
                    to="*@*", rate=0.5, filters={"maternal": pattern}
                )
                modifier = rules.to_zygote_modifier(host)
            assert modifier is not None, pattern
