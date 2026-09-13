"""Tests for natal.frontend.modifiers — CR-1 unified conversion contracts.

Covers: declaration-time validation of the four keyword-only rule
classes, the gamete/zygote cascade engines (exact probabilities over the
unmodified Mendelian baseline), the negative contract for the removed
label-rule/Condition surface, and the surviving key-resolution/write
helpers.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import natal as nt
from natal.frontend.modifiers.conversion_rules import (
    GameteAlleleConversionRule,
    GameteGtypeConversionRule,
    ZygoteAlleleConversionRule,
    ZygoteZtypeConversionRule,
)
from natal.frontend.modifiers.gamete_conversion import GameteConversionRuleSet
from natal.frontend.modifiers.zygote_conversion import ZygoteConversionRuleSet
from natal.frontend.registry.index import IndexRegistry

from natal.frontend.modifiers.module import (  # noqa: F401  (used by kept helper tests)
    _normalize_zygote_val_to_distribution,
    _resolve_gtype_key,
    _write_zygote_distribution,
    evaluate_genotype_filter,
)


@pytest.fixture
def simple_species():
    """Minimal singleton-cached species: chr1/loc, alleles WT/Dr/R2."""
    return nt.Species.from_dict(
        name="SimpleSpecies",
        structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        gamete_labels=["default"],
    )


class TestResolveGtypeKey:
    """Tests for _resolve_gtype_key — gtype key resolution."""

    def test_int_passthrough(self, simple_species):
        """int keys pass through as-is."""
        registry = IndexRegistry()
        assert _resolve_gtype_key(7, registry) == 7

    def test_haploid_genotype_pair(self, simple_species):
        """(HaploidGenotype, glab_str) resolves via gtype_index."""
        hgs = simple_species.get_all_haploid_genotypes()
        registry = IndexRegistry()
        registry.register_gamete_label("default")
        registry.register_haplogenotype(hgs[0])
        registry.register_haplogenotype(hgs[1])

        # We test resolve_gtype_key with a single part
        result = _resolve_gtype_key((hgs[0], "default"), registry)
        assert result == 0  # hg0 * 1 + 0

    def test_int_pair_compressed(self, simple_species):
        """(int, int) pair resolves via registry.gtype_index."""
        hgs = simple_species.get_all_haploid_genotypes()
        registry = IndexRegistry()
        registry.register_gamete_label("default")
        registry.register_haplogenotype(hgs[0])

        result = _resolve_gtype_key((0, 0), registry)
        assert result == registry.gtype_index(hgs[0], "default")

    def test_non_tuple_int_passthrough(self, simple_species):
        """Bare int passes through."""
        registry = IndexRegistry()
        assert _resolve_gtype_key(42, registry) == 42

    def test_unknown_key_raises(self, simple_species):
        """Unrecognized key type raises KeyError."""
        registry = IndexRegistry()
        with pytest.raises(KeyError):
            _resolve_gtype_key(object(), registry)


# ============================================================================
# _normalize_zygote_val_to_distribution
# ============================================================================


class TestNormalizeZygoteVal:
    """Tests for _normalize_zygote_val_to_distribution."""

    def test_int_ztype_index(self, simple_species):
        """Integer ztype index becomes {index: 1.0}."""
        registry = IndexRegistry()
        result = _normalize_zygote_val_to_distribution(5, registry)
        assert result == {5: 1.0}

    def test_dict_distribution(self, simple_species):
        """Dict distribution passes through unchanged."""
        registry = IndexRegistry()
        result = _normalize_zygote_val_to_distribution({3: 0.7, 4: 0.3}, registry)
        assert result == {3: 0.7, 4: 0.3}

    def test_tuple_pair(self, simple_species):
        """(int, prob) tuple becomes {int: prob}."""
        registry = IndexRegistry()
        result = _normalize_zygote_val_to_distribution((3, 0.5), registry)
        assert result == {3: 0.5}

    def test_non_numeric_prob_raises(self, simple_species):
        """Dict with non-numeric probability raises AssertionError."""
        registry = IndexRegistry()
        with pytest.raises(AssertionError, match="probabilities must be numeric"):
            _normalize_zygote_val_to_distribution({3: "bad"}, registry)


# ============================================================================
# _write_zygote_distribution
# ============================================================================


class TestWriteZygoteDistribution:
    """Tests for _write_zygote_distribution."""

    def test_writes_to_tensor(self):
        """Distribution writes correct probabilities into the tensor slice."""
        n_gtypes, n_ztypes = 4, 3
        tensor = np.zeros((n_gtypes, n_gtypes, n_ztypes), dtype=np.float64)

        _write_zygote_distribution(tensor, 0, 1, {0: 1.0})

        assert tensor[0, 1, 0] == 1.0
        assert tensor[0, 1, 1] == 0.0
        assert tensor[0, 1, 2] == 0.0

    def test_zeros_matching_row(self):
        """Writing a distribution first clears the entire row."""
        n_gtypes, n_ztypes = 4, 3
        tensor = np.zeros((n_gtypes, n_gtypes, n_ztypes), dtype=np.float64)
        tensor[1, 2, :] = [0.3, 0.4, 0.3]

        _write_zygote_distribution(tensor, 1, 2, {1: 0.8, 2: 0.2})

        assert tensor[1, 2, 0] == 0.0
        assert tensor[1, 2, 1] == 0.8
        assert tensor[1, 2, 2] == 0.2


# ============================================================================
# evaluate_genotype_filter
# ============================================================================


class TestEvaluateGenotypeFilter:
    """Tests for evaluate_genotype_filter -- genotype filter evaluation."""

    def test_none_always_passes(self, simple_species):
        """None filter always returns (True, None)."""
        genotype = simple_species.get_all_genotypes()[0]
        passed, compiled = evaluate_genotype_filter(None, genotype, None)
        assert passed is True
        assert compiled is None

    def test_callable_true(self, simple_species):
        """Callable returning True."""
        genotype = simple_species.get_all_genotypes()[0]
        passed, compiled = evaluate_genotype_filter(
            lambda g: True, genotype, None
        )
        assert passed is True
        assert compiled is None

    def test_callable_false(self, simple_species):
        """Callable returning False."""
        genotype = simple_species.get_all_genotypes()[0]
        passed, compiled = evaluate_genotype_filter(
            lambda g: False, genotype, None
        )
        assert passed is False
        assert compiled is None


# ============================================================================
# Gamete conversion rules — construction, validation, repr
# ============================================================================





# ============================================================================
# CR-1: declaration-time validation of the four rule classes
# ============================================================================


class TestDeclarationValidation:
    """Keyword-only declarations validate rate/filters/to/side up front."""

    def test_rate_out_of_range(self):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="rate must be in"):
                GameteGtypeConversionRule(to="Dr@*", rate=bad)
            with pytest.raises(ValueError, match="rate must be in"):
                ZygoteAlleleConversionRule(from_allele="WT", to_allele="Dr", rate=bad)

    def test_rate_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            GameteAlleleConversionRule(from_allele="WT", to_allele="Dr", rate=float("inf"))

    def test_filters_unknown_key(self):
        with pytest.raises(ValueError, match="filter key 'bogus'"):
            GameteGtypeConversionRule(to="*@*", rate=0.5, filters={"bogus": "A"})
        with pytest.raises(ValueError, match="filter key 'parent'"):
            ZygoteZtypeConversionRule(to="*@*", rate=0.5, filters={"parent": "A"})

    def test_filters_stage_scoping(self):
        # parent_sex is gamete-stage only; maternal/paternal are zygote-stage only
        with pytest.raises(ValueError, match="filter key 'parent_sex'"):
            ZygoteZtypeConversionRule(to="*@*", rate=0.5, filters={"parent_sex": "both"})
        with pytest.raises(ValueError, match="filter key 'maternal'"):
            GameteAlleleConversionRule(
                from_allele="WT", to_allele="Dr", rate=0.5, filters={"maternal": "*"}
            )

    def test_filters_empty_pattern_and_bad_type(self):
        with pytest.raises(ValueError, match="non-empty pattern"):
            GameteGtypeConversionRule(to="*@*", rate=0.5, filters={"current": ""})
        with pytest.raises(TypeError, match="must be a Mapping"):
            GameteGtypeConversionRule(to="*@*", rate=0.5, filters=["current"])

    def test_target_requires_both_parts(self):
        for bad in ("Dr", "Dr@", "@tagged", "*@"):
            with pytest.raises(ValueError, match="both parts explicit"):
                GameteGtypeConversionRule(to=bad, rate=0.5)
        with pytest.raises(TypeError, match="must be a string"):
            GameteGtypeConversionRule(to=123, rate=0.5)

    def test_side_validation(self):
        with pytest.raises(ValueError, match="side must be one of"):
            ZygoteAlleleConversionRule(from_allele="WT", to_allele="Dr", rate=0.5, side="left")

    def test_allele_names_must_be_strings(self):
        with pytest.raises(TypeError, match="from_allele"):
            GameteAlleleConversionRule(from_allele=object(), to_allele="Dr", rate=0.5)


# ============================================================================
# CR-1: gamete cascade engine
# ============================================================================


def _two_allele_species(name):
    return nt.Species.from_dict(
        name=name, structure={"chr1": {"A": ["WT", "Dr"]}}, gamete_labels=["default", "tagged"]
    )


def _host_for(species):
    """A minimal RecipeHost: species + fresh full registry + baseline draft."""
    from natal.frontend.builder._registry_builder import build_registry
    from natal.frontend.genetics.compile import project_mendelian_maps
    from natal.frontend.model import build_discrete_engine_config

    registry = build_registry(species)
    z2g, g2z = project_mendelian_maps(species, registry)
    config = build_discrete_engine_config(
        n_genotypes=len(registry.index_to_genotype),
        n_gtypes=len(registry.index_to_gtype),
        n_glabs=len(registry.glab_labels),
        n_slabs=len(registry.slab_labels),
        zygotes_to_gametes_map=z2g,
        gametes_to_zygotes_map=g2z,
        has_sex_chromosomes=bool(species.get_sex_chromosome_groups()),
    )
    host = SimpleNamespace(species=species, registry=registry, index_registry=registry, config=config)
    return host


def _gamete_rows(modifier):
    """Invoke a gamete modifier; return {(sex, ztype): {gtype: prob}}."""
    return modifier()


class TestGameteCascade:
    """Cascade semantics over the unmodified Mendelian baseline."""

    def test_allele_rate_is_exact(self, simple_species):
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=0.3)
        rows = _gamete_rows(rs.to_gamete_modifier(host))
        reg = host.registry
        wt = reg.index_to_haplo[0]
        wt_def = reg.gtype_index(wt, "default")
        dr_def = reg.gtype_index(
            simple_species.get_haploid_genotype_from_str("Dr"), "default"
        )
        # Homozygous WT ztype row (female): 0.3 -> Dr, 0.7 stays WT.
        z_wt = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        row = rows[(0, z_wt)]
        assert row[dr_def] == pytest.approx(0.3)
        assert row[wt_def] == pytest.approx(0.7)

    def test_rate_zero_is_identity(self, simple_species):
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=0.0)
        rows = _gamete_rows(rs.to_gamete_modifier(host))
        reg = host.registry
        z_wt = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        row = rows[(0, z_wt)]
        wt_hg = simple_species.get_haploid_genotype_from_str("WT")
        assert row[reg.gtype_index(wt_hg, "default")] == pytest.approx(1.0)
        assert sum(row.values()) == pytest.approx(1.0)

    def test_rate_one_full_conversion(self, simple_species):
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=1.0)
        rows = _gamete_rows(rs.to_gamete_modifier(host))
        reg = host.registry
        z_wt = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        row = rows[(0, z_wt)]
        wt_hg = simple_species.get_haploid_genotype_from_str("WT")
        dr_hg = simple_species.get_haploid_genotype_from_str("Dr")
        assert row[reg.gtype_index(dr_hg, "default")] == pytest.approx(1.0)
        assert row.get(reg.gtype_index(wt_hg, "default"), 0.0) == pytest.approx(0.0)

    def test_gtype_target_keeps_or_swaps_parts(self):
        """``*`` keeps a part, a concrete part replaces it exactly."""
        sp = _two_allele_species("cr1_glabs")
        host = _host_for(sp)
        reg = host.registry
        rs = GameteConversionRuleSet()
        rs.add_gtype_convert(to="*@tagged", rate=1.0)
        rows = _gamete_rows(rs.to_gamete_modifier(host))
        z_wt = reg.ztype_index(sp.get_genotype_from_str("WT|WT"), "default")
        wt_hg = sp.get_haploid_genotype_from_str("WT")
        row = rows[(0, z_wt)]
        assert row[reg.gtype_index(wt_hg, "tagged")] == pytest.approx(1.0)
        assert row.get(reg.gtype_index(wt_hg, "default"), 0.0) == pytest.approx(0.0)

    def test_cascade_order_compounds(self, simple_species):
        """A->B then B->C: the second rule sees the first rule's branches."""
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=0.5)
        rs.add_allele_convert(from_allele="Dr", to_allele="R2", rate=0.5)
        rows = _gamete_rows(rs.to_gamete_modifier(host))
        reg = host.registry
        z_wt = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        wt_hg = simple_species.get_haploid_genotype_from_str("WT")
        dr_hg = simple_species.get_haploid_genotype_from_str("Dr")
        r2_hg = simple_species.get_haploid_genotype_from_str("R2")
        row = rows[(0, z_wt)]
        # 0.5 WT; of the 0.5 Dr, 0.5 -> R2 (0.25) and 0.25 stays Dr.
        assert row[reg.gtype_index(wt_hg, "default")] == pytest.approx(0.5)
        assert row[reg.gtype_index(dr_hg, "default")] == pytest.approx(0.25)
        assert row[reg.gtype_index(r2_hg, "default")] == pytest.approx(0.25)
        assert sum(row.values()) == pytest.approx(1.0)

    def test_current_filter_checks_entering_branch(self):
        """filters['current'] matches the branch state entering the rule."""
        sp = _two_allele_species("cr1_current")
        host = _host_for(sp)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=1.0)
        # Only Dr-carrying gametes get tagged; after rule 1 the converted
        # branches ARE Dr, so this fires on them (0.5 of the row).
        rs.add_gtype_convert(to="*@tagged", rate=1.0, filters={"current": "Dr"})
        rows = _gamete_rows(rs.to_gamete_modifier(host))
        reg = host.registry
        z_wt = reg.ztype_index(sp.get_genotype_from_str("WT|WT"), "default")
        dr_hg = sp.get_haploid_genotype_from_str("Dr")
        wt_hg = sp.get_haploid_genotype_from_str("WT")
        row = rows[(0, z_wt)]
        # Rule 1 converts all WT->Dr (rate 1.0); rule 2 then tags every
        # Dr branch (current=Dr) at 1.0, so the whole row ends tagged.
        assert row[reg.gtype_index(dr_hg, "tagged")] == pytest.approx(1.0)

    def test_parent_sex_filter(self, simple_species):
        """filters['parent_sex'] restricts conversion to one producer sex."""
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(
            from_allele="WT", to_allele="Dr", rate=1.0,
            filters={"parent_sex": "male"},
        )
        rows = _gamete_rows(rs.to_gamete_modifier(host))
        reg = host.registry
        z_wt = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        dr_hg = simple_species.get_haploid_genotype_from_str("Dr")
        wt_hg = simple_species.get_haploid_genotype_from_str("WT")
        assert rows[(1, z_wt)][reg.gtype_index(dr_hg, "default")] == pytest.approx(1.0)
        assert rows[(0, z_wt)][reg.gtype_index(wt_hg, "default")] == pytest.approx(1.0)

    def test_compile_from_baseline_never_stacks(self, simple_species):
        """Re-invoking the modifier re-derives from the Mendelian baseline."""
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=0.5)
        modifier = rs.to_gamete_modifier(host)
        first = _gamete_rows(modifier)
        second = _gamete_rows(modifier)
        assert first == second  # identical: no stacking across invocations

    def test_unknown_target_label_raises(self, simple_species):
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_gtype_convert(to="*@bogus", rate=1.0)
        with pytest.raises(ValueError, match="bogus"):
            rs.to_gamete_modifier(host)

    def test_unknown_target_genotype_raises(self, simple_species):
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_gtype_convert(to="Nope@*", rate=1.0)
        with pytest.raises(ValueError, match="Nope"):
            rs.to_gamete_modifier(host)

    def test_cross_locus_allele_target_raises(self):
        sp = nt.Species.from_dict(
            name="cr1_crossloc",
            structure={"chr1": {"A": ["WT", "Dr"], "B": ["B1", "B2"]}},
        )
        host = _host_for(sp)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="B1", rate=1.0)
        with pytest.raises(ValueError, match="same locus"):
            rs.to_gamete_modifier(host)

    def test_unknown_source_allele_raises(self, simple_species):
        host = _host_for(simple_species)
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="Nope", to_allele="Dr", rate=1.0)
        with pytest.raises(ValueError, match="Nope"):
            rs.to_gamete_modifier(host)


# ============================================================================
# CR-1: zygote cascade engine
# ============================================================================


class TestZygoteCascade:
    """Joint (Genotype, slab) branch semantics of the zygote engine."""

    def test_allele_side_both_splits_independently(self, simple_species):
        host = _host_for(simple_species)
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=0.5)
        rows = rs.to_zygote_modifier(host)()
        reg = host.registry
        # WT|WT x WT|WT: both copies convert at 0.5 independently.
        z = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        row = rows[(z, z)]
        dr = reg.ztype_index(simple_species.get_genotype_from_str("Dr|Dr"), "default")
        het = reg.ztype_index(simple_species.get_genotype_from_str("WT|Dr"), "default")
        assert row[z] == pytest.approx(0.25)
        assert row[het] == pytest.approx(0.5)
        assert row[dr] == pytest.approx(0.25)

    def test_allele_side_maternal_only(self, simple_species):
        host = _host_for(simple_species)
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=1.0, side="maternal")
        rows = rs.to_zygote_modifier(host)()
        reg = host.registry
        z = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        het = reg.ztype_index(simple_species.get_genotype_from_str("Dr|WT"), "default")
        row = rows[(z, z)]
        assert row[het] == pytest.approx(1.0)

    def test_allele_rule_keeps_slab(self, simple_species):
        """An allele conversion never changes the somatic label."""
        host = _host_for(simple_species)
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=1.0, side="both")
        rows = rs.to_zygote_modifier(host)()
        reg = host.registry
        z = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        dr = reg.ztype_index(simple_species.get_genotype_from_str("Dr|Dr"), "default")
        assert set(rows[(z, z)]) == {dr}

    def test_current_filter_fires_on_converted_branch(self, simple_species):
        """A rule with current=B after A->B DOES match (cascade fix)."""
        host = _host_for(simple_species)
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=1.0)
        rs.add_ztype_convert(to="WT|WT@*", rate=1.0, filters={"current": "Dr|Dr"})
        rows = rs.to_zygote_modifier(host)()
        reg = host.registry
        z = reg.ztype_index(simple_species.get_genotype_from_str("WT|WT"), "default")
        # After rule 1 the branch IS Dr|Dr, so the current=B rule reverts it.
        row = rows[(z, z)]
        assert row[z] == pytest.approx(1.0)

    def test_ztype_redirect_label_only(self):
        """``*@I`` moves the branch to the same genotype in slab I."""
        sp = _two_allele_species("cr1_slabs")
        sp.somatic_labels = ["default", "infected"]
        host = _host_for(sp)
        rs = ZygoteConversionRuleSet()
        rs.add_ztype_convert(to="*@infected", rate=1.0)
        rows = rs.to_zygote_modifier(host)()
        reg = host.registry
        z_def = reg.ztype_index(sp.get_genotype_from_str("WT|WT"), "default")
        z_inf = reg.ztype_index(sp.get_genotype_from_str("WT|WT"), "infected")
        assert rows[(z_def, z_def)][z_inf] == pytest.approx(1.0)

    def test_row_probability_conservation(self, simple_species):
        """Every rewritten row still sums to its baseline mass (0 or 1)."""
        host = _host_for(simple_species)
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="WT", to_allele="Dr", rate=0.3)
        rows = rs.to_zygote_modifier(host)()
        for pair, branches in rows.items():
            assert sum(branches.values()) == pytest.approx(1.0), pair

    def test_target_outside_axis_raises(self, simple_species):
        host = _host_for(simple_species)
        rs = ZygoteConversionRuleSet()
        # "Dr|Dr" genotype exists but the ruleset compiles against a
        # species whose registry has it; use a bogus slab instead.
        rs.add_ztype_convert(to="*@bogus", rate=1.0)
        with pytest.raises(ValueError, match="bogus"):
            rs.to_zygote_modifier(host)


# ============================================================================
# Negative contract: the replaced API surface is gone
# ============================================================================


class TestRemovedSurface:
    """CR-1 removed the label rules, aliases, and the Condition DSL."""

    def test_old_rule_classes_are_gone(self):
        import natal.frontend.modifiers as m

        for name in (
            "GameteGlabConversionRule",
            "GameteHaploidGenomeConversionRule",
            "ZygoteGlabRedirectRule",
            "ZygoteGenotypeConversionRule",
        ):
            assert not hasattr(m, name), name

    def test_condition_helpers_are_not_rule_state(self):
        """Rules carry only declarative filter pairs — no Condition objects."""
        rule = GameteGtypeConversionRule(to="*@*", rate=1.0)
        assert not hasattr(rule, "_when")
        assert rule.filter_pairs == ()

    def test_rule_sets_reject_foreign_rules(self):
        rs = GameteConversionRuleSet()
        with pytest.raises(TypeError):
            rs.add_rule(object())
        zs = ZygoteConversionRuleSet()
        with pytest.raises(TypeError):
            zs.add_rule(GameteGtypeConversionRule(to="*@*", rate=1.0))

    def test_convenience_methods_expose_every_field(self):
        """No convenience method fixes rate: it stays required."""
        with pytest.raises(TypeError):
            GameteConversionRuleSet().add_allele_convert(from_allele="WT")
        with pytest.raises(TypeError):
            ZygoteConversionRuleSet().add_ztype_convert(to="*@*")


# ============================================================================
# Wrapper pipeline registration (rewritten to the CR-1 API)
# ============================================================================


def _build_glab_pop():
    """Build a minimal age-structured population with two gamete labels."""
    sp = nt.Species.from_dict(
        name="_glab_test",
        structure={"chr1": {"A": ["WT", "Dr"]}},
        gamete_labels=["default", "tagged"],
    )
    return (
        nt.PopulationBuilder.for_age_structured(sp)
        .setup(stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state({"female": {"WT|WT": [0, 10, 0]}, "male": {"WT|WT": [0, 10, 0]}})
        .competition(carrying_capacity=100, low_density_growth_rate=1)
        .build()
    )


class TestBuildModifierWrappers:
    """Cover build_modifier_wrappers and the wrap_*_modifier pipeline."""

    def test_wraps_gamete_modifier(self):
        pop = _build_glab_pop()
        rs = GameteConversionRuleSet()
        rs.add_gtype_convert(to="*@tagged", rate=0.5, filters={"current": "*@default"})
        modifier = rs.to_gamete_modifier(pop)
        assert modifier is not None
        pop.add_gamete_modifier(modifier, name="test", refresh=True)
        assert len(pop._gamete_modifiers) > 0
        z2g = pop.config.zygotes_to_gametes_map
        assert z2g.shape[2] > 0

    def test_wraps_zygote_modifier(self):
        pop = _build_glab_pop()
        rs = ZygoteConversionRuleSet("test_zyg")
        gt = pop.species.get_genotype_from_str("WT|WT")
        rs.add_ztype_convert(to=f"{gt.to_string()}@*", rate=0.1)
        modifier = rs.to_zygote_modifier(pop)
        assert modifier is not None
        pop.add_zygote_modifier(modifier, name="test", refresh=True)
        assert len(pop._zygote_modifiers) > 0
        g2z = pop.config.gametes_to_zygotes_map
        assert g2z.shape[2] > 0

    def test_multiple_modifiers_compose(self):
        """Two label converts compose: every row sum stays 1.0."""
        pop = _build_glab_pop()
        rs = GameteConversionRuleSet()
        rs.add_gtype_convert(to="*@tagged", rate=0.4, filters={"current": "*@default"})
        rs.add_gtype_convert(to="*@tagged", rate=1.0, filters={"current": "*@tagged"})
        modifier = rs.to_gamete_modifier(pop)
        pop.add_gamete_modifier(modifier, name="test", refresh=True)
        z2g = pop.config.zygotes_to_gametes_map
        for sex in range(z2g.shape[0]):
            for z in range(z2g.shape[1]):
                s = float(z2g[sex, z, :].sum())
                assert s == pytest.approx(1.0) or s == pytest.approx(0.0)


class TestGameteModifierEmptyFreqs:
    """Empty rows fall through the cascade without breaking the build."""

    def test_modifier_with_all_ztypes_iterates(self):
        """Build a population where some ztypes have no gametes."""
        import natal as nt
        from natal.frontend.modifiers.gamete_conversion import GameteConversionRuleSet
        sp = nt.Species.from_dict(
            name="_empty_freqs",
            structure={"chr1": {"A": ["WT", "Dr"]}},
            gamete_labels=["default", "tagged"],
        )
        pop = (
            nt.PopulationBuilder.for_age_structured(sp)
            .setup(stochastic=False)
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state({"female": {"WT|WT": [0, 10, 0]}, "male": {"WT|WT": [0, 10, 0]}})
            .competition(carrying_capacity=100, low_density_growth_rate=1)
            .build()
        )
        # Add a modifier that converts default→tagged at 100% (CR-1 API)
        rs = GameteConversionRuleSet()
        rs.add_gtype_convert(to="*@tagged", rate=1.0, filters={"current": "*@default"})
        modifier = rs.to_gamete_modifier(pop)
        assert modifier is not None
        pop.add_gamete_modifier(modifier, name="test", refresh=True)
        # Operation succeeded — the continue at line 656 was hit for
        # ztypes without WT haplotypes
        assert pop.config.zygotes_to_gametes_map.shape[2] > 0
