"""Tests for natal.modifiers — unified key resolution and write pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from natal.modifiers.module import (
    _normalize_zygote_val_to_distribution,
    _resolve_gtype_key,
    _write_zygote_distribution,
    evaluate_genotype_filter,
)
from natal.registry.index import IndexRegistry

# ============================================================================
# _resolve_gtype_key
# ============================================================================


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
        """Unrecognised key type raises KeyError."""
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


class TestGameteRules:
    """Unit tests for gamete conversion rule classes."""

    def test_ztype_rule_validation_rate(self, simple_species):
        """Invalid rate raises ValueError."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        with pytest.raises(ValueError, match="rate must be in"):
            GameteGtypeConversionRule(hg_match=hg, to_haploid_genotype=hg, rate=1.5)

    def test_ztype_rule_validation_type(self, simple_species):
        """Invalid hg_match type raises TypeError."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        with pytest.raises(TypeError, match="hg_match must be"):
            GameteGtypeConversionRule(hg_match=object(), to_haploid_genotype=hg, rate=0.5)

    def test_ztype_rule_validation_to_type(self, simple_species):
        """Invalid to_haploid_genotype raises TypeError."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        with pytest.raises(TypeError, match="to_haploid_genotype must be"):
            GameteGtypeConversionRule(hg_match=hg, to_haploid_genotype=object(), rate=0.5)

    def test_ztype_rule_matches(self, simple_species):
        """matches() delegates to the match function."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hgs = simple_species.get_all_haploid_genotypes()
        rule = GameteGtypeConversionRule(hg_match=hgs[0], to_haploid_genotype=hgs[1], rate=0.5)
        assert rule.matches(hgs[0]) is True
        assert rule.matches(hgs[1]) is False

    def test_ztype_rule_replacement(self, simple_species):
        """replacement() returns the configured target."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hgs = simple_species.get_all_haploid_genotypes()
        rule = GameteGtypeConversionRule(hg_match=hgs[0], to_haploid_genotype=hgs[1], rate=0.5)
        assert rule.replacement(hgs[0]) is hgs[1]

    def test_ztype_rule_applies_to_sex(self, simple_species):
        """applies_to_sex() respects sex_filter."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        rule = GameteGtypeConversionRule(hg_match=hg, to_haploid_genotype=hg, rate=0.5)
        assert rule.applies_to_sex(0) is True
        assert rule.applies_to_sex("female") is True
        # both sexes by default
        rule_f = GameteGtypeConversionRule(hg_match=hg, to_haploid_genotype=hg, rate=0.5, sex_filter=0)
        assert rule_f.applies_to_sex(0) is True
        assert rule_f.applies_to_sex(1) is False
        assert rule_f.applies_to_sex("male") is False

    def test_ztype_rule_applies_to_genotype(self, simple_species):
        """applies_to_genotype() uses the genotype filter."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        gt = simple_species.get_all_genotypes()[0]
        rule = GameteGtypeConversionRule(hg_match=hg, to_haploid_genotype=hg, rate=0.5)
        assert rule.applies_to_genotype(gt) is True  # no filter = passes

    def test_ztype_rule_repr(self, simple_species):
        """__repr__ includes name and rate."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        rule = GameteGtypeConversionRule(hg_match=hg, to_haploid_genotype=hg, rate=0.3)
        r = repr(rule)
        assert "0.3" in r

    def test_glab_rule_validation_rate(self):
        """Invalid rate raises ValueError."""
        from natal.modifiers.gamete_conversion import GameteGlabConversionRule
        with pytest.raises(ValueError, match="rate must be in"):
            GameteGlabConversionRule(from_glab="a", to_glab="b", rate=2.0)

    def test_glab_rule_matches(self, simple_species):
        """glab rule always matches any haploid genotype."""
        from natal.modifiers.gamete_conversion import GameteGlabConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        rule = GameteGlabConversionRule(from_glab="a", to_glab="b", rate=1.0)
        assert rule.matches(hg) is True

    def test_glab_rule_replacement(self, simple_species):
        """replacement() returns the same haploid genotype unchanged."""
        from natal.modifiers.gamete_conversion import GameteGlabConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        rule = GameteGlabConversionRule(from_glab="a", to_glab="b", rate=1.0)
        assert rule.replacement(hg) is hg

    def test_glab_rule_applies_to_sex(self):
        """applies_to_sex() respects sex_filter."""
        from natal.modifiers.gamete_conversion import GameteGlabConversionRule
        rule = GameteGlabConversionRule(from_glab="a", to_glab="b", rate=1.0, sex_filter=0)
        assert rule.applies_to_sex(0) is True
        assert rule.applies_to_sex(1) is False

    def test_glab_rule_applies_to_genotype(self, simple_species):
        """applies_to_genotype() is True by default."""
        from natal.modifiers.gamete_conversion import GameteGlabConversionRule
        gt = simple_species.get_all_genotypes()[0]
        rule = GameteGlabConversionRule(from_glab="a", to_glab="b", rate=1.0)
        assert rule.applies_to_genotype(gt) is True

    def test_glab_rule_repr(self):
        """__repr__ includes glab names."""
        from natal.modifiers.gamete_conversion import GameteGlabConversionRule
        rule = GameteGlabConversionRule(from_glab="X", to_glab="Y", rate=0.5)
        r = repr(rule)
        assert "X" in r and "Y" in r

    def test_allele_rule_validation_rate(self):
        """Invalid rate raises ValueError."""
        from natal.modifiers.gamete_conversion import GameteAlleleConversionRule
        with pytest.raises(ValueError, match="rate must be in"):
            GameteAlleleConversionRule(from_allele="A", to_allele="B", rate=-0.1)

    def test_allele_rule_repr(self):
        """__repr__ includes conversion info."""
        from natal.modifiers.gamete_conversion import GameteAlleleConversionRule
        rule = GameteAlleleConversionRule(from_allele="A", to_allele="B", rate=0.5)
        r = repr(rule)
        assert "A" in r and "B" in r


# ============================================================================
# Zygote conversion rules — construction, validation, repr
# ============================================================================


class TestZygoteRules:
    """Unit tests for zygote conversion rule classes."""

    def test_ztype_rule_validation_rate(self, simple_species):
        """Invalid rate raises ValueError."""
        from natal.modifiers.zygote_conversion import ZygoteZtypeConversionRule
        gt = simple_species.get_all_genotypes()[0]
        with pytest.raises(ValueError, match="rate must be in"):
            ZygoteZtypeConversionRule(genotype_match=gt, to_genotype=gt, rate=2.0)

    def test_ztype_rule_validation_type(self, simple_species):
        """Invalid genotype_match raises TypeError."""
        from natal.modifiers.zygote_conversion import ZygoteZtypeConversionRule
        gt = simple_species.get_all_genotypes()[0]
        with pytest.raises(TypeError, match="genotype_match must be"):
            ZygoteZtypeConversionRule(genotype_match=object(), to_genotype=gt, rate=0.5)

    def test_ztype_rule_validation_to_type(self, simple_species):
        """Invalid to_genotype raises TypeError."""
        from natal.modifiers.zygote_conversion import ZygoteZtypeConversionRule
        gt = simple_species.get_all_genotypes()[0]
        with pytest.raises(TypeError, match="to_genotype must be"):
            ZygoteZtypeConversionRule(genotype_match=gt, to_genotype=object(), rate=0.5)

    def test_ztype_rule_matches(self, simple_species):
        """matches() by identity when given a Genotype."""
        from natal.modifiers.zygote_conversion import ZygoteZtypeConversionRule
        gts = simple_species.get_all_genotypes()
        rule = ZygoteZtypeConversionRule(genotype_match=gts[0], to_genotype=gts[1], rate=0.5)
        assert rule.matches(gts[0]) is True
        assert rule.matches(gts[1]) is False

    def test_ztype_rule_replacement(self, simple_species):
        """replacement() returns the configured target."""
        from natal.modifiers.zygote_conversion import ZygoteZtypeConversionRule
        gts = simple_species.get_all_genotypes()
        rule = ZygoteZtypeConversionRule(genotype_match=gts[0], to_genotype=gts[1], rate=0.5)
        assert rule.replacement(gts[0]) is gts[1]

    def test_ztype_rule_callable_match(self, simple_species):
        """Callable match predicate works."""
        from natal.modifiers.zygote_conversion import ZygoteZtypeConversionRule
        gts = simple_species.get_all_genotypes()
        rule = ZygoteZtypeConversionRule(
            genotype_match=lambda g: True, to_genotype=gts[0], rate=0.5,
        )
        assert rule.matches(gts[1]) is True

    def test_ztype_rule_repr(self, simple_species):
        """__repr__ includes name and rate."""
        from natal.modifiers.zygote_conversion import ZygoteZtypeConversionRule
        gt = simple_species.get_all_genotypes()[0]
        rule = ZygoteZtypeConversionRule(genotype_match=gt, to_genotype=gt, rate=0.3)
        r = repr(rule)
        assert "0.3" in r

    def test_glab_redirect_validation_rate(self):
        """Invalid rate raises ValueError."""
        from natal.modifiers.zygote_conversion import ZygoteGlabRedirectRule
        with pytest.raises(ValueError, match="rate must be in"):
            ZygoteGlabRedirectRule(from_glab="a", to_glab="b", rate=2.0)

    def test_glab_redirect_matches(self, simple_species):
        """Always matches any genotype."""
        from natal.modifiers.zygote_conversion import ZygoteGlabRedirectRule
        gt = simple_species.get_all_genotypes()[0]
        rule = ZygoteGlabRedirectRule(from_glab="a", to_glab="b", rate=1.0)
        assert rule.matches(gt) is True

    def test_glab_redirect_replacement(self, simple_species):
        """replacement() returns genotype unchanged."""
        from natal.modifiers.zygote_conversion import ZygoteGlabRedirectRule
        gt = simple_species.get_all_genotypes()[0]
        rule = ZygoteGlabRedirectRule(from_glab="a", to_glab="b", rate=1.0)
        assert rule.replacement(gt) is gt

    def test_glab_redirect_repr(self):
        """__repr__ includes glab names."""
        from natal.modifiers.zygote_conversion import ZygoteGlabRedirectRule
        rule = ZygoteGlabRedirectRule(from_glab="X", to_glab="Y", rate=0.5)
        r = repr(rule)
        assert "X" in r and "Y" in r

    def test_allele_rule_validation_rate(self):
        """Invalid rate raises ValueError."""
        from natal.modifiers.zygote_conversion import ZygoteAlleleConversionRule
        with pytest.raises(ValueError, match="rate must be in"):
            ZygoteAlleleConversionRule(from_allele="A", to_allele="B", rate=-0.1)

    def test_allele_rule_repr(self):
        """__repr__ includes allele names."""
        from natal.modifiers.zygote_conversion import ZygoteAlleleConversionRule
        rule = ZygoteAlleleConversionRule(from_allele="A", to_allele="B", rate=0.5)
        r = repr(rule)
        assert "A" in r and "B" in r


# ============================================================================
# RuleSet construction
# ============================================================================


class TestRuleSets:
    """Unit tests for GameteConversionRuleSet and ZygoteConversionRuleSet."""

    def test_gamete_ruleset_add_convert(self, simple_species):
        """add_convert appends a rule and returns self."""
        from natal.modifiers.gamete_conversion import GameteConversionRuleSet
        hg = simple_species.get_all_haploid_genotypes()[0]
        rs = GameteConversionRuleSet()
        result = rs.add_glab_convert(from_glab='a', to_glab='b', rate=0.5)
        assert result is rs
        assert len(rs.rules) == 1

    def test_gamete_ruleset_add_glab_convert(self):
        """add_glab_convert appends a glab rule."""
        from natal.modifiers.gamete_conversion import GameteConversionRuleSet
        rs = GameteConversionRuleSet()
        rs.add_glab_convert(from_glab="a", to_glab="b", rate=1.0)
        assert len(rs.rules) == 1

    def test_gamete_ruleset_add_allele_convert(self):
        """add_allele_convert appends an allele rule."""
        from natal.modifiers.gamete_conversion import GameteConversionRuleSet
        rs = GameteConversionRuleSet()
        rs.add_allele_convert(from_allele="A", to_allele="B", rate=0.5)
        assert len(rs.rules) == 1

    def test_gamete_ruleset_repr(self):
        """__repr__ includes rule count."""
        from natal.modifiers.gamete_conversion import GameteConversionRuleSet
        rs = GameteConversionRuleSet(name="test")
        r = repr(rs)
        assert "test" in r and "0 rules" in r

    def test_zygote_ruleset_add_convert(self, simple_species):
        """add_convert appends a rule and returns self."""
        from natal.modifiers.zygote_conversion import ZygoteConversionRuleSet
        gt = simple_species.get_all_genotypes()[0]
        rs = ZygoteConversionRuleSet()
        result = rs.add_convert(genotype_match=gt, to_genotype=gt, rate=0.5)
        assert result is rs
        assert len(rs.rules) == 1

    def test_zygote_ruleset_add_glab_redirect(self):
        """add_glab_redirect appends a rule."""
        from natal.modifiers.zygote_conversion import ZygoteConversionRuleSet
        rs = ZygoteConversionRuleSet()
        rs.add_glab_redirect(from_glab="a", to_glab="b")
        assert len(rs.rules) == 1

    def test_zygote_ruleset_add_allele_convert(self):
        """add_allele_convert appends an allele rule."""
        from natal.modifiers.zygote_conversion import ZygoteConversionRuleSet
        rs = ZygoteConversionRuleSet()
        rs.add_allele_convert(from_allele="A", to_allele="B", rate=0.5)
        assert len(rs.rules) == 1

    def test_zygote_ruleset_repr(self):
        """__repr__ includes rule count."""
        from natal.modifiers.zygote_conversion import ZygoteConversionRuleSet
        rs = ZygoteConversionRuleSet(name="test")
        r = repr(rs)
        assert "test" in r and "0 rules" in r


# ============================================================================
# Condition base class
# ============================================================================


class TestConditionBase:
    """Tests for the Condition abstract base class."""

    def test_base_matches_raises(self):
        """Calling _matches on the base Condition raises NotImplementedError."""
        from natal.modifiers.conditions import Condition
        from natal.registry.index import IndexRegistry
        c = Condition()
        with pytest.raises(NotImplementedError):
            c._matches(0, 0, None, "", IndexRegistry())  # type: ignore[arg-type]

    def test_and_operator(self):
        """& operator creates _And."""
        from natal.modifiers.conditions import _And, sex
        c = sex("female") & sex("male")  # type: ignore[operator]
        assert isinstance(c, _And)

    def test_or_operator(self):
        """| operator creates _Or."""
        from natal.modifiers.conditions import _Or, sex
        c = sex("female") | sex("male")  # type: ignore[operator]
        assert isinstance(c, _Or)


# ============================================================================
# Gamete rule — callable match/replacement paths
# ============================================================================


class TestGameteRuleCallables:
    """Cover callable-based match and replacement paths."""

    def test_callable_match(self, simple_species):
        """hg_match as a callable."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        rule = GameteGtypeConversionRule(
            hg_match=lambda h: h is hg,
            to_haploid_genotype=hg, rate=0.5,
        )
        assert rule.matches(hg) is True

    def test_callable_replacement(self, simple_species):
        """to_haploid_genotype as a callable."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hgs = simple_species.get_all_haploid_genotypes()
        rule = GameteGtypeConversionRule(
            hg_match=hgs[0],
            to_haploid_genotype=lambda h: hgs[1], rate=0.5,
        )
        assert rule.replacement(hgs[0]) is hgs[1]

    def test_invalid_to_type_raises(self, simple_species):
        """Non-HaploidGenotype/callable raises TypeError."""
        from natal.modifiers.gamete_conversion import GameteGtypeConversionRule
        hg = simple_species.get_all_haploid_genotypes()[0]
        with pytest.raises(TypeError, match="to_haploid_genotype must be"):
            GameteGtypeConversionRule(hg_match=hg, to_haploid_genotype=object(), rate=0.5)


# ============================================================================
# Zygote allele helpers
# ============================================================================


class TestZygoteAlleleHelpers:
    """Cover _replace_allele_in_haploid and _convert_diploid_genotype_to_gts."""

    def test_replace_allele_not_present(self, simple_species):
        """Returns None when allele absent."""
        from natal.modifiers.zygote_conversion import _replace_allele_in_haploid
        hgs = simple_species.get_all_haploid_genotypes()
        result = _replace_allele_in_haploid(hgs[0], "NONEXISTENT", "WT")
        assert result is None

    def test_convert_diploid_no_match(self, simple_species):
        """Returns None when no allele matches."""
        from natal.modifiers.zygote_conversion import (
            ZygoteAlleleConversionRule,
            _convert_diploid_genotype_to_gts,
        )
        gt = simple_species.get_all_genotypes()[0]
        rule = ZygoteAlleleConversionRule("NONEXISTENT", "WT", rate=0.5)
        result = _convert_diploid_genotype_to_gts(gt, rule)
        assert result is None


# ============================================================================
# RuleSet validation
# ============================================================================


class TestRuleSetValidation:
    """Cover RuleSet add_rule with invalid types."""

    def test_gamete_ruleset_bad_rule_raises(self):
        """add_rule with wrong type raises AssertionError."""
        from natal.modifiers.gamete_conversion import GameteConversionRuleSet
        rs = GameteConversionRuleSet()
        with pytest.raises(AssertionError):
            rs.add_rule(object())  # type: ignore[arg-type]

    def test_zygote_ruleset_bad_rule_raises(self):
        """add_rule with wrong type raises AssertionError."""
        from natal.modifiers.zygote_conversion import ZygoteConversionRuleSet
        rs = ZygoteConversionRuleSet()
        with pytest.raises(AssertionError):
            rs.add_rule(object())  # type: ignore[arg-type]
