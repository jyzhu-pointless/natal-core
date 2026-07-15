"""Tests for natal.patterns.individual_selector.IndividualSelector.

Every assertion proves a mathematical invariant — exact mask comparisons,
coordinate-set cardinalities, or algebraic laws (commutativity, associativity,
idempotence of OR on compiled masks).
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.patterns import IndividualSelector
from natal.registry.index import IndexRegistry
from natal.utils.types import Sex

# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def simple_species() -> nt.Species:
    """2-allele unordered species: WT|WT, WT|Dr, Dr|Dr."""
    return nt.Species.from_dict(
        name="test_is_species",
        structure={"chr1": {"loc1": ["WT", "Dr"]}},
    )


@pytest.fixture(scope="module")
def simple_registry(simple_species: nt.Species) -> IndexRegistry:
    """Registry with all 3 genotypes → ztype indices 0, 1, 2."""
    reg = IndexRegistry()
    for g in simple_species.get_all_genotypes():
        reg.register_genotype(g)
    assert reg.n_ztypes == 3
    return reg


# ZType indices for the simple_species (unordered), by genotype string:
#   0 = WT|WT
#   1 = WT|Dr
#   2 = Dr|Dr


# ── Helper ───────────────────────────────────────────────────────────────────


def _assert_mask_equal(
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    """Exact boolean mask equality."""
    assert actual.dtype == bool
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} vs {expected.shape}"
    assert np.array_equal(actual, expected), (
        f"Masks differ.\nActual sum: {actual.sum()}\nExpected sum: {expected.sum()}\n"
        f"Actual True at: {np.argwhere(actual)}\nExpected True at: {np.argwhere(expected)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Construction & Normalization
# ══════════════════════════════════════════════════════════════════════════════


class TestConstructionDefault:
    """Invariants for default (all-None) construction."""

    def test_default_is_all_wildcard(self):
        """Default IndividualSelector() matches every coordinate."""
        s = IndividualSelector()
        assert s.n_atoms == 1
        assert s.is_empty is True  # empty == all-wildcard, matches everything

    def test_default_compile_all_true(self, simple_registry):
        """Identity selector → all-True mask of shape (2, 1, 3)."""
        s = IndividualSelector()
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.ones((2, 1, 3), dtype=bool)
        _assert_mask_equal(mask, expected)

    def test_default_compile_coordinates_cardinality(self, simple_registry):
        """Identity selector selects all 2 × 1 × 3 = 6 coordinates."""
        s = IndividualSelector()
        coords = s.compile_coordinates(simple_registry, n_sexes=2, n_ages=1)
        assert len(coords) == 2 * 1 * 3  # 6


class TestConstructionZtype:
    """Invariants for ztype-only construction."""

    def test_ztype_star_drive_compile(self, simple_registry):
        """'*|Dr' matches both genotypes carrying Dr: WT|Dr (1) and Dr|Dr (2)."""
        s = IndividualSelector(ztype="*|Dr")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        # ZType cols 1, 2 are True; both sexes (0, 1) and age 0
        expected[:, 0, 1] = True
        expected[:, 0, 2] = True
        _assert_mask_equal(mask, expected)

    def test_ztype_wt_wt_compile(self, simple_registry):
        """'WT|WT' selects exactly ztype index 0."""
        s = IndividualSelector(ztype="WT|WT")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[:, 0, 0] = True
        _assert_mask_equal(mask, expected)

    def test_ztype_dr_dr_compile(self, simple_registry):
        """'Dr|Dr' selects exactly ztype index 2."""
        s = IndividualSelector(ztype="Dr|Dr")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[:, 0, 2] = True
        _assert_mask_equal(mask, expected)

    def test_ztype_wildcard_star_compile(self, simple_registry):
        """'*' ztype selects all ztypes."""
        s = IndividualSelector(ztype="*")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.ones((2, 1, 3), dtype=bool)
        _assert_mask_equal(mask, expected)

    def test_ztype_as_genotype_tuple_with_slab(self, simple_species, simple_registry):
        """ztype=(Genotype, slab_label) → normalised to string with @slab.
        A non-existent slab (e.g. 'infected') resolves to nothing → ValueError."""
        gt = simple_species.get_genotype_from_str("WT|WT")
        s = IndividualSelector(ztype=(gt, "infected"))
        with pytest.raises(ValueError, match="selects no"):
            s.compile(simple_registry, n_sexes=2, n_ages=1)

    def test_ztype_as_genotype_tuple_without_slab(self, simple_species, simple_registry):
        """ztype=(Genotype, '') → normalised to genotype string without slab."""
        gt = simple_species.get_genotype_from_str("WT|WT")
        s = IndividualSelector(ztype=(gt, ""))
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        # "WT|WT" → ztype pattern matches index 0
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[:, 0, 0] = True
        _assert_mask_equal(mask, expected)

    def test_ztype_as_zygote_type_pattern_normalizes_via_str(self, simple_species, simple_registry):
        """ztype=ZygoteTypePattern → normalised via str() (produces repr).
        The repr of a ZygoteTypePattern is not a valid pattern syntax,
        so _resolve_ztype will raise PatternParseError at compile time."""
        from natal.patterns import PatternParseError, ZygoteTypePattern
        pattern = ZygoteTypePattern.parse("*|Dr", simple_species)
        s = IndividualSelector(ztype=pattern)
        # _to_tuple_ztype calls str(value) which gives the repr string
        # repr is not re-parseable → PatternParseError in _resolve_ztype
        with pytest.raises(PatternParseError):
            s.compile(simple_registry, n_sexes=2, n_ages=1)

    def test_invalid_ztype_spec_type_raises(self):
        """Non-string, non-tuple, non-ZTP ztype → TypeError."""
        with pytest.raises(TypeError, match="Unsupported ztype spec type"):
            IndividualSelector(ztype=42)  # type: ignore[arg-type]  # int is not a valid ztype spec


class TestConstructionSex:
    """Invariants for sex-only construction."""

    def test_sex_female_str_normalizes_to_female(self, simple_registry):
        """'female' → Sex.FEMALE (0). Mask: only row 0 is True."""
        s = IndividualSelector(sex="female")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[0, :, :] = True
        _assert_mask_equal(mask, expected)

    def test_sex_female_abbrev(self, simple_registry):
        """'f' is accepted as female abbreviation."""
        s = IndividualSelector(sex="f")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[0, :, :] = True
        _assert_mask_equal(mask, expected)

    def test_sex_male_str_normalizes_to_male(self, simple_registry):
        """'male' → Sex.MALE (1). Mask: only row 1 is True."""
        s = IndividualSelector(sex="male")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[1, :, :] = True
        _assert_mask_equal(mask, expected)

    def test_sex_male_abbrev(self, simple_registry):
        """'m' is accepted as male abbreviation."""
        s = IndividualSelector(sex="m")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[1, :, :] = True
        _assert_mask_equal(mask, expected)

    def test_sex_list_normalizes_to_sorted_tuple(self, simple_registry):
        """['male', 'female'] → both sexes selected (rows 0,1)."""
        s = IndividualSelector(sex=["male", "female"])
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.ones((2, 1, 3), dtype=bool)
        _assert_mask_equal(mask, expected)

    def test_sex_number_normalizes(self, simple_registry):
        """Sex numbers: 0 → female, 1 → male."""
        s = IndividualSelector(sex=0)
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[0, :, :] = True
        _assert_mask_equal(mask, expected)

    def test_sex_enum_normalizes(self, simple_registry):
        """Sex.MALE → row 1."""
        s = IndividualSelector(sex=Sex.MALE)
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[1, :, :] = True
        _assert_mask_equal(mask, expected)

    def test_invalid_sex_label_raises(self):
        """Unknown sex label → ValueError."""
        with pytest.raises(ValueError, match="Unknown sex label"):
            IndividualSelector(sex="other")

    def test_sex_collection_with_numbers(self, simple_registry):
        """sex=[0, 1] → both sexes selected (numeric ints in collection)."""
        s = IndividualSelector(sex=[0, 1])
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.ones((2, 1, 3), dtype=bool)
        _assert_mask_equal(mask, expected)

    def test_sex_collection_with_enum(self, simple_registry):
        """sex=[Sex.MALE] → only male, covering (Sex, int) branch."""
        s = IndividualSelector(sex=[Sex.MALE])
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[1, :, :] = True
        _assert_mask_equal(mask, expected)

    def test_invalid_sex_in_collection_raises(self):
        """Invalid sex in a collection → ValueError."""
        with pytest.raises(ValueError, match="Unknown sex label"):
            IndividualSelector(sex=["male", "unknown"])


class TestConstructionAge:
    """Invariants for age-only construction."""

    def test_age_range_normalizes(self, simple_registry):
        """range(2, 5) → (2, 3, 4) as a tuple. Compile with n_ages=6."""
        s = IndividualSelector(age=range(2, 5))
        mask = s.compile(simple_registry, n_sexes=2, n_ages=6)
        expected = np.zeros((2, 6, 3), dtype=bool)
        expected[:, 2:5, :] = True
        _assert_mask_equal(mask, expected)

    def test_age_int_normalizes_to_singleton(self, simple_registry):
        """age=3 → (3,). Compile with n_ages=6."""
        s = IndividualSelector(age=3)
        mask = s.compile(simple_registry, n_sexes=2, n_ages=6)
        expected = np.zeros((2, 6, 3), dtype=bool)
        expected[:, 3, :] = True
        _assert_mask_equal(mask, expected)

    def test_age_list_normalizes_to_sorted(self, simple_registry):
        """age=[5, 1, 3] → sorted (1, 3, 5)."""
        s = IndividualSelector(age=[5, 1, 3])
        mask = s.compile(simple_registry, n_sexes=2, n_ages=7)
        expected = np.zeros((2, 7, 3), dtype=bool)
        expected[:, 1, :] = True
        expected[:, 3, :] = True
        expected[:, 5, :] = True
        _assert_mask_equal(mask, expected)


class TestConstructionAnd:
    """Invariants for multi-field AND construction."""

    def test_ztype_and_sex_and_semantics(self, simple_registry):
        """ztype='WT|WT' AND sex='female' → intersection."""
        s = IndividualSelector(ztype="WT|WT", sex="female")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[0, 0, 0] = True
        _assert_mask_equal(mask, expected)

    def test_ztype_and_sex_and_age_semantics(self, simple_registry):
        """ztype='*|Dr', sex='male', age=2 → single coordinate."""
        s = IndividualSelector(ztype="*|Dr", sex="male", age=2)
        mask = s.compile(simple_registry, n_sexes=2, n_ages=4)
        # sex=male → row 1; age=2 → column 2; ztype *|Dr → cols 1,2
        expected = np.zeros((2, 4, 3), dtype=bool)
        expected[1, 2, 1] = True
        expected[1, 2, 2] = True
        _assert_mask_equal(mask, expected)

    def test_compile_coordinates_cardinality_and(self, simple_registry):
        """ztype='WT|WT' AND sex='female' → exactly 1 coordinate."""
        s = IndividualSelector(ztype="WT|WT", sex="female")
        coords = s.compile_coordinates(simple_registry, n_sexes=2, n_ages=1)
        assert len(coords) == 1
        assert next(iter(coords)) == (0, 0, 0)


# ══════════════════════════════════════════════════════════════════════════════
# Union Operations
# ══════════════════════════════════════════════════════════════════════════════


class TestUnion:
    """Invariants for | and + operations."""

    def test_or_produces_union_mask(self, simple_registry):
        """s1 | s2 → union of compiled masks."""
        s_female = IndividualSelector(sex="female")
        s_male = IndividualSelector(sex="male")
        union = s_female | s_male
        mask = union.compile(simple_registry, n_sexes=2, n_ages=1)
        # Union of female-only and male-only → all sexes selected
        expected = np.ones((2, 1, 3), dtype=bool)
        _assert_mask_equal(mask, expected)

    def test_or_atom_count_is_sum(self):
        """| concatenates atoms: n_atoms(s|t) = n_atoms(s) + n_atoms(t)."""
        s = IndividualSelector(ztype="WT|WT")
        t = IndividualSelector(ztype="Dr|Dr")
        union = s | t
        assert union.n_atoms == s.n_atoms + t.n_atoms  # 2

    def test_plus_equivalent_to_or(self, simple_registry):
        """s + t produces the same compiled mask as s | t."""
        s = IndividualSelector(sex="female")
        t = IndividualSelector(sex="male")
        mask_or = (s | t).compile(simple_registry, n_sexes=2, n_ages=1)
        mask_plus = (s + t).compile(simple_registry, n_sexes=2, n_ages=1)
        _assert_mask_equal(mask_or, mask_plus)

    def test_plus_atom_count_same_as_or(self):
        """s + t has same n_atoms as s | t."""
        s = IndividualSelector(ztype="WT|WT")
        t = IndividualSelector(ztype="Dr|Dr")
        assert (s + t).n_atoms == (s | t).n_atoms

    def test_or_is_idempotent_on_masks(self, simple_registry):
        """(s | s).compile() == s.compile() — OR of a set with itself is itself."""
        s = IndividualSelector(ztype="*|Dr", sex="female")
        mask_s = s.compile(simple_registry, n_sexes=2, n_ages=1)
        mask_ss = (s | s).compile(simple_registry, n_sexes=2, n_ages=1)
        _assert_mask_equal(mask_ss, mask_s)

    def test_or_is_commutative_on_masks(self, simple_registry):
        """(a | b).compile() == (b | a).compile() — union is commutative."""
        a = IndividualSelector(ztype="WT|WT", sex="female")
        b = IndividualSelector(ztype="Dr|Dr", sex="male")
        mask_ab = (a | b).compile(simple_registry, n_sexes=2, n_ages=1)
        mask_ba = (b | a).compile(simple_registry, n_sexes=2, n_ages=1)
        _assert_mask_equal(mask_ab, mask_ba)

    def test_or_is_associative_on_masks(self, simple_registry):
        """((a|b)|c).compile() == (a|(b|c)).compile()."""
        a = IndividualSelector(ztype="WT|WT")
        b = IndividualSelector(ztype="WT|Dr")
        c = IndividualSelector(ztype="Dr|Dr")
        mask_left = ((a | b) | c).compile(simple_registry, n_sexes=2, n_ages=1)
        mask_right = (a | (b | c)).compile(simple_registry, n_sexes=2, n_ages=1)
        _assert_mask_equal(mask_left, mask_right)

    def test_or_chain(self, simple_registry):
        """s1 | s2 | s3 produces correctly unioned mask."""
        s1 = IndividualSelector(ztype="WT|WT")
        s2 = IndividualSelector(ztype="WT|Dr")
        s3 = IndividualSelector(ztype="Dr|Dr")
        union = s1 | s2 | s3
        mask = union.compile(simple_registry, n_sexes=2, n_ages=1)
        # Union of all individual ztypes → all ztypes selected
        expected = np.ones((2, 1, 3), dtype=bool)
        _assert_mask_equal(mask, expected)

    def test_or_with_non_selector_returns_notimplemented(self):
        """| with a non-IndividualSelector returns NotImplemented."""
        s = IndividualSelector()
        result = s.__or__(42)
        assert result is NotImplemented

    def test_plus_with_non_selector_returns_notimplemented(self):
        """+ with a non-IndividualSelector returns NotImplemented."""
        s = IndividualSelector()
        result = s.__add__("not_a_selector")
        assert result is NotImplemented


# ══════════════════════════════════════════════════════════════════════════════
# Immutability & Hash
# ══════════════════════════════════════════════════════════════════════════════


class TestImmutability:
    """Frozen dataclass invariants."""

    def test_can_be_dict_key(self):
        """IndividualSelector instances can be used as dict keys (hashable)."""
        s1 = IndividualSelector(ztype="WT|WT", sex="female")
        s2 = IndividualSelector(ztype="Dr|Dr")
        d = {s1: "a", s2: "b"}
        assert d[s1] == "a"
        assert d[s2] == "b"

    def test_equal_selectors_have_equal_hashes(self):
        """Equal selectors must have equal hash values."""
        s1 = IndividualSelector(ztype="WT|WT", sex="female")
        s2 = IndividualSelector(ztype="WT|WT", sex="female")
        assert hash(s1) == hash(s2)

    def test_different_selectors_may_have_different_hashes(self):
        """Different selectors typically have different hashes."""
        s1 = IndividualSelector(ztype="WT|WT")
        s2 = IndividualSelector(ztype="Dr|Dr")
        # Not strictly required but expected in practice
        assert hash(s1) != hash(s2)

    def test_cannot_assign_attributes(self):
        """Frozen dataclass prevents attribute assignment."""
        from dataclasses import FrozenInstanceError

        s = IndividualSelector()
        with pytest.raises(FrozenInstanceError):
            s._atoms = ()  # type: ignore[misc]  # testing frozen dataclass mutation guard


# ══════════════════════════════════════════════════════════════════════════════
# Equality
# ══════════════════════════════════════════════════════════════════════════════


class TestEquality:
    """Structural equality invariants."""

    def test_same_atoms_equal(self):
        """Identically constructed selectors compare equal."""
        s1 = IndividualSelector(ztype="WT|WT", sex="female")
        s2 = IndividualSelector(ztype="WT|WT", sex="female")
        assert s1 == s2
        assert s1 is not s2  # different objects

    def test_different_atoms_not_equal(self):
        """Different atom content → not equal."""
        s1 = IndividualSelector(ztype="WT|WT")
        s2 = IndividualSelector(ztype="Dr|Dr")
        assert s1 != s2

    def test_union_order_may_differ(self):
        """Selectors with different _atoms ordering are not equal
        (current implementation preserves construction order)."""
        s1 = IndividualSelector(ztype="WT|WT")
        s2 = IndividualSelector(ztype="WT|Dr")
        # s1 | s2 puts s1's atoms first, s2 | s1 puts s2's atoms first
        # These differ structurally, but compiled masks are identical
        assert (s1 | s2) != (s2 | s1)

    def test_union_order_masks_identical(self, simple_registry):
        """Although structural order differs, compiled masks are identical."""
        s1 = IndividualSelector(ztype="WT|WT")
        s2 = IndividualSelector(ztype="WT|Dr")
        mask_12 = (s1 | s2).compile(simple_registry, n_sexes=2, n_ages=1)
        mask_21 = (s2 | s1).compile(simple_registry, n_sexes=2, n_ages=1)
        _assert_mask_equal(mask_12, mask_21)


# ══════════════════════════════════════════════════════════════════════════════
# Introspection
# ══════════════════════════════════════════════════════════════════════════════


class TestIntrospection:
    """n_atoms, is_empty, repr, fingerprint invariants."""

    def test_n_atoms_default(self):
        """Default selector has exactly 1 atom."""
        assert IndividualSelector().n_atoms == 1

    def test_n_atoms_after_union(self):
        """n_atoms after union is sum of operand n_atoms."""
        s = IndividualSelector(ztype="WT|WT") | IndividualSelector(ztype="Dr|Dr")
        assert s.n_atoms == 2

    def test_n_atoms_after_chained_union(self):
        """Chained union accumulates atoms."""
        s = (
            IndividualSelector(ztype="WT|WT")
            | IndividualSelector(ztype="WT|Dr")
            | IndividualSelector(ztype="Dr|Dr")
        )
        assert s.n_atoms == 3

    def test_is_empty_true_only_for_all_wildcard(self):
        """is_empty is True when any atom is fully wildcard."""
        assert IndividualSelector().is_empty is True
        s = IndividualSelector(ztype="WT|WT") | IndividualSelector()
        assert s.is_empty is True  # the wildcard atom makes it "empty"

    def test_is_empty_false_for_specific_selector(self):
        """A selector with any specified field is not empty."""
        assert IndividualSelector(ztype="WT|WT").is_empty is False
        assert IndividualSelector(sex="female").is_empty is False
        assert IndividualSelector(age=5).is_empty is False

    def test_repr_default(self):
        """Default selector repr contains '<all>'."""
        r = repr(IndividualSelector())
        assert "IndividualSelector(" in r
        assert "<all>" in r

    def test_repr_with_ztype(self):
        """Selector with a ztype field shows the ztype in repr."""
        s = IndividualSelector(ztype="*|Dr")
        r = repr(s)
        assert "ztype=" in r
        assert "*|Dr" in r

    def test_repr_with_sex(self):
        """Selector with a sex field shows the normalized sex values in repr."""
        s = IndividualSelector(sex="female")
        r = repr(s)
        assert "sex=" in r
        assert "0" in r  # Sex.FEMALE = 0

    def test_repr_with_union(self):
        """Union selector repr contains 'OR'."""
        s = IndividualSelector(ztype="WT|WT") | IndividualSelector(ztype="Dr|Dr")
        r = repr(s)
        assert " OR " in r

    def test_repr_stable(self):
        """Same construction → same repr string."""
        s1 = IndividualSelector(ztype="WT|WT", sex="female")
        s2 = IndividualSelector(ztype="WT|WT", sex="female")
        assert repr(s1) == repr(s2)

    def test_fingerprint_same_selector_same_fingerprint(self):
        """Same construction parameters → same fingerprint."""
        s1 = IndividualSelector(ztype="WT|WT", sex="female")
        s2 = IndividualSelector(ztype="WT|WT", sex="female")
        assert s1.fingerprint == s2.fingerprint

    def test_fingerprint_different_selector_different_fingerprint(self):
        """Different selectors have different fingerprints."""
        s1 = IndividualSelector(ztype="WT|WT")
        s2 = IndividualSelector(ztype="Dr|Dr")
        assert s1.fingerprint != s2.fingerprint

    def test_fingerprint_length_is_16(self):
        """Fingerprint is exactly 16 hex characters."""
        s = IndividualSelector(ztype="WT|WT", sex="female", age=3)
        assert len(s.fingerprint) == 16
        assert all(c in "0123456789abcdef" for c in s.fingerprint)

    def test_fingerprint_is_hex_string(self):
        """Fingerprint contains only hex characters."""
        s = IndividualSelector(ztype="*|Dr")
        assert all(c in "0123456789abcdef" for c in s.fingerprint)


# ══════════════════════════════════════════════════════════════════════════════
# Compile
# ══════════════════════════════════════════════════════════════════════════════


class TestCompile:
    """compile() and compile_coordinates() invariants."""

    # ── compile() ─────────────────────────────────────────────────────────

    def test_mask_shape(self, simple_registry):
        """compile returns (n_sexes, n_ages, n_ztypes) bool ndarray."""
        s = IndividualSelector()
        mask = s.compile(simple_registry, n_sexes=2, n_ages=5)
        assert mask.shape == (2, 5, 3)
        assert mask.dtype == bool

    def test_specific_ztype_mask(self, simple_registry):
        """ztype='WT|WT' → only ztype col 0 is True."""
        s = IndividualSelector(ztype="WT|WT")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        assert mask.sum() == 2  # 2 sexes × 1 age × 1 ztype

    def test_specific_sex_mask(self, simple_registry):
        """sex='female' → only row 0 is True."""
        s = IndividualSelector(sex="female")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        assert mask.sum() == 3  # 1 sex × 1 age × 3 ztypes

    def test_specific_age_mask(self, simple_registry):
        """age=3 with n_ages=6 → only age column 3 is True."""
        s = IndividualSelector(age=3)
        mask = s.compile(simple_registry, n_sexes=2, n_ages=6)
        assert mask.sum() == 6  # 2 sexes × 1 age × 3 ztypes

    def test_and_combination_cardinality(self, simple_registry):
        """ztype='WT|WT' AND sex='female' → 1 True entry."""
        s = IndividualSelector(ztype="WT|WT", sex="female")
        mask = s.compile(simple_registry, n_sexes=2, n_ages=1)
        assert mask.sum() == 1

    def test_union_mask_cardinality(self, simple_registry):
        """Union of two disjoint ztypes → 2 × 2 = 4 True entries."""
        s1 = IndividualSelector(ztype="WT|WT")
        s2 = IndividualSelector(ztype="Dr|Dr")
        union = s1 | s2
        mask = union.compile(simple_registry, n_sexes=2, n_ages=1)
        # WT|WT: 2 sexes; Dr|Dr: 2 sexes → 4 total
        assert mask.sum() == 4

    # ── compile_coordinates() ─────────────────────────────────────────────

    def test_compile_coordinates_type(self, simple_registry):
        """compile_coordinates returns a frozenset of (int,int,int) tuples."""
        s = IndividualSelector()
        coords = s.compile_coordinates(simple_registry, n_sexes=2, n_ages=1)
        assert isinstance(coords, frozenset)
        for c in coords:
            assert isinstance(c, tuple)
            assert len(c) == 3
            assert all(isinstance(x, int) for x in c)

    def test_compile_coordinates_exact_set(self, simple_registry):
        """ztype='WT|WT', sex='female' → {(0, 0, 0)}."""
        s = IndividualSelector(ztype="WT|WT", sex="female")
        coords = s.compile_coordinates(simple_registry, n_sexes=2, n_ages=1)
        assert coords == frozenset({(0, 0, 0)})

    def test_compile_coordinates_star_drive(self, simple_registry):
        """ztype='*|Dr' → coordinates for ztypes 1 and 2, all sexes, age 0."""
        s = IndividualSelector(ztype="*|Dr")
        coords = s.compile_coordinates(simple_registry, n_sexes=2, n_ages=1)
        expected = frozenset({
            (0, 0, 1), (0, 0, 2),
            (1, 0, 1), (1, 0, 2),
        })
        assert coords == expected

    def test_compile_coordinates_union_of_disjoint(self, simple_registry):
        """Union of two disjoint selectors → coordinates are union of sets."""
        s_female_wt = IndividualSelector(ztype="WT|WT", sex="female")
        s_male_dr = IndividualSelector(ztype="Dr|Dr", sex="male")
        union = s_female_wt | s_male_dr
        coords = union.compile_coordinates(simple_registry, n_sexes=2, n_ages=1)
        expected = frozenset({(0, 0, 0), (1, 0, 2)})
        assert coords == expected

    def test_compile_coordinates_ages(self, simple_registry):
        """age=range(1,3) with n_ages=5 → 2 ages × 2 sexes × 3 ztypes = 12 coords."""
        s = IndividualSelector(age=range(1, 3))
        coords = s.compile_coordinates(simple_registry, n_sexes=2, n_ages=5)
        assert len(coords) == 2 * 2 * 3  # 2 ages × 2 sexes × 3 ztypes = 12


# ══════════════════════════════════════════════════════════════════════════════
# Compile — Special Cases
# ══════════════════════════════════════════════════════════════════════════════


class TestCompileEdgeCases:
    """Edge-case invariants for compile() and compile_coordinates()."""

    def test_empty_selection_raises_value_error(self, simple_registry):
        """A selector targeting a non-existent ztype raises ValueError."""
        s = IndividualSelector(ztype="XX|YY")  # no such genotype
        with pytest.raises(ValueError, match="selects no"):
            s.compile(simple_registry, n_sexes=2, n_ages=1)

    def test_empty_union_raises_if_all_atoms_empty(self, simple_registry):
        """Union of atoms that each resolve to nothing raises ValueError."""
        s = IndividualSelector(ztype="XX|YY") | IndividualSelector(ztype="ZZ|ZZ")
        with pytest.raises(ValueError, match="selects no"):
            s.compile(simple_registry, n_sexes=2, n_ages=1)

    def test_partial_union_skips_empty_atoms(self, simple_registry):
        """Union with one matching and one non-matching atom still succeeds."""
        s_match = IndividualSelector(ztype="WT|WT")
        s_empty = IndividualSelector(ztype="XX|YY")
        union = s_match | s_empty
        mask = union.compile(simple_registry, n_sexes=2, n_ages=1)
        expected = np.zeros((2, 1, 3), dtype=bool)
        expected[:, 0, 0] = True
        _assert_mask_equal(mask, expected)

    def test_sex_out_of_range_filtered(self, simple_registry):
        """Sex indices >= n_sexes are filtered out (no match)."""
        # sex=1 (male) is in range, sex=2 is out of range for n_sexes=2
        # But _to_tuple_sex only accepts (0,1) via labels, numbers go through
        # Let's test: sex=2 with n_sexes=2 → filtered to empty → ValueError
        s = IndividualSelector(sex=2)
        with pytest.raises(ValueError, match="selects no"):
            s.compile(simple_registry, n_sexes=2, n_ages=1)

    def test_age_out_of_range_filtered(self, simple_registry):
        """Age >= n_ages is filtered out."""
        s = IndividualSelector(age=99)
        with pytest.raises(ValueError, match="selects no"):
            s.compile(simple_registry, n_sexes=2, n_ages=1)

    def test_wildcard_ztype_with_empty_registry(self):
        """Wildcard ztype with n_ztypes=0 → empty set → ValueError."""
        reg = IndexRegistry()
        s = IndividualSelector(ztype="*")
        with pytest.raises(ValueError, match="selects no"):
            s.compile(reg, n_sexes=1, n_ages=1)

    def test_specific_ztype_with_empty_registry(self):
        """Non-wildcard, non-* ztype with n_ztypes=0 → species is None
        → _resolve_ztype returns empty → ValueError."""
        reg = IndexRegistry()
        s = IndividualSelector(ztype="WT|WT")
        with pytest.raises(ValueError, match="selects no"):
            s.compile(reg, n_sexes=1, n_ages=1)

    def test_compile_coordinates_empty_raises(self, simple_registry):
        """Empty compile_coordinates also raises ValueError."""
        s = IndividualSelector(ztype="XX|YY")
        with pytest.raises(ValueError, match="selects no"):
            s.compile_coordinates(simple_registry, n_sexes=2, n_ages=1)


# ══════════════════════════════════════════════════════════════════════════════
# Serialization
# ══════════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """to_dict() invariants."""

    def test_default_to_dict(self):
        """Default selector → atoms: [{'all': True}]."""
        d = IndividualSelector().to_dict()
        assert isinstance(d, dict)
        assert "atoms" in d
        assert d["atoms"] == [{"all": True}]

    def test_ztype_only_to_dict(self):
        """ztype-only → dict with ztype key."""
        s = IndividualSelector(ztype="*|Dr")
        d = s.to_dict()
        assert d["atoms"] == [{"ztype": ["*|Dr"]}]

    def test_sex_only_to_dict(self):
        """sex-only → dict with sex key (normalized as int list)."""
        s = IndividualSelector(sex="female")
        d = s.to_dict()
        assert d["atoms"] == [{"sex": [0]}]

    def test_age_only_to_dict(self):
        """age-only → dict with age key."""
        s = IndividualSelector(age=3)
        d = s.to_dict()
        assert d["atoms"] == [{"age": [3]}]

    def test_multifield_to_dict(self):
        """Multiple fields → all appear in the atom dict."""
        s = IndividualSelector(ztype="WT|WT", sex="male", age=5)
        d = s.to_dict()
        atom = d["atoms"][0]
        assert "ztype" in atom
        assert "sex" in atom
        assert "age" in atom
        assert atom["ztype"] == ["WT|WT"]
        assert atom["sex"] == [1]
        assert atom["age"] == [5]

    def test_union_to_dict(self):
        """Union → multiple atoms in the list."""
        s = IndividualSelector(ztype="WT|WT") | IndividualSelector(ztype="Dr|Dr")
        d = s.to_dict()
        assert len(d["atoms"]) == 2
        assert d["atoms"][0] == {"ztype": ["WT|WT"]}
        assert d["atoms"][1] == {"ztype": ["Dr|Dr"]}

    def test_roundtrip_same_args_same_dict(self):
        """Reconstructing with same args produces the same to_dict output."""
        s1 = IndividualSelector(ztype="WT|WT", sex="female", age=[1, 2])
        s2 = IndividualSelector(ztype="WT|WT", sex="female", age=[1, 2])
        assert s1.to_dict() == s2.to_dict()
