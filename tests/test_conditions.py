"""Tests for natal.modifiers.conditions — all Condition subclasses and combinators."""

from __future__ import annotations

import natal as nt
from natal.modifiers.conditions import (
    _And,
    _Maternal,
    _Or,
    _Paternal,
    _Sex,
    _Slab,
    _ZtypeHas,
    is_maternal,
    is_paternal,
    sex,
    slab,
    ztype_has,
)
from natal.registry.index import IndexRegistry

# ---------------------------------------------------------------------------
# Shared test data (module-level — Species is a singleton by name)
# ---------------------------------------------------------------------------

_SPECIES = nt.Species.from_dict(
    name="__test_conditions__",
    structure={"chr1": {"loc1": ["A", "B"]}},
)


def _hap(allele: str) -> nt.HaploidGenome:
    """Create a HaploidGenome for *allele*."""
    return _SPECIES.get_haploid_genotype_from_str(allele)


def _geno(a1: str, a2: str) -> nt.Genotype:
    """Create a canonical unordered Genotype from two alleles."""
    return _SPECIES.unordered_genotype(_hap(a1), _hap(a2))


def _registry(*genos: nt.Genotype, slabs: tuple[str, ...] = ("default",)) -> IndexRegistry:
    """Create a minimal IndexRegistry with ztypes for each (geno × slab) combo."""
    reg = IndexRegistry()
    for g in genos:
        for s in slabs:
            reg.register_ztype(g, s)
    return reg


# Convenience genotypes
_G_AA = _geno("A", "A")
_G_AB = _geno("A", "B")
_G_BB = _geno("B", "B")


# ============================================================================
# _Sex
# ============================================================================


def test_sex_female_matches():
    c = sex("female")
    assert isinstance(c, _Sex)
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_sex_female_single_char():
    c = sex("f")
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_sex_male_matches():
    c = sex("male")
    assert isinstance(c, _Sex)
    assert c._matches(sex_idx=1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_sex_male_single_char():
    c = sex("m")
    assert c._matches(sex_idx=1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_sex_mismatch():
    c = sex("female")
    assert not c._matches(sex_idx=1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_sex_int_index():
    """sex() also works via resolve_sex_label with int 0/1 directly."""
    from natal.modifiers.conditions import _Sex
    c = _Sex(0)
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    assert not c._matches(sex_idx=1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


# ============================================================================
# _ZtypeHas (pattern string)
# ============================================================================


def test_ztype_has_pattern_match():
    c = ztype_has("A|A")
    assert isinstance(c, _ZtypeHas)
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_ztype_has_pattern_mismatch():
    c = ztype_has("A|A")
    assert not c._matches(sex_idx=0, ztype_idx=1, genotype=_G_AB, slab="default", registry=_registry(_G_AA))


def test_ztype_has_pattern_lazy_compile_called_twice():
    """Second call uses compiled filter (no re-parsing)."""
    c = ztype_has("A|A")
    # First call triggers _lazy_compile
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    # Second call uses compiled filter
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    # Verify _compiled_from_pattern is True — internal attribute
    assert c._compiled_from_pattern  # type: ignore[attr-defined]


# ============================================================================
# _ZtypeHas (callable)
# ============================================================================


def test_ztype_has_callable_match():
    c = ztype_has(lambda g: True)
    assert isinstance(c, _ZtypeHas)
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_ztype_has_callable_mismatch():
    c = ztype_has(lambda g: False)
    assert not c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_ztype_has_callable_not_compiled_from_pattern():
    c = ztype_has(lambda g: True)
    assert not c._compiled_from_pattern  # type: ignore[attr-defined]


# ============================================================================
# _Slab
# ============================================================================


def test_slab_match():
    c = slab("infected")
    assert isinstance(c, _Slab)
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="infected", registry=_registry(_G_AA))


def test_slab_mismatch():
    c = slab("infected")
    assert not c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_slab_empty_string():
    c = slab("")
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="", registry=_registry(_G_AA))
    assert not c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="x", registry=_registry(_G_AA))


# ============================================================================
# _Maternal / _Paternal
# ============================================================================


def test_is_maternal():
    c = is_maternal()
    assert isinstance(c, _Maternal)
    # gamete context: female (0) = maternal, male (1) = not maternal
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    assert not c._matches(sex_idx=1, ztype_idx=5, genotype=_G_AB, slab="x", registry=IndexRegistry())
    # zygote context: sex_idx=-1 → always True (both sides present)
    assert c._matches(sex_idx=-1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_is_paternal():
    c = is_paternal()
    assert isinstance(c, _Paternal)
    # gamete context: male (1) = paternal, female (0) = not paternal
    assert not c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    assert c._matches(sex_idx=1, ztype_idx=5, genotype=_G_AB, slab="x", registry=IndexRegistry())
    # zygote context: sex_idx=-1 → always True (both sides present)
    assert c._matches(sex_idx=-1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


# ============================================================================
# _And combinator
# ============================================================================


def test_and_both_true():
    c = sex("female") & ztype_has("A|A")
    assert isinstance(c, _And)
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_and_first_false():
    c = sex("female") & ztype_has("A|A")
    assert not c._matches(sex_idx=1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_and_second_false():
    c = sex("female") & ztype_has("A|A")
    assert not c._matches(sex_idx=0, ztype_idx=1, genotype=_G_AB, slab="default", registry=_registry(_G_AA, _G_AB))


def test_and_both_false():
    c = sex("female") & ztype_has("A|A")
    assert not c._matches(sex_idx=1, ztype_idx=1, genotype=_G_AB, slab="default", registry=_registry(_G_AA, _G_AB))


# ============================================================================
# _Or combinator
# ============================================================================


def test_or_both_true():
    c = sex("female") | ztype_has("A|A")
    assert isinstance(c, _Or)
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_or_first_true_second_false():
    c = sex("female") | ztype_has("A|A")
    assert c._matches(sex_idx=0, ztype_idx=1, genotype=_G_AB, slab="default", registry=_registry(_G_AA, _G_AB))


def test_or_first_false_second_true():
    c = sex("female") | ztype_has("A|A")
    assert c._matches(sex_idx=1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))


def test_or_both_false():
    c = sex("female") | ztype_has("A|A")
    assert not c._matches(sex_idx=1, ztype_idx=1, genotype=_G_AB, slab="default", registry=_registry(_G_AA, _G_AB))


# ============================================================================
# Chained combinators
# ============================================================================


def test_chained_and():
    c = sex("female") & ztype_has("A|A") & slab("default")
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    assert not c._matches(sex_idx=1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    assert not c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="infected", registry=_registry(_G_AA))


def test_chained_or():
    c = sex("female") | ztype_has("A|A") | slab("infected")
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    assert c._matches(sex_idx=1, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    assert c._matches(sex_idx=1, ztype_idx=1, genotype=_G_AB, slab="infected", registry=_registry(_G_AA, _G_AB))
    assert not c._matches(sex_idx=1, ztype_idx=1, genotype=_G_AB, slab="default", registry=_registry(_G_AA, _G_AB))


def test_mixed_and_or():
    """Left-associative: (female & AA) | slab_infected."""
    c = sex("female") & ztype_has("A|A") | slab("infected")
    # OR true via left branch
    assert c._matches(sex_idx=0, ztype_idx=0, genotype=_G_AA, slab="default", registry=_registry(_G_AA))
    # OR true via right branch
    assert c._matches(sex_idx=1, ztype_idx=1, genotype=_G_AB, slab="infected", registry=_registry(_G_AA, _G_AB))
    # Both false
    assert not c._matches(sex_idx=1, ztype_idx=1, genotype=_G_AB, slab="default", registry=_registry(_G_AA, _G_AB))


def test_sex_invalid_label_raises():
    """Invalid sex label raises ValueError via resolve_sex_label."""
    import pytest
    with pytest.raises(ValueError, match="Invalid sex label"):
        sex("unknown")
