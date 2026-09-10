"""Contracts for the unified observation selection representation.

Every observation entry point compiles one selection representation —
:class:`~natal.frontend.patterns.IndividualSelector`.  Legacy dictionary
spellings are normalized at the :meth:`ObservationFilter.build_filter`
boundary.  These tests pin the four checked semantics: age selection
(inclusive range pairs), label selection, empty-match behavior (raise,
not silent zeros), and compressed genotype-index mapping.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

import natal as nt
from natal.frontend.output.observation import ObservationFilter
from natal.frontend.patterns import IndividualSelector
from natal.frontend.registry.index import IndexRegistry


def _species(name: str) -> nt.Species:
    """Two-allele unordered species → genotypes WT|WT, WT|A, A|A."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "A"]}},
    )


def _registry(name: str) -> IndexRegistry:
    """Registry over all genotypes of :func:`_species`."""
    species = _species(name)
    registry = IndexRegistry()
    for genotype in species.get_all_genotypes():
        registry.register_genotype(genotype)
    return registry


def _counts(n_ages: int = 3, n_ztypes: int = 3) -> np.ndarray:
    """Deterministic (2, n_ages, n_ztypes) count tensor."""
    array = np.zeros((2, n_ages, n_ztypes), dtype=np.float64)
    for sex in range(2):
        for age in range(n_ages):
            for ztype in range(n_ztypes):
                array[sex, age, ztype] = float(100 * ztype + 10 * sex + age + 1)
    return array


# ── Age selection: inclusive range pairs keep legacy semantics ───────────────


@pytest.mark.parametrize(
    "age_spec", [(1, 2), [1, 2], (0, 0), [2, 2]], ids=["tuple", "list", "one", "repeat"]
)
def test_age_range_pairs_are_inclusive(age_spec: Any) -> None:
    """A two-integer pair selects the inclusive age range, as before."""
    registry = _registry("uni_age_pairs")
    observation = ObservationFilter(registry).build_filter(
        groups={"adults": {"age": age_spec}},
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    mask = observation.build_mask(2, 3, 3)
    selected_ages = sorted(
        int(age) for age in np.nonzero(mask[0, 0, :, 0])[0]
    )
    start, end = age_spec
    assert selected_ages == list(range(start, end + 1))


def test_age_mixed_list_expands_pairs_and_singletons() -> None:
    """``[0, (1, 2)]`` selects ages {0, 1, 2} — pairs expand, ints stay."""
    registry = _registry("uni_age_mixed")
    observation = ObservationFilter(registry).build_filter(
        groups={"mix": {"age": [0, (1, 2)]}},
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    mask = observation.build_mask(2, 3, 3)
    selected = np.nonzero(mask[0, 0, :, 0])[0]
    np.testing.assert_array_equal(selected, [0, 1, 2])


def test_callable_age_selector_is_rejected() -> None:
    """Callable ages are not part of the unified representation."""
    registry = _registry("uni_age_callable")
    with pytest.raises(TypeError, match="unified"):
        ObservationFilter(registry).build_filter(
            groups={"bad": {"age": lambda age: age > 0}},
            n_sexes=2,
            n_ages=3,
            n_ztypes=3,
        )


def test_empty_age_range_raises_instead_of_silent_zero_row() -> None:
    """An inverted range raises; the legacy compiler selected nothing silently."""
    registry = _registry("uni_age_empty")
    with pytest.raises(ValueError, match="selects no ages"):
        ObservationFilter(registry).build_filter(
            groups={"empty": {"age": (2, 1)}},
            n_sexes=2,
            n_ages=3,
            n_ztypes=3,
        )


# ── Label selection ──────────────────────────────────────────────────────────


def test_dict_labels_are_preserved_in_order() -> None:
    """Dict keys become the group labels in insertion order."""
    registry = _registry("uni_labels_dict")
    observation = ObservationFilter(registry).build_filter(
        groups={
            "wild": {"genotype": "*"},
            "drive": {"genotype": ["A|A"]},
        },
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    assert observation.labels == ("wild", "drive")


def test_sequence_labels_get_group_index_names() -> None:
    """Sequence input keeps the ``group_{i}`` legacy label scheme."""
    registry = _registry("uni_labels_seq")
    observation = ObservationFilter(registry).build_filter(
        groups=[{"genotype": "*"}, {"sex": "female"}],
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    assert observation.labels == ("group_0", "group_1")


def test_none_groups_keep_legacy_identity_labels() -> None:
    """``groups=None`` keeps one ``g{i}`` group per provided genotype."""
    registry = _registry("uni_labels_none")
    species = _species("uni_labels_none")
    genotypes = list(species.iter_genotypes(unordered=species.unordered))
    observation = ObservationFilter(registry).build_filter(
        diploid_genotypes=genotypes,
        groups=None,
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    assert observation.labels == tuple(f"g{i}" for i in range(len(genotypes)))
    # Each identity group selects exactly its genotype's ztype columns.
    mask = observation.build_mask(2, 3, 3)
    for index, label in enumerate(observation.labels):
        columns = np.nonzero(mask[index, 0, 0, :])[0]
        assert len(columns) >= 1
        for column in columns:
            genotype, _slab = registry.index_to_ztype[int(column)]
            assert genotype is genotypes[index]


# ── Empty match: unified raise, no silent all-zero group ─────────────────────


@pytest.mark.parametrize(
    "spec",
    [
        {"genotype": ["missing|allele"]},
        {"sex": "alien"},
        {"age": [7, 9]},
    ],
    ids=["genotype", "sex", "age"],
)
def test_empty_matches_raise(spec: Dict[str, Any]) -> None:
    """A selection matching no coordinate raises instead of yielding zeros."""
    registry = _registry("uni_empty")
    with pytest.raises(ValueError):
        ObservationFilter(registry).build_filter(
            groups={"void": spec},
            n_sexes=2,
            n_ages=3,
            n_ztypes=3,
        )


def test_selector_entry_points_raise_identically() -> None:
    """The builder-side selector entry and the boundary raise the same way."""
    registry = _registry("uni_empty_selector")
    with pytest.raises(ValueError, match="selects no"):
        ObservationFilter(registry).build_from_selectors(
            groups={"void": IndividualSelector(ztype="missing|allele")},
            n_sexes=2,
            n_ages=3,
            n_ztypes=3,
        )


# ── Compressed genotype indices ──────────────────────────────────────────────


def test_integer_genotype_index_maps_to_all_slabs_of_that_genotype() -> None:
    """``genotype: [i]`` selects every slab of registry genotype *i*."""
    registry = _registry("uni_comp_idx")
    observation = ObservationFilter(registry).build_filter(
        groups={"by_index": {"genotype": [0]}},
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    expected_genotype = registry.index_to_genotype[0]
    expected_columns = {
        position
        for position, (genotype, _slab) in enumerate(registry.index_to_ztype)
        if genotype == expected_genotype
    }
    mask = observation.build_mask(2, 3, 3)
    selected_columns = set(np.nonzero(mask[0, 0, 0, :])[0].tolist())
    assert selected_columns == expected_columns


def test_integer_index_group_equals_string_pattern_group() -> None:
    """Index spelling and the equivalent pattern string select identically."""
    registry = _registry("uni_comp_equiv")
    by_index = ObservationFilter(registry).build_filter(
        groups={"g": {"genotype": [1]}},
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    by_pattern = ObservationFilter(registry).build_filter(
        groups={"g": {"genotype": [str(registry.index_to_genotype[1])]}},
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    np.testing.assert_array_equal(
        by_index.build_mask(2, 3, 3), by_pattern.build_mask(2, 3, 3)
    )


def test_duck_typed_group_object_preserves_coordinates() -> None:
    """Duck-typed legacy group objects compile to the same mask as dicts."""
    registry = _registry("uni_duck")
    counts = _counts()

    class _Group:
        genotype = ["A|A"]
        age = [1]
        sex = "female"

    duck = ObservationFilter(registry).build_filter(
        groups={"duck": _Group()},
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    legacy = ObservationFilter(registry).build_filter(
        groups={"duck": {"genotype": ["A|A"], "age": [1], "sex": "female"}},
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    mask_duck = duck.build_mask(2, 3, 3)
    mask_legacy = legacy.build_mask(2, 3, 3)
    np.testing.assert_array_equal(mask_duck, mask_legacy)

    result = duck.apply(counts)
    genotype_column = [
        position
        for position, (genotype, _slab) in enumerate(registry.index_to_ztype)
        if str(genotype) == "A|A"
    ]
    expected = float(counts[0, 1, genotype_column[0]])
    assert float(result[0, 0, 1]) == expected


def test_selector_values_pass_through_the_boundary_unchanged() -> None:
    """Selector values in a mapping are not copied or re-normalized."""
    registry = _registry("uni_passthrough")
    selector = IndividualSelector(ztype="A|A", sex="female")
    observation = ObservationFilter(registry).build_filter(
        groups={"sel": selector},
        n_sexes=2,
        n_ages=3,
        n_ztypes=3,
    )
    assert observation._selectors is not None  # pyright: ignore[reportPrivateUsage]  # boundary stores the unified representation
    assert observation._selectors[0] is selector  # pyright: ignore[reportPrivateUsage]
