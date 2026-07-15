"""Strict tests for Phase 2 Observation changes.

Follows numerical-verification standards — every assertion proves a
mathematical invariant.  Covers:

  - ObservationResult: construction, to_dict, frozen dataclass
  - build_identity_observation: labels, identity_map, n_groups, fingerprint
  - ObservationFilter.build_from_selectors: empty, valid, mask shapes,
    equivalence with legacy dict-based masks
  - Observation.apply() identity path: numerical losslessness,
    total-sum conservation, collapse_age
  - Observation.apply() lazy rebuild (mask=None → auto-rebuild)
  - Observation.project(): correct axes, tick preservation
  - Backward compatibility: build_filter, specs, legacy mask
"""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from typing import Dict

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.output.observation import (
    Observation,
    ObservationFilter,
    ObservationResult,
    apply_rule,
    build_identity_observation,
)
from natal.patterns import IndividualSelector
from natal.registry.index import IndexRegistry

# ── Shared fixtures ──────────────────────────────────────────────────────────


def _fingerprint_hex(*components: object) -> str:
    """Replicate Observation._build_fingerprint for comparison."""
    h = hashlib.sha256()
    for c in components:
        h.update(repr(c).encode("utf-8"))
    return h.hexdigest()[:16]


@pytest.fixture(scope="module")
def phase2_species() -> nt.Species:
    """2-allele unordered species: WT, Dr → 3 unordered genotypes."""
    return nt.Species.from_dict(
        name="Phase2Species",
        structure={"chr1": {"loc1": ["WT", "Dr"]}},
    )


@pytest.fixture(scope="module")
def phase2_registry(phase2_species: nt.Species) -> IndexRegistry:
    """Registry with all 3 genotypes registered → 3 ZTypes."""
    reg = IndexRegistry()
    for g in phase2_species.get_all_genotypes():
        reg.register_genotype(g)
    assert reg.n_ztypes == 3  # WT|WT, WT|Dr, Dr|Dr (unordered)
    return reg


# Slug labels for the 3-genotype registry (unordered: WT, Dr)
_P2_LABELS_A = {
    0: "Dr|Dr",
    1: "WT|Dr",
    2: "WT|WT",
}  # unordered order


def _p2_spec_labels(registry: IndexRegistry) -> Dict[int, str]:
    """Return {ztype_idx: label_str} for the phase2 3-genotype registry."""
    return {i: str(gt) for i, (gt, _slab) in enumerate(registry.index_to_ztype)}


def _make_ind_count_3d(
    n_sexes: int = 2, n_ages: int = 2, n_ztypes: int = 3
) -> NDArray[np.float64]:
    """Create a deterministic 3-D count array for testing.

    Each entry is a unique value based on its position so we can verify
    exact numerical preservation after projection.
    """
    arr = np.zeros((n_sexes, n_ages, n_ztypes), dtype=np.float64)
    base = 1
    for s in range(n_sexes):
        for a in range(n_ages):
            for z in range(n_ztypes):
                arr[s, a, z] = float(base)
                base += 1
    return arr


def _make_ind_count_2d(
    n_sexes: int = 2, n_ztypes: int = 3
) -> NDArray[np.float64]:
    """Create a deterministic 2-D count array (discrete-generation style)."""
    arr = np.zeros((n_sexes, n_ztypes), dtype=np.float64)
    base = 1
    for s in range(n_sexes):
        for z in range(n_ztypes):
            arr[s, z] = float(base)
            base += 1
    return arr


# ══════════════════════════════════════════════════════════════════════════════
# 1. ObservationResult: construction, to_dict, frozen
# ══════════════════════════════════════════════════════════════════════════════


class TestObservationResult:
    """Invariants for ObservationResult dataclass."""

    def test_construction_basic(self) -> None:
        """ObservationResult holds tick, values, axes, labels."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = ObservationResult(
            tick=5,
            values=values,
            axes=("group", "sex"),
            labels={"group": ("g0", "g1")},
        )
        assert result.tick == 5
        np.testing.assert_array_equal(result.values, values)
        assert result.axes == ("group", "sex")
        assert result.labels == {"group": ("g0", "g1")}

    def test_frozen_dataclass(self) -> None:
        """ObservationResult is immutable."""
        result = ObservationResult(
            tick=0,
            values=np.zeros((1, 2), dtype=np.float64),
            axes=("group", "sex"),
            labels={"group": ("g",)},
        )
        with pytest.raises(FrozenInstanceError):
            result.tick = 99  # type: ignore[misc]  # testing frozen

    def test_to_dict_structure(self) -> None:
        """to_dict() returns JSON-serializable dict with correct keys."""
        values = np.array([[10.0, 20.0]], dtype=np.float64)
        result = ObservationResult(
            tick=3,
            values=values,
            axes=("group", "sex"),
            labels={"group": ("carriers",)},
        )
        d = result.to_dict()
        # Invariant: expected keys
        assert set(d.keys()) == {"tick", "values", "axes", "labels"}
        # Invariant: tick preserved
        assert d["tick"] == 3
        # Invariant: values converted to nested list (JSON-serializable)
        assert d["values"] == [[10.0, 20.0]]
        # Invariant: axes is list of strings
        assert d["axes"] == ["group", "sex"]
        # Invariant: labels values are lists
        assert d["labels"] == {"group": ["carriers"]}

    def test_to_dict_empty_groups(self) -> None:
        """to_dict() works with empty values array."""
        result = ObservationResult(
            tick=0,
            values=np.zeros((0, 2), dtype=np.float64),
            axes=("group", "sex"),
            labels={"group": ()},
        )
        d = result.to_dict()
        assert d["values"] == []
        assert d["labels"] == {"group": []}

    def test_round_trip_via_dict(self) -> None:
        """ObservationResult reconstructed from to_dict() preserves values."""
        values = np.array([[7.0, 8.0, 9.0]], dtype=np.float64)
        original = ObservationResult(
            tick=42,
            values=values,
            axes=("group", "sex", "age"),
            labels={"group": ("adults",)},
        )
        d = original.to_dict()
        # Reconstruct
        reconstructed = ObservationResult(
            tick=int(d["tick"]),
            values=np.array(d["values"], dtype=np.float64),
            axes=tuple(d["axes"]),
            labels={k: tuple(v) for k, v in d["labels"].items()},
        )
        assert reconstructed.tick == original.tick
        assert reconstructed.axes == original.axes
        assert reconstructed.labels == original.labels
        np.testing.assert_array_equal(reconstructed.values, original.values)


# ══════════════════════════════════════════════════════════════════════════════
# 2. build_identity_observation
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildIdentityObservation:
    """Invariants for build_identity_observation()."""

    def test_is_identity_true(self, phase2_registry: IndexRegistry) -> None:
        """Identity observation has _is_identity=True."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=1, n_sexes=2
        )
        assert obs._is_identity is True

    def test_identity_map_is_arange(self, phase2_registry: IndexRegistry) -> None:
        """_identity_map is np.arange(n_ztypes) when n_ztypes provided."""
        n = 3
        obs = build_identity_observation(
            phase2_registry, n_ztypes=n, n_ages=1, n_sexes=2
        )
        assert obs._identity_map is not None
        # Invariant: identity_map = [0, 1, 2, ..., n-1]
        expected = np.arange(n, dtype=np.int32)
        np.testing.assert_array_equal(obs._identity_map, expected)

    def test_mask_is_none_for_identity(self, phase2_registry: IndexRegistry) -> None:
        """Identity observations never bake a dense mask."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=1, n_sexes=2
        )
        assert obs.mask is None

    def test_n_groups_equals_n_ztypes(self, phase2_registry: IndexRegistry) -> None:
        """Identity observation has one group per ZType."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=1, n_sexes=2
        )
        assert obs.n_groups == 3

    def test_labels_include_all_genotypes(self, phase2_registry: IndexRegistry) -> None:
        """Labels contain all genotype strings, no @default suffix."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=1, n_sexes=2
        )
        # All 3 genotypes should appear (unordered: WT|WT, WT|Dr, Dr|Dr)
        labels_set = set(obs.labels)
        assert len(labels_set) == 3
        assert "WT|WT" in labels_set
        assert "WT|Dr" in labels_set
        assert "Dr|Dr" in labels_set

    def test_fingerprint_stable(self, phase2_registry: IndexRegistry) -> None:
        """Fingerprint is deterministic for same inputs."""
        obs1 = build_identity_observation(
            phase2_registry, n_ztypes=3, collapse_age=False
        )
        obs2 = build_identity_observation(
            phase2_registry, n_ztypes=3, collapse_age=False
        )
        # Invariant: same inputs → same fingerprint
        assert obs1.population_fingerprint == obs2.population_fingerprint
        # Invariant: fingerprint is 16-char hex
        assert len(obs1.population_fingerprint) == 16
        assert all(c in "0123456789abcdef" for c in obs1.population_fingerprint)

    def test_fingerprint_changes_with_n_ztypes(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Different n_ztypes → different fingerprint."""
        obs_a = build_identity_observation(
            phase2_registry, n_ztypes=3, collapse_age=False
        )
        obs_b = build_identity_observation(
            phase2_registry, n_ztypes=2, collapse_age=False
        )
        assert obs_a.population_fingerprint != obs_b.population_fingerprint

    def test_fingerprint_changes_with_collapse_age(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Different collapse_age → different fingerprint."""
        obs_a = build_identity_observation(
            phase2_registry, n_ztypes=3, collapse_age=False
        )
        obs_b = build_identity_observation(
            phase2_registry, n_ztypes=3, collapse_age=True
        )
        assert obs_a.population_fingerprint != obs_b.population_fingerprint

    def test_collapse_age_flag_preserved(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """collapse_age flag is stored as specified."""
        obs_true = build_identity_observation(
            phase2_registry, n_ztypes=3, collapse_age=True
        )
        obs_false = build_identity_observation(
            phase2_registry, n_ztypes=3, collapse_age=False
        )
        assert obs_true.collapse_age is True
        assert obs_false.collapse_age is False

    def test_n_ztypes_none_lazy(self, phase2_registry: IndexRegistry) -> None:
        """n_ztypes=None → _identity_map is None, mask is None."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=None, collapse_age=False
        )
        assert obs._identity_map is None
        assert obs.mask is None
        assert obs._is_identity is True
        # Labels still computed from registry
        assert obs.n_groups == phase2_registry.n_ztypes

    def test_registry_preserved(self, phase2_registry: IndexRegistry) -> None:
        """_registry reference is stored for lazy rebuild."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=None
        )
        assert obs._registry is phase2_registry

    def test_different_ages_produces_same_labels(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """n_ages does not affect labels — labels are ZType-driven."""
        obs1 = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=1
        )
        obs2 = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=10
        )
        assert obs1.labels == obs2.labels


# ══════════════════════════════════════════════════════════════════════════════
# 3. ObservationFilter.build_from_selectors
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildFromSelectors:
    """Invariants for ObservationFilter.build_from_selectors()."""

    def test_empty_groups_raises_valueerror(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Empty groups dict raises ValueError."""
        compiler = ObservationFilter(phase2_registry)
        with pytest.raises(ValueError, match="groups must be non-empty"):
            compiler.build_from_selectors(
                groups={}, n_ztypes=3, collapse_age=False
            )

    def test_empty_label_raises_valueerror(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Empty string label raises ValueError."""
        compiler = ObservationFilter(phase2_registry)
        with pytest.raises(ValueError, match="must be non-empty strings"):
            compiler.build_from_selectors(
                groups={"": IndividualSelector(ztype="WT|WT")},
                n_ztypes=3,
                collapse_age=False,
            )

    def test_valid_groups_produces_observation(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Valid groups → Observation with correct labels and selectors."""
        compiler = ObservationFilter(phase2_registry)
        groups: Dict[str, IndividualSelector] = {
            "wt": IndividualSelector(ztype="WT|WT"),
            "dr_carriers": IndividualSelector(ztype="*|Dr"),
        }
        obs = compiler.build_from_selectors(
            groups=groups, n_ztypes=3, collapse_age=False
        )
        # Invariant: labels match keys
        assert obs.labels == ("wt", "dr_carriers")
        assert obs.n_groups == 2
        # Invariant: selectors stored
        assert obs._selectors is not None
        assert len(obs._selectors) == 2

    def test_n_ztypes_none_mask_is_none(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """n_ztypes=None → mask is None (lazy)."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_from_selectors(
            groups={"all": IndividualSelector()},
            n_ztypes=None,
            collapse_age=False,
        )
        assert obs.mask is None
        assert obs._registry is not None

    def test_n_ztypes_provided_mask_is_baked(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """n_ztypes=N → mask is pre-baked with correct shape."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_from_selectors(
            groups={"wt": IndividualSelector(ztype="WT|WT")},
            n_ztypes=3,
            n_sexes=2,
            n_ages=1,
            collapse_age=False,
        )
        # Invariant: mask is baked
        assert obs.mask is not None
        # Invariant: 4-D shape (n_groups, n_sexes, n_ages, n_ztypes)
        assert obs.mask.shape == (1, 2, 1, 3)
        # Invariant: mask values are 0.0 or 1.0
        assert np.all((obs.mask == 0.0) | (obs.mask == 1.0))

    def test_selector_mask_equals_dict_mask_wt(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Selector-based mask ≡ equivalent dict-based mask for 'WT|WT'."""
        compiler = ObservationFilter(phase2_registry)
        n_sexes, n_ages, n_ztypes = 2, 2, 3

        # Via selectors
        mask_sel = compiler.build_mask_from_selectors(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            selectors=(IndividualSelector(ztype="WT|WT"),),
            collapse_age=False,
        )

        # Via legacy specs
        mask_legacy = compiler.build_mask_from_specs(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            specs=(("g0", {"genotype": ["WT|WT"]}),),
            collapse_age=False,
        )

        # Invariant: masks are element-for-element identical
        np.testing.assert_array_equal(mask_sel, mask_legacy)

        # Invariant: 'WT|WT' selects exactly 1 ztype column
        assert mask_sel.sum() == float(n_sexes * n_ages * 1)

    def test_selector_mask_equals_dict_mask_star_drive(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Selector '*|Dr' ≡ dict {'genotype': ['*|Dr']}."""
        compiler = ObservationFilter(phase2_registry)
        n_sexes, n_ages, n_ztypes = 2, 2, 3

        mask_sel = compiler.build_mask_from_selectors(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            selectors=(IndividualSelector(ztype="*|Dr"),),
            collapse_age=False,
        )

        mask_legacy = compiler.build_mask_from_specs(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            specs=(("dr", {"genotype": ["*|Dr"]}),),
            collapse_age=False,
        )

        np.testing.assert_array_equal(mask_sel, mask_legacy)

    def test_collapse_age_mask_shape(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """collapse_age=True produces 3-D mask (n_groups, n_sexes, n_ztypes)."""
        compiler = ObservationFilter(phase2_registry)
        mask = compiler.build_mask_from_selectors(
            n_sexes=2,
            n_ages=5,
            n_ztypes=3,
            selectors=(IndividualSelector(),),  # wildcard
            collapse_age=True,
        )
        # Invariant: 3-D shape
        assert mask.ndim == 3
        assert mask.shape == (1, 2, 3)

    def test_collapse_age_false_mask_shape(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """collapse_age=False produces 4-D mask."""
        compiler = ObservationFilter(phase2_registry)
        mask = compiler.build_mask_from_selectors(
            n_sexes=2,
            n_ages=5,
            n_ztypes=3,
            selectors=(IndividualSelector(),),
            collapse_age=False,
        )
        # Invariant: 4-D shape
        assert mask.ndim == 4
        assert mask.shape == (1, 2, 5, 3)

    def test_selector_with_sex_and_age(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Selector with sex+age constraints produces correct mask."""
        compiler = ObservationFilter(phase2_registry)
        # Female (sex=0), age 1 only
        mask = compiler.build_mask_from_selectors(
            n_sexes=2,
            n_ages=3,
            n_ztypes=3,
            selectors=(
                IndividualSelector(
                    ztype="WT|WT", sex="female", age=1
                ),
            ),
            collapse_age=False,
        )
        # Invariant: female (sex=0), age=1 only → exactly one coordinate True
        # WT|WT is ztype index 0 in phase2_registry
        assert mask.sum() == 1.0  # exactly one coordinate
        assert mask[0, 0, 0, :].sum() == 0.0  # sex=0, age=0 → nothing
        assert mask[0, 0, 1, 0] == 1.0  # sex=0, age=1, WT|WT
        assert mask[0, 1, :, :].sum() == 0.0  # sex=1 (male) → nothing selected

    def test_multi_group_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Multiple groups produce mask with correct per-group selection."""
        compiler = ObservationFilter(phase2_registry)
        mask = compiler.build_mask_from_selectors(
            n_sexes=2,
            n_ages=1,
            n_ztypes=3,
            selectors=(
                IndividualSelector(ztype="WT|WT"),
                IndividualSelector(ztype="*|Dr"),
            ),
            collapse_age=False,
        )
        # Invariant: shape = (2, 2, 1, 3)
        assert mask.shape == (2, 2, 1, 3)
        # Invariant: group 0 selects exactly 1 ztype col, group 1 selects 2
        assert mask[0].sum() == pytest.approx(float(2 * 1 * 1))  # 2 sexes × 1 age × 1 ztype
        assert mask[1].sum() == pytest.approx(float(2 * 1 * 2))  # 2 sexes × 1 age × 2 ztypes

    def test_registry_stored_for_lazy(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """_registry reference is stored for lazy rebuild."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_from_selectors(
            groups={"all": IndividualSelector()},
            n_ztypes=None,
        )
        assert obs._registry is phase2_registry


# ══════════════════════════════════════════════════════════════════════════════
# 4. Observation.apply() — identity path
# ══════════════════════════════════════════════════════════════════════════════


class TestApplyIdentity:
    """Invariants for Observation.apply() identity fast path."""

    def test_identity_3d_output_shape_no_collapse(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Identity apply on 3D input → 3D output."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d(n_sexes=2, n_ages=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # Invariant: 3-D output
        assert result.ndim == 3
        # Shape: (n_sexes, n_ages, n_ztypes) = (2, 2, 3) — identity reorders
        assert result.shape == (2, 2, 3)

    def test_identity_3d_output_shape_with_collapse(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Identity apply on 3D input with collapse_age → 2D output."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=True
        )
        ind_count = _make_ind_count_3d(n_sexes=2, n_ages=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # Invariant: 2-D output after collapse
        assert result.ndim == 2
        # Shape: (n_sexes, n_ztypes) = (2, 3)
        assert result.shape == (2, 3)

    def test_identity_total_sum_preserved_3d(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Identity apply preserves total sum (3D input, no collapse)."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d(n_sexes=2, n_ages=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # Invariant: total population conserved
        assert result.sum() == ind_count.sum()

    def test_identity_total_sum_preserved_3d_collapsed(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Identity apply preserves total sum (3D input, collapsed)."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=True
        )
        ind_count = _make_ind_count_3d(n_sexes=2, n_ages=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # Invariant: total population conserved even after age collapse
        assert result.sum() == ind_count.sum()

    def test_identity_3d_no_collapse_values_are_identity_permutation(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Identity projection is numerically lossless: sorted output == sorted input."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d(n_sexes=2, n_ages=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # Invariant: every value in the input appears exactly once in the output
        # (just reordered — identity_map = [0,1,2] maps each ztype to the
        # corresponding group column)
        np.testing.assert_array_equal(
            np.sort(result.ravel()), np.sort(ind_count.ravel())
        )

    def test_identity_collapse_age_sum_invariant(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Collapsed result: each (sex, ztype) value = sum over age axis."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=True
        )
        ind_count = _make_ind_count_3d(n_sexes=2, n_ages=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # result[s, z] should == ind_count[s, :, z].sum()
        for s in range(2):
            for z in range(3):
                expected = ind_count[s, :, z].sum()
                assert result[s, z] == pytest.approx(expected)

    def test_identity_2d_input_no_collapse(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Identity apply on 2D input → 2D output."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=1, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_2d(n_sexes=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # Invariant: 2-D output
        assert result.ndim == 2
        assert result.shape == (2, 3)
        # Invariant: total sum preserved
        assert result.sum() == ind_count.sum()
        # Invariant: numerically lossless
        np.testing.assert_array_equal(
            np.sort(result.ravel()), np.sort(ind_count.ravel())
        )

    def test_identity_2d_input_with_collapse(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Identity apply on 2D input with collapse_age → same shape (no ages)."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=1, n_sexes=2, collapse_age=True
        )
        ind_count = _make_ind_count_2d(n_sexes=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # 2D input with collapse_age still produces 2D output
        assert result.ndim == 2
        assert result.shape == (2, 3)
        assert result.sum() == ind_count.sum()

    def test_identity_map_none_falls_through_to_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """When _identity_map is None (lazy build), apply falls through to mask."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=None, n_ages=2, n_sexes=2, collapse_age=False
        )
        assert obs._identity_map is None

        ind_count = _make_ind_count_3d(n_sexes=2, n_ages=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # Invariant: still produces 3-D output via lazy rebuild path
        assert result.ndim == 3
        # Invariant: total sum preserved (even through lazy path)
        assert result.sum() == ind_count.sum()

    def test_identity_unsupported_ndim_raises(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Unsupported ndim raises ValueError."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2
        )
        with pytest.raises(ValueError, match="Unsupported individual_count ndim"):
            obs.apply(np.zeros((2, 2, 2, 2), dtype=np.float64))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Observation.apply() — legacy mask-based path
# ══════════════════════════════════════════════════════════════════════════════


class TestApplyLegacy:
    """Invariants for legacy (mask-based) Observation.apply()."""

    def test_baked_mask_apply_equals_apply_rule(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation.apply() with baked mask ≡ apply_rule() with same mask."""
        compiler = ObservationFilter(phase2_registry)
        n_sexes, n_ages, n_ztypes = 2, 2, 3

        obs = compiler.build_filter(
            groups={"wt": {"genotype": ["WT|WT"]}},
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            collapse_age=False,
        )
        ind_count = _make_ind_count_3d()

        result_apply = obs.apply(ind_count)
        mask = obs.build_mask(n_sexes, n_ages, n_ztypes)
        result_rule = apply_rule(ind_count, mask)

        # Invariant: both paths produce identical results
        np.testing.assert_array_equal(result_apply, result_rule)

    def test_mask_none_lazy_rebuild(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation with mask=None rebuilds on apply()."""
        compiler = ObservationFilter(phase2_registry)

        obs = compiler.create_observation(
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
        )
        # Invariant: initially mask is None
        assert obs.mask is None

        ind_count = _make_ind_count_3d()
        result = obs.apply(ind_count)
        # Invariant: apply succeeds and produces correct shape
        assert result.ndim == 3
        assert result.shape[0] == 1  # one group

    def test_mask_none_lazy_preserves_total(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Lazy rebuild path still preserves total sum when all selected."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.create_observation(
            groups={"total": {"genotype": "*"}}, collapse_age=False
        )
        ind_count = _make_ind_count_3d()
        result = obs.apply(ind_count)
        # Invariant: with '*' wildcard, sum over ztypes preserves total
        assert result.sum() == ind_count.sum()

    def test_2d_input_apply(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Legacy apply with lazy mask works with 2-D input (discrete-generation).

        Note: 4-D baked mask + 2-D input is unsupported by apply_rule.
        The lazy rebuild path handles 2-D input correctly by inferring
        dimensions at rebuild time.
        """
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.create_observation(
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
        )
        assert obs.mask is None  # lazy

        ind_count_2d = _make_ind_count_2d()
        result = obs.apply(ind_count_2d)
        # Invariant: 2D input → 2D output via lazy rebuild
        assert result.ndim == 2
        assert result.shape[0] == 1  # one group
        assert result.shape[1] == 2  # two sexes
        # Invariant: total sum preserved
        assert result.sum() == ind_count_2d.sum()

    def test_non_identity_without_registry_raises_on_lazy(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation without registry raises on lazy rebuild."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.create_observation(
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
        )
        # Remove _registry to simulate a bad state
        obs_without_registry = Observation(
            labels=obs.labels,
            collapse_age=obs.collapse_age,
            mask=obs.mask,
            _is_identity=False,
            _identity_map=None,
            _registry=None,
        )
        with pytest.raises(ValueError, match="no registry reference"):
            obs_without_registry.apply(_make_ind_count_3d())


# ══════════════════════════════════════════════════════════════════════════════
# 6. Observation.project()
# ══════════════════════════════════════════════════════════════════════════════


class TestProject:
    """Invariants for Observation.project()."""

    def test_project_returns_observation_result(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """project() returns an ObservationResult."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d()
        result = obs.project(ind_count, tick=10)
        assert isinstance(result, ObservationResult)
        assert result.tick == 10

    def test_project_tick_preserved(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Tick value flows through to ObservationResult."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d()
        for tick in (0, 5, 99):
            result = obs.project(ind_count, tick=tick)
            assert result.tick == tick

    def test_project_axes_no_collapse_3d(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """project() with identity on 3D input → correct axes."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d()
        result = obs.project(ind_count, tick=0)
        # identity returns (n_sexes, n_ages, n_ztypes) = (2, 2, 3)
        # ndim=3, collapse_age=False → axes = ("group", "sex", "age")
        assert result.axes == ("group", "sex", "age")

    def test_project_axes_collapse_3d(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """project() with collapse_age → 2D axes."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=True
        )
        ind_count = _make_ind_count_3d()
        result = obs.project(ind_count, tick=0)
        # collapse_age=True, ndim=2 → axes = ("group", "sex")
        assert result.axes == ("group", "sex")

    def test_project_labels_preserved(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation labels appear in project result."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d()
        result = obs.project(ind_count, tick=0)
        assert result.labels["group"] == obs.labels

    def test_project_values_match_apply(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """project().values ≡ apply() output."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d()
        projected = obs.apply(ind_count)
        result = obs.project(ind_count, tick=0)
        np.testing.assert_array_equal(result.values, projected)

    def test_project_legacy_baked_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """project() works with legacy baked-mask observation."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups={"wt": {"genotype": ["WT|WT"]}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
            collapse_age=False,
        )
        ind_count = _make_ind_count_3d(n_ztypes=3)
        result = obs.project(ind_count, tick=7)
        assert result.tick == 7
        assert result.labels["group"] == ("wt",)
        # Legacy apply produces (n_groups, n_sexes, n_ages) = (1, 2, 2)
        assert result.values.shape == (1, 2, 2)
        assert result.axes == ("group", "sex", "age")

    def test_project_to_dict_preserves_values(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """project() → to_dict() preserves all values."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d()
        result = obs.project(ind_count, tick=42)
        d = result.to_dict()
        # Round-trip values
        reconstructed = np.array(d["values"], dtype=np.float64)
        np.testing.assert_array_equal(reconstructed, result.values)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Backward compatibility
# ══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Invariants: legacy API still works unchanged."""

    def test_build_filter_works(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """ObservationFilter.build_filter() builds a valid Observation."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups={"g0": {"genotype": ["WT|WT"]}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
            collapse_age=False,
        )
        assert obs.n_groups == 1
        assert obs.labels == ("g0",)
        assert obs.mask is not None
        assert obs.mask.shape == (1, 2, 2, 3)

    def test_specs_field_accessible(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation.specs is accessible for backward compat."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups={"g0": {"genotype": ["WT|WT"]}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
            collapse_age=False,
        )
        # specs should be populated by build_filter (legacy path)
        assert obs.specs is not None

    def test_legacy_mask_applied_correctly(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Legacy baked-mask Observation.apply() works correctly."""
        compiler = ObservationFilter(phase2_registry)
        n_sexes, n_ages, n_ztypes = 2, 2, 3

        obs = compiler.build_filter(
            groups={"wt": {"genotype": ["WT|WT"]}},
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            collapse_age=False,
        )
        ind_count = np.zeros((n_sexes, n_ages, n_ztypes), dtype=np.float64)
        # Put data only in WT|WT (ztype determined by registry)
        # Find WT|WT index
        for z in range(n_ztypes):
            gt, _slab = phase2_registry.index_to_ztype[z]
            if str(gt) == "WT|WT":
                ind_count[:, :, z] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        result = obs.apply(ind_count)
        # Invariant: shape = (1, 2, 2) = (n_groups, n_sexes, n_ages)
        assert result.shape == (1, 2, 2)
        # Invariant: values match input for WT|WT ztype
        np.testing.assert_array_equal(
            result[0], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        )

    def test_build_mask_with_baked_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """build_mask() returns existing mask when already baked."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups={"total": {"genotype": "*"}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
            collapse_age=False,
        )
        mask1 = obs.build_mask(2, 2, 3)
        mask2 = obs.build_mask(2, 2, 3)
        # Invariant: consecutive calls return same mask
        np.testing.assert_array_equal(mask1, mask2)

    def test_observation_to_dict(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation.to_dict() returns expected metadata."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups={"g0": {"genotype": ["WT|WT"]}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
            collapse_age=True,
        )
        d = obs.to_dict()
        assert d["labels"] == ["g0"]
        assert d["collapse_age"] is True
        assert d["n_groups"] == 1

    def test_observation_to_dict_identity(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation.to_dict() for identity includes 'identity': True."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2
        )
        d = obs.to_dict()
        assert d.get("identity") is True

    def test_create_observation_returns_valid_observation(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """create_observation() (with mask=None) still works and can apply."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.create_observation(
            groups={"total": {"genotype": "*"}}, collapse_age=False
        )
        assert obs.mask is None
        result = obs.apply(_make_ind_count_3d())
        assert result.sum() == _make_ind_count_3d().sum()

    def test_apply_rule_standalone(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """apply_rule() standalone function works correctly."""
        n_sexes, n_ages, n_ztypes = 2, 2, 3
        ind_count = _make_ind_count_3d()
        mask = np.ones((1, n_sexes, n_ages, n_ztypes), dtype=np.float64)
        result = apply_rule(ind_count, mask)
        # Invariant: with all-ones mask, result sums all ztypes
        assert result.shape == (1, n_sexes, n_ages)
        assert result.sum() == ind_count.sum()

    def test_apply_rule_3d_input_4d_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """apply_rule with 3D input + 4D mask sums over ztype axis."""
        ind = np.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            dtype=np.float64,
        )  # (2, 2, 2)
        mask = np.ones((2, 2, 2, 2), dtype=np.float64)
        result = apply_rule(ind, mask)
        # (2, 2, 2) → sum over last axis → (2, 2, 2)
        assert result.shape == (2, 2, 2)
        # Invariant: total sum = 2 * input sum (2 groups, each sums all ztypes)
        assert result.sum() == 2 * ind.sum()

    def test_apply_rule_2d_input_3d_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """apply_rule with 2D input + 3D mask."""
        ind = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        mask = np.ones((1, 2, 3), dtype=np.float64)
        result = apply_rule(ind, mask)
        assert result.shape == (1, 2)
        assert result.sum() == ind.sum()

    def test_apply_rule_3d_input_3d_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """apply_rule with 3D input + 3D mask (collapse_age style)."""
        ind = _make_ind_count_3d()  # (2, 2, 3)
        mask = np.ones((1, 2, 3), dtype=np.float64)  # collapsed mask
        result = apply_rule(ind, mask)
        # expanded to (1, 2, 1, 3) → prod → sum(-1) → sum(-1) → (1, 2)
        assert result.shape == (1, 2)
        # Each sex × ztype sums over ages
        assert result.sum() == ind.sum()

    def test_apply_rule_unsupported_dimensions_raise(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """apply_rule raises ValueError for unsupported ndim combos."""
        ind = _make_ind_count_3d()
        mask_2d = np.ones((1, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="Unsupported rule ndim"):
            apply_rule(ind, mask_2d)

    def test_build_filter_with_species_diploid(
        self, phase2_species: nt.Species, phase2_registry: IndexRegistry
    ) -> None:
        """build_filter with diploid_genotypes=Species resolves all genotypes."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            diploid_genotypes=phase2_species,
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
            collapse_age=False,
        )
        # Without explicit groups, it auto-generates one group per genotype
        assert obs.n_groups == 3
        assert obs.mask is not None

    def test_legacy_group_list_input(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Legacy list-format groups work."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups=[
                {"genotype": ["WT|WT"]},
                {"genotype": ["*|Dr"]},
            ],
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
            collapse_age=False,
        )
        assert obs.n_groups == 2
        assert obs.mask.shape == (2, 2, 2, 3)

    def test_selectors_stored_in_backward_compat(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """_selectors field is None for legacy (build_filter) observations."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups={"g0": {"genotype": ["WT|WT"]}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
        )
        # Legacy path does not populate _selectors
        assert obs._selectors is None


# ══════════════════════════════════════════════════════════════════════════════
# 8. Observation.n_groups property
# ══════════════════════════════════════════════════════════════════════════════


class TestNGroups:
    """Invariant: n_groups == len(labels)."""

    def test_single_group(self, phase2_registry: IndexRegistry) -> None:
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups={"g0": {"genotype": ["WT|WT"]}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
        )
        assert obs.n_groups == 1
        assert obs.n_groups == len(obs.labels)

    def test_multi_group(self, phase2_registry: IndexRegistry) -> None:
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups=[
                {"genotype": ["WT|WT"]},
                {"genotype": ["WT|Dr"]},
                {"genotype": ["Dr|Dr"]},
            ],
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
        )
        assert obs.n_groups == 3
        assert obs.n_groups == len(obs.labels)

    def test_identity_n_groups(self, phase2_registry: IndexRegistry) -> None:
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3
        )
        assert obs.n_groups == 3
        assert obs.n_groups == len(obs.labels)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Edge cases & error handling
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and error handling invariants."""

    def test_n_ztypes_zero_raises(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """build_from_selectors with n_ztypes=0 raises ValueError."""
        compiler = ObservationFilter(phase2_registry)
        with pytest.raises(ValueError, match="n_ztypes <= 0"):
            compiler.build_from_selectors(
                groups={"a": IndividualSelector()},
                n_ztypes=0,
            )

    def test_build_filter_n_ztypes_zero_raises(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """build_filter with n_ztypes=0 raises ValueError."""
        compiler = ObservationFilter(phase2_registry)
        with pytest.raises(ValueError, match="n_ztypes <= 0"):
            compiler.build_filter(
                groups={"total": {"genotype": "*"}},
                n_sexes=2,
                n_ages=2,
                n_ztypes=0,
            )

    def test_build_filter_without_groups_or_diploid_raises(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """build_filter without groups and without diploid_genotypes raises."""
        compiler = ObservationFilter(phase2_registry)
        with pytest.raises(
            ValueError, match="diploid_genotypes required"
        ):
            compiler.build_filter(
                groups=None,
                diploid_genotypes=None,
                n_sexes=2,
                n_ages=2,
                n_ztypes=3,
            )

    def test_single_age_observation_applies_correctly(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation with n_ages=1 works correctly."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=1, n_sexes=2
        )
        ind_count = _make_ind_count_2d(n_sexes=2, n_ztypes=3)
        # Should dispatch as 2D input
        result = obs.apply(ind_count)
        assert result.ndim == 2
        assert result.sum() == ind_count.sum()

    def test_identity_groups_individually_correct(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Each identity group column = corresponding ztype column of input."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=3, n_ages=2, n_sexes=2, collapse_age=False
        )
        ind_count = _make_ind_count_3d(n_sexes=2, n_ages=2, n_ztypes=3)
        result = obs.apply(ind_count)
        # result shape: (n_sexes, n_ages, n_ztypes) since identity_map is [0,1,2]
        # So result[:, :, z] should == ind_count[:, :, z]
        for z in range(3):
            np.testing.assert_array_equal(result[:, :, z], ind_count[:, :, z])

    def test_lazy_rebuild_with_selectors(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Lazy rebuild for selector-based observation works."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_from_selectors(
            groups={
                "wt": IndividualSelector(ztype="WT|WT"),
                "dr": IndividualSelector(ztype="*|Dr"),
            },
            n_ztypes=None,  # lazy
            collapse_age=False,
        )
        assert obs.mask is None
        assert obs._selectors is not None

        ind_count = _make_ind_count_3d(n_ztypes=3)
        result = obs.apply(ind_count)
        # Lazy rebuild via _rebuild_mask_dim
        assert result.ndim == 3
        assert result.shape[0] == 2  # 2 groups

    def test_fingerprint_format(self, phase2_registry: IndexRegistry) -> None:
        """All fingerprints are 16-char hex strings."""
        # Identity
        obs_id = build_identity_observation(
            phase2_registry, n_ztypes=3
        )
        assert len(obs_id.population_fingerprint) == 16
        assert all(c in "0123456789abcdef" for c in obs_id.population_fingerprint)

        # Legacy
        compiler = ObservationFilter(phase2_registry)
        obs_legacy = compiler.build_filter(
            groups={"g0": {"genotype": ["WT|WT"]}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
        )
        assert len(obs_legacy.population_fingerprint) == 16
        assert all(
            c in "0123456789abcdef" for c in obs_legacy.population_fingerprint
        )

    def test_to_dict_non_identity(self, phase2_registry: IndexRegistry) -> None:
        """Observation.to_dict() for non-identity does not include 'identity' key."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_filter(
            groups={"g0": {"genotype": ["WT|WT"]}},
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
        )
        d = obs.to_dict()
        assert "identity" not in d

    def test_collapse_age_mask_any_true_semantics(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Collapse_age=True mask: ztype selected if any age matches."""
        compiler = ObservationFilter(phase2_registry)
        # Select age=2 only, with 5 age classes
        mask = compiler.build_mask_from_selectors(
            n_sexes=2,
            n_ages=5,
            n_ztypes=3,
            selectors=(
                IndividualSelector(ztype="WT|WT", age=2),
            ),
            collapse_age=True,
        )
        # After collapse: mask shape = (1, 2, 3)
        # Since age=2 exists and WT|WT is present, ztype column should be 1.0
        # for all sexes
        assert mask.shape == (1, 2, 3)
        # Find WT|WT ztype index
        wt_z_idx = 0
        for z in range(3):
            gt, _ = phase2_registry.index_to_ztype[z]
            if str(gt) == "WT|WT":
                wt_z_idx = z
                break
        # Both sexes should have WT|WT ztype set to 1.0
        assert mask[0, 0, wt_z_idx] == 1.0
        assert mask[0, 1, wt_z_idx] == 1.0
        # Other ztypes should be 0.0
        for z in range(3):
            if z != wt_z_idx:
                assert mask[0, 0, z] == 0.0
                assert mask[0, 1, z] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 10. Additional coverage for Phase-2-specific paths
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def phase2_registry_multi_slab() -> IndexRegistry:
    """Registry with both 'default' and 'infected' slab labels."""
    reg = IndexRegistry()
    # Register with default slab
    from natal.genetics import Species as _Species
    sp = _Species.from_dict(
        name="MultiSlabSpecies",
        structure={"chr1": {"loc1": ["WT", "Dr"]}},
        somatic_labels=["default", "infected"],
    )
    for g in sp.get_all_genotypes():
        for slab in ["default", "infected"]:
            reg.register_ztype(g, slab)
    # n_ztypes = 3 genotypes × 2 slabs = 6
    assert reg.n_ztypes == 6
    return reg


class TestIdentityWithNonDefaultSlab:
    """Invariants for identity observation with non-default slab labels."""

    def test_labels_include_at_slab_suffix(
        self, phase2_registry_multi_slab: IndexRegistry
    ) -> None:
        """Labels include '@infected' suffix for non-default slabs."""
        obs = build_identity_observation(
            phase2_registry_multi_slab,
            n_ztypes=phase2_registry_multi_slab.n_ztypes,
            n_ages=1,
            n_sexes=2,
        )
        labels_set = set(obs.labels)
        # 6 labels total: 3 default + 3 infected
        assert len(labels_set) == 6
        # Check @infected labels exist
        infected_labels = [label for label in labels_set if "@infected" in label]
        assert len(infected_labels) == 3
        # Check default labels (no @default suffix)
        default_labels = [label for label in labels_set if "@" not in label]
        assert len(default_labels) == 3
        # 'WT|WT@infected' format
        assert any("WT|WT@infected" in label for label in labels_set)

    def test_n_groups_equals_n_ztypes_multi_slab(
        self, phase2_registry_multi_slab: IndexRegistry
    ) -> None:
        """n_groups == n_ztypes even with multiple slabs."""
        obs = build_identity_observation(
            phase2_registry_multi_slab,
            n_ztypes=phase2_registry_multi_slab.n_ztypes,
            n_ages=1,
            n_sexes=2,
        )
        assert obs.n_groups == phase2_registry_multi_slab.n_ztypes == 6

    def test_identity_map_covers_all_ztypes(
        self, phase2_registry_multi_slab: IndexRegistry
    ) -> None:
        """Identity map covers all 6 ztypes."""
        obs = build_identity_observation(
            phase2_registry_multi_slab,
            n_ztypes=phase2_registry_multi_slab.n_ztypes,
        )
        assert obs._identity_map is not None
        expected = np.arange(6, dtype=np.int32)
        np.testing.assert_array_equal(obs._identity_map, expected)


class TestBuildMaskWhenNone:
    """Invariant: build_mask() with no pre-baked mask triggers rebuild."""

    def test_build_mask_from_lazy_observation(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """build_mask() works when Observation was created lazily (mask=None)."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.create_observation(
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
        )
        assert obs.mask is None

        # build_mask should rebuild and return a 4-D mask
        mask = obs.build_mask(n_sexes=2, n_ages=2, n_ztypes=3)
        assert mask.ndim == 4
        assert mask.shape == (1, 2, 2, 3)
        # Invariant: '*' selects all ztypes → mask sum = n_sexes * n_ages * n_ztypes
        assert mask.sum() == pytest.approx(float(2 * 2 * 3))

    def test_build_mask_from_lazy_selector_obs(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """build_mask() works for lazy selector-based observation."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_from_selectors(
            groups={"dr": IndividualSelector(ztype="*|Dr")},
            n_ztypes=None,  # lazy
            collapse_age=False,
        )
        assert obs.mask is None
        assert obs._selectors is not None

        mask = obs.build_mask(n_sexes=2, n_ages=3, n_ztypes=3)
        assert mask.ndim == 4
        assert mask.shape == (1, 2, 3, 3)
        # *|Dr selects 2 ztypes out of 3 → 1 * 2 * 3 * 2 = 12 selected
        assert mask.sum() == pytest.approx(float(2 * 3 * 2))


class TestBuildFromSelectorsIdentity:
    """Invariants for build_from_selectors with is_identity=True."""

    def test_is_identity_flag(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """build_from_selectors with is_identity=True stores the flag."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_from_selectors(
            groups={
                "wt": IndividualSelector(ztype="WT|WT"),
                "dr": IndividualSelector(ztype="Dr|Dr"),
            },
            n_ztypes=3,
            is_identity=True,
        )
        assert obs._is_identity is True

    def test_identity_map_set(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """is_identity with n_ztypes provided → _identity_map is set."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_from_selectors(
            groups={
                "wt": IndividualSelector(ztype="WT|WT"),
                "dr": IndividualSelector(ztype="Dr|Dr"),
            },
            n_ztypes=3,
            is_identity=True,
        )
        assert obs._identity_map is not None
        expected = np.arange(3, dtype=np.int32)
        np.testing.assert_array_equal(obs._identity_map, expected)
        # Invariant: mask stays None for identity (avoids dense mask)
        assert obs.mask is None

    def test_to_dict_identity_flag(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """Observation.to_dict() includes identity flag."""
        compiler = ObservationFilter(phase2_registry)
        obs = compiler.build_from_selectors(
            groups={
                "wt": IndividualSelector(ztype="WT|WT"),
            },
            n_ztypes=3,
            is_identity=True,
        )
        d = obs.to_dict()
        assert d.get("identity") is True


class TestApplyRuleEdgeCases:
    """Cover remaining apply_rule branches."""

    def test_apply_rule_2d_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """apply_rule with 2D input and 2D mask → (n_groups, n_sexes)."""
        ind = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        mask = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        result = apply_rule(ind, mask)
        # mask[:, None, :] * arr[None, ...] → (2, 1, 3) * (1, 2, 3)
        # → broadcast → (2, 2, 3) → sum over ztype → (2, 2)
        assert result.ndim == 2
        assert result.shape == (2, 2)  # (n_groups, n_sexes)
        # Group 0 selects ztype 0: [1, 2, 3]⋅[1,0,0]=1, [4,5,6]⋅[1,0,0]=4
        assert result[0, 0] == 1.0
        assert result[0, 1] == 4.0
        # Group 1 selects ztype 1: [1, 2, 3]⋅[0,1,0]=2, [4,5,6]⋅[0,1,0]=5
        assert result[1, 0] == 2.0
        assert result[1, 1] == 5.0

    def test_apply_rule_unsupported_ndim_raises(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """apply_rule with 1D input raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported individual_count ndim"):
            apply_rule(
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                np.ones((1, 3), dtype=np.float64),
            )

    def test_apply_rule_3d_input_unsupported_mask(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """apply_rule with 3D input + 2D mask raises."""
        ind = _make_ind_count_3d()
        mask_2d = np.ones((1, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="Unsupported rule ndim"):
            apply_rule(ind, mask_2d)


class TestBuildMaskIdentity:
    """Invariant: Observation.build_mask() works for identity observations."""

    def test_build_mask_on_identity(
        self, phase2_registry: IndexRegistry
    ) -> None:
        """build_mask() on identity observation rebuilds from selectors."""
        obs = build_identity_observation(
            phase2_registry, n_ztypes=None, n_ages=2, n_sexes=2
        )
        assert obs.mask is None
        mask = obs.build_mask(n_sexes=2, n_ages=2, n_ztypes=3)
        # Rebuilt from selectors via _rebuild_mask_dim
        assert mask.ndim == 4
        assert mask.shape == (3, 2, 2, 3)
        # Identity: one ztype per group → each group has exactly 1 ztype col
        for g in range(3):
            assert mask[g].sum() == pytest.approx(float(2 * 2 * 1))
