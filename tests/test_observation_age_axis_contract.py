"""Age-axis contract for observation counts and rules (evaluator regression).

Pins the contract from the age-axis unification: counts and rules **always**
carry the age axis; a 2-D ``(sex, ztype)`` count is read as a single age class
and therefore takes a ``(n_groups, sex, 1, ztype)`` rule; a rule that drops the
axis is rejected instead of being inferred.

These tests target the failure modes the unification removes, and the
normalization convergence between :func:`apply_rule` and
:meth:`Observation.apply`:

* an age-collapsed selector mask is byte-identical to an "age-free" rule, so
  accepting a 3-D rule silently sums every age of the matched ZType;
* a rule whose extents differ from the counts but whose *element count* is a
  whole number of planes used to reach the native projection and produce a
  silently mis-mapped result;
* both entry points must lift a 2-D count the same way.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.frontend.output.observation import ObservationFilter, apply_rule
from natal.frontend.registry.index import IndexRegistry

_AGE_AXIS_MESSAGE = "^rule must carry the age axis"
_EXTENT_MESSAGE = "^rule shape does not match the counts"


@pytest.fixture(scope="module")
def registry() -> IndexRegistry:
    """Registry with the three unordered WT/Dr genotypes."""
    species = nt.Species.from_dict(
        name="AgeAxisContractSpecies",
        structure={"chr1": {"loc1": ["WT", "Dr"]}},
    )
    reg = IndexRegistry()
    for genotype in species.get_all_genotypes():
        reg.register_genotype(genotype)
    assert reg.n_ztypes == 3
    return reg


def _count_3d(
    n_sexes: int = 2, n_ages: int = 2, n_ztypes: int = 3
) -> NDArray[np.float64]:
    """Deterministic ``(sex, age, ztype)`` count with unique entries."""
    arr = np.zeros((n_sexes, n_ages, n_ztypes), dtype=np.float64)
    for sex in range(n_sexes):
        for age in range(n_ages):
            for ztype in range(n_ztypes):
                arr[sex, age, ztype] = float(100 * ztype + 10 * sex + age + 1)
    return arr


def _count_2d(
    n_sexes: int = 2, n_ztypes: int = 3
) -> NDArray[np.float64]:
    """Deterministic ``(sex, ztype)`` count (discrete-generation spelling)."""
    arr = np.zeros((n_sexes, n_ztypes), dtype=np.float64)
    for sex in range(n_sexes):
        for ztype in range(n_ztypes):
            arr[sex, ztype] = float(100 * ztype + 10 * sex + 1)
    return arr


class TestRuleCarriesTheAgeAxis:
    """A rule without the age axis is rejected, never inferred."""

    def test_rejects_two_d_and_three_d_rules(self) -> None:
        """Both lower-rank rule spellings fail for both count ranks."""
        for count in (_count_3d(), _count_2d()):
            for rule in (np.ones((2, 3)), np.ones((2, 2, 3))):
                with pytest.raises(ValueError, match=_AGE_AXIS_MESSAGE):
                    apply_rule(count, rule)

    def test_collapsed_selector_mask_is_rejected(
        self, registry: IndexRegistry
    ) -> None:
        """An OR'd selector mask is not an age-free rule.

        A mask selecting female/age-0/WT|WT OR-ed over the age axis has shape
        ``(groups, sexes, ztypes)`` and is byte-identical to a rule that never
        had an age axis.  It must be rejected: accepting it sums every age of
        the matched ZType (3.0 here) instead of the selected age class (1.0).
        """
        count = _count_3d()
        selector = np.zeros((1, 2, 2, 3), dtype=np.float64)
        selector[0, 0, 0, 0] = 1.0
        collapsed = selector.any(axis=2)

        assert apply_rule(count, selector).ravel().tolist() == [1.0, 0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match=_AGE_AXIS_MESSAGE):
            apply_rule(count, collapsed)

    def test_rejection_message_names_the_expected_shape_and_the_fix(self) -> None:
        """The error has to be actionable, not just a rank complaint."""
        with pytest.raises(ValueError) as excinfo:
            apply_rule(_count_3d(), np.ones((2, 2, 3)))
        message = str(excinfo.value)
        assert "expected 4-D (n_groups, 2, 2, 3)" in message
        assert "got 3-D shape (2, 2, 3)" in message
        assert "None" in message

    def test_rejects_a_rule_with_an_unsupported_rank(self) -> None:
        """A 5-D rule is a rank error, reported by the same message."""
        with pytest.raises(ValueError, match=_AGE_AXIS_MESSAGE):
            apply_rule(_count_3d(), np.ones((1, 2, 2, 3, 1)))


class TestRuleExtents:
    """Extents must match; mismatches never reach the native projection."""

    @pytest.mark.parametrize(
        ("count", "rule"),
        [
            pytest.param(_count_3d(), np.ones((1, 3, 2, 3)), id="sex-mismatch"),
            pytest.param(_count_3d(), np.ones((1, 2, 1, 3)), id="age-mismatch"),
            pytest.param(_count_3d(), np.ones((1, 2, 2, 4)), id="ztype-mismatch"),
            pytest.param(_count_2d(), np.ones((1, 3, 1, 3)), id="2d-count-sex"),
            pytest.param(_count_2d(), np.ones((1, 2, 1, 4)), id="2d-count-ztype"),
        ],
    )
    def test_extent_mismatch_is_rejected(
        self, count: NDArray[np.float64], rule: NDArray[np.float64]
    ) -> None:
        """A shape mismatch names both the expected and the actual shape."""
        with pytest.raises(ValueError, match=_EXTENT_MESSAGE) as excinfo:
            apply_rule(count, rule)
        message = str(excinfo.value)
        assert f"expected (n_groups, {count.shape[0]}, " in message
        assert f"got {rule.shape}" in message

    def test_size_preserving_mismatch_is_rejected(self) -> None:
        """A mismatch that keeps the element count a whole number of planes.

        ``(1, 1, 4, 3)`` has 12 elements, exactly the plane of a ``(2, 2, 3)``
        count, so the native projection's ``mask.len() % plane == 0`` guard
        passes and the mask is re-indexed in the wrong layout.  The extent check
        must reject it before that happens.
        """
        count = _count_3d()
        rule = np.arange(1, 13, dtype=np.float64).reshape(1, 1, 4, 3)
        assert rule.size == count.size  # the old guard could not tell

        with pytest.raises(ValueError, match=_EXTENT_MESSAGE):
            apply_rule(count, rule)


class TestSingleAgeClassShape:
    """A count (or rule) can legitimately have exactly one age class."""

    def test_three_d_count_with_one_age_takes_a_one_age_rule(self) -> None:
        """``(2, 1, 3)`` count + ``(2, 2, 1, 3)`` rule -> ``(2, 2, 1)``."""
        count = _count_3d(n_ages=1)
        rule = np.ones((2, 2, 1, 3), dtype=np.float64)

        result = apply_rule(count, rule)

        assert result.shape == (2, 2, 1)
        expected = np.repeat(count.sum(axis=2)[np.newaxis, :, :], 2, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_one_age_count_still_rejects_a_three_d_rule(self) -> None:
        """A degenerate axis is still an axis: ``(2, 2, 3)`` is not accepted."""
        with pytest.raises(ValueError, match=_AGE_AXIS_MESSAGE):
            apply_rule(_count_3d(n_ages=1), np.ones((2, 2, 3)))


class TestSharedNormalization:
    """Both entry points lift a 2-D count through the same normalization."""

    def test_two_d_and_its_explicit_three_d_spelling_agree(self) -> None:
        """``(S, Z)`` and ``(S, 1, Z)`` are the same observation."""
        count_2d = _count_2d()
        rule = np.ones((1, 2, 1, 3), dtype=np.float64)

        from_2d = apply_rule(count_2d, rule)
        from_3d = apply_rule(count_2d[:, np.newaxis, :], rule)

        assert from_2d.shape == (1, 2)
        assert from_3d.shape == (1, 2, 1)
        np.testing.assert_array_equal(from_2d, from_3d[:, :, 0])

    def test_apply_rule_and_observation_apply_agree(
        self, registry: IndexRegistry
    ) -> None:
        """The standalone helper and ``Observation.apply`` share the lift.

        Both are driven with the same 2-D count and the same 4-D mask; a drift
        between the two normalizations shows up as a shape or value difference.
        """
        count_2d = _count_2d()
        observation = ObservationFilter(registry).build_filter(
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
            n_sexes=2,
            n_ages=1,
            n_ztypes=3,
        )
        mask = observation.build_mask(2, 1, 3)
        assert mask.shape == (1, 2, 1, 3)

        from_apply = observation.apply(count_2d)
        from_rule = apply_rule(count_2d, mask)

        assert from_apply.shape == (1, 2)
        assert from_rule.shape == (1, 2)
        np.testing.assert_array_equal(from_apply, from_rule)

    def test_observation_apply_rejects_a_mask_age_mismatch(
        self, registry: IndexRegistry
    ) -> None:
        """A baked mask with a different age extent is rejected, not reshaped.

        The observation is compiled for ``n_ages=2`` and then fed a 2-D count,
        whose lifted age axis is length 1.  The native projection used to run
        first and the failure surfaced later as an opaque reshape error.
        """
        observation = ObservationFilter(registry).build_filter(
            groups={"total": {"genotype": "*"}},
            collapse_age=False,
            n_sexes=2,
            n_ages=2,
            n_ztypes=3,
        )
        with pytest.raises(ValueError, match=_EXTENT_MESSAGE):
            observation.apply(_count_2d())


class TestUnsupportedCountRanks:
    """The pinned rank message is unchanged for the standalone helper."""

    @pytest.mark.parametrize("ndim", [1, 4, 5])
    def test_unsupported_count_ndim_message(self, ndim: int) -> None:
        """``apply_rule`` keeps the ``Unsupported individual_count ndim`` text."""
        shape = {1: (6,), 4: (1, 2, 2, 3), 5: (1, 1, 2, 2, 3)}[ndim]
        with pytest.raises(
            ValueError, match=rf"^Unsupported individual_count ndim: {ndim}$"
        ):
            apply_rule(np.ones(shape), np.ones((1, 2, 2, 3)))
