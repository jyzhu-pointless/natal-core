"""Adversarial-review regression tests for the PointMutation preset.

These tests encode the declared contract of
``src/natal/frontend/presets/point_mutation.py`` (TODO.legacy.md ARCH-021) at the
points the author's suite does not exercise.  They are written to fail for
the *claimed* contract, not to restate current behavior:

1. ``_resolve_rate_pair`` documents ``ValueError`` for "a declaration [that]
   names an unknown sex"; a per-sex mapping key that is not a
   ``Sex``/``int``/``str`` used to leak ``AssertionError`` out of the public
   constructor (and out of ``reconfigure_preset``).
2. ``rate_mode="strict"`` means the rates are probabilities (sum at most 1).
   A non-finite rate used to be neither rejected nor harmless: ``NaN``
   silently became the maximum rate 1.0, so *every* source gamete mutated.
3. The compensation clamp is documented as absorbing "floating-point dust
   from the division", but it used to rewrite a declared rate above 1 into 1.0
   whenever a negative sibling rate kept the sum at or below 1.

The embryonic channel (``zygotic_mutation_rate``) that this review also
covered was subsequently withdrawn by the maintainer (TODO.legacy.md ARCH-021), so the
zygotic cases were removed together with the parameter; the preset is now
germline-only and ``zygote_modifier`` always returns ``None``.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _species(name: str) -> nt.Species:
    """Return a single-locus species with the mutation alleles."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["A", "B", "C", "D"]}},
        gamete_labels=["default"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Error contract: a malformed per-sex mapping key must raise ValueError
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("key", [2.5, 1.0, 0.0, None, b"f", (0, 1)])
def test_malformed_per_sex_rate_key_raises_value_error(key: object) -> None:
    """A mapping key that is not a Sex/int/str is a ValueError, not an assert.

    ``PointMutation._resolve_rate_pair`` documents ``ValueError`` for a key
    that "must name female or male (or 0/1)" and catches
    ``(TypeError, ValueError)`` around ``resolve_sex_label``.  The helper
    guards its input with ``assert``, so ``AssertionError`` escapes instead —
    and under ``python -O`` the assert is stripped and an ``AttributeError``
    escapes.  Either way the documented public error contract is not met.
    """
    with pytest.raises(ValueError):
        nt.PointMutation(
            "BadKeyMut",
            source_allele="A",
            target_allele="B",
            mutation_rate={key: 0.1},
        )


# ══════════════════════════════════════════════════════════════════════════════
# Non-finite and out-of-range declarations must not become probabilities
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rate_is_rejected(bad: float) -> None:
    """``strict`` rates are probabilities, so a non-finite rate must be refused.

    Observed before the fix: ``mutation_rate=float('nan')`` is accepted and
    ``effective_rates() == ((1.0, 1.0),)`` — a NaN silently becomes the
    maximum rate, converting every source gamete.
    """
    with pytest.raises(ValueError):
        nt.PointMutation(
            "NonFiniteMut",
            source_allele="A",
            target_allele="B",
            mutation_rate=bad,
        )


def test_negative_rate_cannot_smuggle_an_out_of_range_rate() -> None:
    """A declared rate above 1 must raise even when a sibling rate is negative.

    ``strict`` mode checks only the *sum*, so ``[-0.5, 1.4]`` passes the
    ``sum > 1`` gate and the documented dust clamp then rewrites 1.4 to 1.0:
    ``effective_rates() == ((0.0, 0.0), (1.0, 1.0))`` and the realized
    distribution of an ``A|A`` parent is 100% target C.  A rate of 1.4 is not
    a probability and 0.0 is not the declared -0.5, so the declaration must
    be rejected instead of silently replaced.
    """
    with pytest.raises(ValueError):
        nt.PointMutation(
            "SmuggleMut",
            source_allele="A",
            target_alleles=["B", "C"],
            mutation_rates=[-0.5, 1.4],
        )


def test_negative_rate_is_not_silently_read_as_zero() -> None:
    """A negative declared rate is invalid, not an alias for "no conversion"."""
    with pytest.raises(ValueError):
        nt.PointMutation(
            "NegativeMut",
            source_allele="A",
            target_alleles=["B", "C"],
            mutation_rates=[-0.1, 0.5],
        )


# ══════════════════════════════════════════════════════════════════════════════
# Sanity anchors for the review (must keep passing)
# ══════════════════════════════════════════════════════════════════════════════


def test_malformed_string_and_int_keys_still_raise_value_error() -> None:
    """The already-correct rejections stay ValueError after any repair."""
    for key in ("woman", 2, -1):
        with pytest.raises(ValueError):
            nt.PointMutation(
                "BadLabelMut",
                source_allele="A",
                target_allele="B",
                mutation_rate={key: 0.1},
            )


def test_valid_boundary_declarations_still_construct() -> None:
    """Rates summing to exactly 1 and a zero rate stay valid declarations."""
    nt.PointMutation(
        "BoundaryMut", source_allele="A", target_alleles=["B", "C"],
        mutation_rates=[0.5, 0.5],
    )
    nt.PointMutation(
        "ZeroMut", source_allele="A", target_alleles=["B", "C"],
        mutation_rates=[0.0, 0.7],
    )
    assert _species("_point_mutation_adversarial_anchor").name == (
        "_point_mutation_adversarial_anchor"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Repair verification: the rejections must hold in every declaration shape
# and must not over-reach into valid declarations
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rate_rejected_in_proportional_mode(bad: float) -> None:
    """Non-finite declarations are refused in ``proportional`` mode too.

    ``proportional`` accepts weights above 1, so its only absolute constraint
    is finiteness: ``[nan, 1.0]`` must not slip through normalization.
    """
    with pytest.raises(ValueError):
        nt.PointMutation(
            "PropNonFinite", source_allele="A", target_alleles=["B", "C"],
            mutation_rates=[bad, 1.0], rate_mode="proportional",
        )


def test_negative_rate_rejected_in_every_declaration_shape() -> None:
    """A negative rate is invalid wherever it is written, not only as a scalar.

    A per-sex pair with one negative side, a per-sex mapping with a negative
    value, a multi-target list, and a proportional weight must all fail loudly
    instead of silently becoming 0.0.
    """
    with pytest.raises(ValueError):
        nt.PointMutation(
            "PairNegative", source_allele="A", target_allele="B",
            mutation_rate=(0.1, -0.2),
        )
    with pytest.raises(ValueError):
        nt.PointMutation(
            "MapNegative", source_allele="A", target_allele="B",
            mutation_rate={"male": -0.1},
        )
    with pytest.raises(ValueError):
        nt.PointMutation(
            "MultiNegative", source_allele="A", target_alleles=["B", "C"],
            mutation_rates=[0.2, -0.1],
        )
    with pytest.raises(ValueError):
        nt.PointMutation(
            "PropNegative", source_allele="A", target_alleles=["B", "C"],
            mutation_rates=[-1.0, 5.0], rate_mode="proportional",
        )


def test_valid_declarations_are_not_over_rejected() -> None:
    """The non-finite/negative guards must not narrow the accepted input space.

    Pins: strict sums of exactly 1 (including ten competing targets), zero
    rates, a zero-rate sex, and proportional weights well above 1 — the
    ratio-style form the docs advertise.
    """
    ten = [f"T{i}" for i in range(1, 11)]
    species = nt.Species.from_dict(
        name="_point_mutation_adversarial_overreach",
        structure={"chr1": {"loc": ["A", *ten]}},
        gamete_labels=["default"],
    )
    preset = nt.PointMutation(
        "TenTargets", source_allele="A", target_alleles=ten,
        mutation_rates=[0.1] * 10, species=species,
    )
    assert len(preset.effective_rates()) == 10
    assert preset.effective_rates()[-1][0] == pytest.approx(1.0)

    nt.PointMutation(
        "ExactOne", source_allele="A", target_alleles=["B", "C", "D"],
        mutation_rates=[0.1, 0.2, 0.7],
    )
    nt.PointMutation(
        "ZeroSex", source_allele="A", target_allele="B", mutation_rate=(0.0, 0.25),
    )
    proportional = nt.PointMutation(
        "Ratios", source_allele="A", target_alleles=["B", "C", "D"],
        mutation_rates=[2.0, 3.0, 5.0], rate_mode="proportional",
    )
    assert proportional.effective_rates()[0] == pytest.approx((0.2, 0.2))


def test_reconfigure_rejects_invalid_declarations_with_value_error() -> None:
    """The runtime write path raises ValueError and leaves the config untouched.

    ``reconfigure_preset`` writes the raw value through ``setattr`` and
    re-compiles, so this is the second public entry point that must surface
    ``ValueError`` (never ``AssertionError``) and must not commit a partially
    applied state.
    """
    species = _species("_point_mutation_adversarial_reconfigure")
    preset = nt.PointMutation(
        "ReconfigureContract", source_allele="A", target_alleles=["B", "C"],
        mutation_rates=[0.1, 0.1],
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="AdversarialReconfPop", stochastic=False
        )
        .initial_state(individual_count={"female": {"A|A": 10}, "male": {"A|A": 10}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .presets(preset)
        .build()
    )
    registry = pop.index_registry
    parent = registry.ztype_index(species.get_genotype_from_str("A|A"), "default")
    before = [float(p) for p in pop.config.zygotes_to_gametes_map[0, parent, :]]

    for bad in ({2.5: 0.1}, [float("nan"), 0.2], [-0.2, 0.4], [0.6, 0.5], "half"):
        with pytest.raises(ValueError):
            pop.update().reconfigure_preset(preset, mutation_rates=bad)
        assert preset.mutation_rates == ((0.1, 0.1), (0.1, 0.1))
        assert [
            float(p) for p in pop.config.zygotes_to_gametes_map[0, parent, :]
        ] == before

    pop.update().reconfigure_preset(preset, mutation_rates=[0.2, 0.3])
    assert preset.effective_rates() == ((0.2, 0.2), (0.3 / 0.8, 0.3 / 0.8))
    pop.run(n_steps=1)


# ══════════════════════════════════════════════════════════════════════════════
# Scope change: germline-only (the embryonic channel was withdrawn)
# ══════════════════════════════════════════════════════════════════════════════


def test_withdrawn_embryonic_channel_leaves_no_zygote_stage_effect() -> None:
    """The withdrawn channel must be gone on *every* public entry point.

    The constructor rejection is pinned by the author's suite; this test adds
    the three remaining ways a withdrawn feature can leak back in: a lingering
    attribute, a zygote modifier reaching a compiled population, and the
    runtime updater writing a stray attribute instead of refusing the name.
    """
    species = _species("_point_mutation_adversarial_germline_only")

    def build(name: str, preset: nt.PointMutation | None) -> nt.DiscreteGenerationPopulation:
        builder = (
            nt.DiscreteGenerationPopulation.setup(
                species=species, name=name, stochastic=False
            )
            .initial_state(individual_count={"female": {"A|A": 20}, "male": {"A|A": 20}})
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2, sex_ratio=0.5)
            .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        )
        return (builder.presets(preset) if preset is not None else builder).build()

    preset = nt.PointMutation(
        "GermlineOnlyContract", source_allele="A", target_alleles=["B", "C"],
        mutation_rates=[0.3, 0.5],
    )
    assert not hasattr(preset, "zygotic_mutation_rate")
    assert not hasattr(preset, "_zygotic_rates")

    baseline = build("AdversarialGermlineBase", None)
    population = build("AdversarialGermlinePop", preset)

    # A zygote-stage conversion would rewrite the fusion map; the germline
    # channel must leave it byte-identical to the preset-free baseline.
    np.testing.assert_array_equal(
        np.asarray(population.config.gametes_to_zygotes_map),
        np.asarray(baseline.config.gametes_to_zygotes_map),
    )
    assert preset.zygote_modifier(population) is None

    # The preset is still active in the germline (guards against a vacuous test).
    registry = population.index_registry
    parent = registry.ztype_index(species.get_genotype_from_str("A|A"), "default")
    gametes = {
        registry.index_to_gtype[k][0].to_string(): pytest.approx(float(v))
        for k, v in enumerate(population.config.zygotes_to_gametes_map[0, parent, :])
        if v > 0
    }
    assert gametes == {"A": pytest.approx(0.2), "B": pytest.approx(0.3), "C": pytest.approx(0.5)}

    # The runtime updater must refuse the withdrawn name, not create it.
    with pytest.raises(AttributeError):
        population.update().reconfigure_preset(preset, zygotic_mutation_rate=0.3)
    assert not hasattr(preset, "zygotic_mutation_rate")
    assert preset.mutation_rates == ((0.3, 0.3), (0.5, 0.5))
    population.update().reconfigure_preset(preset, mutation_rates=[0.2, 0.3])
    assert preset.effective_rates() == ((0.2, 0.2), (0.3 / 0.8, 0.3 / 0.8))
