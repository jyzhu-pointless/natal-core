"""R3-06: conversion algebra -- per-copy vs per-gamete semantics.

Contracts under attack (docstrings of the two conversion rule sets):

- ``ZygoteConversionRuleSet.add_allele_convert(..., side=...)`` converts
  *each zygote copy independently* with the declared rate: ``_copy_options``
  yields ``[(original, 1 - rate), (converted, rate)]`` per copy, and the two
  copies' options are combined independently.  ``side`` selects which copy
  (maternal / paternal / both).  ``HomingDrive`` documents its
  ``embryo_resistance_formation_rate`` as "probability of resistance
  formation in embryos *per target copy*" (src/natal/frontend/presets/homing.py),
  so independent per-copy editing is the documented model -- the
  per-individual conversion probability of a wild-type embryo is
  ``1 - (1 - rate)^2``, not ``rate``.
- ``GameteConversionRuleSet.add_allele_convert`` converts *gamete branches*:
  every gamete carrying the source allele converts with the declared rate,
  independent of the parent genotype, unless a ``parent`` filter restricts
  the rule.  A ``W|W`` homozygote therefore transmits drive gametes at the
  declared rate when no filter is given, and stays Mendelian when the rule
  is gated by a carrier pattern (what ``HomingDrive`` does).

Independently derived expectations: binomial per-copy algebra
(``(1-e)^2, 2e(1-e), e^2``), one-sided conversion (``1-e, e``), and the
exact 0/1 boundaries.  Wrong results rejected: conversion applied to the
whole zygote instead of per copy (``(1-e), e`` for a both-sided rule),
conversion re-applied to already-converted copies, one-sided rules leaking
to the other copy, or per-gamete rules silently skipping homozygotes.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers_r3 import discrete_pop, genotype_masses, species_locus
from natal.frontend.modifiers.gamete_conversion import GameteConversionRuleSet
from natal.frontend.modifiers.zygote_conversion import ZygoteConversionRuleSet

TOL = 1e-9


def _zygote_cross(name: str, *, rate: float, side: str):
    """WT|WT x WT|WT with one zygote allele-conversion rule.

    The rule set is compiled against the built population and attached
    through the runtime updater -- the mount route that the codebase's own
    tests use (tests/test_runtime_updater_contracts.py).  The chainable
    ``builder.modifiers(...)`` route shown in the rule sets' docstrings
    raises TypeError; see the EVIDENCE_F4 test below.
    """
    rules = ZygoteConversionRuleSet(f"{name}_rules")
    rules.add_allele_convert(from_allele="W", to_allele="D", rate=rate, side=side)
    population = discrete_pop(
        name,
        species=species_locus(f"{name}_sp", ["W", "D"]),
        female={"W|W": 500.0},
        male={"W|W": 500.0},
        eggs_per_female=2.0,
        growth_mode="no_competition",
    )
    population.update().modifiers(zygote_modifiers=[rules.to_zygote_modifier(population)])
    return population


def _gamete_homozygote_cross(name: str, *, rate: float, carrier_filter: bool = False):
    species = species_locus(f"{name}_sp", ["W", "D"])
    rules = GameteConversionRuleSet(f"{name}_rules")
    filters = None
    if carrier_filter:
        from natal.frontend.presets._types import carrier_pattern

        filters = {"parent": carrier_pattern(species, "D")}
    rules.add_allele_convert(from_allele="W", to_allele="D", rate=rate, filters=filters)

    population = discrete_pop(
        name,
        species=species,
        female={"W|W": 500.0},
        male={"W|W": 500.0},
        eggs_per_female=2.0,
        growth_mode="no_competition",
    )
    population.update().modifiers(gamete_modifiers=[rules.to_gamete_modifier(population)])
    return population


@pytest.mark.parametrize("rate", [0.4, 0.25])
def test_zygote_conversion_is_per_copy_both_sides(rate: float) -> None:
    """WT|WT x WT|WT with side="both": (1-e)^2 / 2e(1-e) / e^2."""
    pop = _zygote_cross(f"R3_06_zyg_both_{rate}", rate=rate, side="both")
    pop.run(1)
    masses = genotype_masses(pop)
    total = sum(masses.values())
    assert total == pytest.approx(1000.0, abs=TOL)
    assert masses["W|W"] / total == pytest.approx((1.0 - rate) ** 2, abs=TOL)
    single = masses["W|D"] + masses.get("D|W", 0.0)
    assert single / total == pytest.approx(2.0 * rate * (1.0 - rate), abs=TOL)
    assert masses["D|D"] / total == pytest.approx(rate**2, abs=TOL)
    # Row sums conserve mass in every cell (no conversion leak).
    assert sum(masses.values()) == pytest.approx(1000.0, abs=TOL)


@pytest.mark.parametrize("side", ["maternal", "paternal"])
def test_zygote_conversion_side_converts_one_copy_only(side: str) -> None:
    """A one-sided rule converts exactly one copy: (1-e) unchanged, e edited."""
    rate = 0.4
    pop = _zygote_cross(f"R3_06_zyg_{side}", rate=rate, side=side)
    pop.run(1)
    masses = genotype_masses(pop)
    total = sum(masses.values())
    assert total == pytest.approx(1000.0, abs=TOL)
    edited = masses["W|D"] + masses.get("D|W", 0.0)
    assert edited / total == pytest.approx(rate, abs=TOL)
    assert masses["W|W"] / total == pytest.approx(1.0 - rate, abs=TOL)
    assert masses["D|D"] / total == pytest.approx(0.0, abs=TOL)


@pytest.mark.parametrize("rate,expected", [(0.0, 0.0), (1.0, 1.0)])
def test_zygote_conversion_boundaries(rate: float, expected: float) -> None:
    """e = 0 and e = 1 are exact no-op / full-conversion boundaries."""
    pop = _zygote_cross(f"R3_06_zyg_bound_{rate}", rate=rate, side="both")
    pop.run(1)
    masses = genotype_masses(pop)
    total = sum(masses.values())
    if rate == 0.0:
        assert masses["W|W"] / total == pytest.approx(1.0, abs=TOL)
    else:
        assert masses["D|D"] / total == pytest.approx(1.0, abs=TOL)


def test_unfiltered_gamete_conversion_acts_on_homozygotes() -> None:
    """No parent filter: a W|W parent transmits D gametes at the declared rate.

    Both parents are W|W, so every gamete starts as W; each converts with
    probability e.  The offspring distribution is the product of the two
    gamete pools: (1-e)^2 / 2e(1-e) / e^2 with e = 0.4.
    """
    rate = 0.4
    pop = _gamete_homozygote_cross(f"R3_06_gam_unfiltered_{rate}", rate=rate)
    pop.run(1)
    masses = genotype_masses(pop)
    total = sum(masses.values())
    assert masses["W|W"] / total == pytest.approx((1.0 - rate) ** 2, abs=TOL)
    assert (masses["W|D"] + masses.get("D|W", 0.0)) / total == pytest.approx(
        2.0 * rate * (1.0 - rate), abs=TOL
    )
    assert masses["D|D"] / total == pytest.approx(rate**2, abs=TOL)


def test_carrier_filtered_gamete_conversion_skips_wild_type_homozygotes() -> None:
    """A carrier-gated rule (HomingDrive's form) leaves W|W parents Mendelian."""
    rate = 0.4
    pop = _gamete_homozygote_cross(
        f"R3_06_gam_filtered_{rate}", rate=rate, carrier_filter=True
    )
    pop.run(1)
    masses = genotype_masses(pop)
    total = sum(masses.values())
    assert masses["W|W"] / total == pytest.approx(1.0, abs=TOL)
    assert masses.get("D|D", 0.0) == pytest.approx(0.0, abs=TOL)


def test_per_embryo_editing_probability_is_derived_from_the_copy_rate() -> None:
    """The declared rate is a *copy* rate; 1-(1-e)^2 is a derived quantity.

    Like the gamete-stage drive conversion rate (whose derived consequence is
    the (1+c)/2 inheritance rate of a carrier, not c itself), the zygote rate
    is defined per target copy and has no per-individual reading.  A wild-type
    embryo is therefore edited with probability 1 - (1 - e)^2 = 0.64 at
    e = 0.4 -- a consequence of the definition, not a second parameter to be
    converted, and the HomingDrive docstring already says "per target copy".
    """
    e = 0.4
    pop = _zygote_cross(f"R3_06_zyg_percopy_note_{e}", rate=e, side="both")
    pop.run(1)
    masses = genotype_masses(pop)
    total = sum(masses.values())
    edited_embryos = total - masses["W|W"]
    assert edited_embryos / total == pytest.approx(1.0 - (1.0 - e) ** 2, abs=TOL)
    assert edited_embryos / total == pytest.approx(0.64, abs=TOL)
    # The rate that would reproduce a per-individual probability of 0.4:
    equivalent = 1.0 - np.sqrt(1.0 - 0.4)
    assert equivalent == pytest.approx(0.2254033307585166, rel=1e-12)


def test_documented_ruleset_mount_form_works() -> None:
    """The rule sets' docstring mount form builds, runs, and converts.

    Both docstrings now show the working route (compile against the built
    population, then ``pop.add_gamete_modifier`` / ``add_zygote_modifier``);
    this test executes it so the documented example cannot drift back into a
    TypeError.  The previously documented
    ``builder.modifiers(gamete_modifiers=[rs.to_gamete_modifier])`` (bound
    method, never called) fails at build time because the pipeline invokes it
    with the population and requires a Mapping back, while
    ``to_gamete_modifier(host)`` returns a ``CompiledRuleModifier``.
    """
    species = species_locus("R3_06_mount_doc_sp", ["W", "D"])
    rules = GameteConversionRuleSet("R3_06_mount_doc")
    rules.add_allele_convert(from_allele="W", to_allele="D", rate=0.4)

    population = discrete_pop(
        "R3_06_mount_doc",
        species=species,
        female={"W|W": 500.0},
        male={"W|W": 500.0},
        eggs_per_female=2.0,
        growth_mode="no_competition",
    )
    population.add_gamete_modifier(rules.to_gamete_modifier(population))
    population.run(1)

    # Unfiltered per-gamete conversion: both gamete pools are (0.6 W, 0.4 D),
    # so the offspring are the product of those pools.
    masses = genotype_masses(population)
    total = sum(masses.values())
    assert total == pytest.approx(1000.0, abs=TOL)
    assert masses["W|W"] / total == pytest.approx(0.36, abs=TOL)
    assert masses["W|D"] / total == pytest.approx(0.48, abs=TOL)
    assert masses["D|D"] / total == pytest.approx(0.16, abs=TOL)
