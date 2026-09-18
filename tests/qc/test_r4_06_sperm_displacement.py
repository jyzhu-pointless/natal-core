"""R4-06: sperm storage, remating and displacement in the age-structured path.

Documented contract (rust ``sample_mating`` + ``fertilize``):

- a female that has mated stores sperm *per male genotype* and keeps it
  across ticks; her offspring come from the stored sperm, not from the males
  present,
- a female mates again only through the displacement channel, so the
  per-tick replacement probability is ``sperm_displacement_rate x female
  mating rate at her age``; the replaced mass is re-drawn from the
  *current* male pool,
- a female that did not mate stays a virgin and mates later at her
  *current* age's rate (no female is permanently lost),
- female counts are conserved by the bookkeeping (mated females are counted
  once; the kernel raises on a negative virgin count).

The probe changes the male composition between two ticks with the public
``Op.convert`` hook (every ``A|A`` male -> ``a|a`` at tick 1).  The age-1
mating rate is fixed at 1 so the first tick's maternity is complete; then
the second tick's newborns must split exactly as

    A|A = 1000 (1 - d p2),   A|a = 1000 d p2 + 1000

for displacement ``d`` and the mothers' age-2 mating rate ``p2``.

Wrong results rejected: displacement that also replaces sperm of females
that did not mate (the ``p2`` factor), offspring drawn from the present
males instead of the stored sperm, stored sperm lost at aging, the
displaced mass not re-drawn (which would shrink the offspring total), a
"displacement" that removes sperm without adding new sperm, and unmated
females becoming permanently sterile.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt

from _helpers_r4 import age_pop, species_locus

EGGS = 2.0
MOTHERS = 500.0


def _make_converter():
    return nt.Op.convert(
        from_=nt.IndividualSelector(ztype="A|A", sex="male"),
        to=nt.IndividualSelector(ztype="a|a", sex="male"),
        probability=1.0,
        when="tick == 1",
        event="first",
    )


def _build(name: str, *, displacement: float, mothers_rate: float):
    species = species_locus(f"R4_06_{name}", ["A", "a"])
    return age_pop(
        name,
        species=species,
        n_ages=3,
        new_adult_age=1,
        initial={
            "female": {"A|A": {1: MOTHERS}},
            "male": {"A|A": {1: MOTHERS}},
        },
        survival_f=[1.0, 1.0, 0.0],
        survival_m=[1.0, 1.0, 0.0],
        mating_f=[0.0, 1.0, mothers_rate],
        mating_m=[0.0, 1.0, 1.0],
        eggs_per_female=EGGS,
        sex_ratio=0.5,
        growth_mode="no_competition",
        extra=lambda b: b.reproduction(sperm_displacement_rate=displacement).hooks(
            _make_converter(), event="first"
        ),
    )


def _newborn_split(pop) -> dict[str, float]:
    """Newborn mass per genotype, read at age 1 after *aging* ran.

    The tick ends with aging, which advances the age-0 cohort to age 1 and
    clears age 0, so the age-1 slot holds exactly this tick's newborns (the
    previous cohort moved to age 2).
    """
    counts = np.asarray(pop.state.individual_count)
    out: dict[str, float] = {}
    for genotype, _slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype, _slab)
        out[genotype.to_string()] = out.get(genotype.to_string(), 0.0) + float(
            counts[:, 1, idx].sum()
        )
    return out


def _run(name: str, *, displacement: float, mothers_rate: float):
    pop = _build(name, displacement=displacement, mothers_rate=mothers_rate)
    pop.run(2)
    return pop


@pytest.mark.parametrize(
    ("displacement", "mothers_rate"),
    [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.0, 0.4), (0.5, 0.6), (0.25, 0.8)],
)
def test_displacement_replaces_exactly_d_times_p2(
    displacement: float, mothers_rate: float
) -> None:
    tag = f"{str(displacement).replace('.', 'p')}_{str(mothers_rate).replace('.', 'p')}"
    pop = _run(f"disp_{tag}", displacement=displacement, mothers_rate=mothers_rate)
    split = _newborn_split(pop)
    replaced = displacement * mothers_rate
    old = MOTHERS * EGGS * (1.0 - replaced)
    new = MOTHERS * EGGS * replaced + MOTHERS * EGGS
    assert split.get("A|A", 0.0) == pytest.approx(old, rel=1e-9)
    assert split.get("A|a", 0.0) == pytest.approx(new, rel=1e-9)


def test_stored_sperm_survives_without_remating() -> None:
    """d = 0: the tick-0 paternity is still in force one tick later."""
    pop = _build("keep", displacement=0.0, mothers_rate=1.0)
    pop.run(1)
    first = _newborn_split(pop)
    assert first.get("A|A", 0.0) == pytest.approx(MOTHERS * EGGS, rel=1e-9)
    assert first.get("A|a", 0.0) == pytest.approx(0.0, abs=1e-12)

    pop.run(1)
    split = _newborn_split(pop)
    # The stored A|A sperm still fathers the mothers' clutch; the tick-0
    # newborn females (now age 1) contribute A|a offspring.
    assert split.get("A|A", 0.0) == pytest.approx(MOTHERS * EGGS, rel=1e-9)
    assert split.get("A|a", 0.0) == pytest.approx(MOTHERS * EGGS, rel=1e-9)


def test_unmated_females_mate_later_at_their_current_age_rate() -> None:
    """Half the mothers mate in tick 0; the virgins stay available.

    Tick 0: age-1 mating rate 0.5 -> 250 pairs -> 500 newborns (250 of each
    sex).  Tick 1 newborns = 500 from the 250 already-mated mothers + 500
    from the 250 never-mated mothers (age-2 rate 1) + 250 from the 250
    daughters (age-1 rate 0.5) = 1250.
    """
    species = species_locus("R4_06_late", ["A", "a"])
    pop = age_pop(
        "late_mating",
        species=species,
        n_ages=3,
        new_adult_age=1,
        initial={"female": {"A|A": {1: MOTHERS}}, "male": {"A|A": {1: MOTHERS}}},
        survival_f=[1.0, 1.0, 0.0],
        survival_m=[1.0, 1.0, 0.0],
        mating_f=[0.0, 0.5, 1.0],
        mating_m=[0.0, 1.0, 1.0],
        eggs_per_female=EGGS,
        sex_ratio=0.5,
        growth_mode="no_competition",
    )
    pop.run(1)
    first = _newborn_split(pop)
    assert sum(first.values()) == pytest.approx(MOTHERS * 0.5 * EGGS, rel=1e-9)

    pop.run(1)
    second = _newborn_split(pop)
    mated_mothers = MOTHERS * 0.5
    late_mothers = MOTHERS * 0.5
    daughters = MOTHERS * 0.5 * EGGS * 0.5  # newborn females of tick 0
    expected = (
        mated_mothers * EGGS          # still carrying stored sperm
        + late_mothers * 1.0 * EGGS   # mated for the first time at age 2
        + daughters * 0.5 * EGGS      # daughters at the age-1 rate
    )
    assert sum(second.values()) == pytest.approx(expected, rel=1e-9)


def test_female_count_is_conserved_by_displacement() -> None:
    """Displacement must not create or destroy mated females."""
    for displacement in (0.0, 0.5, 1.0):
        pop = _build(
            f"cons_{str(displacement).replace('.', 'p')}",
            displacement=displacement,
            mothers_rate=1.0,
        )
        before = float(np.asarray(pop.state.individual_count)[0].sum())
        pop.run(1)
        after = float(np.asarray(pop.state.individual_count)[0].sum())
        # Tick 0 adds the newborn females (500 x 2 eggs x 0.5 sex ratio).
        assert after == pytest.approx(before + MOTHERS * EGGS * 0.5, rel=1e-9)
