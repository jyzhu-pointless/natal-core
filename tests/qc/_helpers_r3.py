"""Shared builders for the 2026-09-16 adversarial QC round (R3).

This round deliberately attacks *iterated* numerical behaviour: the
per-tick recursions, their fixed points, stability boundaries, growth
rates and sampling moments, with expectations taken from the classical
population-genetics / population-dynamics literature rather than from
the implementation's own algebra:

- Beverton-Holt and Ricker stock-recruitment maps (Ricker 1954; May 1976),
- Burt (2003) / Unckless et al. (2015) homing-drive recursion and its
  ``e/s`` threshold,
- Euler-Lotka / Leslie dominant eigenvalue (Euler 1760; Leslie 1945),
- Wright-Fisher drift variance ``p(1-p)/(2N)`` (Wright 1931),
- migration mixing decay ``(1-2m)^t``,
- Fisherian 1:1 sex ratio under XY sex determination.

All probes build populations through the public builder chain only and
never write to product code.  Species names are prefixed ``R3_`` so the
singleton ``Species`` cache cannot collide with earlier rounds.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import natal as nt


# --------------------------------------------------------------------------
# species
# --------------------------------------------------------------------------
def species_locus(name: str, alleles: Sequence[str]) -> nt.Species:
    """Single autosomal locus with the given allele list."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": list(alleles)}},
        gamete_labels=["default"],
    )


def species_xy(name: str, alleles: Sequence[str] = ("W", "D")) -> nt.Species:
    """XY species with one autosomal locus plus the sex chromosomes."""
    return nt.Species.from_dict(
        name=name,
        structure={
            "chr1": {"loc": list(alleles)},
            "sex": {"X": ["X"], "Y": ["Y"]},
        },
        gamete_labels=["default"],
    )


# --------------------------------------------------------------------------
# discrete-generation populations
# --------------------------------------------------------------------------
def discrete_pop(
    name: str,
    *,
    species: nt.Species,
    female: Mapping[str, float],
    male: Mapping[str, float],
    eggs_per_female: float,
    sex_ratio: float = 0.5,
    survival: float = 1.0,
    growth_mode: str | int = "no_competition",
    carrying_capacity: float = 1e12,
    low_density_growth_rate: float = 2.0,
    stochastic: bool = False,
    fixed_egg_count: bool | None = None,
    extra: Any = None,
) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic-by-default discrete-generation population."""
    setup_kwargs: dict[str, Any] = {
        "species": species,
        "name": name,
        "stochastic": stochastic,
    }
    if fixed_egg_count is not None:
        setup_kwargs["fixed_egg_count"] = fixed_egg_count
    builder = nt.DiscreteGenerationPopulation.setup(**setup_kwargs)
    builder = (
        builder.initial_state(individual_count={"female": dict(female), "male": dict(male)})
        .survival(female_age0_survival=survival, male_age0_survival=survival)
        .reproduction(eggs_per_female=eggs_per_female, sex_ratio=sex_ratio)
        .competition(
            juvenile_growth_mode=growth_mode,
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=low_density_growth_rate,
        )
    )
    if extra is not None:
        builder = extra(builder)
    return builder.build()


def zidx(pop: Any, genotype: str) -> int:
    """Ztype index whose genotype string matches exactly."""
    for genotype_obj, slab in pop.registry.index_to_ztype:
        if genotype_obj.to_string() == genotype:
            return pop.registry.ztype_index(genotype_obj, slab)
    raise KeyError(f"no ztype with genotype string {genotype!r}")


def adult_total(pop: Any) -> float:
    """Total adult (age 1) individuals summed over sex and genotype."""
    counts = np.asarray(pop.state.individual_count)
    return float(counts[:, 1, :].sum())


def genotype_masses(pop: Any) -> dict[str, float]:
    """Total (both sexes, all ages) mass per genotype string."""
    counts = np.asarray(pop.state.individual_count)
    masses: dict[str, float] = {}
    for genotype_obj, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype_obj, slab)
        key = genotype_obj.to_string()
        masses[key] = masses.get(key, 0.0) + float(counts[:, :, idx].sum())
    return masses


def allele_frequency(pop: Any, allele: str) -> float:
    """Copy-frequency of *allele* over all genotypes present in the state."""
    counts = np.asarray(pop.state.individual_count)
    total = float(counts.sum())
    copies = 0.0
    for genotype_obj, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype_obj, slab)
        tokens = [
            token
            for chunk in genotype_obj.to_string().split(";")
            for token in chunk.split("|")
        ]
        # Two copies for a homozygote, one for a heterozygote or haploid match.
        copies += float(tokens.count(allele)) * float(counts[:, :, idx].sum())
    return copies / (2.0 * total) if total > 0 else float("nan")


# --------------------------------------------------------------------------
# age-structured populations
# --------------------------------------------------------------------------
def age_pop(
    name: str,
    *,
    species: nt.Species,
    n_ages: int,
    new_adult_age: int,
    initial: Mapping[str, Mapping[str, Mapping[int, float]]],
    survival_f: Sequence[float],
    survival_m: Sequence[float],
    mating_f: Sequence[float],
    mating_m: Sequence[float],
    eggs_per_female: float,
    sex_ratio: float = 0.5,
    repro_rate: Sequence[float] | float | None = None,
    fertility: Sequence[float] | float | None = None,
    growth_mode: str | int = "no_competition",
    carrying_capacity: float = 1e12,
    low_density_growth_rate: float = 2.0,
    stochastic: bool = False,
    extra: Any = None,
) -> nt.AgeStructuredPopulation:
    """Build a deterministic age-structured population with explicit rates."""
    builder = (
        nt.AgeStructuredPopulation.setup(species=species, name=name, stochastic=stochastic)
        .age_structure(n_ages=n_ages, new_adult_age=new_adult_age)
        .initial_state(individual_count=dict(initial))
        .survival(
            female_age_based_survival=list(survival_f),
            male_age_based_survival=list(survival_m),
        )
        .reproduction(
            female_age_based_mating_rate=list(mating_f),
            male_age_based_mating_rate=list(mating_m),
            eggs_per_female=eggs_per_female,
            sex_ratio=sex_ratio,
            age_based_reproduction_rate=repro_rate,
            female_age_based_fertility=fertility,
        )
        .competition(
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=low_density_growth_rate,
            growth_mode=growth_mode,
        )
    )
    if extra is not None:
        builder = extra(builder)
    return builder.build()


def female_age_counts(pop: Any) -> np.ndarray:
    """Female individuals per age class."""
    return np.asarray(pop.get_age_distribution("female"), dtype=float)


def male_age_counts(pop: Any) -> np.ndarray:
    """Male individuals per age class."""
    return np.asarray(pop.get_age_distribution("male"), dtype=float)
