"""Shared helpers for the 2026-09-15 adversarial QC probe suite.

These probes are read-only with respect to product code: they only build
populations through the public builder chain and assert independently
derived closed-form expectations (Mendelian/Hardy-Weinberg ratios, drive
conversion algebra, Beverton-Holt fixed-point recursion, migration mass
conservation).

All species names are prefixed with ``QC0915`` so the singleton Species
cache cannot collide with the main test suite.
"""

from __future__ import annotations

import natal as nt


def qc_species_3(name: str) -> nt.Species:
    """Autosomal single locus with WT / Dr / R2 (drive-target-resistance)."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        gamete_labels=["default"],
    )


def qc_species_2(name: str) -> nt.Species:
    """Autosomal single locus with two alleles W / D."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["W", "D"]}},
        gamete_labels=["default"],
    )


def zidx(pop: nt.DiscreteGenerationPopulation, genotype_string: str) -> int:
    """Return the ztype index whose genotype string matches exactly."""
    for genotype, slab in pop.registry.index_to_ztype:
        if genotype.to_string() == genotype_string:
            return pop.registry.ztype_index(genotype, slab)
    raise KeyError(f"no ztype with genotype string {genotype_string!r}")


def neutral_population(
    species: nt.Species,
    name: str,
    *,
    female: dict[str, float],
    male: dict[str, float],
    eggs_per_female: float = 10.0,
    sex_ratio: float = 0.5,
    survival: float = 1.0,
    growth_mode: str = "no_competition",
    carrying_capacity: float = 1e12,
    low_density_growth_rate: float = 2.0,
    stochastic: bool = False,
    extra_steps=None,
    extreme_speed_mode: int | None = None,
) -> nt.DiscreteGenerationPopulation:
    """Build a discrete-generation population with explicit parameters.

    ``extra_steps`` is an optional callable applied to the builder right
    before ``build()`` (used to attach presets or fitness writes).
    """
    setup_kwargs: dict = {"species": species, "name": name, "stochastic": stochastic}
    if extreme_speed_mode is not None:
        setup_kwargs["extreme_speed_mode"] = extreme_speed_mode
    builder = nt.DiscreteGenerationPopulation.setup(**setup_kwargs)
    builder = (
        builder.initial_state(individual_count={"female": female, "male": male})
        .survival(female_age0_survival=survival, male_age0_survival=survival)
        .reproduction(eggs_per_female=eggs_per_female, sex_ratio=sex_ratio)
        .competition(
            juvenile_growth_mode=growth_mode,
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=low_density_growth_rate,
        )
    )
    if extra_steps is not None:
        builder = extra_steps(builder)
    return builder.build()
