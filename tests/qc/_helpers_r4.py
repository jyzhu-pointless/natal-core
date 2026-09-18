"""Shared builders for the 2026-09-17 adversarial QC round (R4).

This round attacks *fresh* ground relative to rounds P (0915), QC (0915)
and R3 (0916).  In particular:

- Burt (2003) doubling closed form ``1 - q_t = (1 - q_0)^(2^t)`` and the
  heterozygote-cost invasion threshold ``(1 + e)(1 - h s) > 1``,
- Turelli & Hoffmann (1995) cytoplasmic-incompatibility threshold
  ``p* = s_f / s_h`` built from public primitives (paternal gamete tags +
  cross-keyed zygote rules + slab viability),
- Kimura (1962) fixation probability under selection,
- Haldane (1927) mutation-selection balance ``q_hat = mu / s``,
- sex-chromosome species calibration (``sex_ratio`` used by
  ``equilibrium_metrics`` while the engine ignores it),
- sperm displacement / remating bookkeeping in the age-structured path,
- continuous (Beta/Dirichlet/Gamma) sampling moments,
- spatial density equilibrium per deme plus a zero-habitat sink deme.

All probes build populations through the public builder chain only and
never write to product code.  Species names are prefixed ``R4_``.
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


def species_xy(name: str, loci: Sequence[str] = ("A", "a")) -> nt.Species:
    """XY species: one autosomal locus (loci[0] vs loci[1]) plus X/Y chromosomes.

    Genotype keys read ``"A|a;X1|X1"`` (female) and ``"A|a;X1|Y1"`` (male).
    """
    x_alleles = [f"X{i + 1}" for i in range(max(1, len(loci) - 1))]
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrA": {"loci": {"A": list(loci)}},
            "chrX": {"sex_type": "X", "loci": {"sx": x_alleles}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
    )


def species_cyto(name: str, *, slabs: Sequence[str], glabs: Sequence[str]) -> nt.Species:
    """Single-allele species carrying somatic slabs and gamete labels."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["W"]}},
        gamete_labels=list(glabs),
        somatic_labels=list(slabs),
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
    continuous_sampling: bool = False,
    fixed_egg_count: bool | None = None,
    extreme_speed_mode: int | None = None,
    seed: int | None = None,
    extra: Any = None,
) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic-by-default discrete-generation population."""
    setup_kwargs: dict[str, Any] = {
        "species": species,
        "name": name,
        "stochastic": stochastic,
        "continuous_sampling": continuous_sampling,
    }
    if fixed_egg_count is not None:
        setup_kwargs["fixed_egg_count"] = fixed_egg_count
    if extreme_speed_mode is not None:
        setup_kwargs["extreme_speed_mode"] = extreme_speed_mode
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
    pop = builder.build()
    if seed is not None:
        pop._initialize_session(seed=seed)  # noqa: SLF001 - QC harness needs explicit seeds
    return pop


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


# --------------------------------------------------------------------------
# state readers
# --------------------------------------------------------------------------
def _counts(pop: Any) -> np.ndarray:
    return np.asarray(pop.state.individual_count, dtype=float)


def genotype_masses(pop: Any) -> dict[str, float]:
    """Total (both sexes, all ages, all slabs) mass per genotype string."""
    counts = _counts(pop)
    masses: dict[str, float] = {}
    for genotype_obj, _slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype_obj, _slab)
        key = genotype_obj.to_string()
        masses[key] = masses.get(key, 0.0) + float(counts[:, :, idx].sum())
    return masses


def ztype_masses(pop: Any, label: str) -> float:
    """Total mass of every ztype whose slab equals *label*."""
    counts = _counts(pop)
    total = 0.0
    for genotype_obj, slab in pop.registry.index_to_ztype:
        if slab == label:
            idx = pop.registry.ztype_index(genotype_obj, slab)
            total += float(counts[:, :, idx].sum())
    return total


def allele_frequency(pop: Any, allele: str) -> float:
    """Copy-frequency of *allele* over every genotype present in the state."""
    counts = _counts(pop)
    total = float(counts.sum())
    copies = 0.0
    for genotype_obj, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype_obj, slab)
        tokens = [
            token
            for chunk in genotype_obj.to_string().split(";")
            for token in chunk.split("|")
        ]
        copies += float(tokens.count(allele)) * float(counts[:, :, idx].sum())
    return copies / (2.0 * total) if total > 0 else float("nan")


def adult_total(pop: Any) -> float:
    """Total adult (age 1) individuals summed over sex, genotype and slab."""
    return float(_counts(pop)[:, 1, :].sum())


def female_age_counts(pop: Any) -> np.ndarray:
    """Female individuals per age class."""
    return np.asarray(pop.get_age_distribution("female"), dtype=float)


def ad_slab_frequency(pop: Any, label: str) -> float:
    """Fraction of the total adult population carrying somatic slab *label*."""
    total = adult_total(pop)
    if total <= 0.0:
        return float("nan")
    counts = _counts(pop)
    mass = 0.0
    for genotype_obj, slab in pop.registry.index_to_ztype:
        if slab == label:
            idx = pop.registry.ztype_index(genotype_obj, slab)
            mass += float(counts[:, 1, idx].sum())
    return mass / total


# --------------------------------------------------------------------------
# cytoplasmic incompatibility built from public primitives
# --------------------------------------------------------------------------
def build_ci_population(
    name: str,
    *,
    p0: float,
    total: float = 4000.0,
    s_f: float,
    s_h: float,
    eggs: float = 4.0,
    male_p0: float | None = None,
    n_ticks: int = 0,
) -> nt.DiscreteGenerationPopulation:
    """Wolbachia-style CI: fecundity cost ``s_f``, incompatible-embryo death ``s_h``.

    Construction (all public primitives):
    - gamete labels ``mat``/``pat`` tag gametes of infected females/males,
    - a maternal tag redirects ``normal`` zygotes to the ``infected`` slab
      (perfect maternal transmission),
    - a paternal tag redirects ``normal`` zygotes to a ``dead`` slab at rate
      ``s_h`` *after* the maternal rule has already moved infected-mother
      offspring out of ``normal`` (the cascade's ``current`` filter), so only
      the uninfected-female x infected-male cross is killed,
    - the ``dead`` slab carries viability 0 and the ``infected`` slab a female
      fecundity factor ``1 - s_f``.
    """
    sp = species_cyto(
        f"{name}_sp", slabs=["normal", "infected", "dead"], glabs=["default", "mat", "pat"]
    )
    p_male = p0 if male_p0 is None else male_p0
    counts = {
        "female": {
            f"W|W@normal": total * (1.0 - p0),
            f"W|W@infected": total * p0,
        },
        "male": {
            f"W|W@normal": total * (1.0 - p_male),
            f"W|W@infected": total * p_male,
        },
    }

    from natal.frontend.modifiers.gamete_conversion import GameteConversionRuleSet
    from natal.frontend.modifiers.zygote_conversion import ZygoteConversionRuleSet

    def extra(builder: Any) -> Any:
        builder = builder.fitness(
            viability={"W|W@dead": 0.0},
            fecundity={"W|W@infected": {"female": 1.0 - s_f}},
            mode="multiply",
        )
        return builder

    pop = discrete_pop(
        name,
        species=sp,
        female=counts["female"],
        male=counts["male"],
        eggs_per_female=eggs,
        growth_mode="no_competition",
        extra=extra,
    )

    gametes = GameteConversionRuleSet(f"{name}_tag")
    gametes.add_gtype_convert(
        to="*@mat", rate=1.0,
        filters={"parent_sex": "female", "parent": "*@infected", "current": "*@default"},
    )
    gametes.add_gtype_convert(
        to="*@pat", rate=1.0,
        filters={"parent_sex": "male", "parent": "*@infected", "current": "*@default"},
    )
    pop.add_gamete_modifier(gametes.to_gamete_modifier(pop))

    zygotes = ZygoteConversionRuleSet(f"{name}_ci")
    zygotes.add_ztype_convert(
        to="*@infected", rate=1.0,
        filters={"maternal": "*@mat", "current": "*@normal"},
    )
    zygotes.add_ztype_convert(
        to="*@dead", rate=s_h,
        filters={"paternal": "*@pat", "current": "*@normal"},
    )
    pop.add_zygote_modifier(zygotes.to_zygote_modifier(pop))

    if n_ticks:
        pop.run(n_ticks)
    return pop


def ci_map(p: float, s_f: float, s_h: float) -> float:
    """Turelli-Hoffmann infected-frequency recursion (random mating, perfect transmission).

    ``p' = p (1 - s_f) / [p (1 - s_f) + (1 - p)^2 + (1 - p) p (1 - s_h)]``
    """
    infected = p * (1.0 - s_f)
    uninfected = (1.0 - p) ** 2 + (1.0 - p) * p * (1.0 - s_h)
    return infected / (infected + uninfected)


def ci_threshold(s_f: float, s_h: float) -> float:
    """Interior unstable equilibrium of :func:`ci_map`: ``p* = s_f / s_h``."""
    return s_f / s_h
