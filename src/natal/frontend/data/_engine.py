"""Engine-level building and compression helpers.

This private module contains functions for initializing gamete/zygote maps,
building a discrete-generation ``ModelDraft``, validating custom slot
values, and compressing drafts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, List, Optional, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Genotype, HaploidGenotype
from natal.frontend.utils.types import Sex

from ._config import (
    build_config_maps,
)
from .config import ModelDraft

__all__ = [
    "derive_equilibrium_metrics_from_draft",
    "equilibrium_metrics_dispatch",
    "initialize_gamete_map",
    "initialize_zygote_map",
    "recompute_offspring_tensor",
    "validate_meiosis_table",
]


def equilibrium_metrics_dispatch(
    carrying_capacity: float,
    eggs_per_female: float,
    sex_ratio: float,
    survival_rates: NDArray[np.float64],
    reproduction_rates: NDArray[np.float64],
    fertility: NDArray[np.float64],
    competition_weights: NDArray[np.float64],
    new_adult_age: int,
    n_ages: int,
    declared_distribution: NDArray[np.float64] | None,
    external_expected_eggs: float | None,
) -> tuple[float, float] | None:
    """Run the Rust equilibrium kernel; ``None`` when it is unavailable.

    Single dispatch point for the equilibrium calibration (plan 5.2):
    the sensitive-parameter sync path and the build-time map computation
    both funnel through here so the kernel choice cannot drift apart.
    Callers feed already-resolved reproduction vectors (the None-fallback
    to the female mating row is caller policy) and translate ``None``
    into their pure-Python fallback.

    Args:
        carrying_capacity: Carrying capacity K (age-1 total).
        eggs_per_female: Baseline offspring count per female.
        sex_ratio: Female proportion.
        survival_rates: ``(2, n_ages)`` survival matrix.
        reproduction_rates: Resolved ``(n_ages,)`` participation vector.
        fertility: ``(n_ages,)`` relative female fertility.
        competition_weights: ``(n_ages,)`` juvenile competition weights.
        new_adult_age: First adult age index.
        n_ages: Total age classes.
        declared_distribution: ``None`` or empty means derive mode.
        external_expected_eggs: Champer egg override (``None`` = unused).

    Returns:
        ``(expected_competition_strength, expected_survival_rate)`` from
        the Rust kernel, or ``None`` when the extension is absent.
    """
    try:
        from natal._engine_rs import equilibrium_metrics_flat as rust_metrics
    except ImportError:
        return None
    declared = (
        np.ascontiguousarray(declared_distribution, dtype=np.float64)
        if declared_distribution is not None and declared_distribution.size > 0
        else None
    )
    return rust_metrics(
        float(carrying_capacity),
        float(eggs_per_female),
        float(sex_ratio),
        np.ascontiguousarray(survival_rates, dtype=np.float64),
        np.ascontiguousarray(reproduction_rates, dtype=np.float64),
        np.ascontiguousarray(fertility, dtype=np.float64),
        np.ascontiguousarray(competition_weights, dtype=np.float64),
        int(new_adult_age),
        int(n_ages),
        declared,
        external_expected_eggs,
    )


def _rust_offspring_kernel(
    meiosis: NDArray[np.float64], fusion: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Run the Rust offspring kernel when the extension is available.

    Args:
        meiosis: Meiosis table of shape ``(2, n_ztypes, n_gtypes)``.
        fusion: Fusion table of shape ``(n_gtypes, n_gtypes, n_ztypes)``.

    Returns:
        The flat kernel result reshaped to ``(n_ztypes,)*3``, or ``None``
        when the extension is not importable (the pure-Python fallback
        remains until the Rust-only stage retires it).
    """
    try:
        from natal._engine_rs import compute_offspring_tensor as rust_kernel
    except ImportError:
        return None
    flat = rust_kernel(
        np.ascontiguousarray(meiosis, dtype=np.float64),
        np.ascontiguousarray(fusion, dtype=np.float64),
    )
    z = int(meiosis.shape[1])
    return np.asarray(flat, dtype=np.float64).reshape(z, z, z)


def recompute_offspring_tensor(
    meiosis: NDArray[np.float64],
    fusion: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Recompute the derived offspring tensor from the live tables.

    Single owner of the derivation
    ``P[i,j,k] = sum(meiosis_f[i,a] * meiosis_m[j,b] * fusion[a,b,k])``;
    every caller — the writer channel, the spatial variant channel, the
    modifier refresh, the registry compression, and the build-time map
    computation — funnels through this one spelling so they cannot drift
    apart.  The numeric kernel lives in Rust (plan 5.2); the pure-Python
    spelling below is the extension-less fallback and both are
    statement-for-statement identical, so results are bit-equal either
    way.

    Args:
        meiosis: Meiosis table of shape ``(2, n_ztypes, n_gtypes)``.
        fusion: Fusion table of shape ``(n_gtypes, n_gtypes, n_ztypes)``.

    Returns:
        The recomputed offspring tensor ``(n_ztypes, n_ztypes, n_ztypes)``.
    """
    meiosis = np.ascontiguousarray(meiosis, dtype=np.float64)
    fusion = np.ascontiguousarray(fusion, dtype=np.float64)
    rust_result = _rust_offspring_kernel(meiosis, fusion)
    if rust_result is not None:
        return rust_result
    # Extension-less fallback: same statement order and zero-skips as the
    # Rust kernel (bit-identical results).
    from natal.backends.reference.simulation.age_structured import (
        compute_offspring_probability_tensor,
    )

    # Single-gamete-label layouts collapse the label axis, so the
    # ztype/gtype counts come from the meiosis table itself.
    n_z = int(meiosis.shape[1])
    n_g = int(meiosis.shape[2])
    return np.ascontiguousarray(
        compute_offspring_probability_tensor(
            meiosis_f=meiosis[0],
            meiosis_m=meiosis[1],
            haplo_to_genotype_map=fusion,
            n_ztypes=n_z,
            n_gtypes=n_g,
        )
    )


def derive_equilibrium_metrics_from_draft(
    draft: ModelDraft,
) -> tuple[float, float]:
    """Derive the equilibrium metrics from a draft's current values.

    Single read-side derivation shared by the sensitive-write sync and
    the ``pop.params`` query surface (plan 5.2: one numeric source; the
    draft's stored copies retire with slice 2).  The declared
    distribution and Champer override are read from the draft itself,
    and the reproduction fallback (female mating row) is resolved here.

    Args:
        draft: The draft whose ecology drives the derivation.

    Returns:
        ``(expected_competition_strength, expected_survival_rate)`` —
        always freshly computed, never a cached copy.
    """
    reproduction = (
        draft.age_based_reproduction_rates
        if draft.age_based_reproduction_rates is not None
        else draft.age_based_mating_rates[0]
    )
    metrics = equilibrium_metrics_dispatch(
        draft.carrying_capacity,
        draft.eggs_per_female,
        draft.sex_ratio,
        draft.age_based_survival_rates,
        reproduction,
        draft.female_age_based_fertility,
        draft.age_based_relative_competition_strength,
        int(draft.new_adult_age),
        int(draft.n_ages),
        draft.equilibrium_individual_distribution,
        draft.external_expected_eggs,
    )
    if metrics is not None:
        return metrics
    # Extension-less fallback: the pure-Python spelling (retired at S6).
    from natal.backends.reference.simulation.age_structured import (
        compute_equilibrium_metrics,
    )

    return compute_equilibrium_metrics(
        carrying_capacity=float(draft.carrying_capacity),
        eggs_per_female=float(draft.eggs_per_female),
        age_based_survival_rates=draft.age_based_survival_rates,
        age_based_mating_rates=draft.age_based_mating_rates,
        age_based_reproduction_rates=draft.age_based_reproduction_rates,
        female_age_based_fertility=draft.female_age_based_fertility,
        relative_competition_strength=draft.age_based_relative_competition_strength,
        sex_ratio=float(draft.sex_ratio),
        new_adult_age=int(draft.new_adult_age),
        n_ages=int(draft.n_ages),
        equilibrium_individual_count=draft.equilibrium_individual_distribution,
        external_expected_eggs=draft.external_expected_eggs,
    )


def validate_meiosis_table(candidate: NDArray[np.float64]) -> None:
    """Reject a meiosis table whose rows are not distributions.

    Single validation point shared by every meiosis write channel, so
    they all refuse the same inputs.

    Args:
        candidate: The candidate ``(2, n_ztypes, n_gtypes)`` table.

    Raises:
        ValueError: If any (sex, ztype) row does not sum to 1 or
            contains a negative entry — meiosis always produces
            exactly one gamete with non-negative probability.
    """
    row_sums = candidate.sum(axis=-1)
    ok = np.isclose(row_sums, 1.0, rtol=1e-9, atol=1e-12)
    if not ok.all():
        bad = int(np.count_nonzero(~ok))
        raise ValueError(
            "meiosis_map rows must be probability distributions "
            f"(each (sex, ztype) row sums to 1); {bad} row(s) violate this"
        )
    if (candidate < 0.0).any():
        bad = int(np.count_nonzero(candidate < 0.0))
        raise ValueError(
            "meiosis_map entries must be non-negative; "
            f"{bad} entr(ies) violate this"
        )


def initialize_zygote_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    n_slabs: int = 1,
    unordered: bool = False,
    zygote_modifiers: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
) -> NDArray[np.float64]:
    """Initialize the ``gametes_to_zygotes_map`` tensor.

    Builds baseline Mendelian inheritance for all haplotype pairs and
    gamete-label combinations on ``n_genotypes * n_slabs`` zygote-type
    columns, then applies optional zygote modifiers.  The baseline maps
    each genotype pair to the **default** slab (index 0 of each genotype
    group); zygote modifiers may redirect probability to other slab
    indices within the same genotype group.

    When *unordered* is True, uses ``unordered_genotype()`` so that
    ``(hg_a, hg_b)`` and ``(hg_b, hg_a)`` map to the same unordered
    genotype index, collapsing symmetric pairs.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects (unique
            set, without slab variants).
        n_glabs: Number of gamete labels (default: 1).
        n_slabs: Number of somatic slab variants per genotype (≥ 1).
        unordered: If True, use unordered genotype canonicalization.
        zygote_modifiers: Optional sequence of callables that accept and
            return a modified ``gametes_to_zygotes_map`` tensor.

    Returns:
        Array of shape ``(n_gtypes, n_gtypes, n_genotypes * n_slabs)``.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    n_ztypes = n_genotypes * n_slabs
    n_gtypes = n_hg * n_glabs
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    gametes_to_zygotes_map: NDArray[np.float64] = np.zeros(
        (n_gtypes, n_gtypes, n_ztypes), dtype=np.float64,
    )

    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi
        for hi in range(n_hg) for gi in range(n_glabs)
    }

    for idx_hg1, hg1 in enumerate(haploid_genotypes):
        for idx_hg2, hg2 in enumerate(haploid_genotypes):
            if unordered:
                zygote_gt = hg1.species.unordered_genotype(hg1, hg2)
            else:
                zygote_gt = Genotype(
                    species=hg1.species,
                    maternal=hg1,
                    paternal=hg2,
                )

            if zygote_gt in diploid_genotypes:
                idx_gt = diploid_genotypes.index(zygote_gt)
                ztype_idx = idx_gt * n_slabs
                for glab1 in range(n_glabs):
                    for glab2 in range(n_glabs):
                        compressed_idx1 = _gtype_index[(idx_hg1, glab1)]
                        compressed_idx2 = _gtype_index[(idx_hg2, glab2)]
                        gametes_to_zygotes_map[
                            compressed_idx1, compressed_idx2, ztype_idx
                        ] = 1.0

    if zygote_modifiers:
        for modifier in zygote_modifiers:
            gametes_to_zygotes_map = modifier(gametes_to_zygotes_map)

    return gametes_to_zygotes_map


def initialize_gamete_map(
    haploid_genotypes: List[HaploidGenotype],
    diploid_genotypes: List[Genotype],
    n_glabs: int = 1,
    n_slabs: int = 1,
    gamete_modifiers: Optional[List[Callable[[NDArray[np.float64]], NDArray[np.float64]]]] = None,
) -> NDArray[np.float64]:
    """Create and return a ``zygotes_to_gametes_map`` tensor.

    Builds a baseline mapping from each diploid genotype's gamete
    production, replicated across *n_slabs* somatic slab variants, then
    applies optional modifier callables on the full ``n_genotypes *
    n_slabs``-wide tensor.

    Mendelian gamete production is identical across slab variants, so each
    genotype's baseline frequencies are replicated to all slab indices
    within its genotype group.

    Args:
        haploid_genotypes: List of all haploid genotype objects.
        diploid_genotypes: List of all diploid genotype objects (unique
            set, without slab variants).
        n_glabs: Number of gamete labels (default: 1).
        n_slabs: Number of somatic slab variants per genotype (≥ 1).
        gamete_modifiers: Optional sequence of callables that accept and
            return a modified ``zygotes_to_gametes_map`` tensor.

    Returns:
        ``(n_sexes, n_genotypes * n_slabs, n_gtypes)`` float64 array.
    """
    n_hg = len(haploid_genotypes)
    n_genotypes = len(diploid_genotypes)
    n_ztypes = n_genotypes * n_slabs
    if n_hg <= 0:
        raise ValueError("haploid_genotypes must be non-empty")
    if n_genotypes <= 0:
        raise ValueError("diploid_genotypes must be non-empty")
    if n_glabs <= 0:
        raise ValueError("n_glabs must be positive")

    n_sexes = max(int(s.value) for s in Sex) + 1
    n_gtypes = n_hg * n_glabs

    zygotes_to_gametes_map: NDArray[np.float64] = np.zeros(
        (n_sexes, n_ztypes, n_gtypes), dtype=np.float64,
    )
    haplo_to_idx = {hg: idx for idx, hg in enumerate(haploid_genotypes)}

    _gtype_index: dict[tuple[int, int], int] = {
        (hi, gi): hi * n_glabs + gi
        for hi in range(n_hg) for gi in range(n_glabs)
    }

    allowed_haplotypes_by_sex: dict[int, set[HaploidGenotype]] = {}
    if haploid_genotypes:
        species = haploid_genotypes[0].species
        try:
            female_allowed = set(species.get_maternal_haploid_genotypes())
            male_allowed = set(species.get_paternal_haploid_genotypes())
            if female_allowed:
                allowed_haplotypes_by_sex[int(Sex.FEMALE)] = female_allowed
            if male_allowed:
                allowed_haplotypes_by_sex[int(Sex.MALE)] = male_allowed
        except Exception:
            allowed_haplotypes_by_sex = {}

    for idx_genotype, genotype in enumerate(diploid_genotypes):
        base_gametes = genotype.produce_gametes()
        for sex_idx in range(n_sexes):
            allowed = allowed_haplotypes_by_sex.get(sex_idx)
            if allowed is None:
                filtered_gametes = base_gametes
            else:
                filtered_gametes = {
                    gamete: freq for gamete, freq in base_gametes.items() if gamete in allowed
                }

            total_freq = float(sum(filtered_gametes.values()))
            if total_freq <= 0.0:
                continue

            inv_total = 1.0 / total_freq
            for gamete, freq in filtered_gametes.items():
                idx_hg = haplo_to_idx.get(gamete)
                if idx_hg is None:
                    continue
                compressed_idx = _gtype_index[(idx_hg, 0)]
                baseline_freq = float(freq) * inv_total
                for slab_idx in range(n_slabs):
                    ztype_idx = idx_genotype * n_slabs + slab_idx
                    zygotes_to_gametes_map[sex_idx, ztype_idx, compressed_idx] = baseline_freq

    if gamete_modifiers:
        for modifier in gamete_modifiers:
            zygotes_to_gametes_map = modifier(zygotes_to_gametes_map)

    return zygotes_to_gametes_map


# ==================================================================
# Haplotype compression helpers — used by gamete-map construction
# during species blueprint build (before IndexRegistry exists).
# Use IndexRegistry.gtype_index() for runtime dict-based lookups once
# the registry is established.
# ==================================================================


def compress_hl(hg_idx: int, glab_idx: int, n_glabs: int) -> int:
    """Compress a (haplogenotype, glab) pair into a flat index.

    The compressed representation is *hg_idx × n_glabs + glab_idx*,
    used to index the ``HL = n_hap × n_glabs`` axis of gamete maps.

    Args:
        hg_idx: Haplogenotype index.
        glab_idx: Gamete-label index.
        n_glabs: Number of distinct gamete labels.

    Returns:
        int: The flat combined index.
    """
    return int(hg_idx) * int(n_glabs) + int(glab_idx)


def decompress_hl(compressed_idx: int, n_glabs: int) -> tuple[int, int]:
    """Decompress a flat HL index back into (hg_idx, glab_idx).

    Args:
        compressed_idx: The flat integer index.
        n_glabs: Number of gamete labels used during compression.

    Returns:
        tuple[int, int]: ``(hg_idx, glab_idx)``.
    """
    hg_idx = int(compressed_idx) // int(n_glabs)
    glab_idx = int(compressed_idx) % int(n_glabs)
    return hg_idx, glab_idx


# ── Discrete-generation variant ──────────────────────────────────────────────


def build_discrete_engine_config(
    *,
    n_genotypes: int,
    n_gtypes: int,
    n_glabs: int,
    n_slabs: int = 1,
    gamete_labels: Optional[list[str]] = None,
    somatic_labels: Optional[list[str]] = None,
    zygotes_to_gametes_map: NDArray[np.float64],
    gametes_to_zygotes_map: NDArray[np.float64],
    carrying_capacity: float | None = None,
    has_sex_chromosomes: bool = False,
    **kwargs: Any,
) -> ModelDraft:
    """Build a discrete-generation :class:`ModelDraft`.

    Discrete-specific defaults (juvenile survival=1.0, adult survival=0.0,
    new_adult_age=1) are applied before the shared computation runs, and
    the result stays in the unified ``(2, n_ages)`` vector schema — no
    decomposition into discrete-only scalars or per-sex views.

    Args:
        n_genotypes: Diploid genotype count before slab expansion.
        n_gtypes: Total gamete types (haplotypes x gamete labels).
        n_glabs: Gamete-label variants per haplotype.
        n_slabs: Somatic-label variants per genotype.
        gamete_labels: Registered gamete label strings (unused here;
            kept for signature parity with the age-structured factory).
        somatic_labels: Registered somatic label strings (as above).
        zygotes_to_gametes_map: (2, z, g) meiosis probabilities.
        gametes_to_zygotes_map: (hl, hl, z) gamete pair mapping.
        carrying_capacity: Environment capacity K.
        has_sex_chromosomes: Sex-chromosome constraints active.
        **kwargs: Remaining shared parameters (survival/mating vectors
            are discrete-fixed; see body).

    Returns:
        A unified ``ModelDraft`` in the discrete normalization.
    """
    # Discrete-generation defaults.
    n_ages = int(kwargs.pop("n_ages", 2))
    new_adult_age = int(kwargs.pop("new_adult_age", 1))

    # Survival: juveniles survive to adult (1.0), adults replaced every tick (0.0).
    survival = np.ones((2, n_ages), dtype=np.float64)
    survival[:, 0] = 1.0
    survival[:, 1] = 0.0

    # Mating / reproduction: only adults (age 1) participate.
    mating = np.ones((2, n_ages), dtype=np.float64)
    mating[:, 0] = 0.0
    reproduction = np.ones(n_ages, dtype=np.float64)
    reproduction[0] = 0.0
    fertility = np.ones(n_ages, dtype=np.float64)
    fertility[0] = 0.0

    # Pop stochastic / continuous_sampling once into locals; both
    # build_config_maps and the ModelDraft constructor need them, and a
    # double kwargs.pop would silently fall back to the default on the
    # second read (a pre-existing bug that gave stochastic=True even
    # when the caller passed False).
    stochastic_val = bool(kwargs.pop("stochastic", True))
    continuous_sampling_val = bool(kwargs.pop("continuous_sampling", False))
    equilibrium_val = kwargs.pop("equilibrium_individual_distribution", None)

    m = build_config_maps(
        n_genotypes=n_genotypes,
        n_gtypes=n_gtypes,
        n_sexes=2,
        n_ages=n_ages,
        n_glabs=n_glabs,
        n_slabs=n_slabs,
        gamete_labels=gamete_labels,
        somatic_labels=somatic_labels,
        new_adult_age=new_adult_age,
        stochastic=stochastic_val,
        continuous_sampling=continuous_sampling_val,
        age_based_mating_rates=mating,
        age_based_reproduction_rates=reproduction,
        age_based_survival_rates=survival,
        female_age_based_fertility=fertility,
        viability_fitness=kwargs.pop("viability_fitness", None),
        fecundity_fitness=kwargs.pop("fecundity_fitness", None),
        sexual_selection_fitness=kwargs.pop("sexual_selection_fitness", None),
        zygote_viability_fitness=kwargs.pop("zygote_viability_fitness", None),
        age_based_relative_competition_strength=kwargs.pop("age_based_relative_competition_strength", None),
        sperm_displacement_rate=float(kwargs.pop("sperm_displacement_rate", 0.05)),
        eggs_per_female=float(kwargs.pop("eggs_per_female", 100.0)),
        fixed_egg_count=bool(kwargs.pop("fixed_egg_count", False)),
        carrying_capacity=carrying_capacity or 1000.0,
        sex_ratio=float(kwargs.pop("sex_ratio", 0.5)),
        low_density_growth_rate=float(kwargs.pop("low_density_growth_rate", 6.0)),
        juvenile_growth_mode=int(kwargs.pop("juvenile_growth_mode", 0)),  # LOGISTIC
        has_sex_chromosomes=has_sex_chromosomes,
        zygotes_to_gametes_map=zygotes_to_gametes_map,
        gametes_to_zygotes_map=gametes_to_zygotes_map,
        initial_individual_count=kwargs.pop("initial_individual_count", None),
        initial_sperm_storage=kwargs.pop("initial_sperm_storage", None),
        age_1_carrying_capacity=kwargs.pop("age_1_carrying_capacity", None),
        old_juvenile_carrying_capacity=kwargs.pop("old_juvenile_carrying_capacity", None),
        infer_capacity_from_initial_state=bool(kwargs.pop("infer_capacity_from_initial_state", True)),
        equilibrium_individual_distribution=equilibrium_val,
        external_expected_eggs=kwargs.pop("external_expected_eggs", None),
        pre_expanded=zygotes_to_gametes_map.shape[1] > n_genotypes,
    )

    extreme_speed = int(kwargs.pop("extreme_speed_mode", 0))
    hook_slot_val = int(kwargs.pop("hook_slot", 0))
    resolved_z_names = kwargs.pop("ztype_names", None)
    resolved_g_names = kwargs.pop("gtype_names", None)
    if resolved_z_names is None:
        resolved_z_names = tuple(f"ztype_{i}" for i in range(m.n_g_compressed))
    if resolved_g_names is None:
        resolved_g_names = tuple(f"gtype_{i}" for i in range(m.n_gtypes))

    return ModelDraft(
        stochastic=stochastic_val,
        continuous_sampling=continuous_sampling_val,
        n_sexes=m.n_sexes,
        n_ages=m.n_ages,
        n_ztypes=m.n_g_compressed,
        n_gtypes=m.n_gtypes,
        n_glabs=m.n_glabs,
        n_slabs=m.n_slabs,
        new_adult_age=m.new_adult_age,
        adult_ages=m.adult_ages.copy(),
        extreme_speed_mode=extreme_speed,
        ztype_names=resolved_z_names,
        gtype_names=resolved_g_names,
        age_based_survival_rates=m.survival,
        age_based_mating_rates=m.mating,
        age_based_reproduction_rates=m.reproduction,
        female_age_based_fertility=m.female_fertility,
        age_based_relative_competition_strength=m.competition,
        carrying_capacity=m.carrying_capacity,
        eggs_per_female=m.eggs_per_female,
        sex_ratio=m.sex_ratio,
        sperm_displacement_rate=m.sperm_displacement_rate,
        low_density_growth_rate=m.low_density_growth_rate,
        juvenile_growth_mode=int(m.juvenile_growth_mode),
        expected_competition_strength=m.expected_competition_strength,
        expected_survival_rate=m.expected_survival_rate,
        generation_time=0.0,
        viability_fitness=m.viability,
        fecundity_fitness=m.fecundity,
        sexual_selection_fitness=m.sexual,
        zygote_viability_fitness=m.zygote,
        zygotes_to_gametes_map=np.stack([m.meiosis_f, m.meiosis_m], axis=0),
        gametes_to_zygotes_map=m.zygote_map,
        offspring_tensor=m.offspring_tensor,
        female_ztype_compatibility=m.female_ztype_compatibility,
        male_ztype_compatibility=m.male_ztype_compatibility,
        female_only_by_sex_chrom=m.female_only_by_sex_chrom,
        male_only_by_sex_chrom=m.male_only_by_sex_chrom,
        initial_individual_count=m.initial_individual_count,
        initial_sperm_storage=m.initial_sperm_storage,
        equilibrium_individual_distribution=equilibrium_val,
        hook_slot=hook_slot_val,
        custom={},
        fixed_egg_count=m.fixed_egg_count,
        has_sex_chromosomes=m.has_sex_chromosomes,
        external_expected_eggs=None,
        discrete_generation=True,
    )



def build_custom_slots(
    specs: Mapping[str, object],
) -> dict[str, bool | int | float | NDArray[np.float64]]:
    """Validate and normalize user custom slot values.

    Called by :meth:`Configurator.custom` and the legacy
    ``PopulationBuilderBase.custom``.  The draft's ``custom`` field is a
    plain ``{name: value}`` dict (the runtime ``Params.custom_slots``
    contract); this helper is the single validation point for what may
    enter it.

    Normalization per entry:

    - NumPy scalars (``np.generic``) → native Python values via ``item()``
      (``np.bool_`` → ``bool``, ``np.integer`` → ``int``, ``np.floating``
      → ``float``).
    - Native ``bool`` / ``int`` / ``float`` pass through unchanged.
    - 3-D ``np.ndarray`` → fresh float64 C-contiguous copy (the draft
      owns its arrays; callers never share storage with user input).

    Args:
        specs: ``{name: value}`` mapping of custom field names to values.

    Returns:
        A fresh normalized ``{name: value}`` mapping.

    Raises:
        TypeError: If a value has an unsupported type.
    """
    slots: dict[str, bool | int | float | NDArray[np.float64]] = {}
    for name, val in specs.items():
        if isinstance(val, np.ndarray):
            array_val = cast(np.ndarray[Any, np.dtype[Any]], val)
            if len(array_val.shape) != 3:
                raise TypeError(
                    f"custom field '{name}' is a {len(array_val.shape)}-D ndarray. "
                    f"Only 3-D (sex, age, genotype) arrays are supported."
                )
            slots[str(name)] = np.array(val, dtype=np.float64, order="C")
        elif isinstance(val, np.generic):
            # NumPy scalar → native Python value via .item() (np.bool_ →
            # bool, np.integer → int, np.floating → float).
            slots[str(name)] = val.item()
        elif isinstance(val, (bool, int, float)):
            # Native Python values pass through unchanged.  bool is a
            # subclass of int, but the value is stored as given, so the
            # runtime type is preserved either way.
            slots[str(name)] = val
        else:
            raise TypeError(
                f"custom field '{name}' has unsupported type {type(val).__name__!r}. "
                f"Supported types: bool, int, float (including NumPy scalars), "
                f"or 3-D np.ndarray."
            )
    return slots


# ---------------------------------------------------------------------------
# Config compression helper
# ---------------------------------------------------------------------------


def compress_config(
    config: ModelDraft,
    ztype_mask: NDArray[np.int32],
) -> ModelDraft:
    """Subslice the Z-axis fitness/state fields and the name directory.

    Pure function — returns a new draft via ``_replace`` without mutating
    the original.

    Scope note: only the fields listed in the body are subsliced.  The
    meiosis/gamete maps and ``offspring_tensor`` are intentionally NOT
    touched here — the build pipeline (``rebuild_config_maps``) slices
    and recomputes them from the modifier maps before calling this.
    Calling this function standalone therefore leaves those maps at the
    pre-compression axis size.

    Args:
        config: Draft to compress.
        ztype_mask: ``(n_ztypes,)`` int32 array — -1 = pruned.

    Returns:
        A new draft with the Z-axis fitness/state fields compressed.
    """
    _z_active = ztype_mask >= 0
    n_g = int(_z_active.sum())

    overrides: dict[str, Any] = {
        "n_ztypes": n_g,
        "initial_individual_count": config.initial_individual_count[:, :, _z_active],
        "viability_fitness": config.viability_fitness[:, :, _z_active],
        "fecundity_fitness": config.fecundity_fitness[:, _z_active],
        "sexual_selection_fitness": config.sexual_selection_fitness[_z_active, :][:, _z_active],
        "zygote_viability_fitness": config.zygote_viability_fitness[:, _z_active],
        "female_ztype_compatibility": config.female_ztype_compatibility[_z_active],
        "male_ztype_compatibility": config.male_ztype_compatibility[_z_active],
        "female_only_by_sex_chrom": config.female_only_by_sex_chrom[_z_active],
        "male_only_by_sex_chrom": config.male_only_by_sex_chrom[_z_active],
        "initial_sperm_storage": config.initial_sperm_storage[:, _z_active, :][:, :, _z_active],
        "ztype_names": tuple(
            name
            for name, active in zip(config.ztype_names, _z_active.tolist())
            if active
        ),
    }

    return config._replace(**overrides)
