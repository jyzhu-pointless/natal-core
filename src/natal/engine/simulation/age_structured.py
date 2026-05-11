"""Simulation helpers used by cohort-based (absolute population size)
population simulations.

This module provides Numba-accelerated helper functions for computing
mating/sperm matrices, updating sperm storage and occupancy, generating
offspring distributions, and other population genetics operations. All
functions are written to be shape-defensive and to integrate with the
`PopulationState` data structures.
"""
from typing import Annotated, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from natal import numba_compat as nbc
from natal.numba_compat import njit_switch

# ============================================================================
# Continuous distribution helper functions (for use_continuous_sampling=True)
# ============================================================================
# Very small threshold to prevent numerical errors when distribution parameters are 0
EPS = 1e-10

# 1. Prepare male gamete pool
@njit_switch(cache=True)
def compute_mating_probability_matrix(
    sexual_selection_matrix: Annotated[NDArray[np.float64], "shape=(g,g)"],
    male_counts: Annotated[NDArray[np.float64], "shape=(g,)"],
    n_genotypes: int
) -> Annotated[NDArray[np.float64], "shape=(g,g)"]:
    """Compute a row-normalized mating probability matrix.

    The function computes A = alpha * diag(M) (implemented as column-wise
    scaling) and returns a row-normalized matrix P where each row sums to 1.

    Args:
        sexual_selection_matrix: Preference weights with shape ``(g, g)``.
            Rows correspond to female genotypes, columns to male genotypes.
        male_counts: Male counts vector with shape ``(g,)``.
        n_genotypes: Number of genotypes ``g`` used for shape validation.

    Returns:
        np.ndarray: Row-normalized mating probability matrix ``P`` with shape
            ``(g, g)``. Any zero rows in the intermediate matrix are preserved
            as zero rows in the output.
    """
    A = np.asarray(sexual_selection_matrix)
    M = np.asarray(male_counts)
    g = n_genotypes

    assert A.shape == (g, g)
    assert M.shape == (g,)

    # Allocate one output matrix and fill it in-place to avoid temporary
    # weighted / row_sums arrays.
    P = np.empty((g, g), dtype=np.float64)
    for gf in range(g):
        row_sum = 0.0
        for gm in range(g):
            val = A[gf, gm] * M[gm]
            P[gf, gm] = val
            row_sum += val

        if row_sum > 0.0:
            inv = 1.0 / row_sum
            for gm in range(g):
                P[gf, gm] *= inv
        else:
            for gm in range(g):
                P[gf, gm] = 0.0
    return P

@njit_switch(cache=True)
def sample_mating(
    female_counts: Annotated[NDArray[np.float64], "shape=(A,g)"],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    mating_prob: Annotated[NDArray[np.float64], "shape=(g,g)"],
    female_mating_rates_by_age: Annotated[NDArray[np.float64], "shape=(A,)"],
    sperm_displacement_rate: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
) -> Annotated[NDArray[np.float64], "shape=(A,g,g)"]:
    """Vectorized version: batch sampling of mating events (monogamous). (67.0x speedup)

    Assumption: Each female mates at most once per tick.
    Sampling process consists of two steps:
    1. Determine how many females of each genotype participate in mating (Binomial)
    2. These mating females choose which male genotype to mate with (Multinomial)

    Args:
        female_counts: Female counts array with shape (A, g) where A is number of ages
        sperm_store: Sperm storage array with shape (A, g, g) tracking mated females by male genotype
        mating_prob: Mating probability matrix with shape (g, g)
        female_mating_rates_by_age: Age-specific female mating rates with shape (A,)
        sperm_displacement_rate: Rate controlling remating displacement.
            The effective remating probability is
            ``p_remating = p_mating(age) * sperm_displacement_rate``.
        adult_start_idx: Starting age index for adults
        n_ages: Total number of age classes
        n_genotypes: Number of genotypes g
        is_stochastic: If True, use stochastic sampling; if False, use deterministic expectations
        use_continuous_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling. Currently not implemented (will use discrete).

    Returns:
        Updated sperm storage array with shape (A, g, g) containing mated female allocations

    Note:
        - `S[a, gf, :]` is interpreted as a partition of *mated* females of
          (age=a, female_genotype=gf) by male genotype.
                - Virgins are represented implicitly as:
                    `virgins = female_count[a, gf] - sum_gm(S[a, gf, gm])`
                - Historical sperm storage is preserved across ticks.
                    Remating displaces an expected fraction of existing sperm allocation,
                    then adds newly formed matings.
        - Previous implementation used a single scalar mating rate for all
          adult ages. Now we use age-specific female mating rates:
          `p_mating(age) = female_mating_rates_by_age[age]`.
    """
    sperm = sperm_store.copy()
    females = female_counts
    female_rates = np.asarray(female_mating_rates_by_age)
    mating_prob_mat = np.asarray(mating_prob)
    assert female_rates.shape[0] == n_ages

    # Sperm storage tracks mated females partitioned by male genotype.
    # Historical mated state is preserved across ticks; remating displaces
    # existing sperm allocations controlled by sperm_displacement_rate.

    if is_stochastic:
        tmp = np.zeros(n_genotypes, dtype=np.float64)

        for age in range(adult_start_idx, n_ages):
            p_mating = nbc.clamp01(float(female_rates[age]))
            p_displace = nbc.clamp01(float(sperm_displacement_rate))

            for gf in range(n_genotypes):
                n_female = float(females[age, gf])

                mated_count = 0.0
                for gm in range(n_genotypes):
                    mated_count += sperm[age, gf, gm]
                virgins = max(0.0, n_female - mated_count)

                # Virgin mating: how many virgins mate this tick.
                if use_continuous_sampling:
                    n_mating_virgins = nbc.continuous_binomial(virgins, p_mating)
                else:
                    n_mating_virgins = float(nbc.binomial(int(round(virgins)), p_mating))

                # Sperm displacement: remating displaces existing sperm.
                p_remating = p_displace * p_mating

                if mated_count > EPS and p_remating > EPS:
                    if use_continuous_sampling:
                        removed_frac = min(1.0, p_remating)
                        for gm in range(n_genotypes):
                            sperm[age, gf, gm] -= sperm[age, gf, gm] * removed_frac
                        n_remating = mated_count * removed_frac
                    else:
                        total_removed = 0.0
                        for gm in range(n_genotypes):
                            count = sperm[age, gf, gm]
                            if count > EPS:
                                n_remove = float(nbc.binomial(int(round(count)), p_remating))
                                sperm[age, gf, gm] = max(0.0, sperm[age, gf, gm] - n_remove)
                                total_removed += n_remove
                        n_remating = total_removed
                else:
                    n_remating = 0.0

                # Allocate new matings to male genotypes.
                n_new = n_mating_virgins + n_remating
                if n_new > EPS:
                    if use_continuous_sampling:
                        nbc.continuous_multinomial(n_new, mating_prob_mat[gf, :], tmp)
                        for gm in range(n_genotypes):
                            sperm[age, gf, gm] += tmp[gm]
                    else:
                        n_int = int(round(n_new))
                        if n_int > 0:
                            draws = nbc.multinomial(n_int, mating_prob_mat[gf, :])
                            for gm in range(n_genotypes):
                                sperm[age, gf, gm] += float(draws[gm])

        return sperm

    else:
        for age in range(adult_start_idx, n_ages):
            p_mating = nbc.clamp01(float(female_rates[age]))
            p_displace = nbc.clamp01(float(sperm_displacement_rate))

            for gf in range(n_genotypes):
                n_female = float(females[age, gf])
                mated_count = 0.0
                for gm in range(n_genotypes):
                    mated_count += sperm[age, gf, gm]

                virgins = max(0.0, n_female - mated_count)
                n_mating_virgins = virgins * p_mating
                p_remating = p_displace * p_mating
                n_remating = mated_count * p_remating

                if n_remating > EPS and mated_count > EPS:
                    frac = min(1.0, n_remating / mated_count)
                    for gm in range(n_genotypes):
                        sperm[age, gf, gm] -= sperm[age, gf, gm] * frac

                n_new = n_mating_virgins + n_remating
                if n_new > EPS:
                    for gm in range(n_genotypes):
                        sperm[age, gf, gm] += n_new * mating_prob_mat[gf, gm]

        return sperm

@njit_switch(cache=True)
def compute_offspring_probability_tensor(
    meiosis_f: Annotated[NDArray[np.float64], "shape=(g,hl)"],
    meiosis_m: Annotated[NDArray[np.float64], "shape=(g,hl)"],
    haplo_to_genotype_map: Annotated[NDArray[np.float64], "shape=(hl,hl,g)"],
    n_genotypes: int,
    n_haplogenotypes: int,
    n_glabs: int = 1,
) -> Annotated[NDArray[np.float64], "shape=(g,g,g)"]:
    """Precompute offspring genotype probabilities for all (gf, gm) pairs.

    Constructs the tensor P[gf, gm, g_off] where each entry represents the
    probability of offspring genotype g_off from the cross (gf × gm).

    The computation leverages tensor operations to compute all gamete pairs
    simultaneously: P(g_off | gf, gm) = Σ_hf,hm P(hf | gf) × P(hm | gm) × I[hf ⊗ hm = g_off]

    Args:
        meiosis_f: Female meiosis probability tensor with shape (g, hl),
            where entry [g, h] = P(haplotype h | genotype g) for females.
        meiosis_m: Male meiosis probability tensor with shape (g, hl),
            where entry [g, h] = P(haplotype h | genotype g) for males.
        haplo_to_genotype_map: Haplotype-pair to diploid-genotype mapping with
            shape (hl, hl, g). Entry [h1, h2, g] = 1 if haplotypes h1, h2
            combine to form genotype g, else 0.
        n_genotypes: Number of diploid genotypes.
        n_haplogenotypes: Number of haploid genotypes.
        n_glabs: Number of gamete-label variants per haplotype (default 1).
            If > 1, the total haplotype space is hl = n_haplogenotypes * n_glabs.

    Returns:
        Offspring probability tensor with shape (g, g, g), where
        out[gf, gm, g_off] = P(g_off | gf, gm).
    """
    meio_f = np.asarray(meiosis_f, dtype=np.float64)
    meio_m = np.asarray(meiosis_m, dtype=np.float64)
    zygote_map = np.asarray(haplo_to_genotype_map, dtype=np.float64)

    hl = n_haplogenotypes * n_glabs
    zygote_flat = np.ascontiguousarray(zygote_map).reshape(hl * hl, n_genotypes)

    # Tensor product: P[gf, gm, go] = Σ_{hf,hm} meio_f[gf,hf] · meio_m[gm,hm] · zygote[hf,hm,go]
    f_exp = meio_f[:, None, :, None]   # (g, 1, hl, 1)
    m_exp = meio_m[None, :, None, :]   # (1, g, 1, hl)
    gamete_pairs = (f_exp * m_exp).reshape(n_genotypes * n_genotypes, hl * hl)
    offspring_flat = np.dot(gamete_pairs, zygote_flat)
    return offspring_flat.reshape(n_genotypes, n_genotypes, n_genotypes)


@njit_switch(cache=True)
def _fertilize_with_precomputed_offspring_probability(
    sperm_storage_by_male_genotype: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    fertility_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fertility_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    offspring_probability: Annotated[NDArray[np.float64], "shape=(g,g,g)"],
    average_eggs_per_wt_female: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    female_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    female_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    male_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    has_sex_chromosomes: bool = False,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
) -> tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Generate offspring counts using precomputed offspring probabilities.

    This is the core Numba-optimized fertilization kernel. It processes all
    mating pairs (gf, gm) across all adult ages, computes egg production,
    samples offspring genotypes, and assigns sex based on genotype compatibility.

    The function implements a complex egg-production and zygote-formation pipeline:
    1. For each (age, gf, gm) mating pair, compute expected eggs via Poisson sampling
    2. Apply viability filtering if zygote fitness is incomplete
    3. Sample offspring genotypes from the precomputed probability tensor
    4. Assign offspring sex based on genotype-sex compatibility or global sex_ratio

    Sex assignment strategy:
    - If any offspring genotype has asymmetric sex-compatibility (one sex OK, other
      not), treat the system as sex-chromosome-constrained. For each genotype:
      - If only one sex is compatible, allocate all offspring to that sex
      - If both are compatible, use weighted ratio (f_w / (f_w + m_w))
    - Otherwise (no sex-chromosomes), use global sex_ratio parameter for all genotypes

    Args:
        sperm_storage_by_male_genotype: Sperm storage reservoir indexed by
            (age, female_genotype, male_genotype) with shape (A, g, g).
        fertility_f: Female fertility rates (relative to wild-type) with shape (g,).
        fertility_m: Male fertility rates (relative to wild-type) with shape (g,).
        offspring_probability: Precomputed (gf, gm, g_off) probability tensor,
            where entry [gf, gm, g_off] = P(g_off | gf, gm).
        average_eggs_per_wt_female: Expected egg count per reproducing wild-type female.
        adult_start_idx: First age class that reproduces (typically 1).
        n_ages: Total number of age classes.
        n_genotypes: Number of diploid genotypes.
        female_genotype_compatibility: Sex-compatibility weight for each genotype
            in females (row sums of genotype_to_gametes_map[0]).
        male_genotype_compatibility: Sex-compatibility weight for each genotype
            in males (row sums of genotype_to_gametes_map[1]).
        female_only_by_sex_chrom: Precomputed boolean mask where True means
            genotype is female-only under sex-chromosome constraints.
        male_only_by_sex_chrom: Precomputed boolean mask where True means
            genotype is male-only under sex-chromosome constraints.

        fixed_eggs: If False (default), sample egg counts from Poisson; if True,
            use deterministic expected values.
        sex_ratio: Offspring female fraction (0 to 1). Used only when
            has_sex_chromosomes is False. Otherwise ignored.
        has_sex_chromosomes: If True, offspring sex is determined by genotype-specific
            compatibility weights (f_w, m_w). If False, all offspring sex allocation
            uses the global sex_ratio parameter. This flag is independent of
            gamete modifier effects or temporary lethality (default False).
        is_stochastic: If False, use deterministic expectations without sampling.
        use_continuous_sampling: If True (with is_stochastic=True), use continuous
            distributions (Beta, Dirichlet) instead of discrete (Binomial, Multinomial).

    Returns:
        Tuple (n_offspring_female, n_offspring_male):
        - n_offspring_female: Female offspring counts per genotype with shape (g,)
        - n_offspring_male: Male offspring counts per genotype with shape (g,)
    """
    sperm = np.asarray(sperm_storage_by_male_genotype, dtype=np.float64)
    fert_f_arr = np.asarray(fertility_f, dtype=np.float64)
    fert_m_arr = np.asarray(fertility_m, dtype=np.float64)
    offspring_prob = np.asarray(offspring_probability, dtype=np.float64)

    offspring_acc = np.zeros(n_genotypes, dtype=np.float64)
    prob_norm = np.zeros(n_genotypes, dtype=np.float64)
    tmp = np.zeros(n_genotypes, dtype=np.float64)

    has_any = False
    for age in range(adult_start_idx, n_ages):
        for gf in range(n_genotypes):
            ff = fert_f_arr[gf]
            for gm in range(n_genotypes):
                n_pairs = float(sperm[age, gf, gm])
                if n_pairs <= 0.0:
                    continue
                has_any = True

                eggs_per_pair = average_eggs_per_wt_female * ff * fert_m_arr[gm]

                if is_stochastic:
                    n_pairs_eff = n_pairs if use_continuous_sampling else np.round(n_pairs)
                    if n_pairs_eff <= 0.0:
                        continue
                    total_lambda = float(n_pairs_eff * eggs_per_pair)
                    if fixed_eggs:
                        n_total = float(total_lambda) if use_continuous_sampling else float(np.round(total_lambda))
                    else:
                        n_total = (
                            nbc.continuous_poisson(total_lambda)
                            if use_continuous_sampling
                            else float(np.random.poisson(total_lambda))
                        )
                else:
                    n_total = float(n_pairs * eggs_per_pair)

                if n_total <= EPS:
                    continue

                p_surv = 0.0
                for go in range(n_genotypes):
                    p_surv += offspring_prob[gf, gm, go]

                if is_stochastic:
                    if p_surv <= EPS:
                        continue
                    n_viable = (
                        n_total
                        if p_surv >= 1.0 - EPS
                        else (
                            nbc.continuous_binomial(n_total, p_surv)
                            if use_continuous_sampling
                            else float(nbc.binomial(int(round(n_total)), p_surv))
                        )
                    )
                    if n_viable <= EPS:
                        continue

                    inv = 1.0 / p_surv
                    for go in range(n_genotypes):
                        prob_norm[go] = offspring_prob[gf, gm, go] * inv

                    if use_continuous_sampling:
                        nbc.continuous_multinomial(n_viable, prob_norm, tmp)
                        for go in range(n_genotypes):
                            offspring_acc[go] += tmp[go]
                    else:
                        draws = nbc.multinomial(int(round(n_viable)), prob_norm)
                        for go in range(n_genotypes):
                            offspring_acc[go] += float(draws[go])
                else:
                    for go in range(n_genotypes):
                        offspring_acc[go] += n_total * offspring_prob[gf, gm, go]

    if not has_any:
        return np.zeros(n_genotypes), np.zeros(n_genotypes)

    total = offspring_acc.sum()
    if total <= EPS:
        return np.zeros(n_genotypes), np.zeros(n_genotypes)

    sr = nbc.clamp01(float(sex_ratio))
    n_f = np.zeros(n_genotypes, dtype=np.float64)
    n_m = np.zeros(n_genotypes, dtype=np.float64)

    for go in range(n_genotypes):
        n_g = offspring_acc[go]
        if n_g <= EPS:
            continue

        f_w = female_genotype_compatibility[go]
        m_w = male_genotype_compatibility[go]

        if has_sex_chromosomes and female_only_by_sex_chrom[go]:
            n_f[go] = n_g
        elif has_sex_chromosomes and male_only_by_sex_chrom[go]:
            n_m[go] = n_g
        else:
            if has_sex_chromosomes:
                denom = f_w + m_w
                p_f = nbc.clamp01(f_w / denom) if denom > EPS else 0.5
            else:
                p_f = sr

            if is_stochastic:
                n_fem = (
                    nbc.continuous_binomial(n_g, p_f)
                    if use_continuous_sampling
                    else float(nbc.binomial(int(round(n_g)), p_f))
                )
            else:
                n_fem = n_g * p_f
            n_f[go] = n_fem
            n_m[go] = n_g - n_fem

    return n_f, n_m


# Forward declaration for the internal function
@njit_switch(cache=True)
def _fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction(
    sperm_storage_by_male_genotype: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    fertility_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fertility_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    offspring_probability: Annotated[NDArray[np.float64], "shape=(g,g,g)"],
    average_eggs_per_wt_female: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    female_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    female_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    male_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    n_glabs: int = 1,
    age_based_reproduction_rates: Optional[NDArray[np.float64]] = None,  # (age,)
    female_age_based_relative_fertility: Optional[NDArray[np.float64]] = None,  # (age,)
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    has_sex_chromosomes: bool = False,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
) -> tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Generate offspring using age-specific reproduction rates for consistency with equilibrium inference.

    This variant uses age-specific reproduction rates directly, ensuring consistency between
    equilibrium inference and actual reproduction behavior.

    Args:
        sperm_storage_by_male_genotype: Sperm storage reservoir indexed by (age, female_genotype, male_genotype).
        fertility_f: Female fertility rates (relative to wild-type).
        fertility_m: Male fertility rates (relative to wild-type).
        offspring_probability: Precomputed offspring probability tensor.
        average_eggs_per_wt_female: Expected egg count per reproducing wild-type female.
        adult_start_idx: First age class that reproduces.
        n_ages: Total number of age classes.
        n_genotypes: Number of diploid genotypes.
        female_genotype_compatibility: Female sex-compatibility weights.
        male_genotype_compatibility: Male sex-compatibility weights.
        female_only_by_sex_chrom: Female-only genotype mask.
        male_only_by_sex_chrom: Male-only genotype mask.
        n_glabs: Unused parameter for API compatibility.
        age_based_reproduction_rates: Age-specific reproduction participation rates (n_ages,).
            If None, falls back to all females reproducing (equivalent to proportion=1.0).
        female_age_based_relative_fertility: Age-specific relative fertility rates (n_ages,).
            If None, falls back to all females having full fertility (equivalent to 1.0).
        fixed_eggs: Whether to use fixed egg counts.
        sex_ratio: Offspring female ratio.
        has_sex_chromosomes: Whether offspring sex is genotype-constrained.
        is_stochastic: Whether to sample stochastically.
        use_continuous_sampling: Whether to use continuous sampling.

    Returns:
        Tuple containing female and male offspring counts with shape (g,).
    """
    sperm = np.asarray(sperm_storage_by_male_genotype, dtype=np.float64)
    fert_f_arr = np.asarray(fertility_f, dtype=np.float64)
    fert_m_arr = np.asarray(fertility_m, dtype=np.float64)
    offspring_prob = np.asarray(offspring_probability, dtype=np.float64)

    repro_rates = (
        np.asarray(age_based_reproduction_rates)
        if age_based_reproduction_rates is not None
        else np.ones(n_ages, dtype=np.float64)
    )
    rel_fert = (
        np.asarray(female_age_based_relative_fertility)
        if female_age_based_relative_fertility is not None
        else np.ones(n_ages, dtype=np.float64)
    )

    offspring_acc = np.zeros(n_genotypes, dtype=np.float64)
    prob_norm = np.zeros(n_genotypes, dtype=np.float64)
    tmp = np.zeros(n_genotypes, dtype=np.float64)

    has_any = False
    for age in range(adult_start_idx, n_ages):
        p_reproduce = nbc.clamp01(float(repro_rates[age]))
        fertility_factor = nbc.clamp01(float(rel_fert[age]))

        for gf in range(n_genotypes):
            ff = fert_f_arr[gf]
            for gm in range(n_genotypes):
                n_pairs = float(sperm[age, gf, gm])
                if n_pairs <= 0.0:
                    continue
                has_any = True

                eggs_per_pair = (
                    average_eggs_per_wt_female * ff * fert_m_arr[gm] * fertility_factor
                )

                if is_stochastic:
                    n_pairs_eff = n_pairs if use_continuous_sampling else np.round(n_pairs)
                    if n_pairs_eff <= 0.0:
                        continue
                    n_reproducing = float(n_pairs_eff)
                    if p_reproduce < 1.0 - EPS:
                        n_reproducing = (
                            nbc.continuous_binomial(n_pairs_eff, p_reproduce)
                            if use_continuous_sampling
                            else float(nbc.binomial(int(n_pairs_eff), p_reproduce))
                        )
                    total_lambda = float(n_reproducing * eggs_per_pair)
                    if fixed_eggs:
                        n_total = float(total_lambda) if use_continuous_sampling else float(np.round(total_lambda))
                    else:
                        n_total = (
                            nbc.continuous_poisson(total_lambda)
                            if use_continuous_sampling
                            else float(np.random.poisson(total_lambda))
                        )
                else:
                    n_reproducing = float(n_pairs * p_reproduce)
                    n_total = n_reproducing * eggs_per_pair

                if n_total <= EPS:
                    continue

                p_surv = 0.0
                for go in range(n_genotypes):
                    p_surv += offspring_prob[gf, gm, go]

                if is_stochastic:
                    if p_surv <= EPS:
                        continue
                    n_viable = (
                        n_total
                        if p_surv >= 1.0 - EPS
                        else (
                            nbc.continuous_binomial(n_total, p_surv)
                            if use_continuous_sampling
                            else float(nbc.binomial(int(round(n_total)), p_surv))
                        )
                    )
                    if n_viable <= EPS:
                        continue
                    inv = 1.0 / p_surv
                    for go in range(n_genotypes):
                        prob_norm[go] = offspring_prob[gf, gm, go] * inv
                    if use_continuous_sampling:
                        nbc.continuous_multinomial(n_viable, prob_norm, tmp)
                        for go in range(n_genotypes):
                            offspring_acc[go] += tmp[go]
                    else:
                        draws = nbc.multinomial(int(round(n_viable)), prob_norm)
                        for go in range(n_genotypes):
                            offspring_acc[go] += float(draws[go])
                else:
                    for go in range(n_genotypes):
                        offspring_acc[go] += n_total * offspring_prob[gf, gm, go]

    if not has_any:
        return np.zeros(n_genotypes), np.zeros(n_genotypes)

    total = offspring_acc.sum()
    if total <= EPS:
        return np.zeros(n_genotypes), np.zeros(n_genotypes)

    sr = nbc.clamp01(float(sex_ratio))
    n_f = np.zeros(n_genotypes, dtype=np.float64)
    n_m = np.zeros(n_genotypes, dtype=np.float64)

    for go in range(n_genotypes):
        n_g = offspring_acc[go]
        if n_g <= EPS:
            continue

        f_w = female_genotype_compatibility[go]
        m_w = male_genotype_compatibility[go]

        if has_sex_chromosomes and female_only_by_sex_chrom[go]:
            n_f[go] = n_g
        elif has_sex_chromosomes and male_only_by_sex_chrom[go]:
            n_m[go] = n_g
        else:
            p_f = sr
            if has_sex_chromosomes:
                denom = f_w + m_w
                p_f = nbc.clamp01(f_w / denom) if denom > EPS else 0.5
            if is_stochastic:
                n_fem = (
                    nbc.continuous_binomial(n_g, p_f)
                    if use_continuous_sampling
                    else float(nbc.binomial(int(round(n_g)), p_f))
                )
            else:
                n_fem = n_g * p_f
            n_f[go] = n_fem
            n_m[go] = n_g - n_fem

    return n_f, n_m


@njit_switch(cache=True)
def fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction(
    female_counts: Annotated[NDArray[np.float64], "shape=(A,g)"],
    sperm_storage_by_male_genotype: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    fertility_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fertility_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    offspring_probability: Annotated[NDArray[np.float64], "shape=(g,g,g)"],
    average_eggs_per_wt_female: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    n_haplogenotypes: int,
    female_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    female_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    male_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    n_glabs: int = 1,
    age_based_reproduction_rates: Optional[NDArray[np.float64]] = None,  # (age,)
    female_age_based_relative_fertility: Optional[NDArray[np.float64]] = None,  # (age,)
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    has_sex_chromosomes: bool = False,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
) -> tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Public interface for fertilization with age-specific reproduction rates.

    This function ensures consistency between equilibrium inference and actual reproduction
    by using age-specific reproduction rates and relative fertility rates directly.

    Args:
        female_counts: Female counts array (unused, for API compatibility).
        sperm_storage_by_male_genotype: Sperm storage array.
        fertility_f: Female fertility rates.
        fertility_m: Male fertility rates.
        offspring_probability: Precomputed offspring probability tensor.
        average_eggs_per_wt_female: Expected eggs per wild-type female.
        adult_start_idx: First reproductive age class.
        n_ages: Total number of age classes.
        n_genotypes: Number of diploid genotypes.
        n_haplogenotypes: Unused parameter for API compatibility.
        female_genotype_compatibility: Female sex-compatibility weights.
        male_genotype_compatibility: Male sex-compatibility weights.
        female_only_by_sex_chrom: Female-only genotype mask.
        male_only_by_sex_chrom: Male-only genotype mask.
        n_glabs: Unused parameter for API compatibility.
        age_based_reproduction_rates: Age-specific reproduction participation rates.
        female_age_based_relative_fertility: Age-specific relative fertility rates.
        fixed_eggs: Whether to use fixed egg counts.
        sex_ratio: Offspring female ratio.
        has_sex_chromosomes: Whether offspring sex is genotype-constrained.
        is_stochastic: Whether to sample stochastically.
        use_continuous_sampling: Whether to use continuous sampling.

    Returns:
        Tuple containing female and male offspring counts.
    """
    _ = female_counts
    _ = n_haplogenotypes
    _ = n_glabs

    return _fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction(
        sperm_storage_by_male_genotype=sperm_storage_by_male_genotype,
        fertility_f=fertility_f,
        fertility_m=fertility_m,
        offspring_probability=offspring_probability,
        average_eggs_per_wt_female=average_eggs_per_wt_female,
        adult_start_idx=adult_start_idx,
        n_ages=n_ages,
        n_genotypes=n_genotypes,
        female_genotype_compatibility=female_genotype_compatibility,
        male_genotype_compatibility=male_genotype_compatibility,
        female_only_by_sex_chrom=female_only_by_sex_chrom,
        male_only_by_sex_chrom=male_only_by_sex_chrom,
        age_based_reproduction_rates=age_based_reproduction_rates,
        female_age_based_relative_fertility=female_age_based_relative_fertility,
        fixed_eggs=fixed_eggs,
        sex_ratio=sex_ratio,
        has_sex_chromosomes=has_sex_chromosomes,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling,
    )


@njit_switch(cache=True)
def fertilize_with_mating_genotype(
    female_counts: Annotated[NDArray[np.float64], "shape=(A,g)"],
    sperm_storage_by_male_genotype: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    fertility_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fertility_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    meiosis_f: Annotated[NDArray[np.float64], "shape=(g,hl)"],
    meiosis_m: Annotated[NDArray[np.float64], "shape=(g,hl)"],
    haplo_to_genotype_map: Annotated[NDArray[np.float64], "shape=(hl,hl,g)"],
    average_eggs_per_wt_female: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    n_haplogenotypes: int,
    female_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    female_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    male_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    n_glabs: int = 1,
    age_based_reproduction_rates: Optional[NDArray[np.float64]] = None,  # (age,)
    female_age_based_relative_fertility: Optional[NDArray[np.float64]] = None,  # (age,)
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    has_sex_chromosomes: bool = False,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
) -> tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Fertilization using meiosis matrices (on-the-fly probability computation).

    Vectorized offspring generation with batch multinomial sampling. This variant
    computes offspring probabilities on-the-fly from meiosis matrices, then
    delegates to the core _fertilize_with_precomputed_offspring_probability kernel.

    Achieves ~60.9x speedup through vectorization:
    - Pre-compute expected egg counts per mating pair
    - Batch Poisson sampling to avoid individual per-pair sampling
    - Single multinomial draw per viable egg count
    - Vectorized accumulation of genotype counts

    Sex-chromosome compatibility is inferred from meiosis row sums: genotypes
    that cannot produce gametes of one sex are marked as sex-incompatible.

    Args:
        female_counts: Female genotype counts, shape (A, g) (unused, for API compatibility).
        sperm_storage_by_male_genotype: Sperm storage reservoir, shape (A, g, g).
        fertility_f: Female fertility rates relative to wild-type, shape (g,).
        fertility_m: Male fertility rates relative to wild-type, shape (g,).
        meiosis_f: Female meiosis probabilities (genotype → haplotype), shape (g, hl).
            Row sums indicate whether a genotype can produce female gametes.
        meiosis_m: Male meiosis probabilities (genotype → haplotype), shape (g, hl).
            Row sums indicate whether a genotype can produce male gametes.
        haplo_to_genotype_map: Haplotype pair → genotype membership, shape (hl, hl, g).
        average_eggs_per_wt_female: Expected eggs per reproducing wild-type female.
        adult_start_idx: First reproductive age class.
        n_ages: Total age classes.
        n_genotypes: Number of diploid genotypes.
        n_haplogenotypes: Number of haploid genotypes.
        female_genotype_compatibility: Female compatibility weight per genotype.
        male_genotype_compatibility: Male compatibility weight per genotype.
        female_only_by_sex_chrom: Precomputed female-only genotype mask.
        male_only_by_sex_chrom: Precomputed male-only genotype mask.
        n_glabs: Gamete-label variants per haplotype (default 1).
        age_based_reproduction_rates: Age-specific reproduction rates, shape (age,).
        female_age_based_relative_fertility: Age-specific relative fertility rates, shape (age,).

        fixed_eggs: Use deterministic eggs if True, Poisson if False.
        sex_ratio: Offspring female fraction (used if no sex-chromosomes).
        has_sex_chromosomes: Whether offspring sex is genotype-constrained.
        is_stochastic: Use sampling if True, deterministic if False.
        use_continuous_sampling: Use Beta/Dirichlet if True, Binomial/Multinomial if False.

    Returns:
        Tuple (n_offspring_female, n_offspring_male) with shape (g,) each.
    """

    # F = np.asarray(female_counts, dtype=np.float64)
    P_offspring = compute_offspring_probability_tensor(
        meiosis_f=meiosis_f,
        meiosis_m=meiosis_m,
        haplo_to_genotype_map=haplo_to_genotype_map,
        n_genotypes=n_genotypes,
        n_haplogenotypes=n_haplogenotypes,
        n_glabs=n_glabs,
    )

    return _fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction(
        sperm_storage_by_male_genotype=sperm_storage_by_male_genotype,
        fertility_f=fertility_f,
        fertility_m=fertility_m,
        offspring_probability=P_offspring,
        average_eggs_per_wt_female=average_eggs_per_wt_female,
        adult_start_idx=adult_start_idx,
        n_ages=n_ages,
        n_genotypes=n_genotypes,
        female_genotype_compatibility=female_genotype_compatibility,
        male_genotype_compatibility=male_genotype_compatibility,
        female_only_by_sex_chrom=female_only_by_sex_chrom,
        male_only_by_sex_chrom=male_only_by_sex_chrom,
        n_glabs=n_glabs,
        age_based_reproduction_rates=age_based_reproduction_rates,
        female_age_based_relative_fertility=female_age_based_relative_fertility,
        fixed_eggs=fixed_eggs,
        sex_ratio=sex_ratio,
        has_sex_chromosomes=has_sex_chromosomes,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling,
    )


@njit_switch(cache=True)
def fertilize_with_precomputed_offspring_probability(
    female_counts: Annotated[NDArray[np.float64], "shape=(A,g)"],
    sperm_storage_by_male_genotype: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    fertility_f: Annotated[NDArray[np.float64], "shape=(g,)"],
    fertility_m: Annotated[NDArray[np.float64], "shape=(g,)"],
    offspring_probability: Annotated[NDArray[np.float64], "shape=(g,g,g)"],
    average_eggs_per_wt_female: float,
    adult_start_idx: int,
    n_ages: int,
    n_genotypes: int,
    n_haplogenotypes: int,
    female_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_genotype_compatibility: Annotated[NDArray[np.float64], "shape=(g,)"],
    female_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    male_only_by_sex_chrom: Annotated[NDArray[np.bool_], "shape=(g,)"],
    n_glabs: int = 1,
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    has_sex_chromosomes: bool = False,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
) -> tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Fertilization wrapper using externally precomputed offspring probabilities.

    Args:
        female_counts: Female counts array with shape (A, g). Reserved for API
            compatibility with the non-precomputed variant.
        sperm_storage_by_male_genotype: Sperm storage array with shape (A, g, g).
        fertility_f: Female fertility rates with shape (g,).
        fertility_m: Male fertility rates with shape (g,).
        offspring_probability: Precomputed offspring tensor (g, g, g).
        average_eggs_per_wt_female: Average eggs produced per wild-type female.
        adult_start_idx: Starting age index for adults.
        n_ages: Total number of age classes.
        n_genotypes: Number of genotypes.
        n_haplogenotypes: Unused here; kept for signature parity.
        female_genotype_compatibility: Female-compatible weight per genotype.
            If sex-chromosome constraints are present, this overrides global
            ``sex_ratio`` for offspring sex assignment.
        male_genotype_compatibility: Male-compatible weight per genotype.
            If sex-chromosome constraints are present, this overrides global
            ``sex_ratio`` for offspring sex assignment.
        female_only_by_sex_chrom: Precomputed female-only genotype mask.
        male_only_by_sex_chrom: Precomputed male-only genotype mask.
        n_glabs: Unused here; kept for signature parity.

        fixed_eggs: Whether to use fixed egg counts.
        sex_ratio: Offspring female ratio. Used only when has_sex_chromosomes is False.
        has_sex_chromosomes: Whether offspring sex is genotype-constrained.
        is_stochastic: Whether to sample stochastically.
        use_continuous_sampling: Whether to use continuous sampling.

    Returns:
        Tuple containing female and male offspring counts with shape (g,).
    """
    _ = female_counts
    _ = n_haplogenotypes
    _ = n_glabs

    return _fertilize_with_precomputed_offspring_probability(
        sperm_storage_by_male_genotype=sperm_storage_by_male_genotype,
        fertility_f=fertility_f,
        fertility_m=fertility_m,
        offspring_probability=offspring_probability,
        average_eggs_per_wt_female=average_eggs_per_wt_female,
        adult_start_idx=adult_start_idx,
        n_ages=n_ages,
        n_genotypes=n_genotypes,
        female_genotype_compatibility=female_genotype_compatibility,
        male_genotype_compatibility=male_genotype_compatibility,
        female_only_by_sex_chrom=female_only_by_sex_chrom,
        male_only_by_sex_chrom=male_only_by_sex_chrom,

        fixed_eggs=fixed_eggs,
        sex_ratio=sex_ratio,
        has_sex_chromosomes=has_sex_chromosomes,
        is_stochastic=is_stochastic,
        use_continuous_sampling=use_continuous_sampling,
    )


@njit_switch(cache=True)
def compute_age_based_survival_rates(
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)"],
    n_ages: int,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,)"], Annotated[NDArray[np.float64], "shape=(A,)"]]:
    """Return age-specific survival rate arrays (no sampling).

    Args:
        female_survival_rates: Female survival rates shape (n_ages,)
        male_survival_rates: Male survival rates shape (n_ages,)
        n_ages: Number of ages

    Returns:
        Tuple[survival_rates_f, survival_rates_m]: Two arrays with shape (n_ages,)
    """
    return np.asarray(female_survival_rates), np.asarray(male_survival_rates)


@njit_switch(cache=True)
def compute_viability_survival_rates(
    female_viability_rates: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_viability_rates: Annotated[NDArray[np.float64], "shape=(g,)"],
    n_genotypes: int,
    target_age: int,
    n_ages: int,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]]:
    """Return viability survival rate matrices (non-zero only at target age).

    Args:
        female_viability_rates: Female viability genotype-specific rates shape (g,)
        male_viability_rates: Male viability genotype-specific rates shape (g,)
        n_genotypes: Number of genotypes
        target_age: Age index where viability is applied
        n_ages: Total number of ages

    Returns:
        Tuple[survival_rates_f, survival_rates_m]: Two matrices with shape (n_ages, n_genotypes),
            all rows are 1.0 except target_age row
    """
    v_f = np.asarray(female_viability_rates)
    v_m = np.asarray(male_viability_rates)

    # Initialize as all 1.0 matrices
    surv_f = np.ones((n_ages, n_genotypes), dtype=np.float64)
    surv_m = np.ones((n_ages, n_genotypes), dtype=np.float64)

    # Set viability survival rates only at target age
    surv_f[target_age, :] = v_f
    surv_m[target_age, :] = v_m

    return surv_f, surv_m


@njit_switch(cache=True)
def apply_survival_rates_deterministic(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    n_genotypes: int,
    n_ages: int,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]]:
    """Deterministically apply survival rates (direct multiplication, no sampling).

    Supports two input formats:
    - 1D array shape (A,): Apply by age, broadcast to all genotypes
    - 2D array shape (A,g): Directly apply to each (age, genotype)

    Args:
        population: (female, male) tuple
        female_survival_rates: Female survival rates
        male_survival_rates: Male survival rates
        n_genotypes: Number of genotypes
        n_ages: Number of ages

    Returns:
        Tuple[female_new, male_new]: Population multiplied by survival rates
    """
    female, male = population
    f_result = np.asarray(female).copy()
    m_result = np.asarray(male).copy()
    s_f = np.asarray(female_survival_rates)
    s_m = np.asarray(male_survival_rates)

    assert f_result.shape == (n_ages, n_genotypes)
    assert m_result.shape == (n_ages, n_genotypes)

    if s_f.ndim == 1:
        # 1D array: Apply by age
        assert s_f.shape == (n_ages,)
        f_result = f_result * s_f[:, None]
    else:
        # 2D array: Direct application
        assert s_f.shape == (n_ages, n_genotypes)
        f_result = f_result * s_f

    if s_m.ndim == 1:
        # 1D array: Apply by age
        assert s_m.shape == (n_ages,)
        m_result = m_result * s_m[:, None]
    else:
        # 2D array: Direct application
        assert s_m.shape == (n_ages, n_genotypes)
        m_result = m_result * s_m

    return f_result, m_result


@njit_switch(cache=True)
def apply_survival_rates_deterministic_with_sperm_storage(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    n_genotypes: int,
    n_ages: int,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g,g)"]]:
    """Deterministically apply survival rates with consistent scaling of sperm storage (no sampling).

    Key: sperm storage is scaled by the same survival rates.

    Args:
        population: (female, male) tuple
        sperm_store: Sperm storage array shape (n_ages, n_genotypes, n_genotypes)
        female_survival_rates: Female survival rates (supports 1D or 2D)
        male_survival_rates: Male survival rates (supports 1D or 2D)
        n_genotypes: Number of genotypes
        n_ages: Number of ages

    Returns:
        Tuple[female_new, male_new, sperm_store_new]
    """
    females, males = population
    f_pop = np.asarray(females).copy()
    m_pop = np.asarray(males).copy()
    s_pop = np.asarray(sperm_store).copy()
    surv_f = np.asarray(female_survival_rates)
    surv_m = np.asarray(male_survival_rates)

    assert f_pop.shape == (n_ages, n_genotypes)
    assert m_pop.shape == (n_ages, n_genotypes)
    assert s_pop.shape == (n_ages, n_genotypes, n_genotypes)

    # Normalize to 2D for uniform access.
    surv_f_2d = surv_f.reshape(n_ages, 1) if surv_f.ndim == 1 else surv_f
    surv_m_2d = surv_m.reshape(n_ages, 1) if surv_m.ndim == 1 else surv_m

    for age in range(n_ages):
        for g in range(n_genotypes):
            gf_idx = g % surv_f_2d.shape[1]
            f_rate = float(surv_f_2d[age, gf_idx])
            f_pop[age, g] *= f_rate
            s_pop[age, g, :] *= f_rate

    for age in range(n_ages):
        for g in range(n_genotypes):
            gm_idx = g % surv_m_2d.shape[1]
            m_rate = float(surv_m_2d[age, gm_idx])
            m_pop[age, g] *= m_rate

    return f_pop, m_pop, s_pop


@njit_switch(cache=True)
def sample_survival_with_sperm_storage(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    n_genotypes: int,
    n_ages: int,
    use_continuous_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g,g)"]]:
    """Randomly apply survival rates with consistent sampling of sperm storage.

    Key: For each (age, gf) pair, use the **same sampling result** to update individual counts and sperm storage.

    Args:
        population: (female, male) tuple
        sperm_store: Sperm storage array shape (n_ages, n_genotypes, n_genotypes)
        female_survival_rates: Female survival rates (supports 1D or 2D)
        male_survival_rates: Male survival rates (supports 1D or 2D)
        n_genotypes: Number of genotypes
        use_continuous_sampling: If True, use Dirichlet distribution instead of discrete sampling.
            Currently not implemented (will use discrete).
        n_ages: Number of ages

    Returns:
        Tuple[female_new, male_new, sperm_store_new]

    Important implementation notes:
        - Invariant check for `n_virgins` uses raw floating-point mass:
          `n_virgins_raw = F_raw - sum(S_raw)`.
          This avoids false negatives from per-cell rounding in stochastic mode.
        - Discrete sampling still rounds each category to integer trial counts
          before binomial draws, but only after the raw-mass invariant check.
    """
    females, males = population
    f_pop = np.asarray(females).copy()
    m_pop = np.asarray(males).copy()
    s_pop = np.asarray(sperm_store).copy()
    surv_f = np.asarray(female_survival_rates)
    surv_m = np.asarray(male_survival_rates)

    assert f_pop.shape == (n_ages, n_genotypes)
    assert m_pop.shape == (n_ages, n_genotypes)
    assert s_pop.shape == (n_ages, n_genotypes, n_genotypes)

    surv_f_2d = surv_f.reshape(n_ages, 1) if surv_f.ndim == 1 else surv_f
    surv_m_2d = surv_m.reshape(n_ages, 1) if surv_m.ndim == 1 else surv_m

    for age in range(n_ages):
        for g in range(n_genotypes):
            # ── females + sperm ──
            n_f_raw = float(f_pop[age, g])
            gf_idx = g % surv_f_2d.shape[1]
            p_f = nbc.clamp01(float(surv_f_2d[age, gf_idx]))

            total_sperm = 0.0
            for gm in range(n_genotypes):
                total_sperm += float(s_pop[age, g, gm])

            n_virgins_raw = n_f_raw - total_sperm
            if n_virgins_raw < -EPS:
                raise ValueError(
                    f"Invalid state: n_virgins < 0 in sample_survival_with_sperm_storage "
                    f"(age={age}, g={g}, n_f_raw={n_f_raw}, total_sperm={total_sperm})"
                )
            n_virgins_raw = max(0.0, n_virgins_raw)
            n_virgins = n_virgins_raw if use_continuous_sampling else float(int(round(n_virgins_raw)))

            new_sperm_sum = 0.0
            for gm in range(n_genotypes):
                n_sperm = s_pop[age, g, gm] if use_continuous_sampling else float(int(round(s_pop[age, g, gm])))
                if n_sperm > EPS:
                    s_pop[age, g, gm] = (
                        nbc.continuous_binomial(n_sperm, p_f)
                        if use_continuous_sampling
                        else float(nbc.binomial(int(n_sperm), p_f))
                    )
                else:
                    s_pop[age, g, gm] = 0.0
                new_sperm_sum += s_pop[age, g, gm]

            surv_virgins = 0.0
            if n_virgins > EPS:
                surv_virgins = (
                    nbc.continuous_binomial(n_virgins, p_f)
                    if use_continuous_sampling
                    else float(nbc.binomial(int(n_virgins), p_f))
                )

            f_pop[age, g] = new_sperm_sum + surv_virgins

            # ── males ──
            n_m = m_pop[age, g] if use_continuous_sampling else float(int(round(m_pop[age, g])))
            gm_idx = g % surv_m_2d.shape[1]
            p_m = nbc.clamp01(float(surv_m_2d[age, gm_idx]))

            if n_m > EPS:
                m_pop[age, g] = (
                    nbc.continuous_binomial(n_m, p_m)
                    if use_continuous_sampling
                    else float(nbc.binomial(int(n_m), p_m))
                )
            else:
                m_pop[age, g] = 0.0

    return f_pop, m_pop, s_pop

@njit_switch(cache=True)
def recruit_juveniles_sampling(
    age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    carrying_capacity: int,
    n_genotypes: int,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Cohort-mode recruitment to carrying capacity.

    If total juveniles <= carrying_capacity, returns float copies. If
    greater, deterministically scale down to K while preserving genotype proportions
    (with remainder distribution), unless `is_stochastic` is True in which case
    exactly `K` juveniles are sampled by multinomial.
    Returns float64 arrays (containing integral values if stochastic).

    Args:
        age_0_juvenile_counts: Tuple of (female_0, male_0) age-0 juvenile counts
        carrying_capacity: Carrying capacity K
        n_genotypes: Number of genotypes
        is_stochastic: If True, use stochastic sampling; if False, use deterministic scaling
        use_continuous_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling

    Returns:
        Tuple[female_new, male_new]: Recruited juvenile counts with shape (g,) each
    """
    female_0, male_0 = age_0_juvenile_counts
    # Keep deterministic paths on raw expected counts; only stochastic-discrete
    # paths require integerized trials.
    if is_stochastic and (not use_continuous_sampling):
        female_arr = np.rint(np.asarray(female_0))
        male_arr = np.rint(np.asarray(male_0))
    else:
        female_arr = np.asarray(female_0)
        male_arr = np.asarray(male_0)

    assert female_arr.shape == (n_genotypes,)
    assert male_arr.shape == (n_genotypes,)

    total = float(female_arr.sum() + male_arr.sum())
    K = float(carrying_capacity)

    if total <= 0:
        return np.zeros_like(female_arr), np.zeros_like(male_arr)

    if total <= K:
        return female_arr.copy(), male_arr.copy()

    # Flatten to vector of length 2g for probability weights
    counts = np.concatenate((female_arr, male_arr))
    probs = counts / total

    if is_stochastic:
        if use_continuous_sampling:
            # Continuous sampling: use Dirichlet instead of Multinomial
            out_counts = np.zeros(2 * n_genotypes, dtype=np.float64)
            nbc.continuous_multinomial(K, probs, out_counts)
            draws = out_counts
        else:
            # Discrete sampling: standard Multinomial
            draws = nbc.multinomial(int(round(K)), probs).astype(np.float64)
        f_new = draws[:n_genotypes]
        m_new = draws[n_genotypes:]
        return f_new, m_new

    # Deterministic scaling
    scaled = counts * (K / total)
    f_new = scaled[:n_genotypes]
    m_new = scaled[n_genotypes:]
    return f_new, m_new


@njit_switch(cache=True)
def recruit_juveniles_given_scaling_factor_sampling(
    age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    scaling_factor: float,
    n_genotypes: int,
    is_stochastic: bool = True,
    use_continuous_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Scale age-0 juveniles by `scaling_factor`.

    If `is_stochastic` is True, sample exactly `round(total * scaling_factor)`
    juveniles by multinomial according to genotype-by-sex proportions.

    Args:
        age_0_juvenile_counts: Tuple of (female_0, male_0) age-0 juvenile counts
        scaling_factor: Scaling factor to apply to total juvenile count
        n_genotypes: Number of genotypes
        is_stochastic: If True, use stochastic sampling; if False, use deterministic scaling
        use_continuous_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling. Currently not implemented (will use discrete).

    Returns:
        Tuple[female_new, male_new]: Scaled juvenile counts with shape (g,) each.
            Returns float64 arrays (containing integral values if stochastic).
    """
    female_0, male_0 = age_0_juvenile_counts
    if is_stochastic and (not use_continuous_sampling):
        female_arr = np.rint(np.asarray(female_0))
        male_arr = np.rint(np.asarray(male_0))
    else:
        female_arr = np.asarray(female_0)
        male_arr = np.asarray(male_0)

    assert female_arr.shape == (n_genotypes,)
    assert male_arr.shape == (n_genotypes,)

    total = float(female_arr.sum() + male_arr.sum())
    if total <= 0:
        return np.zeros_like(female_arr), np.zeros_like(male_arr)

    if is_stochastic and (not use_continuous_sampling):
        desired = float(int(round(total * float(scaling_factor))))
    else:
        desired = total * float(scaling_factor)

    if desired <= 0:
        return np.zeros_like(female_arr), np.zeros_like(male_arr)

    counts = np.concatenate((female_arr, male_arr))
    # Key fix: Ensure division uses Python float scalar instead of 0-d array
    # counts.sum() may return 0-d array, causing Numba type inference issues
    total_counts = float(counts.sum())
    probs = counts / total_counts

    if is_stochastic:
        # Use nbc.multinomial instead of np.random.multinomial
        # This avoids Numba nested JIT dynamic probability array type inference bug
        if use_continuous_sampling:
            # Continuous sampling: use Dirichlet instead of Multinomial
            temp_counts = np.zeros(2 * n_genotypes, dtype=np.float64)
            nbc.continuous_multinomial(float(desired), probs, temp_counts)
            f_new = temp_counts[:n_genotypes].astype(np.float64)
            m_new = temp_counts[n_genotypes:].astype(np.float64)
        else:
            # Discrete sampling: standard Multinomial
            draws = nbc.multinomial(int(round(desired)), probs)
            f_new = draws[:n_genotypes].astype(np.float64)
            m_new = draws[n_genotypes:].astype(np.float64)
        return f_new, m_new

    # Deterministic: use scaled value directly without rounding
    scaled = counts * (desired / float(total))
    f_new = scaled[:n_genotypes]
    m_new = scaled[n_genotypes:]
    return f_new, m_new

@njit_switch(cache=True)
def compute_equilibrium_metrics(
    carrying_capacity: float,
    expected_eggs_per_female: float,
    age_based_survival_rates: NDArray[np.float64], # (sex, age)
    age_based_mating_rates: NDArray[np.float64],   # (sex, age)
    female_age_based_relative_fertility: NDArray[np.float64], # (age,)
    relative_competition_strength: NDArray[np.float64], # (age,)
    sex_ratio: float,
    new_adult_age: int,
    n_ages: int,
    age_based_reproduction_rates: Optional[NDArray[np.float64]] = None, # (age,)
    equilibrium_individual_count: Optional[NDArray[np.float64]] = None, # (sex, age, genotype_sum)
    external_expected_eggs: Optional[float] = None,  # overrides produced_age_0 for survival rate
) -> Tuple[float, float]:
    """Calculate competition strength and survival rate metrics under equilibrium.

    These metrics are used for LOGISTIC and BEVERTON_HOLT density-dependent modes.

    The equilibrium distribution (from ``equilibrium_individual_count`` or built from
    ``carrying_capacity``) is always used for ``expected_competition_strength``.
    The ``expected_survival_rate`` is computed as ``total_age_1 / (produced_age_0 * s_0_avg)``.
    When ``external_expected_eggs`` is given, it replaces ``produced_age_0`` in the survival
    rate formula but NOT in the competition strength formula (competition uses the actual
    equilibrium distribution's egg production).

    Args:
        carrying_capacity: Total carrying capacity K based on age=1
        expected_eggs_per_female: Basic offspring count
        age_based_survival_rates: Survival rate matrix (2, n_ages)
        age_based_mating_rates: Mating rate matrix (2, n_ages)
        age_based_reproduction_rates: Female age-specific reproduction participation
            rates with shape (n_ages,). If None, falls back to
            ``age_based_mating_rates[0]``.
        female_age_based_relative_fertility: Female age-dependent relative fertility (n_ages,)
        relative_competition_strength: Competition weights for each age (n_ages,)
        sex_ratio: Sex ratio (female proportion)
        new_adult_age: Adult starting age
        n_ages: Total number of ages
        equilibrium_individual_count: Optional user-provided equilibrium distribution (2, n_ages)
        external_expected_eggs: If provided, overrides ``produced_age_0`` in the survival
            rate computation. This enables ``expected_num_adult_females`` to independently
            determine expected egg production separate from the equilibrium distribution.

    Returns:
        Tuple[expected_competition_strength, expected_survival_rate]
    """
    # Use age-specific reproduction participation rates for equilibrium
    # calibration. If not provided, fall back to female mating rates.
    p_reproducing = np.zeros(n_ages, dtype=np.float64)
    if age_based_reproduction_rates is not None:
        reproduce_rates = np.asarray(age_based_reproduction_rates)
    else:
        reproduce_rates = age_based_mating_rates[0]
    for age in range(new_adult_age, n_ages):
        p_reproducing[age] = nbc.clamp01(float(reproduce_rates[age]))

    if equilibrium_individual_count is not None:
        # 1. Use user-provided equilibrium distribution
        expected_distribution = equilibrium_individual_count
        # Calculate produced age-0 numbers: only adult females
        produced_age_0 = 0.0
        for age in range(new_adult_age, n_ages):
            n_f = expected_distribution[0, age]
            # Use per-tick reproducing fraction and relative fertility
            # Contribution of this age to age0 production:
            # n_f * P(reproducing_this_tick) * relative_fertility * eggs_per_female
            produced_age_0 += n_f * p_reproducing[age] * female_age_based_relative_fertility[age] * expected_eggs_per_female

        total_age_1 = expected_distribution[0, 1] + expected_distribution[1, 1]
    else:
        # 2. Automatically derive equilibrium distribution
        # Derive based on age=1 total count = K
        total_age_1 = carrying_capacity
        expected_distribution = np.zeros((2, n_ages), dtype=np.float64)

        # Age 1: Allocate females and males
        # Age1 baseline allocation:
        # female_age1 = total_age1 * sex_ratio
        # male_age1   = total_age1 * (1-sex_ratio)
        expected_distribution[0, 1] = total_age_1 * sex_ratio
        expected_distribution[1, 1] = total_age_1 * (1.0 - sex_ratio)

        # Derive subsequent ages (based on survival rates)
        for age in range(2, n_ages):
            # Age progression expectation:
            # N(age) = N(age-1) * survival(age-1)
            expected_distribution[0, age] = expected_distribution[0, age - 1] * age_based_survival_rates[0, age - 1]
            expected_distribution[1, age] = expected_distribution[1, age - 1] * age_based_survival_rates[1, age - 1]

        # Calculate produced Egg count (produced_age_0)
        produced_age_0 = 0.0
        for age in range(new_adult_age, n_ages):
            n_f = expected_distribution[0, age]
            produced_age_0 += n_f * p_reproducing[age] * female_age_based_relative_fertility[age] * expected_eggs_per_female

    # Calculate total expected competition strength (limited to larvae participating in competition, i.e., age < new_adult_age)
    # Age 0 is produced Egg count; Age 1+ are survivors in distribution
    # Competition strength is weighted sum of "larvae count * corresponding competition weight".
    # NOTE: competition strength always uses the equilibrium distribution's own produced_age_0,
    # NOT external_expected_eggs (which only affects the survival rate).
    expected_competition_strength = produced_age_0 * relative_competition_strength[0]
    for age in range(1, new_adult_age):
        n_total = expected_distribution[0, age] + expected_distribution[1, age]
        expected_competition_strength += n_total * relative_competition_strength[age]

    # Calculate expected survival rate (scaling factor from Egg production to entering age=1)
    # Under equilibrium: total_age_1 = produced_age_0 * expected_survival_rate * s_0_avg
    # Where s_0_avg is base survival rate from Age 0 to Age 1
    s_0_avg = sex_ratio * age_based_survival_rates[0, 0] + (1.0 - sex_ratio) * age_based_survival_rates[1, 0]

    # When external_expected_eggs is provided (from expected_num_adult_females),
    # use it for the survival rate formula instead of the distribution-computed produced_age_0.
    # This allows independent specification of capacity (K) and expected egg production.
    survival_eggs = external_expected_eggs if external_expected_eggs is not None else produced_age_0

    if survival_eggs > 0 and s_0_avg > 1e-10:
        expected_survival_rate = total_age_1 / (survival_eggs * s_0_avg)
    else:
        expected_survival_rate = 1.0

    return expected_competition_strength, expected_survival_rate


# ============================================================================
# Scaling factor calculation functions (for larval recruitment)
# ============================================================================

# Growth mode constants
NO_COMPETITION = 0
FIXED = 1
LOGISTIC = LINEAR = 2
CONCAVE = BEVERTON_HOLT = 3


@njit_switch(cache=True)
def compute_scaling_factor_fixed(
    total_age_0: float,
    carrying_capacity: float,
) -> float:
    """Calculate scaling factor for FIXED mode.

    When total_age_0 > K, scale down proportionally to K; otherwise keep unchanged.

    Args:
        total_age_0: Total age-0 larvae count
        carrying_capacity: Carrying capacity K

    Returns:
        scaling_factor = min(1.0, K / total)
    """
    if total_age_0 > 0:
        return min(1.0, carrying_capacity / total_age_0)
    else:
        return 1.0


@njit_switch(cache=True)
def compute_actual_competition_strength(
    juvenile_counts_by_age: NDArray[np.float64],
    relative_competition_strength: NDArray[np.float64],
    new_adult_age: int,
) -> float:
    """Compute current total competition strength metrics.

    Args:
        juvenile_counts_by_age: Juvenile counts by age with shape (n_ages,)
        relative_competition_strength: Competition weights for each age with shape (n_ages,)
        new_adult_age: Starting age index for adults

    Returns:
        Total competition strength as weighted sum of juvenile counts
    """
    actual_competition_strength = 0.0
    for age in range(new_adult_age):
        actual_competition_strength += juvenile_counts_by_age[age] * relative_competition_strength[age]
    return actual_competition_strength


@njit_switch(cache=True)
def compute_scaling_factor_logistic(
    actual_competition_strength: float,
    expected_competition_strength: float,
    expected_survival_rate: float,
    low_density_growth_rate: float,
) -> float:
    """Compute LOGISTIC (LINEAR) mode scaling factor.

    Args:
        actual_competition_strength: Current competition strength
        expected_competition_strength: Expected competition strength at equilibrium
        expected_survival_rate: Expected survival rate at equilibrium
        low_density_growth_rate: Growth rate at low population density

    Returns:
        Scaling factor for larval recruitment in LOGISTIC mode
    """
    if expected_competition_strength > 0:
        competition_ratio = actual_competition_strength / expected_competition_strength
    else:
        competition_ratio = 1.0

    # Logistic (Linear): growth rate decreases linearly with competition
    r = low_density_growth_rate
    actual_growth_rate = max(0.0, -competition_ratio * (r - 1) + r)

    return actual_growth_rate * expected_survival_rate


@njit_switch(cache=True)
def compute_scaling_factor_beverton_holt(
    actual_competition_strength: float,
    expected_competition_strength: float,
    expected_survival_rate: float,
    low_density_growth_rate: float,
) -> float:
    """Compute BEVERTON_HOLT (CONCAVE) mode scaling factor.

    Args:
        actual_competition_strength: Current competition strength
        expected_competition_strength: Expected competition strength at equilibrium
        expected_survival_rate: Expected survival rate at equilibrium
        low_density_growth_rate: Growth rate at low population density

    Returns:
        Scaling factor for larval recruitment in BEVERTON_HOLT mode
    """
    if expected_competition_strength > 0:
        competition_ratio = actual_competition_strength / expected_competition_strength
    else:
        competition_ratio = 1.0

    # Beverton-Holt (Concave): growth rate follows a hyperbolic curve
    r = low_density_growth_rate
    denominator = competition_ratio * (r - 1) + 1
    actual_growth_rate = r / denominator

    return actual_growth_rate * expected_survival_rate
