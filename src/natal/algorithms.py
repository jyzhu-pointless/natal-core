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
# Continuous distribution helper functions (for use_dirichlet_sampling=True)
# ============================================================================
# Very small threshold to prevent numerical errors when distribution parameters are 0
EPS = 1e-10

@njit_switch(cache=True)
def _clamp01(x: float) -> float:
    """Clamp a probability-like value into [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x

@njit_switch(cache=True)
def continuous_poisson(lam: float) -> float:
    """Use Gamma distribution to continuousize Poisson distribution.

    Moments matching: Poisson(λ) -> Gamma(λ, 1)
    Mean and variance are both λ.

    Args:
        lam: Poisson parameter λ

    Returns:
        Value sampled from Gamma(λ, 1)
    """
    if lam <= EPS:
        return 0.0
    return np.random.gamma(lam, 1.0)


@njit_switch(cache=True)
def continuous_binomial(n: float, p: float) -> float:
    """Use Beta distribution to continuousize Binomial distribution.

    Moments matching: Binomial(n, p) -> Beta((n-1)*p, (n-1)*(1-p))
    Multiply the sampled proportion by n to get "continuous count".

    Args:
        n: Binomial sample size
        p: Binomial success probability (0 < p < 1)

    Returns:
        Continuous count value (float between 0 and n)
    """
    if p <= EPS:
        return 0.0
    if p >= 1.0 - EPS:
        return float(n)

    # When n <= 1, the concentration (n-1) is non-positive, making it impossible to perform effective moment matching via Beta distribution.
    # In this case, forced sampling would cause severe numerical bias (tending towards 0.5*n), so we fall back to deterministic expected value.
    if n <= 1.0 + EPS:
        return n * p


    # Moment matching: map Binomial(n, p) to proportion variable r~Beta(alpha,beta), then return n*r.
    # The larger the concentration, the smaller the fluctuation in r (closer to deterministic p).
    concentration = n - 1.0
    # alpha / (alpha + beta) = p, ensuring the proportion mean is p
    alpha = p * concentration
    beta_val = (1.0 - p) * concentration

    # Numerical protection
    alpha = max(alpha, EPS)
    beta_val = max(beta_val, EPS)

    proportion = np.random.beta(alpha, beta_val)
    # Return "continuous count" rather than proportion: count = n * proportion
    return proportion * n


@njit_switch(cache=True)
def continuous_multinomial(n: float, p_array: NDArray[np.float64], out_counts: NDArray[np.float64]) -> None:
    """Use Dirichlet distribution to continuousize Multinomial distribution.

    Moments matching: Multinomial(n, p) -> Dirichlet((n-1)*p)
    Use Gamma component-wise method to generate Dirichlet, avoiding direct calls that may allocate memory.
    Results are stored in pre-allocated array out_counts (in-place operation).

    Args:
        n: Multinomial total count
        p_array: Probability vector with shape (k,)
        out_counts: Output array to store results with shape (k,), will be modified in-place
    """
    k = len(p_array)

    # Performance optimization and numerical protection: for extremely small sample sizes, use deterministic allocation directly.
    if n <= 1.0 + EPS:
        for i in range(k):
            out_counts[i] = n * p_array[i]
        return

    # Similar to continuous_binomial, Dirichlet total concentration is set to (n-1).
    # Each category concentration alpha_i = p_i * (n-1), so the mean is p_i.
    concentration = n - 1.0
    sum_gamma = 0.0

    # Generate k Gamma(α_i, 1) variables
    for i in range(k):
        alpha = p_array[i] * concentration

        if alpha <= EPS:
            # If probability is extremely low, set to 0 directly
            val = 0.0
        else:
            val = float(np.random.gamma(alpha, 1.0))  # pyright: ignore

        out_counts[i] = val
        sum_gamma += val

    # Normalize and multiply by total n:
    # If g_i ~ Gamma(alpha_i,1), then g_i/sum(g) ~ Dirichlet(alpha)
    # Finally out_i = n * g_i/sum(g) is the continuous "category count".
    if sum_gamma > EPS:
        factor = n / sum_gamma
        for i in range(k):
            out_counts[i] *= factor
    else:
        # Extreme case (all alpha close to 0）
        # Use original probability vector for fallback to maintain total approximately n
        for i in range(k):
            out_counts[i] = n * p_array[i]

    # Final numerical validation: ensure output sum is approximately equal to n, avoiding cumulative numerical errors
    total = 0.0
    for i in range(k):
        total += out_counts[i]

    # If total is very small or already within reasonable error range, no additional processing needed
    tol = 1e-6 * max(1.0, n)
    if total > EPS and abs(total - n) > tol:
        # Lightweight rescaling to correct deviations caused by floating point errors
        correction = n / total
        for i in range(k):
            out_counts[i] *= correction

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
    use_dirichlet_sampling: bool = False,
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
        sperm_displacement_rate: Rate of sperm displacement (unused in current implementation)
        adult_start_idx: Starting age index for adults
        n_ages: Total number of age classes
        n_genotypes: Number of genotypes g
        is_stochastic: If True, use stochastic sampling; if False, use deterministic expectations
        use_dirichlet_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling. Currently not implemented (will use discrete).

    Returns:
        Updated sperm storage array with shape (A, g, g) containing mated female allocations

    Note:
        - `S[a, gf, :]` is interpreted as a partition of *mated* females of
          (age=a, female_genotype=gf) by male genotype.
        - Virgins are represented implicitly as:
          `virgins = female_count[a, gf] - sum_gm(S[a, gf, gm])`
        - To preserve this meaning, this function rebuilds adult rows each tick
          from current mating outcomes, rather than blending with previous rows.
        - Previous implementation used a single scalar mating rate for all
          adult ages. Now we use age-specific female mating rates:
          `p_mating(age) = female_mating_rates_by_age[age]`.
    """
    S = sperm_store.copy()
    F = female_counts
    female_rates = np.asarray(female_mating_rates_by_age)

    # n_f_int = np.round(F).astype(np.int64)
    P = np.asarray(mating_prob)  # (g, g)
    assert female_rates.shape[0] == n_ages

    # NOTE:
    # sperm_store stores only mated-female allocations by male genotype.
    # Therefore each tick we rebuild adult rows from current mating outcomes,
    # instead of mixing with previous rows. Keep sperm_displacement_rate for
    # API compatibility, but it is intentionally unused here.

    if is_stochastic:
        # ===== Monogamous random mode =====
        # Steps: (1) Determine mating count (2) Choose mating partner genotype
        temp_mating = np.zeros(n_genotypes, dtype=np.float64)
        draws_int = np.zeros(n_genotypes, dtype=np.int64)
        for a in range(adult_start_idx, n_ages):
            # Rebuild this age-slice from current outcomes.
            S[a, :, :] = 0.0

            for gf in range(n_genotypes):
                # Step 1: How many females of this genotype participate in mating?
                # _n1: Current (age, gf) female count
                _n1 = float(F[a, gf])
                # Age-specific female mating probability at age a.
                _p1 = _clamp01(float(female_rates[a]))

                if use_dirichlet_sampling:
                    # Continuous sampling: use Beta instead of Binomial
                    n_mating = continuous_binomial(_n1, _p1)
                else:
                    # Discrete sampling: standard Binomial
                    # n_mating ~ Binomial(_n1, _p1)
                    # Semantics: number of females that mate in this tick for this age/female genotype.
                    n_mating = float(nbc.binomial(int(round(_n1)), _p1))

                # Step 2: Which male genotype do these mating females mate with respectively?
                if n_mating > EPS:
                    if use_dirichlet_sampling:
                        # Continuous sampling: use Dirichlet instead of Multinomial
                        continuous_multinomial(n_mating, P[gf, :], temp_mating)
                        S[a, gf, :] = temp_mating
                    else:
                        # Discrete sampling: standard Multinomial
                        # actual_matings[gf,gm]:
                        # Allocation of mated females gf paired with males gm.
                        draws_int = nbc.multinomial(int(round(n_mating)), P[gf, :])
                        for gm in range(n_genotypes):
                            S[a, gf, gm] = float(draws_int[gm])

        return S

    else:
        # ===== Monogamous deterministic mode =====
        # Mating count = female count * mating rate * P[gf, gm]
        for a in range(adult_start_idx, n_ages):
            for gf in range(n_genotypes):
                # Deterministic version:
                # E[n_mating(age)] = female_count(age) * mating_rate(age)
                n_mating = F[a, gf] * _clamp01(float(female_rates[a]))
                # E[matings(gf,gm)] = E[n_mating] * P(gm|gf)
                for gm in range(n_genotypes):
                    S[a, gf, gm] = n_mating * P[gf, gm]

        return S

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

    Args:
        meiosis_f: Female meiosis probability matrix with shape ``(g, hl)``.
        meiosis_m: Male meiosis probability matrix with shape ``(g, hl)``.
        haplo_to_genotype_map: Haplotype-pair to genotype map with shape
            ``(hl, hl, g)``.
        n_genotypes: Number of diploid genotypes ``g``.
        n_haplogenotypes: Number of haploid genotypes.
        n_glabs: Number of gamete-label variants per haplotype.

    Returns:
        Offspring probability tensor with shape ``(g, g, g)`` where
        ``out[gf, gm, g_off] = P(g_off | gf, gm)``.
    """
    G_f = np.asarray(meiosis_f, dtype=np.float64)
    G_m = np.asarray(meiosis_m, dtype=np.float64)
    H = np.asarray(haplo_to_genotype_map, dtype=np.float64)

    hl = n_haplogenotypes * n_glabs
    H_contig = np.ascontiguousarray(H)
    H_flat = H_contig.reshape(hl * hl, n_genotypes)

    G_f_expanded = G_f[:, None, :, None]      # (g, 1, hl, 1)
    G_m_expanded = G_m[None, :, None, :]      # (1, g, 1, hl)
    all_gamete_pairs = G_f_expanded * G_m_expanded  # (g, g, hl, hl)

    all_gamete_pairs_flat = all_gamete_pairs.reshape(n_genotypes * n_genotypes, hl * hl)
    P_offspring_flat = np.dot(all_gamete_pairs_flat, H_flat)
    return P_offspring_flat.reshape(n_genotypes, n_genotypes, n_genotypes)


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
    proportion_of_females_that_reproduce: float = 1.0,
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
) -> tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Generate offspring counts using precomputed offspring probabilities."""
    S = np.asarray(sperm_storage_by_male_genotype, dtype=np.float64)
    phi_f = np.asarray(fertility_f, dtype=np.float64)
    phi_m = np.asarray(fertility_m, dtype=np.float64)
    P_offspring = np.asarray(offspring_probability, dtype=np.float64)

    n_offspring_by_geno = np.zeros(n_genotypes, dtype=np.float64)
    p_norm = np.zeros(n_genotypes, dtype=np.float64)
    temp_offspring = np.zeros(n_genotypes, dtype=np.float64)
    draws_int = np.zeros(n_genotypes, dtype=np.int64)
    p_reproduce = _clamp01(float(proportion_of_females_that_reproduce))

    has_combo = False
    for a in range(adult_start_idx, n_ages):
        for gf in range(n_genotypes):
            fertility_factor_f = phi_f[gf]
            for gm in range(n_genotypes):
                pair_count = float(S[a, gf, gm])
                if pair_count <= 0.0:
                    continue

                has_combo = True

                lambda_pair = average_eggs_per_wt_female * fertility_factor_f * phi_m[gm]
                n_total = 0.0

                if is_stochastic:
                    if use_dirichlet_sampling:
                        n_pairs_for_sampling = float(pair_count)
                    else:
                        n_pairs_for_sampling = float(np.round(pair_count))

                    if proportion_of_females_that_reproduce < 1.0:
                        if use_dirichlet_sampling:
                            n_reproducing = float(continuous_binomial(n_pairs_for_sampling, p_reproduce))
                        else:
                            n_reproducing = float(nbc.binomial(int(n_pairs_for_sampling), p_reproduce))
                    else:
                        n_reproducing = float(n_pairs_for_sampling)

                    total_lambda = float(n_reproducing * lambda_pair)

                    if fixed_eggs:
                        if use_dirichlet_sampling:
                            n_total = float(total_lambda)
                        else:
                            n_total = float(np.round(total_lambda))
                    else:
                        if use_dirichlet_sampling:
                            n_total = float(continuous_poisson(total_lambda))
                        else:
                            n_total = float(np.random.poisson(float(total_lambda)))
                else:
                    n_reproducing = float(pair_count * proportion_of_females_that_reproduce)
                    n_total = float(n_reproducing * lambda_pair)

                if n_total <= EPS:
                    continue

                p_surv = 0.0
                for g_off in range(n_genotypes):
                    p_surv += P_offspring[gf, gm, g_off]

                if is_stochastic:
                    if p_surv <= EPS:
                        continue

                    n_viable = 0.0

                    if p_surv >= 1.0 - EPS:
                        n_viable = float(n_total)
                    else:
                        if use_dirichlet_sampling:
                            n_viable = float(continuous_binomial(n_total, p_surv))
                        else:
                            n_viable = float(nbc.binomial(int(round(n_total)), p_surv))

                    if n_viable <= EPS:
                        continue

                    inv_surv = 1.0 / p_surv
                    for g_off in range(n_genotypes):
                        p_norm[g_off] = P_offspring[gf, gm, g_off] * inv_surv

                    if use_dirichlet_sampling:
                        continuous_multinomial(n_viable, p_norm, temp_offspring)
                        for g_off in range(n_genotypes):
                            n_offspring_by_geno[g_off] += temp_offspring[g_off]
                    else:
                        draws_int = nbc.multinomial(int(round(n_viable)), p_norm)
                        for g_off in range(n_genotypes):
                            n_offspring_by_geno[g_off] += float(draws_int[g_off])
                else:
                    for g_off in range(n_genotypes):
                        n_offspring_by_geno[g_off] += n_total * P_offspring[gf, gm, g_off]

    if not has_combo:
        return np.zeros(n_genotypes), np.zeros(n_genotypes)

    total_offspring = n_offspring_by_geno.sum()
    if total_offspring > EPS:
        if is_stochastic:
            sex_ratio_scalar = _clamp01(float(sex_ratio))
            if use_dirichlet_sampling:
                n_females_total = continuous_binomial(total_offspring, sex_ratio_scalar)
            else:
                n_females_total = float(nbc.binomial(int(total_offspring), sex_ratio_scalar))
        else:
            n_females_total = total_offspring * sex_ratio

        n_males_total = total_offspring - n_females_total
        n_offspring_female = np.zeros(n_genotypes, dtype=np.float64)
        n_offspring_male = np.zeros(n_genotypes, dtype=np.float64)

        nonzero_mask = n_offspring_by_geno > 0
        if nonzero_mask.any():
            proportions = n_offspring_by_geno / n_offspring_by_geno.sum()
            n_offspring_female = proportions * n_females_total
            n_offspring_male = proportions * n_males_total

        return n_offspring_female, n_offspring_male

    return np.zeros(n_genotypes), np.zeros(n_genotypes)


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
    n_glabs: int = 1,
    proportion_of_females_that_reproduce: float = 1.0,
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
) -> tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Vectorized version: batch Multinomial sampling, reducing Python loop layers. (60.9x speedup)

    Key improvements:
    1. Pre-compute expected egg counts for all (age, gf, gm) combinations → (n_adult_combos,) vector
    2. Batch Poisson sampling of all egg counts at once → avoid individual sampling
       *Note*: If `P_sums < 1.0` (Zygote Fitness/Lethality) at this stage, binomial filtering
       will be applied first to reduce egg counts. This is Pre-competition (Hard Selection) filtering.
    3. Use np.random.multinomial() directly to sample offspring genotypes
    4. Vectorized accumulation (instead of individual accumulation)

    Args:
        female_counts: Female counts array with shape (A, g)
        sperm_storage_by_male_genotype: Sperm storage array with shape (A, g, g)
        fertility_f: Female fertility rates with shape (g,)
        fertility_m: Male fertility rates with shape (g,)
        meiosis_f: Female meiosis probability matrix with shape (g, hl)
        meiosis_m: Male meiosis probability matrix with shape (g, hl)
        haplo_to_genotype_map: Haplotype to genotype mapping with shape (hl, hl, g)
        average_eggs_per_wt_female: Average eggs produced per wild-type female
        adult_start_idx: Starting age index for adults
        n_ages: Total number of age classes
        n_genotypes: Number of genotypes g
        n_haplogenotypes: Number of haploid genotypes hl
        n_glabs: Number of genetic loci (default: 1)
        proportion_of_females_that_reproduce: Proportion of females that reproduce (default: 1.0)
        fixed_eggs: If True, use fixed egg counts; if False, use Poisson sampling (default: False)
        sex_ratio: Sex ratio of offspring (default: 0.5)
        is_stochastic: If True, use stochastic sampling; if False, use deterministic expectations (default: True)
        use_dirichlet_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling. Currently not implemented (will use discrete).

    Returns:
        Tuple containing:
        - Female offspring counts with shape (g,)
        - Male offspring counts with shape (g,)
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

    return _fertilize_with_precomputed_offspring_probability(
        sperm_storage_by_male_genotype=sperm_storage_by_male_genotype,
        fertility_f=fertility_f,
        fertility_m=fertility_m,
        offspring_probability=P_offspring,
        average_eggs_per_wt_female=average_eggs_per_wt_female,
        adult_start_idx=adult_start_idx,
        n_ages=n_ages,
        n_genotypes=n_genotypes,
        proportion_of_females_that_reproduce=proportion_of_females_that_reproduce,
        fixed_eggs=fixed_eggs,
        sex_ratio=sex_ratio,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling,
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
    n_glabs: int = 1,
    proportion_of_females_that_reproduce: float = 1.0,
    fixed_eggs: bool = False,
    sex_ratio: float = 0.5,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
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
        n_glabs: Unused here; kept for signature parity.
        proportion_of_females_that_reproduce: Proportion of females that reproduce.
        fixed_eggs: Whether to use fixed egg counts.
        sex_ratio: Offspring female ratio.
        is_stochastic: Whether to sample stochastically.
        use_dirichlet_sampling: Whether to use continuous sampling.

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
        proportion_of_females_that_reproduce=proportion_of_females_that_reproduce,
        fixed_eggs=fixed_eggs,
        sex_ratio=sex_ratio,
        is_stochastic=is_stochastic,
        use_dirichlet_sampling=use_dirichlet_sampling,
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
    female, male = population
    F = np.asarray(female).copy()
    M = np.asarray(male).copy()
    S = np.asarray(sperm_store).copy()
    s_f = np.asarray(female_survival_rates)
    s_m = np.asarray(male_survival_rates)

    assert F.shape == (n_ages, n_genotypes)
    assert M.shape == (n_ages, n_genotypes)
    assert S.shape == (n_ages, n_genotypes, n_genotypes)

    # === Female: Normalize to 2D array ===
    if s_f.ndim == 1:
        assert s_f.shape == (n_ages,)
        # Convert to 2D for unified processing
        s_f_2d = s_f.reshape(n_ages, 1)
    else:
        assert s_f.shape == (n_ages, n_genotypes)
        s_f_2d = s_f

    # === Male: Normalize to 2D array ===
    if s_m.ndim == 1:
        assert s_m.shape == (n_ages,)
        s_m_2d = s_m.reshape(n_ages, 1)
    else:
        assert s_m.shape == (n_ages, n_genotypes)
        s_m_2d = s_m

    # === Apply female survival rates (loop to handle possible broadcasting) ===
    for age in range(n_ages):
        for g in range(n_genotypes):
            # Use modulo operation to handle broadcasted (n_ages, 1) case
            g_idx = g % s_f_2d.shape[1]
            rate = float(s_f_2d[age, g_idx])
            F[age, g] *= rate
            S[age, g, :] *= rate

    # === Apply male survival rates ===
    for age in range(n_ages):
        for g in range(n_genotypes):
            g_idx = g % s_m_2d.shape[1]
            rate = float(s_m_2d[age, g_idx])
            M[age, g] *= rate

    return F, M, S


@njit_switch(cache=True)
def sample_survival_with_sperm_storage(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    female_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    male_survival_rates: Annotated[NDArray[np.float64], "shape=(A,)|(A,g)"],
    n_genotypes: int,
    n_ages: int,
    use_dirichlet_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g,g)"]]:
    """Randomly apply survival rates with consistent sampling of sperm storage.

    Key: For each (age, gf) pair, use the **same sampling result** to update individual counts and sperm storage.

    Args:
        population: (female, male) tuple
        sperm_store: Sperm storage array shape (n_ages, n_genotypes, n_genotypes)
        female_survival_rates: Female survival rates (supports 1D or 2D)
        male_survival_rates: Male survival rates (supports 1D or 2D)
        n_genotypes: Number of genotypes
        use_dirichlet_sampling: If True, use Dirichlet distribution instead of discrete sampling.
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
    female, male = population
    F = np.asarray(female).copy()
    M = np.asarray(male).copy()
    S = np.asarray(sperm_store).copy()
    s_f = np.asarray(female_survival_rates)
    s_m = np.asarray(male_survival_rates)

    assert F.shape == (n_ages, n_genotypes)
    assert M.shape == (n_ages, n_genotypes)
    assert S.shape == (n_ages, n_genotypes, n_genotypes)

    # Normalize survival rates to 2D arrays (must be done outside loop to avoid Numba type issues)
    # This ensures s_f_2d and s_m_2d always have consistent types in the loop
    if s_f.ndim == 1:
        # If 1D, expand to 2D: (n_ages,) -> (n_ages, 1)
        s_f_2d = s_f.reshape(n_ages, 1)
    else:
        # If already 2D, use directly
        s_f_2d = s_f

    if s_m.ndim == 1:
        s_m_2d = s_m.reshape(n_ages, 1)
    else:
        s_m_2d = s_m

    # Sample by (age, genotype) pair
    for age in range(n_ages):
        for g in range(n_genotypes):
            # ===== Sample females and their sperm storage =====
            n_f_raw = float(F[age, g])

            g_idx_f = g % s_f_2d.shape[1]
            p_f = _clamp01(float(s_f_2d[age, g_idx_f]))

            # Calculate number of virgin females (females without stored sperm)
            # total_sperm_count = sum_gm S[age,g,gm]
            # Semantics: 'mated female mass' in this (age,g).
            total_sperm_count = 0.0
            for gm in range(n_genotypes):
                total_sperm_count += float(S[age, g, gm])

            # Validate on raw mass (not per-cell rounded mass), then convert for sampling.
            n_virgins_raw = n_f_raw - total_sperm_count
            if n_virgins_raw < -EPS:
                print(
                    "n_virgins<0 in sample_survival_with_sperm_storage:",
                    n_virgins_raw,
                    "age=",
                    age,
                    "g=",
                    g,
                    "n_f_raw=",
                    n_f_raw,
                    "total_sperm=",
                    total_sperm_count,
                )
                raise ValueError("Invalid state: n_virgins < 0 in sample_survival_with_sperm_storage")
            # In discrete mode, number of trials for binomial should be integer; continuous mode maintains floating point mass.
            n_virgins = n_virgins_raw if use_dirichlet_sampling else float(int(round(n_virgins_raw)))

            # Sample each sperm storage separately (independently using same survival rate p_f):
            # S_new[gm] ~ Binomial(S_old[gm], p_f)
            new_sperm_sum = 0.0
            for gm in range(n_genotypes):
                if use_dirichlet_sampling:
                    n_sperm = S[age, g, gm]
                else:
                    n_sperm = float(int(round(S[age, g, gm])))

                if n_sperm > EPS:
                    if use_dirichlet_sampling:
                        # Continuous sampling: use Beta instead of Binomial
                        S[age, g, gm] = continuous_binomial(n_sperm, p_f)
                    else:
                        # Discrete sampling: standard Binomial
                        S[age, g, gm] = float(nbc.binomial(int(n_sperm), p_f))
                else:
                    S[age, g, gm] = 0.0
                new_sperm_sum += S[age, g, gm]

            # Sample virgin females (also using same survival rate p_f)
            if n_virgins > EPS:
                if use_dirichlet_sampling:
                    # Continuous sampling: use Beta instead of Binomial
                    survivors_virgins = continuous_binomial(n_virgins, p_f)
                else:
                    # Discrete sampling: standard Binomial
                    survivors_virgins = float(nbc.binomial(int(n_virgins), p_f))
            else:
                survivors_virgins = 0.0

            # Mass conservation reconstruction:
            # F_new = (sum of mated survivors) + (number of virgin survivors)
            F[age, g] = new_sperm_sum + survivors_virgins

            # ===== Sample males =====
            if use_dirichlet_sampling:
                n_m = M[age, g]
            else:
                n_m = float(int(round(M[age, g])))

            g_idx_m = g % s_m_2d.shape[1]
            p_m = _clamp01(float(s_m_2d[age, g_idx_m]))

            if n_m > EPS:
                if use_dirichlet_sampling:
                    # Continuous sampling: use Beta instead of Binomial
                    M[age, g] = continuous_binomial(n_m, p_m)
                else:
                    # Discrete sampling: standard Binomial
                    M[age, g] = float(nbc.binomial(int(n_m), p_m))
            else:
                M[age, g] = 0.0

    return F, M, S

# deprecated
@njit_switch(cache=True)
def sample_viability_with_sperm_storage(
    population: Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"]],
    sperm_store: Annotated[NDArray[np.float64], "shape=(A,g,g)"],
    female_viability_rates: Annotated[NDArray[np.float64], "shape=(g,)"],
    male_viability_rates: Annotated[NDArray[np.float64], "shape=(g,)"],
    n_genotypes: int,
    n_ages: int,
    target_age: int,
    use_dirichlet_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g)"], Annotated[NDArray[np.float64], "shape=(A,g,g)"]]:
    """Randomly apply viability with consistent sampling of sperm storage (only at target_age).

    Similar to apply_viability_sampling, but also returns updated sperm_store.

    Args:
        population: (female, male) tuple
        sperm_store: Sperm storage array shape (n_ages, n_genotypes, n_genotypes)
        female_viability_rates: Female viability genotype-specific rates shape (g,)
        male_viability_rates: Male viability genotype-specific rates shape (g,)
        n_genotypes: Number of genotypes
        n_ages: Number of ages
        target_age: Age index where viability is applied
        use_dirichlet_sampling: If True, use Dirichlet distribution instead of discrete sampling.

    Returns:
        Tuple[female_new, male_new, sperm_store_new]

    Note:
        - This legacy path mirrors `sample_survival_with_sperm_storage`.
        - Negative `n_virgins_raw` is treated as invalid state and raises.
    """
    female, male = population
    F = np.asarray(female).copy()
    M = np.asarray(male).copy()
    S = np.asarray(sperm_store).copy()
    v_f = np.asarray(female_viability_rates)
    v_m = np.asarray(male_viability_rates)

    assert F.shape == (n_ages, n_genotypes)
    assert M.shape == (n_ages, n_genotypes)
    assert S.shape == (n_ages, n_genotypes, n_genotypes)
    assert v_f.shape == (n_genotypes,)
    assert v_m.shape == (n_genotypes,)

    # Sample only at target_age
    for g in range(n_genotypes):
        n_f_raw = float(F[target_age, g])
        if use_dirichlet_sampling:
            n_m_val = M[target_age, g]
        else:
            n_m_val = float(int(round(M[target_age, g])))

        p_f_val = _clamp01(float(v_f[g]))
        p_m_val = _clamp01(float(v_m[g]))

        # ===== Sample females and their sperm storage =====
        # Calculate number of virgin females
        total_sperm_count = 0.0
        for gm in range(n_genotypes):
            total_sperm_count += float(S[target_age, g, gm])

        # Validate on raw mass (not per-cell rounded mass), then convert for sampling.
        n_virgins_raw = n_f_raw - total_sperm_count
        if n_virgins_raw < -EPS:
            print(
                "n_virgins<0 in sample_viability_with_sperm_storage:",
                n_virgins_raw,
                "target_age=",
                target_age,
                "g=",
                g,
                "n_f_raw=",
                n_f_raw,
                "total_sperm=",
                total_sperm_count,
            )
            raise ValueError("Invalid state: n_virgins < 0 in sample_viability_with_sperm_storage")
        n_virgins = n_virgins_raw if use_dirichlet_sampling else float(int(round(n_virgins_raw)))

        # Sample each sperm storage separately (independently using survival rate p_f_val)
        new_sperm_sum = 0.0
        for gm in range(n_genotypes):
            if use_dirichlet_sampling:
                n_sperm = S[target_age, g, gm]
            else:
                n_sperm = float(int(round(S[target_age, g, gm])))

            if n_sperm > EPS:
                if use_dirichlet_sampling:
                    # Continuous sampling: use Beta instead of Binomial
                    S[target_age, g, gm] = continuous_binomial(n_sperm, p_f_val)
                else:
                    # Discrete sampling: standard Binomial
                    S[target_age, g, gm] = float(nbc.binomial(int(n_sperm), p_f_val))
            else:
                S[target_age, g, gm] = 0.0
            new_sperm_sum += S[target_age, g, gm]

        # Sample virgin females
        if n_virgins > EPS:
            if use_dirichlet_sampling:
                # Continuous sampling: use Beta instead of Binomial
                survivors_virgins = continuous_binomial(n_virgins, p_f_val)
            else:
                # Discrete sampling: standard Binomial
                survivors_virgins = float(nbc.binomial(int(n_virgins), p_f_val))
        else:
            survivors_virgins = 0.0

        # F[target_age, g] = survived females with stored sperm + survived virgin females
        F[target_age, g] = new_sperm_sum + survivors_virgins

        # ===== Sample males =====
        if n_m_val > EPS:
            if use_dirichlet_sampling:
                # Continuous sampling: use Beta instead of Binomial
                M[target_age, g] = continuous_binomial(n_m_val, p_m_val)
            else:
                # Discrete sampling: standard Binomial
                M[target_age, g] = float(nbc.binomial(int(n_m_val), p_m_val))
        else:
            M[target_age, g] = 0.0

    return F, M, S

@njit_switch(cache=True)
def recruit_juveniles_sampling(
    age_0_juvenile_counts: Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]],
    carrying_capacity: int,
    n_genotypes: int,
    is_stochastic: bool = True,
    use_dirichlet_sampling: bool = False,
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
        use_dirichlet_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling

    Returns:
        Tuple[female_new, male_new]: Recruited juvenile counts with shape (g,) each
    """
    female_0, male_0 = age_0_juvenile_counts
    # Ensure inputs are treated as flattened counts
    if use_dirichlet_sampling:
        female_arr = np.asarray(female_0)
        male_arr = np.asarray(male_0)
    else:
        female_arr = np.rint(np.asarray(female_0))
        male_arr = np.rint(np.asarray(male_0))

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
        if use_dirichlet_sampling:
            # Continuous sampling: use Dirichlet instead of Multinomial
            out_counts = np.zeros(2 * n_genotypes, dtype=np.float64)
            continuous_multinomial(K, probs, out_counts)
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
    use_dirichlet_sampling: bool = False,
) -> Tuple[Annotated[NDArray[np.float64], "shape=(g,)"], Annotated[NDArray[np.float64], "shape=(g,)"]]:
    """Scale age-0 juveniles by `scaling_factor`.

    If `is_stochastic` is True, sample exactly `round(total * scaling_factor)`
    juveniles by multinomial according to genotype-by-sex proportions.

    Args:
        age_0_juvenile_counts: Tuple of (female_0, male_0) age-0 juvenile counts
        scaling_factor: Scaling factor to apply to total juvenile count
        n_genotypes: Number of genotypes
        is_stochastic: If True, use stochastic sampling; if False, use deterministic scaling
        use_dirichlet_sampling: If True and is_stochastic=True, use Dirichlet distribution
            instead of discrete sampling. Currently not implemented (will use discrete).

    Returns:
        Tuple[female_new, male_new]: Scaled juvenile counts with shape (g,) each.
            Returns float64 arrays (containing integral values if stochastic).
    """
    female_0, male_0 = age_0_juvenile_counts
    if use_dirichlet_sampling:
        female_arr = np.asarray(female_0)
        male_arr = np.asarray(male_0)
    else:
        female_arr = np.rint(np.asarray(female_0))
        male_arr = np.rint(np.asarray(male_0))

    assert female_arr.shape == (n_genotypes,)
    assert male_arr.shape == (n_genotypes,)

    total = float(female_arr.sum() + male_arr.sum())
    if total <= 0:
        return np.zeros_like(female_arr), np.zeros_like(male_arr)

    if use_dirichlet_sampling:
        desired = total * float(scaling_factor)
    else:
        desired = float(int(round(total * float(scaling_factor))))

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
        if use_dirichlet_sampling:
            # Continuous sampling: use Dirichlet instead of Multinomial
            temp_counts = np.zeros(2 * n_genotypes, dtype=np.float64)
            continuous_multinomial(float(desired), probs, temp_counts)
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
    equilibrium_individual_count: Optional[NDArray[np.float64]] = None, # (sex, age, genotype_sum)
) -> Tuple[float, float]:
    """Calculate competition strength and survival rate metrics under equilibrium.

    These metrics are used for LOGISTIC and BEVERTON_HOLT density-dependent modes.

    Args:
        carrying_capacity: Total carrying capacity K based on age=1
        expected_eggs_per_female: Basic offspring count
        age_based_survival_rates: Survival rate matrix (2, n_ages)
        age_based_mating_rates: Mating rate matrix (2, n_ages)
        female_age_based_relative_fertility: Female age-dependent relative fertility (n_ages,)
        relative_competition_strength: Competition weights for each age (n_ages,)
        sex_ratio: Sex ratio (female proportion)
        new_adult_age: Adult starting age
        n_ages: Total number of ages
        equilibrium_individual_count: Optional user-provided equilibrium distribution (2, n_ages)

    Returns:
        Tuple[expected_competition_strength, expected_survival_rate]
    """
    # Pre-calculate cumulative mating rates for females of each age (i.e., proportion holding sperm, assuming no sperm depletion)
    # Individuals hold sperm if they have mated before
    p_mated = np.zeros(n_ages, dtype=np.float64)
    p_unmated = 1.0
    for age in range(new_adult_age, n_ages):
        m_rate = age_based_mating_rates[0, age]
        # Recursion: P(unmated until age a) = Π_{k<=a}(1 - m_rate[k])
        p_unmated *= (1.0 - m_rate)
        # P(mated by age a) = 1 - P(unmated by age a)
        p_mated[age] = 1.0 - p_unmated

    if equilibrium_individual_count is not None:
        # 1. Use user-provided equilibrium distribution
        expected_distribution = equilibrium_individual_count
        # Calculate produced age-0 numbers: only adult females
        produced_age_0 = 0.0
        for age in range(new_adult_age, n_ages):
            n_f = expected_distribution[0, age]
            # Here consider cumulative mating rate (proportion holding sperm) and relative fertility
            # Contribution of this age to age0 production:
            # n_f * P(mated) * relative_fertility * eggs_per_female
            produced_age_0 += n_f * p_mated[age] * female_age_based_relative_fertility[age] * expected_eggs_per_female

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
            produced_age_0 += n_f * p_mated[age] * female_age_based_relative_fertility[age] * expected_eggs_per_female

    # Calculate total expected competition strength (limited to larvae participating in competition, i.e., age < new_adult_age)
    # Age 0 is produced Egg count; Age 1+ are survivors in distribution
    # Competition strength is weighted sum of "larvae count * corresponding competition weight".
    expected_competition_strength = produced_age_0 * relative_competition_strength[0]
    for age in range(1, new_adult_age):
        n_total = expected_distribution[0, age] + expected_distribution[1, age]
        expected_competition_strength += n_total * relative_competition_strength[age]

    # Calculate expected survival rate (scaling factor from Egg production to entering age=1)
    # Under equilibrium: total_age_1 = produced_age_0 * expected_survival_rate * s_0_avg
    # Where s_0_avg is base survival rate from Age 0 to Age 1
    s_0_avg = sex_ratio * age_based_survival_rates[0, 0] + (1.0 - sex_ratio) * age_based_survival_rates[1, 0]

    if produced_age_0 > 0 and s_0_avg > 1e-10:
        # Derive from equilibrium relationship:
        # total_age_1 = produced_age_0 * expected_survival_rate * s_0_avg
        # => expected_survival_rate = total_age_1 / (produced_age_0 * s_0_avg)
        expected_survival_rate = total_age_1 / (produced_age_0 * s_0_avg)
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
