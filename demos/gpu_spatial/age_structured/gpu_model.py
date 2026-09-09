"""PyTorch XPU implementation of the spatial age-structured deterministic model.

This model mirrors the CPU reference in ``reference_cpu.py`` and follows the
same deterministic lifecycle order as natal-core:

    first hooks (omitted in GPU prototype)
    → reproduction (mating/sperm update + offspring production)
    → survival (juvenile competition + age/viability survival)
    → aging
    → migration

Only deterministic mode is implemented here. The stochastic path is left for a
later extension; ``stochastic=True`` is rejected for now.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# natal growth mode constants.
NO_COMPETITION = 0
FIXED = 1
LOGISTIC = 2
BEVERTON_HOLT = 3


def _as_tensor(
    arr: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Move a NumPy array to a torch tensor on *device*."""
    return torch.as_tensor(arr, dtype=dtype, device=device)


class SpatialAgeStructuredXPU:
    """Deterministic spatial age-structured model on XPU.

    State tensors:

    - ``individual_count``: (D, sex=2, age=A, ztype=G)
    - ``sperm_storage``:    (D, age=A, female_ztype=G, male_ztype=G)

    All lifecycle stages are vectorised across the deme dimension ``D``.
    """

    def __init__(
        self,
        individual_count: np.ndarray,
        sperm_storage: np.ndarray,
        config: object,
        adjacency: np.ndarray | None = None,
        *,
        migration_rate: float | np.ndarray,
        n_ticks: int,
        device: torch.device | None = None,
        stochastic: bool = False,
        seed: int | None = None,
        grid_shape: tuple[int, int] | None = None,
        wrap: bool = False,
        migration_kernel: np.ndarray | None = None,
        adjust_migration_on_edge: bool = True,
    ) -> None:
        if device is None:
            device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        self.device = device
        self.stochastic = bool(stochastic)
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "xpu":
                torch.xpu.manual_seed_all(seed)
            elif device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        self.dtype = torch.float32
        self.n_ticks = n_ticks

        self.ind_count = _as_tensor(individual_count, device=device)
        self.sperm = _as_tensor(sperm_storage, device=device)

        self.n_demes = int(self.ind_count.shape[0])
        self.n_sexes = int(self.ind_count.shape[1])
        self.n_ages = int(self.ind_count.shape[2])
        self.n_ztypes = int(self.ind_count.shape[3])

        # ---------------- natal config fields ---------------------------------
        self.adult_start = int(config.new_adult_age)
        self.has_sex_chromosomes = bool(config.has_sex_chromosomes)

        # Reproduction / mating tensors.
        self.eggs_per_female = float(config.eggs_per_female[()])
        self.sex_ratio = float(config.sex_ratio[()])
        self.sperm_displacement_rate = float(config.sperm_displacement_rate[()])
        self.fixed_egg_count = bool(config.fixed_egg_count)
        self.offspring_tensor = _as_tensor(
            np.asarray(config.offspring_tensor, dtype=np.float32), device=device
        )
        self.fecundity_f = _as_tensor(
            np.asarray(config.fecundity_fitness[0], dtype=np.float32), device=device
        )
        self.fecundity_m = _as_tensor(
            np.asarray(config.fecundity_fitness[1], dtype=np.float32), device=device
        )
        self.sexual_selection = _as_tensor(
            np.asarray(config.sexual_selection_fitness, dtype=np.float32),
            device=device,
        )
        self.female_mating_rate = _as_tensor(
            np.asarray(config.age_based_mating_rates[0], dtype=np.float32),
            device=device,
        )
        self.male_mating_rate = _as_tensor(
            np.asarray(config.age_based_mating_rates[1], dtype=np.float32),
            device=device,
        )
        self.reproduction_rate = _as_tensor(
            np.asarray(config.age_based_reproduction_rates, dtype=np.float32),
            device=device,
        )
        self.female_fertility = _as_tensor(
            np.asarray(config.female_age_based_fertility, dtype=np.float32),
            device=device,
        )

        # Survival tensors.
        target_viability_age = int(config.new_adult_age) - 1
        self.age_survival_f = _as_tensor(
            np.asarray(config.age_based_survival_rates[0], dtype=np.float32),
            device=device,
        )
        self.age_survival_m = _as_tensor(
            np.asarray(config.age_based_survival_rates[1], dtype=np.float32),
            device=device,
        )
        self.viability_f = _as_tensor(
            np.asarray(
                config.viability_fitness[0, target_viability_age, :],
                dtype=np.float32,
            ),
            device=device,
        )
        self.viability_m = _as_tensor(
            np.asarray(
                config.viability_fitness[1, target_viability_age, :],
                dtype=np.float32,
            ),
            device=device,
        )
        self.zygote_viability_f = _as_tensor(
            np.asarray(config.zygote_viability_fitness[0], dtype=np.float32),
            device=device,
        )
        self.zygote_viability_m = _as_tensor(
            np.asarray(config.zygote_viability_fitness[1], dtype=np.float32),
            device=device,
        )

        # Sex-chromosome metadata.
        self.female_only = torch.as_tensor(
            np.asarray(config.female_only_by_sex_chrom, dtype=np.bool_),
            device=device,
        )
        self.male_only = torch.as_tensor(
            np.asarray(config.male_only_by_sex_chrom, dtype=np.bool_),
            device=device,
        )
        self.female_compat = _as_tensor(
            np.asarray(config.female_ztype_compatibility, dtype=np.float32),
            device=device,
        )
        self.male_compat = _as_tensor(
            np.asarray(config.male_ztype_compatibility, dtype=np.float32),
            device=device,
        )

        # Competition tensors.
        self.juvenile_growth_mode = int(config.juvenile_growth_mode[()])
        self.expected_competition_strength = float(
            config.expected_competition_strength[()]
        )
        self.expected_survival_rate = float(config.expected_survival_rate[()])
        self.low_density_growth_rate = float(config.low_density_growth_rate[()])
        self.carrying_capacity = float(config.carrying_capacity[()])
        self.relative_competition_strength = _as_tensor(
            np.asarray(
                config.age_based_relative_competition_strength, dtype=np.float32
            ),
            device=device,
        )

        # Migration.
        self.migration_rate = migration_rate
        self.migration_rate_by_age = self._normalize_migration_rate()

        self.grid_shape = grid_shape
        self.wrap = bool(wrap)
        self.adjust_migration_on_edge = bool(adjust_migration_on_edge)
        self.migration_kernel = (
            None if migration_kernel is None else np.asarray(migration_kernel)
        )
        self.use_stencil = self.grid_shape is not None

        if self.use_stencil:
            rows, cols = int(self.grid_shape[0]), int(self.grid_shape[1])
            if rows * cols != self.n_demes:
                raise ValueError(
                    f"grid_shape {self.grid_shape} does not match n_demes={self.n_demes}"
                )
            if self.migration_kernel is None:
                raise ValueError("migration_kernel is required for stencil migration")
            kernel = self.migration_kernel
            if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
                raise ValueError("migration_kernel must be a 2D odd-sized array")
            self.grid_rows = rows
            self.grid_cols = cols
            self.kernel_pad_r = kernel.shape[0] // 2
            self.kernel_pad_c = kernel.shape[1] // 2
            self.kernel_total_sum = float(kernel.sum())
            self.stencil_offsets: list[tuple[int, int, float]] = []
            for kr in range(kernel.shape[0]):
                for kc in range(kernel.shape[1]):
                    if kr == self.kernel_pad_r and kc == self.kernel_pad_c:
                        continue
                    weight = float(kernel[kr, kc])
                    if weight <= 0.0:
                        continue
                    self.stencil_offsets.append(
                        (kr - self.kernel_pad_r, kc - self.kernel_pad_c, weight)
                    )

            # Degree map = number/weight sum of valid neighbors per source cell.
            ones_grid = torch.ones(
                (1, rows, cols), dtype=self.dtype, device=device
            )
            degree = torch.zeros_like(ones_grid)
            for dr, dc, weight in self.stencil_offsets:
                degree += weight * self._shift_grid(ones_grid, dr, dc)
            self.degree_map = degree.clamp_min(1e-10)

            neighbor_indices, neighbor_probs = self._build_stencil_neighbor_arrays()

            # Dense fallback is not used in stencil mode.
            self.adjacency_T = None
            self.eye = None
        else:
            if adjacency is None:
                raise ValueError(
                    "adjacency is required when grid_shape is not provided"
                )
            adjacency_t = torch.as_tensor(
                adjacency, dtype=self.dtype, device=device
            )
            # natal's deterministic migration output for one bucket x is:
            #   y_d = (1-r) x_d + r * sum_i adj[i,d] x_i
            # = ((1-r) I + r * adj^T) x
            self.eye = torch.eye(
                adjacency_t.shape[0], dtype=self.dtype, device=device
            )
            self.adjacency_T = adjacency_t.T
            neighbor_indices, neighbor_probs = (
                self._build_adjacency_neighbor_arrays(
                    np.asarray(adjacency, dtype=np.float64), self.n_demes
                )
            )

        self.neighbor_indices = torch.as_tensor(
            neighbor_indices, device=device
        )
        self.neighbor_probs = torch.as_tensor(
            neighbor_probs, dtype=self.dtype, device=device
        )

        self.tick = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _normalize_migration_rate(self) -> torch.Tensor:
        """Return per-age migration rates.

        Scalar migration applies only to age >= ``new_adult_age``; younger ages
        get zero migration, matching natal-core's ``_normalize_migration_rate``.
        """
        if np.ndim(self.migration_rate) == 0:
            base = float(self.migration_rate)
            rate = np.zeros(self.n_ages, dtype=np.float32)
            rate[self.adult_start:] = base
        else:
            rate = np.asarray(self.migration_rate, dtype=np.float32)
            if rate.shape != (self.n_ages,):
                raise ValueError(
                    f"migration_rate shape {rate.shape} != (n_ages={self.n_ages},)"
                )
        self.migration_rate_np = rate
        self.active_ages_np = np.flatnonzero(rate > 0.0)
        if self.active_ages_np.size > 0:
            first_rate = float(rate[self.active_ages_np[0]])
            self.uniform_active_rate = bool(
                np.allclose(rate[self.active_ages_np], first_rate)
            )
            self.uniform_rate_value = first_rate
        else:
            self.uniform_active_rate = False
            self.uniform_rate_value = 0.0
        return _as_tensor(rate, device=self.device)

    def _shift_grid(
        self, x: torch.Tensor, dr: int, dc: int
    ) -> torch.Tensor:
        """Shift a (C, rows, cols) grid by (dr, dc).

        ``wrap=True`` uses periodic (torus) boundaries via ``torch.roll``.
        ``wrap=False`` uses zero padding so out-of-grid neighbors contribute 0.
        """
        if self.wrap:
            return torch.roll(x, shifts=(dr, dc), dims=(-2, -1))
        padded = F.pad(
            x,
            (self.kernel_pad_c, self.kernel_pad_c, self.kernel_pad_r, self.kernel_pad_r),
        )
        return padded[
            :,
            self.kernel_pad_r + dr : self.kernel_pad_r + dr + self.grid_rows,
            self.kernel_pad_c + dc : self.kernel_pad_c + dc + self.grid_cols,
        ]

    def _migration_stencil(self, flat: torch.Tensor, rate: float) -> torch.Tensor:
        """Apply one stencil migration step to (D, C) counts.

        With ``adjust_migration_on_edge=True`` this matches natal-core's
        kernel path: each source deme sends ``rate`` of its mass, split over
        its *valid* neighbors by kernel weight. With ``False`` it sends
        ``rate * valid_weight_sum / kernel_total_sum`` instead.
        """
        rows, cols = self.grid_rows, self.grid_cols
        channels = flat.shape[1]
        x_grid = flat.T.reshape(channels, rows, cols)

        source = x_grid
        if self.adjust_migration_on_edge:
            source = x_grid / self.degree_map

        neighbor_sum = torch.zeros_like(source)
        for dr, dc, weight in self.stencil_offsets:
            neighbor_sum = neighbor_sum + weight * self._shift_grid(source, dr, dc)

        if self.adjust_migration_on_edge:
            incoming = neighbor_sum
        else:
            incoming = neighbor_sum / self.kernel_total_sum

        out_grid = (1.0 - rate) * x_grid + rate * incoming
        return out_grid.reshape(channels, rows * cols).T

    def _build_stencil_neighbor_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Build sparse neighbour index/probability rows from the stencil.

        Used by the stochastic migration path so it never needs a dense
        ``(D, D)`` adjacency matrix.
        """
        rows, cols = self.grid_rows, self.grid_cols
        n_demes = rows * cols
        max_nnz = len(self.stencil_offsets)
        indices = np.full((n_demes, max_nnz), -1, dtype=np.int64)
        probs = np.zeros((n_demes, max_nnz), dtype=np.float32)

        for src in range(n_demes):
            src_row, src_col = divmod(src, cols)
            valid: list[tuple[int, float]] = []
            for dr, dc, weight in self.stencil_offsets:
                dst_row = src_row + dr
                dst_col = src_col + dc
                if self.wrap:
                    dst_row %= rows
                    dst_col %= cols
                elif (
                    dst_row < 0
                    or dst_row >= rows
                    or dst_col < 0
                    or dst_col >= cols
                ):
                    continue
                valid.append((dst_row * cols + dst_col, weight))

            total = sum(weight for _dst, weight in valid)
            if self.adjust_migration_on_edge:
                denom = total if total > 0.0 else 1.0
            else:
                denom = self.kernel_total_sum if self.kernel_total_sum > 0.0 else 1.0
            for k, (dst, weight) in enumerate(valid):
                indices[src, k] = dst
                probs[src, k] = weight / denom
        return indices, probs

    @staticmethod
    def _build_adjacency_neighbor_arrays(
        adjacency: np.ndarray, n_demes: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build sparse neighbour rows from an explicit dense adjacency."""
        neighbor_lists: list[np.ndarray] = []
        prob_lists: list[np.ndarray] = []
        for i in range(n_demes):
            nz = np.flatnonzero(adjacency[i] > 0.0)
            neighbor_lists.append(nz.astype(np.int64))
            row = adjacency[i, nz]
            if row.sum() > 0.0:
                row = row / row.sum()
            prob_lists.append(row.astype(np.float32))
        max_nnz = max((len(x) for x in neighbor_lists), default=0)
        indices = np.full((n_demes, max_nnz), -1, dtype=np.int64)
        probs = np.zeros((n_demes, max_nnz), dtype=np.float32)
        for i in range(n_demes):
            n = len(neighbor_lists[i])
            indices[i, :n] = neighbor_lists[i]
            probs[i, :n] = prob_lists[i]
        return indices, probs

    def _mating_probability(self) -> torch.Tensor:
        """Effective-male-based mating probability matrix P (D, Gf, Gm).

        Corresponds to:
        ``natal.engine.simulation.age_structured.compute_mating_probability_matrix``
        over all adult ages.
        """
        adult_ages = list(range(self.adult_start, self.n_ages))
        adult_males = self.ind_count[:, 1, adult_ages, :]  # (D, Aad, G)
        male_rates = self.male_mating_rate[adult_ages]     # (Aad,)
        effective_males = (adult_males * male_rates[None, :, None]).sum(dim=1)

        A = self.sexual_selection.unsqueeze(0) * effective_males.unsqueeze(1)
        row_sum = A.sum(dim=-1)  # (D, Gf)
        eps = 1e-10
        safe = row_sum.clamp_min(eps)
        P = torch.where(
            (row_sum > eps).unsqueeze(-1),
            A / safe.unsqueeze(-1),
            torch.zeros_like(A),
        )
        return P

    # ------------------------------------------------------------------
    # stochastic sampling helpers (same approach as the discrete model)
    # ------------------------------------------------------------------
    @staticmethod
    def _round_counts(counts: torch.Tensor) -> torch.Tensor:
        """Round float counts to non-negative integers (as natal does)."""
        return counts.round().clamp_min(0.0)

    @staticmethod
    def _sample_binomial(
        counts: torch.Tensor,
        probs: torch.Tensor | float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample Binomial(counts, probs), preserving natal's round-first style."""
        n = SpatialAgeStructuredXPU._round_counts(counts)
        if isinstance(probs, float):
            if probs <= 1e-10:
                return torch.zeros_like(n)
            if probs >= 1.0 - 1e-10:
                return n
            p = torch.full_like(n, probs)
        else:
            p = probs.to(device=device, dtype=dtype)
        return torch.distributions.Binomial(total_count=n, probs=p).sample()

    @staticmethod
    def _sample_multinomial_rows(
        totals: torch.Tensor,
        probs: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample independent Multinomial rows with different per-row totals.

        Uses the conditional-binomial decomposition (same helper as the
        discrete model) so it stays O(K) per row and needs no CPU sync.
        """
        totals = SpatialAgeStructuredXPU._round_counts(totals)
        n_rows, n_categories = probs.shape
        if n_rows == 0 or n_categories == 0:
            return torch.zeros_like(probs)

        row_sum = probs.sum(dim=1, keepdim=True)
        safe_probs = probs / row_sum.clamp_min(1e-10)
        active = (totals > 0.0) & (row_sum.squeeze(1) > 1e-10)
        safe_probs = torch.where(
            active.unsqueeze(1), safe_probs, torch.zeros_like(safe_probs)
        )
        safe_probs = torch.nan_to_num(
            safe_probs, nan=0.0, posinf=0.0, neginf=0.0
        )

        out = torch.zeros_like(probs)
        remaining = totals.clamp_min(0.0)
        for k in range(n_categories - 1):
            tail = safe_probs[:, k:].sum(dim=1)
            cond_p = torch.where(
                tail > 1e-10,
                safe_probs[:, k] / tail.clamp_min(1e-10),
                torch.zeros_like(tail),
            )
            remaining_int = remaining.round().clamp_min(0.0)
            draws = torch.distributions.Binomial(
                total_count=remaining_int, probs=cond_p
            ).sample()
            draws = torch.minimum(draws, remaining_int)
            out[:, k] = draws
            remaining = remaining_int - draws
        out[:, n_categories - 1] = remaining.round().clamp_min(0.0)
        return out

    # ------------------------------------------------------------------
    # stage: reproduction
    # ------------------------------------------------------------------
    def _reproduction(self) -> None:
        """Deterministic mating/sperm update + offspring production.

        Matches CPU:
        ``run_reproduction_with_precomputed_offspring_probability`` →
        ``sample_mating(stochastic=False)`` +
        ``fertilize_with_precomputed_offspring_probability_and_age_specific_reproduction``
        """
        P = self._mating_probability()  # (D, Gf, Gm)
        eps = 1e-10
        adult = slice(self.adult_start, self.n_ages)

        # 1. Update sperm storage for all adult female ages at once.
        S_adult = self.sperm[:, adult, :, :]          # (D, Aad, Gf, Gm)
        F_adult = self.ind_count[:, 0, adult, :]      # (D, Aad, Gf)
        mated = S_adult.sum(dim=-1)                   # (D, Aad, Gf)
        virgins = (F_adult - mated).clamp_min(0.0)    # (D, Aad, Gf)

        p_mating = self.female_mating_rate[adult]     # (Aad,)
        p_remating = p_mating * self.sperm_displacement_rate
        n_mating_virgins = virgins * p_mating[None, :, None]
        n_remating = mated * p_remating[None, :, None]
        frac = torch.where(
            mated > eps,
            (n_remating / mated.clamp_min(eps)).clamp_max(1.0),
            torch.zeros_like(mated),
        )
        n_new = n_mating_virgins + n_remating         # (D, Aad, Gf)

        S_new = S_adult * (1.0 - frac.unsqueeze(-1))
        S_new = S_new + n_new.unsqueeze(-1) * P.unsqueeze(1)
        self.sperm[:, adult, :, :] = S_new

        # 2. Deterministic offspring production for all adult ages at once.
        repro = self.reproduction_rate[adult]         # (Aad,)
        fert_factor = self.female_fertility[adult]    # (Aad,)
        pair_fertility = (
            S_new
            * self.fecundity_f[None, None, :, None]
            * self.fecundity_m[None, None, None, :]
            * self.eggs_per_female
            * repro[None, :, None, None]
            * fert_factor[None, :, None, None]
        )  # (D, Aad, Gf, Gm)
        offspring_by_age = torch.einsum(
            "dafm,fmo->dao", pair_fertility, self.offspring_tensor
        )
        offspring = offspring_by_age.sum(dim=1)      # (D, Go)

        # 3. Sex assignment.
        if self.has_sex_chromosomes:
            denom = self.female_compat + self.male_compat
            p_f = torch.where(
                denom > eps,
                (self.female_compat / denom.clamp_min(eps)).clamp(0.0, 1.0),
                torch.full_like(denom, 0.5),
            )
            female_off = torch.where(
                self.female_only,
                offspring,
                torch.where(
                    self.male_only,
                    torch.zeros_like(offspring),
                    offspring * p_f.unsqueeze(0),
                ),
            )
        else:
            female_off = offspring * self.sex_ratio
        male_off = offspring - female_off

        # 4. Place offspring at age 0 and apply zygote viability.
        self.ind_count[:, 0, 0, :] = female_off * self.zygote_viability_f.unsqueeze(0)
        self.ind_count[:, 1, 0, :] = male_off * self.zygote_viability_m.unsqueeze(0)

    def _reproduction_stochastic(self) -> None:
        """Stochastic mating/sperm update + offspring production.

        Mirrors natal-core's stochastic age-structured reproduction:
        Binomial mating/remating, Multinomial male allocation, Poisson egg
        counts, Multinomial offspring genotypes, Binomial sex assignment and
        Binomial zygote viability.
        """
        P = self._mating_probability()  # (D, Gf, Gm)
        eps = 1e-10
        adult = slice(self.adult_start, self.n_ages)
        D = self.n_demes
        Aad = self.n_ages - self.adult_start
        G = self.n_ztypes

        # 1. Stochastic sperm-store update for all adult ages.
        S_adult = self.sperm[:, adult, :, :]          # (D, Aad, Gf, Gm)
        F_adult = self.ind_count[:, 0, adult, :]      # (D, Aad, Gf)
        mated = S_adult.sum(dim=-1)                   # (D, Aad, Gf)
        virgins = (F_adult - mated).clamp_min(0.0)    # (D, Aad, Gf)

        p_mating = self.female_mating_rate[adult]     # (Aad,)
        p_remating = p_mating * self.sperm_displacement_rate

        n_mating_virgins = self._sample_binomial(
            virgins, p_mating[None, :, None], self.device, self.dtype
        )

        sperm_int = self._round_counts(S_adult)
        n_remove = self._sample_binomial(
            sperm_int,
            p_remating[None, :, None, None],
            self.device,
            self.dtype,
        )
        n_remating = n_remove.sum(dim=-1)             # (D, Aad, Gf)
        S_after_displace = (S_adult - n_remove).clamp_min(0.0)
        n_new = n_mating_virgins + n_remating         # (D, Aad, Gf)

        # Allocate new matings over male genotypes with a multinomial per row.
        pair_totals = n_new.reshape(-1)
        pair_probs = (
            P.unsqueeze(1)
            .expand(D, Aad, G, G)
            .reshape(-1, G)
        )
        new_pairs = self._sample_multinomial_rows(
            pair_totals, pair_probs, self.dtype
        ).reshape(D, Aad, G, G)
        S_new = S_after_displace + new_pairs
        self.sperm[:, adult, :, :] = S_new

        # 2. Stochastic offspring production for all adult ages.
        n_pairs = self._round_counts(S_new)           # (D, Aad, Gf, Gm)
        p_reproduce = self.reproduction_rate[adult]   # (Aad,)
        n_reproducing = self._sample_binomial(
            n_pairs,
            p_reproduce[None, :, None, None],
            self.device,
            self.dtype,
        )

        eggs_per_pair = (
            self.eggs_per_female
            * self.fecundity_f[None, None, :, None]
            * self.fecundity_m[None, None, None, :]
            * self.female_fertility[adult][None, :, None, None]
        )
        total_lambda = n_reproducing * eggs_per_pair

        if self.fixed_egg_count:
            n_total = self._round_counts(total_lambda)
        else:
            n_total = torch.distributions.Poisson(total_lambda).sample()

        p_surv = self.offspring_tensor.sum(dim=2)      # (Gf, Gm)
        p_surv_b = p_surv[None, None, :, :]
        n_total_int = self._round_counts(n_total)
        n_viable = torch.where(
            p_surv_b >= 1.0 - eps,
            n_total_int,
            self._sample_binomial(
                n_total_int, p_surv_b, self.device, self.dtype
            ),
        )

        offspring_probs = (
            self.offspring_tensor
            / p_surv.clamp_min(eps).unsqueeze(-1)
        )
        offspring_probs = (
            offspring_probs.unsqueeze(0)
            .unsqueeze(0)
            .expand(D, Aad, G, G, G)
            .reshape(-1, G)
        )
        offspring_counts = self._sample_multinomial_rows(
            n_viable.reshape(-1), offspring_probs, self.dtype
        ).reshape(D, Aad, G, G, G).sum(dim=(1, 2, 3))  # (D, Go)

        # 3. Stochastic sex assignment.
        if self.has_sex_chromosomes:
            denom = self.female_compat + self.male_compat
            p_f = torch.where(
                denom > eps,
                (self.female_compat / denom.clamp_min(eps)).clamp(0.0, 1.0),
                torch.full_like(denom, 0.5),
            )
            female_off = torch.where(
                self.female_only,
                offspring_counts,
                torch.where(
                    self.male_only,
                    torch.zeros_like(offspring_counts),
                    self._sample_binomial(
                        offspring_counts,
                        p_f.unsqueeze(0),
                        self.device,
                        self.dtype,
                    ),
                ),
            )
        else:
            female_off = self._sample_binomial(
                offspring_counts, self.sex_ratio, self.device, self.dtype
            )
        male_off = offspring_counts - female_off

        # 4. Stochastic zygote viability.
        female_off = self._sample_binomial(
            female_off,
            self.zygote_viability_f.unsqueeze(0),
            self.device,
            self.dtype,
        )
        male_off = self._sample_binomial(
            male_off,
            self.zygote_viability_m.unsqueeze(0),
            self.device,
            self.dtype,
        )
        self.ind_count[:, 0, 0, :] = female_off
        self.ind_count[:, 1, 0, :] = male_off

    # ------------------------------------------------------------------
    # stage: survival
    # ------------------------------------------------------------------
    def _survival(self) -> None:
        """Density-dependent juvenile regulation + deterministic survival.

        Matches CPU ``run_survival(stochastic=False)``.
        """
        # 1. Density regulation on age-0 offspring.
        f0 = self.ind_count[:, 0, 0, :]
        m0 = self.ind_count[:, 1, 0, :]
        scaling = self._density_scaling(f0, m0)

        self.ind_count[:, 0, 0, :] = f0 * scaling.unsqueeze(-1)
        self.ind_count[:, 1, 0, :] = m0 * scaling.unsqueeze(-1)

        # 2. Age-based survival x viability at target age.
        # Viability is only applied at age = new_adult_age - 1.
        target = self.adult_start - 1
        via_f = torch.ones(
            (self.n_ages, self.n_ztypes), dtype=self.dtype, device=self.device
        )
        via_m = torch.ones_like(via_f)
        via_f[target, :] = self.viability_f
        via_m[target, :] = self.viability_m

        combined_f = self.age_survival_f[:, None] * via_f  # (A, G)
        combined_m = self.age_survival_m[:, None] * via_m  # (A, G)

        # Female individuals and associated sperm use the female survival rate.
        self.ind_count[:, 0, :, :] = (
            self.ind_count[:, 0, :, :] * combined_f.unsqueeze(0)
        )
        # sperm[age, gf, gm] is scaled by female survival of (age, gf).
        self.sperm = self.sperm * combined_f[None, :, :, None]

        # Male individuals use the male survival rate.
        self.ind_count[:, 1, :, :] = (
            self.ind_count[:, 1, :, :] * combined_m.unsqueeze(0)
        )

    def _density_scaling(
        self, f0: torch.Tensor, m0: torch.Tensor
    ) -> torch.Tensor:
        """Return the juvenile density-regulation scaling factor per deme."""
        total_age0 = f0.sum(dim=-1) + m0.sum(dim=-1)  # (D,)

        # Juvenile competition includes all ages < new_adult_age.
        juvenile_ind = self.ind_count[:, :, : self.adult_start, :]  # (D, 2, Aj, G)
        juvenile_totals = juvenile_ind.sum(dim=(1, 3))              # (D, Aj)
        actual_comp = (
            juvenile_totals
            * self.relative_competition_strength[: self.adult_start].unsqueeze(0)
        ).sum(dim=-1)

        mode = self.juvenile_growth_mode
        if mode == NO_COMPETITION:
            return torch.ones_like(total_age0)
        if mode == FIXED:
            return torch.where(
                total_age0 > 1e-10,
                torch.clamp(
                    self.carrying_capacity / total_age0.clamp_min(1e-10), max=1.0
                ),
                torch.ones_like(total_age0),
            )
        if mode == LOGISTIC:
            if self.expected_competition_strength > 0:
                competition_ratio = actual_comp / self.expected_competition_strength
            else:
                competition_ratio = torch.ones_like(actual_comp)
            r = self.low_density_growth_rate
            actual_growth_rate = torch.clamp(
                -competition_ratio * (r - 1.0) + r, min=0.0
            )
            return actual_growth_rate * self.expected_survival_rate
        if mode == BEVERTON_HOLT:
            if self.expected_competition_strength > 0:
                competition_ratio = actual_comp / self.expected_competition_strength
            else:
                competition_ratio = torch.ones_like(actual_comp)
            r = self.low_density_growth_rate
            denominator = competition_ratio * (r - 1.0) + 1.0
            return (r / denominator) * self.expected_survival_rate
        raise NotImplementedError(f"Unsupported juvenile_growth_mode: {mode}")

    def _survival_stochastic(self) -> None:
        """Stochastic density regulation and survival.

        Mirrors natal-core's ``run_survival(stochastic=True)``: multinomial
        recruitment followed by Binomial survival of virgin females, sperm
        blocks and males.
        """
        f0 = self.ind_count[:, 0, 0, :]
        m0 = self.ind_count[:, 1, 0, :]
        scaling = self._density_scaling(f0, m0)

        # 1. Multinomial recruitment of age-0 juveniles.
        f0_int = self._round_counts(f0)
        m0_int = self._round_counts(m0)
        total_int = f0_int.sum(dim=-1) + m0_int.sum(dim=-1)
        desired = self._round_counts(total_int * scaling)

        counts = torch.cat([f0_int, m0_int], dim=1)  # (D, 2G)
        probs = counts / total_int.clamp_min(1.0).unsqueeze(1)
        probs = torch.where(
            (total_int > 1e-10).unsqueeze(1),
            probs,
            torch.zeros_like(probs),
        )
        recruited = self._sample_multinomial_rows(desired, probs, self.dtype)
        f_rec = recruited[:, : self.n_ztypes]
        m_rec = recruited[:, self.n_ztypes :]

        # 2. Binomial survival of females + their stored sperm.
        target = self.adult_start - 1
        via_f = torch.ones(
            (self.n_ages, self.n_ztypes), dtype=self.dtype, device=self.device
        )
        via_m = torch.ones_like(via_f)
        via_f[target, :] = self.viability_f
        via_m[target, :] = self.viability_m
        combined_f = self.age_survival_f[:, None] * via_f  # (A, G)
        combined_m = self.age_survival_m[:, None] * via_m  # (A, G)

        sperm_int = self._round_counts(self.sperm)
        sperm_surv = self._sample_binomial(
            sperm_int,
            combined_f[:, :, None],
            self.device,
            self.dtype,
        )
        mated_surv = sperm_surv.sum(dim=-1)              # (D, A, G)
        mated_int = sperm_int.sum(dim=-1)                # (D, A, G)

        female_raw = self.ind_count[:, 0, :, :]
        virgins_int = (female_raw - mated_int).round().clamp_min(0.0)
        virgin_surv = self._sample_binomial(
            virgins_int, combined_f, self.device, self.dtype
        )
        female_surv = mated_surv + virgin_surv

        # Age 0 females are the freshly recruited juveniles, not the previous
        # raw age-0 bucket.
        female_surv[:, 0, :] = self._sample_binomial(
            f_rec, combined_f[0], self.device, self.dtype
        )
        sperm_surv[:, 0, :, :] = 0.0

        male_int = self._round_counts(self.ind_count[:, 1, :, :])
        male_surv = self._sample_binomial(
            male_int, combined_m, self.device, self.dtype
        )
        male_surv[:, 0, :] = self._sample_binomial(
            m_rec, combined_m[0], self.device, self.dtype
        )

        self.ind_count[:, 0, :, :] = female_surv
        self.ind_count[:, 1, :, :] = male_surv
        self.sperm = sperm_surv

    def new_adult_age_for_competition(self) -> int:
        """Compatibility helper for the juvenile-age tensor width."""
        return self.adult_start

    # ------------------------------------------------------------------
    # stage: aging
    # ------------------------------------------------------------------
    def _aging(self) -> None:
        """Advance all age classes by one.

        Matches CPU ``run_aging``.
        """
        new_ind = torch.zeros_like(self.ind_count)
        new_sperm = torch.zeros_like(self.sperm)

        new_ind[:, :, 1:, :] = self.ind_count[:, :, :-1, :]
        new_sperm[:, 1:, :, :] = self.sperm[:, :-1, :, :]

        self.ind_count = new_ind
        self.sperm = new_sperm

    # ------------------------------------------------------------------
    # stage: migration
    # ------------------------------------------------------------------
    def _migrate_age_stencil(self, age: int, rate: float) -> None:
        """Migrate one age's female/male/sperm channels in one stencil call."""
        D = self.n_demes
        f = self.ind_count[:, 0, age, :]                 # (D, G)
        m = self.ind_count[:, 1, age, :]                 # (D, G)
        s = self.sperm[:, age, :, :].reshape(D, -1)      # (D, Gf*Gm)
        combined = torch.cat([f, m, s], dim=1)
        out = self._migration_stencil(combined, rate)

        cf = f.shape[1]
        cm = m.shape[1]
        self.ind_count[:, 0, age, :] = out[:, :cf]
        self.ind_count[:, 1, age, :] = out[:, cf : cf + cm]
        self.sperm[:, age, :, :] = out[:, cf + cm :].reshape_as(
            self.sperm[:, age, :, :]
        )

    def _migration(self) -> None:
        """Apply deterministic migration to adult ages.

        Stencil mode fuses all active ages and all channels into a single
        ``_migration_stencil`` call whenever the per-age migration rate is
        uniform (the normal scalar-rate case). This drastically reduces the
        number of small XPU kernels and CPU launches per tick.
        """
        active = self.active_ages_np
        if active.size == 0:
            return

        if self.use_stencil:
            if (
                self.uniform_active_rate
                and np.array_equal(active, np.arange(active[0], active[-1] + 1))
            ):
                start = int(active[0])
                end = int(active[-1]) + 1
                D = self.n_demes

                f = self.ind_count[:, 0, start:end, :].reshape(D, -1)
                m = self.ind_count[:, 1, start:end, :].reshape(D, -1)
                s = self.sperm[:, start:end, :, :].reshape(D, -1)
                cf, cm = f.shape[1], m.shape[1]
                combined = torch.cat([f, m, s], dim=1)

                migrated = self._migration_stencil(
                    combined, self.uniform_rate_value
                )

                self.ind_count[:, 0, start:end, :] = migrated[:, :cf].reshape_as(
                    self.ind_count[:, 0, start:end, :]
                )
                self.ind_count[:, 1, start:end, :] = migrated[
                    :, cf : cf + cm
                ].reshape_as(self.ind_count[:, 1, start:end, :])
                self.sperm[:, start:end, :, :] = migrated[:, cf + cm :].reshape_as(
                    self.sperm[:, start:end, :, :]
                )
            else:
                for age in active:
                    self._migrate_age_stencil(
                        int(age), float(self.migration_rate_np[age])
                    )
            return

        # Dense fallback: keep the original per-age matrix behavior.
        assert self.eye is not None and self.adjacency_T is not None
        for age in active:
            r = float(self.migration_rate_np[age])
            T = (1.0 - r) * self.eye + r * self.adjacency_T
            age_idx = int(age)
            self.ind_count[:, 0, age_idx, :] = T @ self.ind_count[:, 0, age_idx, :]
            self.ind_count[:, 1, age_idx, :] = T @ self.ind_count[:, 1, age_idx, :]
            flat_s = self.sperm[:, age_idx, :, :].reshape(self.n_demes, -1)
            migrated_s = T @ flat_s
            self.sperm[:, age_idx, :, :] = migrated_s.reshape_as(
                self.sperm[:, age_idx, :, :]
            )

    def _migrate_bucket_stochastic(
        self, values: torch.Tensor, rate: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Migrate one ``(D, C)`` bucket with Binomial + Multinomial sampling.

        Returns ``(stay, incoming)`` where both tensors have shape ``(D, C)``.
        """
        D = values.shape[0]
        C = values.shape[1]
        max_nnz = self.neighbor_probs.shape[1]

        values_int = self._round_counts(values)
        # For adjust_migration_on_edge=False, neighbour probabilities sum to
        # less than 1 at the boundary; scale the outbound probability by that
        # row sum so the stochastic semantics match natal-core.
        outbound_prob = (
            self.neighbor_probs.sum(dim=1) * rate
        ).clamp(0.0, 1.0)
        outbound = self._sample_binomial(
            values_int,
            outbound_prob.unsqueeze(1),
            self.device,
            self.dtype,
        )
        stay = values_int - outbound

        totals = outbound.reshape(-1)
        probs = self.neighbor_probs.repeat_interleave(C, dim=0)
        sampled = self._sample_multinomial_rows(totals, probs, self.dtype)

        dst = self.neighbor_indices.repeat_interleave(C, dim=0)
        cat = (
            torch.arange(C, device=self.device)
            .unsqueeze(0)
            .expand(D, -1)
            .reshape(-1)
        )
        cat = cat.unsqueeze(1).expand(-1, max_nnz)

        valid = dst >= 0
        index_flat = (dst * C + cat).clamp_min(0).reshape(-1)
        counts_flat = torch.where(
            valid.reshape(-1),
            sampled.reshape(-1),
            torch.zeros_like(sampled.reshape(-1)),
        )

        incoming_flat = torch.zeros(
            D * C, dtype=self.dtype, device=self.device
        )
        incoming_flat.scatter_add_(0, index_flat, counts_flat)
        incoming = incoming_flat.reshape(D, C)
        return stay, incoming

    def _migration_stochastic(self) -> None:
        """Stochastic age-structured migration with virgin/sperm coupling.

        When all active ages share one migration rate (the normal scalar-rate
        case), the three bucket types are each migrated in one call over all
        active ages. This keeps the number of small random kernels low.
        """
        active = self.active_ages_np
        if active.size == 0:
            return

        if (
            self.uniform_active_rate
            and np.array_equal(active, np.arange(active[0], active[-1] + 1))
        ):
            start = int(active[0])
            end = int(active[-1]) + 1
            rate = self.uniform_rate_value
            D = self.n_demes
            G = self.n_ztypes

            female = self.ind_count[:, 0, start:end, :]          # (D, Aad, G)
            sperm = self.sperm[:, start:end, :, :]               # (D, Aad, G, G)
            male = self.ind_count[:, 1, start:end, :]            # (D, Aad, G)

            mated = sperm.sum(dim=-1)                            # (D, Aad, G)
            virgins = (female - mated).clamp_min(0.0)

            stay_v, in_v = self._migrate_bucket_stochastic(
                virgins.reshape(D, -1), rate
            )
            stay_s, in_s = self._migrate_bucket_stochastic(
                sperm.reshape(D, -1), rate
            )
            stay_m, in_m = self._migrate_bucket_stochastic(
                male.reshape(D, -1), rate
            )

            Aad = end - start
            stay_v = stay_v.reshape(D, Aad, G)
            in_v = in_v.reshape(D, Aad, G)
            stay_s = stay_s.reshape(D, Aad, G, G)
            in_s = in_s.reshape(D, Aad, G, G)
            stay_m = stay_m.reshape(D, Aad, G)
            in_m = in_m.reshape(D, Aad, G)

            self.ind_count[:, 0, start:end, :] = (
                stay_v + stay_s.sum(dim=-1) + in_v + in_s.sum(dim=-1)
            )
            self.ind_count[:, 1, start:end, :] = stay_m + in_m
            self.sperm[:, start:end, :, :] = stay_s + in_s
            return

        # Fallback: per-age migration when rates differ between ages.
        for age in active:
            age_idx = int(age)
            rate = float(self.migration_rate_np[age_idx])

            female = self.ind_count[:, 0, age_idx, :]          # (D, G)
            sperm_age = self.sperm[:, age_idx, :, :]           # (D, Gf, Gm)
            mated = sperm_age.sum(dim=-1)                      # (D, G)
            virgins = (female - mated).clamp_min(0.0)          # (D, G)

            stay_v, in_v = self._migrate_bucket_stochastic(virgins, rate)
            stay_s, in_s = self._migrate_bucket_stochastic(
                sperm_age.reshape(self.n_demes, -1), rate
            )
            stay_s = stay_s.reshape(self.n_demes, self.n_ztypes, self.n_ztypes)
            in_s = in_s.reshape(self.n_demes, self.n_ztypes, self.n_ztypes)

            stay_m, in_m = self._migrate_bucket_stochastic(
                self.ind_count[:, 1, age_idx, :], rate
            )

            new_female = stay_v + stay_s.sum(dim=-1) + in_v + in_s.sum(dim=-1)
            new_sperm = stay_s + in_s
            new_male = stay_m + in_m

            self.ind_count[:, 0, age_idx, :] = new_female
            self.ind_count[:, 1, age_idx, :] = new_male
            self.sperm[:, age_idx, :, :] = new_sperm

    # ------------------------------------------------------------------
    # public run
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance one age-structured tick (deterministic or stochastic)."""
        if self.stochastic:
            self._reproduction_stochastic()
            self._survival_stochastic()
            self._aging()
            self._migration_stochastic()
        else:
            self._reproduction()
            self._survival()
            self._aging()
            self._migration()
        self.tick += 1

    def run_no_history(self, n_ticks: int | None = None) -> None:
        """Run ticks without returning state to CPU."""
        n = self.n_ticks if n_ticks is None else n_ticks
        for _ in range(n):
            self.step()
        if self.device.type == "xpu":
            torch.xpu.synchronize()

    def run_history(self) -> tuple[np.ndarray, np.ndarray]:
        """Run all ticks; return (individual_count_history, sperm_history).

        History axes are ``(record, D, sex, age, ztype)`` and
        ``(record, D, age, female_ztype, male_ztype)``, matching natal raw
        spatial history.
        """
        # Keep snapshots on the accelerator and perform one CPU transfer at
        # the end instead of syncing once per tick.
        ind_hist = [self.ind_count.clone()]
        sperm_hist = [self.sperm.clone()]
        for _ in range(self.n_ticks):
            self.step()
            ind_hist.append(self.ind_count.clone())
            sperm_hist.append(self.sperm.clone())

        ind_stacked = torch.stack(ind_hist, dim=0).detach().cpu().numpy()
        sperm_stacked = torch.stack(sperm_hist, dim=0).detach().cpu().numpy()
        return ind_stacked, sperm_stacked
