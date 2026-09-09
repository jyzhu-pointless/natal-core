"""PyTorch XPU implementation of the spatial discrete-generation model.

This is intentionally aligned with the CPU reference in ``reference_cpu.py``:

- 25 demes on a 5x5 square grid
- ``stochastic=False``: deterministic discrete-generation lifecycle
- ``stochastic=True``: Binomial / Multinomial / Poisson sampling aligned with
  natal-core's stochastic discrete lifecycle
- fixed juvenile density regulation
- row-normalized adjacency migration

The implementation uses only standard PyTorch tensor operations so it can run
on ``xpu`` locally and later on ``cuda`` with a device change.

Stochastic multinomial sampling uses a conditional-binomial decomposition
instead of expanding each row to ``max_total`` categorical draws; this avoids
both the O(max_total * n_rows) cost and any per-sample CPU synchronisation.

This demo file is outside natal-core's strict ``src`` type-check scope, and it
dynamically reads natal config objects that Pylance cannot fully infer.
"""

# pyright: reportUndefinedVariable=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportCallIssue=false, reportUntypedFunctionDecorator=false, reportAttributeAccessIssue=false

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import reference_cpu

# natal constants used by the engine.
FIXED_MODE = 1


def _torch_tensor_from_numpy(
    arr: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Move a NumPy array to a torch tensor on *device*."""
    return torch.as_tensor(arr, dtype=dtype, device=device)


class SpatialDiscreteXPU:
    """Small XPU/CUDA-ready deterministic spatial discrete-generation model."""

    def __init__(
        self,
        state: np.ndarray,
        config: object,
        adjacency: np.ndarray | None = None,
        *,
        migration_rate: float,
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
        self.dtype = torch.float32
        self.n_ticks = n_ticks
        self.stochastic = bool(stochastic)
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "xpu":
                torch.xpu.manual_seed_all(seed)
            elif device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        self.state = _torch_tensor_from_numpy(
            state, device=device, dtype=self.dtype
        )
        self.config = config

        # --- genetic tensors -------------------------------------------------
        self.offspring_tensor = _torch_tensor_from_numpy(
            np.asarray(config.offspring_tensor, dtype=np.float32),
            device=device,
        )
        self.fecundity_f = _torch_tensor_from_numpy(
            np.asarray(config.fecundity_f, dtype=np.float32), device=device
        )
        self.fecundity_m = _torch_tensor_from_numpy(
            np.asarray(config.fecundity_m, dtype=np.float32), device=device
        )
        self.viability_f = _torch_tensor_from_numpy(
            np.asarray(config.viability_f, dtype=np.float32), device=device
        )
        self.viability_m = _torch_tensor_from_numpy(
            np.asarray(config.viability_m, dtype=np.float32), device=device
        )
        self.sexual_selection = _torch_tensor_from_numpy(
            np.asarray(config.sexual_selection_fitness, dtype=np.float32),
            device=device,
        )
        self.female_compat = _torch_tensor_from_numpy(
            np.asarray(config.female_ztype_compatibility, dtype=np.float32),
            device=device,
        )
        self.male_compat = _torch_tensor_from_numpy(
            np.asarray(config.male_ztype_compatibility, dtype=np.float32),
            device=device,
        )
        self.female_only = torch.as_tensor(
            np.asarray(config.female_only_by_sex_chrom, dtype=np.bool_),
            device=device,
        )
        self.male_only = torch.as_tensor(
            np.asarray(config.male_only_by_sex_chrom, dtype=np.bool_),
            device=device,
        )

        # --- scalar demography ----------------------------------------------
        self.eggs_per_female = float(config.eggs_per_female[()])
        self.sex_ratio = float(config.sex_ratio[()])
        self.female_adult_mating_rate = float(config.female_adult_mating_rate)
        self.male_adult_mating_rate = float(config.male_adult_mating_rate)
        self.reproduction_rate = float(config.reproduction_rate)
        self.female_age0_survival = float(config.female_age0_survival)
        self.male_age0_survival = float(config.male_age0_survival)
        self.carrying_capacity = float(config.carrying_capacity[()])
        self.juvenile_growth_mode = int(config.juvenile_growth_mode[()])
        self.has_sex_chromosomes = bool(config.has_sex_chromosomes)

        self.migration_rate = float(migration_rate)

        # --- migration setup --------------------------------------------------
        # Stencil mode avoids a dense (D, D) adjacency/migration matrix and is
        # the default for the large-grid benchmarks. Dense mode is kept as a
        # backward-compatible fallback for explicit adjacency input.
        self.grid_shape = grid_shape
        self.wrap = bool(wrap)
        self.adjust_migration_on_edge = bool(adjust_migration_on_edge)
        self.migration_kernel = (
            None if migration_kernel is None else np.asarray(migration_kernel)
        )
        self.use_stencil = grid_shape is not None
        n_demes = int(state.shape[0])

        if self.use_stencil:
            rows, cols = int(grid_shape[0]), int(grid_shape[1])
            if rows * cols != n_demes:
                raise ValueError(
                    f"grid_shape {grid_shape} does not match n_demes={n_demes}"
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

            ones_grid = torch.ones(
                (1, rows, cols), dtype=self.dtype, device=device
            )
            degree = torch.zeros_like(ones_grid)
            for dr, dc, weight in self.stencil_offsets:
                degree += weight * self._shift_grid(ones_grid, dr, dc)
            self.degree_map = degree.clamp_min(1e-10)

            neighbor_indices, neighbor_probs = self._build_stencil_neighbor_arrays()
            self.adjacency = None
            self.migration_matrix = None
        else:
            if adjacency is None:
                raise ValueError(
                    "adjacency is required when grid_shape is not provided"
                )
            self.adjacency = torch.as_tensor(
                adjacency, dtype=self.dtype, device=device
            )
            adj = self.adjacency
            eye = torch.eye(adj.shape[0], dtype=self.dtype, device=device)
            self.migration_matrix = (
                (1.0 - self.migration_rate) * eye
                + self.migration_rate * adj.T
            )
            neighbor_indices, neighbor_probs = (
                self._build_adjacency_neighbor_arrays(
                    np.asarray(adjacency, dtype=np.float64), n_demes
                )
            )

        self.neighbor_indices = torch.as_tensor(
            neighbor_indices, device=device
        )
        self.neighbor_probs = torch.as_tensor(
            neighbor_probs, dtype=self.dtype, device=device
        )

        # Discrete generation uses two age slots: age 0 juveniles do not
        # migrate; age 1 adults receive the full scalar migration rate.
        self.migration_rate_by_age = [0.0, float(migration_rate)]

        self.tick = 0

    # ------------------------------------------------------------------
    # migration helpers
    # ------------------------------------------------------------------
    def _shift_grid(self, x: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
        """Shift a ``(C, rows, cols)`` grid by ``(dr, dc)``.

        ``wrap=True`` uses periodic (torus) boundaries via ``torch.roll``.
        ``wrap=False`` uses zero padding so out-of-grid neighbours contribute 0.
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
        """Apply one stencil migration step to ``(D, C)`` counts.

        With ``adjust_migration_on_edge=True`` each source sends ``rate`` of
        its mass split over its valid neighbours; with ``False`` it sends
        ``rate * valid_weight_sum / kernel_total_sum`` instead.
        """
        rows, cols = self.grid_rows, self.grid_cols
        channels = flat.shape[1]
        x_grid = flat.T.reshape(channels, rows, cols)

        source = x_grid
        if self.adjust_migration_on_edge:
            source = x_grid / self.degree_map

        neighbour_sum = torch.zeros_like(source)
        for dr, dc, weight in self.stencil_offsets:
            neighbour_sum = neighbour_sum + weight * self._shift_grid(source, dr, dc)

        if self.adjust_migration_on_edge:
            incoming = neighbour_sum
        else:
            incoming = neighbour_sum / self.kernel_total_sum

        out_grid = (1.0 - rate) * x_grid + rate * incoming
        return out_grid.reshape(channels, rows * cols).T

    def _build_stencil_neighbor_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Build sparse neighbour index/probability rows from the stencil.

        This keeps stochastic migration O(D * K) in memory instead of building
        a dense adjacency matrix.
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

    # ------------------------------------------------------------------
    # stochastic sampling helpers
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
        n = SpatialDiscreteXPU._round_counts(counts)
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

        PyTorch's ``torch.distributions.Multinomial`` does not support
        inhomogeneous per-row total counts. Instead of expanding every row to
        ``max_total`` categorical draws (which is O(max_total * n_rows) and
        previously required a CPU sync), this implementation uses the standard
        conditional-binomial decomposition:

        ``(X_1, ..., X_K) ~ Multinomial(n, p)`` is equivalent to sequentially
        drawing

        ``X_k ~ Binomial(n_remaining, p_k / sum_{j>=k} p_j)``

        for ``k = 1..K-1``, with ``X_K`` equal to the remaining count.

        This keeps the work at O(K) vectorised binomial draws per row and does
        not require any CPU synchronisation.

        Args:
            totals: 1-D tensor of total counts per row.
            probs: 2-D tensor ``(n_rows, n_categories)``.

        Returns:
            Count tensor with the same shape as *probs*.
        """
        totals = SpatialDiscreteXPU._round_counts(totals)
        n_rows, n_categories = probs.shape
        if n_rows == 0 or n_categories == 0:
            return torch.zeros_like(probs)

        # Normalize each row; invalid rows (total 0 or zero probability mass)
        # become all-zero probability rows and therefore produce all-zero
        # counts without any data-dependent CPU branch.
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

        # Conditional binomial decomposition. K is small in all current calls
        # (genotype count or neighbour count), so K-1 vectorised draws is cheap.
        for k in range(n_categories - 1):
            tail = safe_probs[:, k:].sum(dim=1)  # (n_rows,)
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
    # deterministic lifecycle
    # ------------------------------------------------------------------
    def _reproduction_deterministic(self) -> None:
        """Run deterministic reproduction on age-1 adults."""
        females = self.state[:, 0, 1, :]  # (D, G)
        males = self.state[:, 1, 1, :]    # (D, G)
        effective_males = males * self.male_adult_mating_rate

        # A[gf, gm] = sexual_selection[gf, gm] * effective_male[gm]
        A = self.sexual_selection.unsqueeze(0) * effective_males.unsqueeze(1)
        row_sum = A.sum(dim=2)  # (D, Gf)

        eps = 1e-10
        safe_row_sum = row_sum.clamp_min(eps)
        P = torch.where(
            (row_sum > eps).unsqueeze(2),
            A / safe_row_sum.unsqueeze(2),
            torch.zeros_like(A),
        )

        # Deterministic number of mating females.
        n_mating = females * self.female_adult_mating_rate  # (D, Gf)
        pairs = n_mating.unsqueeze(2) * P  # (D, Gf, Gm)

        # Expected offspring before sex assignment.
        pair_fertility = (
            pairs
            * self.fecundity_f.unsqueeze(0).unsqueeze(2)
            * self.fecundity_m.unsqueeze(0).unsqueeze(0)
            * self.eggs_per_female
            * self.reproduction_rate
        )  # (D, Gf, Gm)
        offspring = torch.einsum(
            "dab,abc->dc", pair_fertility, self.offspring_tensor
        )  # (D, Go)

        # Sex assignment.
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

        self.state[:, 0, 0, :] = female_off
        self.state[:, 1, 0, :] = male_off

    def _survival_deterministic(self) -> None:
        """Apply fixed-cap density regulation and age-0 survival."""
        f0 = self.state[:, 0, 0, :]
        m0 = self.state[:, 1, 0, :]
        total0 = f0.sum(dim=1) + m0.sum(dim=1)  # (D,)

        mode = self.juvenile_growth_mode
        if mode == 0:  # NO_COMPETITION
            scaling = torch.ones_like(total0)
        elif mode == FIXED_MODE:
            scaling = torch.where(
                total0 > 1e-10,
                torch.clamp(
                    self.carrying_capacity / total0.clamp_min(1e-10),
                    max=1.0,
                ),
                torch.ones_like(total0),
            )
        else:
            raise NotImplementedError(
                f"GPU model currently supports mode 0/1 only; got {mode}"
            )

        survival_f = self.female_age0_survival * self.viability_f  # (G,)
        survival_m = self.male_age0_survival * self.viability_m

        self.state[:, 0, 0, :] = f0 * scaling.unsqueeze(1) * survival_f.unsqueeze(0)
        self.state[:, 1, 0, :] = m0 * scaling.unsqueeze(1) * survival_m.unsqueeze(0)

    def _aging(self) -> None:
        """Promote age-0 individuals to the next generation's adults."""
        age0 = self.state[:, :, 0, :].clone()
        self.state[:, :, 1, :] = age0
        self.state[:, :, 0, :] = 0.0

    def _migration_deterministic(self) -> None:
        """Apply deterministic migration to adult (age-1) individuals."""
        # Flatten (sex, ztype) into one category axis.
        D = self.state.shape[0]
        age1 = self.state[:, :, 1, :].reshape(D, -1)
        if self.use_stencil:
            age1 = self._migration_stencil(age1, self.migration_rate)
        else:
            assert self.migration_matrix is not None
            age1 = self.migration_matrix @ age1
        self.state[:, :, 1, :] = age1.reshape_as(self.state[:, :, 1, :])

    # ------------------------------------------------------------------
    # stochastic lifecycle
    # ------------------------------------------------------------------
    def _reproduction_stochastic(self) -> None:
        """Stochastic discrete reproduction, aligned with natal semantics."""
        D, _, _, G = self.state.shape
        females = self.state[:, 0, 1, :]  # (D, G)
        males = self.state[:, 1, 1, :]    # (D, G)
        effective_males = males * self.male_adult_mating_rate

        # Mating probability matrix (D, Gf, Gm).
        A = self.sexual_selection.unsqueeze(0) * effective_males.unsqueeze(1)
        row_sum = A.sum(dim=2)
        eps = 1e-10
        P = torch.where(
            (row_sum > eps).unsqueeze(2),
            A / row_sum.clamp_min(eps).unsqueeze(2),
            torch.zeros_like(A),
        )

        # 1. How many females of each genotype mate?
        n_mating = self._sample_binomial(
            females,
            self.female_adult_mating_rate,
            self.device,
            self.dtype,
        )  # (D, Gf)

        # 2. Distribute mating females among male genotypes.
        n_flat = n_mating.reshape(-1)
        row_sum_flat = row_sum.reshape(-1)
        n_flat = torch.where(
            row_sum_flat > eps,
            n_flat,
            torch.zeros_like(n_flat),
        )
        P_flat = P.reshape(D * G, G)
        pairs_flat = self._sample_multinomial_rows(n_flat, P_flat, self.dtype)
        pairs = pairs_flat.reshape(D, G, G)

        # 3. Stochastic fertilization per (gf, gm) pair.
        pair_counts_flat = pairs.reshape(-1)

        # Pair-specific fertility factor.
        fecundity_pair = (
            self.fecundity_f.unsqueeze(1) * self.fecundity_m.unsqueeze(0)
        )  # (Gf, Gm)
        fertility_flat = (
            fecundity_pair.reshape(-1)
            .unsqueeze(0)
            .expand(D, -1)
            .reshape(-1)
            * self.eggs_per_female
        )

        # Number of reproducing pairs.
        if self.reproduction_rate >= 1.0 - eps:
            n_reproducing = pair_counts_flat
        else:
            n_reproducing = self._sample_binomial(
                pair_counts_flat,
                self.reproduction_rate,
                self.device,
                self.dtype,
            )

        total_lambda = n_reproducing * fertility_flat
        n_total = torch.distributions.Poisson(total_lambda).sample()

        # Viability through offspring tensor (usually all rows sum to 1).
        p_surv_flat = (
            self.offspring_tensor.sum(dim=2)
            .reshape(-1)
            .unsqueeze(0)
            .expand(D, -1)
            .reshape(-1)
        )
        n_viable = torch.where(
            p_surv_flat > 1.0 - eps,
            n_total,
            self._sample_binomial(
                n_total,
                p_surv_flat,
                self.device,
                self.dtype,
            ),
        )

        # Sample offspring genotype counts for every (deme, gf, gm) row.
        offspring_probs = (
            self.offspring_tensor.reshape(G * G, G)
            .unsqueeze(0)
            .expand(D, -1, -1)
            .reshape(-1, G)
        )
        offspring_counts_flat = self._sample_multinomial_rows(
            n_viable, offspring_probs, self.dtype
        )
        offspring = offspring_counts_flat.reshape(D, G, G, G).sum(dim=(1, 2))

        # 4. Sex assignment.
        if self.has_sex_chromosomes:
            denom = self.female_compat + self.male_compat
            p_f = torch.where(
                denom > eps,
                (self.female_compat / denom.clamp_min(eps)).clamp(0.0, 1.0),
                torch.full_like(denom, 0.5),
            )
            # For stochastic sex-chromosome models, sample from the applicable
            # female probability per genotype.
            p_f_row = p_f.unsqueeze(0).expand(D, -1)
            female_off = torch.where(
                self.female_only.unsqueeze(0).expand(D, -1),
                offspring,
                torch.where(
                    self.male_only.unsqueeze(0).expand(D, -1),
                    torch.zeros_like(offspring),
                    self._sample_binomial(
                        offspring,
                        p_f_row,
                        self.device,
                        self.dtype,
                    ),
                ),
            )
        else:
            female_off = self._sample_binomial(
                offspring,
                self.sex_ratio,
                self.device,
                self.dtype,
            )
        male_off = offspring - female_off

        self.state[:, 0, 0, :] = female_off
        self.state[:, 1, 0, :] = male_off

    def _survival_stochastic(self) -> None:
        """Stochastic survival with fixed density regulation and viability."""
        D, _, _, G = self.state.shape
        f0 = self.state[:, 0, 0, :]
        m0 = self.state[:, 1, 0, :]
        total0 = f0.sum(dim=1) + m0.sum(dim=1)

        mode = self.juvenile_growth_mode
        if mode == 0:
            scaling = torch.ones_like(total0)
        elif mode == FIXED_MODE:
            scaling = torch.where(
                total0 > 1e-10,
                torch.clamp(
                    self.carrying_capacity / total0.clamp_min(1e-10),
                    max=1.0,
                ),
                torch.ones_like(total0),
            )
        else:
            raise NotImplementedError(
                f"GPU stochastic model currently supports mode 0/1 only; got {mode}"
            )

        # natal's stochastic juvenile recruitment rounds counts first and then
        # draws one multinomial over the concatenated female/male categories.
        f0_int = self._round_counts(f0)
        m0_int = self._round_counts(m0)
        total_int = f0_int.sum(dim=1) + m0_int.sum(dim=1)
        desired = self._round_counts(total_int * scaling)

        counts = torch.cat([f0_int, m0_int], dim=1)  # (D, 2G)
        probs = counts / total_int.clamp_min(1.0).unsqueeze(1)
        probs = torch.where(
            (total_int > 1e-10).unsqueeze(1),
            probs,
            torch.zeros_like(probs),
        )
        recruited = self._sample_multinomial_rows(desired, probs, self.dtype)

        f_rec = recruited[:, :G]
        m_rec = recruited[:, G:]

        survival_f = (self.female_age0_survival * self.viability_f).unsqueeze(0)
        survival_m = (self.male_age0_survival * self.viability_m).unsqueeze(0)

        f_surv = self._sample_binomial(
            f_rec, survival_f.expand(D, -1), self.device, self.dtype
        )
        m_surv = self._sample_binomial(
            m_rec, survival_m.expand(D, -1), self.device, self.dtype
        )

        self.state[:, 0, 0, :] = f_surv
        self.state[:, 1, 0, :] = m_surv

    def _migration_stochastic(self) -> None:
        """Stochastic migration using one augmented multinomial per row.

        natal's stochastic migration is equivalent to: first draw the number of
        outbound individuals ``X ~ Binomial(n, rate)``, then split ``X`` over
        neighbours with a multinomial. The joint distribution of
        ``(stay, neighbour_1, ..., neighbour_K)`` is exactly

        ``Multinomial(n, [1-rate, rate*p_1, ..., rate*p_K])``

        so this implementation uses a single multinomial per (deme, category)
        row. That removes one sampling call per row and lets the conditional-
        binomial multinomial helper handle the split without any CPU sync.
        """
        D = self.state.shape[0]
        C = self.state.shape[1] * self.state.shape[3]  # sex * ztype per age
        max_nnz = self.neighbor_probs.shape[1]

        for age_idx, rate in enumerate(self.migration_rate_by_age):
            if rate <= 0.0:
                continue
            value = self.state[:, :, age_idx, :]  # (D, sex, G)
            value_flat = value.reshape(D, C)      # (D, categories)

            # Augmented probabilities: [stay, neighbour_1, ..., neighbour_K].
            # Using 1 - rate*sum(neighbour_probs) also handles the
            # adjust_migration_on_edge=False case where neighbour probabilities
            # sum to less than 1 at the boundary.
            neighbour_prob = self.neighbor_probs * rate  # (D, max_nnz)
            stay_prob = (1.0 - neighbour_prob.sum(dim=1, keepdim=True)).clamp_min(
                0.0
            )
            probs_aug = torch.cat([stay_prob, neighbour_prob], dim=1)

            # Each (deme, category) row uses its deme's neighbour distribution.
            probs_flat = (
                probs_aug.unsqueeze(1)
                .expand(D, C, -1)
                .reshape(D * C, max_nnz + 1)
            )
            totals = value_flat.reshape(-1)  # D*C

            sampled = self._sample_multinomial_rows(
                totals, probs_flat, self.dtype
            )  # (D*C, max_nnz+1)

            stay = sampled[:, 0].reshape(D, C)
            neighbour_counts = sampled[:, 1:].reshape(D, C, max_nnz)

            # Map sampled neighbour counts back to destination demes.
            dst = self.neighbor_indices.unsqueeze(1).expand(D, C, max_nnz)
            cat = (
                torch.arange(C, device=self.device)
                .unsqueeze(0)
                .expand(D, -1)
            )  # (D, C)
            cat = cat.unsqueeze(2).expand(-1, -1, max_nnz)  # (D, C, max_nnz)

            valid = dst >= 0
            index_flat = (dst * C + cat).clamp_min(0).reshape(-1)
            counts_flat = torch.where(
                valid.reshape(-1),
                neighbour_counts.reshape(-1),
                torch.zeros_like(neighbour_counts.reshape(-1)),
            )

            incoming_flat = torch.zeros(
                D * C, dtype=self.dtype, device=self.device
            )
            incoming_flat.scatter_add_(0, index_flat, counts_flat)
            incoming = incoming_flat.reshape(D, C)

            migrated = stay + incoming
            self.state[:, :, age_idx, :] = migrated.reshape_as(value)

    def step(self) -> None:
        """Advance one generation (deterministic or stochastic)."""
        if self.stochastic:
            self._reproduction_stochastic()
            self._survival_stochastic()
            self._aging()
            self._migration_stochastic()
        else:
            self._reproduction_deterministic()
            self._survival_deterministic()
            self._aging()
            self._migration_deterministic()
        self.tick += 1

    def run_no_history(self, n_ticks: int | None = None) -> None:
        """Run ticks without moving state back to the CPU each tick.

        This is the closest analogue to natal-core's ``record_every=0`` path
        for pure simulation-time comparison.
        """
        n = self.n_ticks if n_ticks is None else n_ticks
        for _ in range(n):
            self.step()
        if self.device.type == "xpu":
            torch.xpu.synchronize()

    def run_history(self) -> np.ndarray:
        """Run all ticks and return a (n_ticks+1, D, 2, 2, G) NumPy history.

        History index 0 is the initial state; indices 1..n_ticks are the
        states after each tick, matching ``reference_cpu.npy``.
        """
        history = [self.state.clone()]
        for _ in range(self.n_ticks):
            self.step()
            history.append(self.state.clone())
        return torch.stack(history, axis=0).detach().cpu().numpy()


# Backward-compatible alias for code written before stochastic support was
# added and the class was renamed.
DeterministicSpatialDiscreteXPU = SpatialDiscreteXPU


def build_gpu_from_reference(
    *,
    device: torch.device | None = None,
) -> tuple[SpatialDiscreteXPU, object]:
    """Build the GPU model from the same natal-core CPU reference definition."""
    population = reference_cpu.build_spatial_population()

    state = np.stack(
        [deme.state.individual_count for deme in population.demes], axis=0
    )
    cfg = population.deme(0).config

    model = SpatialDiscreteXPU(
        state=state,
        config=cfg,
        migration_rate=reference_cpu.MIGRATION_RATE,
        n_ticks=reference_cpu.N_TICKS,
        device=device,
        grid_shape=(reference_cpu.N_ROWS, reference_cpu.N_COLS),
        wrap=False,
        migration_kernel=reference_cpu.MIGRATION_KERNEL,
        adjust_migration_on_edge=reference_cpu.MIGRATION_ADJUST_ON_EDGE,
    )
    return model, population


if __name__ == "__main__":
    model, _cpu_pop = build_gpu_from_reference()
    hist = model.run_history()
    print("GPU history shape:", hist.shape)
    print("Final total adults:", float(hist[-1, :, :, 1, :].sum()))
    print("Final Dr frequency:",
          float(hist[-1, :, :, 1, 1].sum() + 2.0 * hist[-1, :, :, 1, 2].sum())
          / float(2.0 * hist[-1, :, :, 1, :].sum()))
