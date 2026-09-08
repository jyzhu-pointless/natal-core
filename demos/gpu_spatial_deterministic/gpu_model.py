"""PyTorch XPU implementation of the deterministic spatial discrete model.

This is intentionally aligned with the CPU reference in ``reference_cpu.py``:

- 25 demes on a 5x5 square grid
- ``stochastic=False`` discrete-generation lifecycle
- fixed juvenile density regulation
- row-normalized adjacency migration

The implementation uses only standard PyTorch tensor operations so it can run
on ``xpu`` locally and later on ``cuda`` with a device change.

This demo file is outside natal-core's strict ``src`` type-check scope, and it
dynamically reads natal config objects that Pylance cannot fully infer.
"""

# pyright: reportUndefinedVariable=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportCallIssue=false, reportUntypedFunctionDecorator=false, reportAttributeAccessIssue=false

from __future__ import annotations

import numpy as np
import torch

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
        adjacency: np.ndarray,
        *,
        migration_rate: float,
        n_ticks: int,
        device: torch.device | None = None,
        stochastic: bool = False,
        seed: int | None = None,
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

        # --- migration matrix ------------------------------------------------
        # natal's deterministic adjacency migration is:
        #   next[d] = (1-rate) * current[d]
        #             + rate * sum_s current[s] * adj[s, d]
        # which in column-vector form is:
        #   M = (1-rate) I + rate * adj^T
        self.adjacency = torch.as_tensor(
            adjacency, dtype=self.dtype, device=device
        )
        adj = self.adjacency
        eye = torch.eye(adj.shape[0], dtype=self.dtype, device=device)
        self.migration_matrix = (
            (1.0 - self.migration_rate) * eye
            + self.migration_rate * adj.T
        )

        # Sparse row data for stochastic migration.  natal's stochastic
        # migration only routes outbound mass to actual neighbours, so using a
        # dense D x D multinomial is wasteful.
        adj_np = np.asarray(adjacency, dtype=np.float64)
        n_demes = int(state.shape[0])
        neighbor_lists: list[np.ndarray] = []
        prob_lists: list[np.ndarray] = []
        for i in range(n_demes):
            nz = np.flatnonzero(adj_np[i] > 0.0)
            neighbor_lists.append(nz.astype(np.int64))
            row = adj_np[i, nz]
            if row.sum() > 0.0:
                row = row / row.sum()
            prob_lists.append(row.astype(np.float32))
        max_nnz = max((len(x) for x in neighbor_lists), default=0)
        neighbor_indices = np.full((n_demes, max_nnz), -1, dtype=np.int64)
        neighbor_probs = np.zeros((n_demes, max_nnz), dtype=np.float32)
        for i in range(n_demes):
            n = len(neighbor_lists[i])
            neighbor_indices[i, :n] = neighbor_lists[i]
            neighbor_probs[i, :n] = prob_lists[i]
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
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample independent Multinomial rows with different per-row totals.

        PyTorch's ``torch.distributions.Multinomial`` does not support
        inhomogeneous per-row total counts, so this helper implements the
        equivalent multinomial sampling manually:

        1. Find ``max_total``, the largest per-row total.
        2. Draw ``max_total`` Categorical samples for every row.
        3. Keep only the first ``total[i]`` draws for row ``i``.
        4. Count the kept category draws with ``scatter_add_``.

        This is correct but costs ``O(max_total * n_rows)`` random draws and
        temporary memory, which is a major reason the stochastic XPU prototype
        is slow.

        Args:
            totals: 1-D tensor of total counts per row.
            probs: 2-D tensor ``(n_rows, n_categories)``.

        Returns:
            Count tensor with the same shape as *probs*.
        """
        totals = SpatialDiscreteXPU._round_counts(totals)
        n_rows, n_categories = probs.shape
        row_sum = probs.sum(dim=1)
        active = (totals > 0.0) & (row_sum > 1e-10)

        safe_probs = probs.clone()
        safe_probs = safe_probs / safe_probs.sum(dim=1, keepdim=True).clamp_min(1e-10)
        safe_probs = torch.where(
            active.unsqueeze(1), safe_probs, torch.zeros_like(safe_probs)
        )
        # Defensive cleanup: some XPU reduction edge cases can leave NaNs in
        # otherwise inactive rows; make inactive rows safe for Categorical.
        safe_probs = torch.nan_to_num(
            safe_probs, nan=0.0, posinf=1.0, neginf=0.0
        )
        zero_rows = safe_probs.sum(dim=1) <= 1e-10
        if bool(zero_rows.any().cpu().item()):
            safe_probs[zero_rows, 0] = 1.0

        max_total = int(totals.max().item()) if totals.numel() > 0 else 0
        if max_total <= 0:
            return torch.zeros_like(probs)

        # Sample a Categorical draw for every row enough times to cover each
        # row's total, then mask draws beyond that row's total and count them.
        cat = torch.distributions.Categorical(probs=safe_probs)
        draws = cat.sample((max_total,))  # (max_total, n_rows)

        row_pos = torch.arange(max_total, device=device).unsqueeze(1)
        valid = row_pos < totals.unsqueeze(0)  # (max_total, n_rows)
        draws = torch.where(valid, draws, torch.zeros_like(draws))

        row_ids = (
            torch.arange(n_rows, device=device)
            .unsqueeze(0)
            .expand(max_total, -1)
            .reshape(-1)
        )
        cat_ids = draws.reshape(-1)
        flat_index = row_ids * n_categories + cat_ids
        flat_valid = valid.reshape(-1).to(dtype)

        flat_counts = torch.zeros(
            n_rows * n_categories, dtype=dtype, device=device
        )
        flat_counts.scatter_add_(0, flat_index, flat_valid)
        return flat_counts.reshape(n_rows, n_categories)

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
        # Flatten (sex, ztype) into one category axis for a clean matmul.
        D = self.state.shape[0]
        age1 = self.state[:, :, 1, :].reshape(D, -1)
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
        pairs_flat = self._sample_multinomial_rows(
            n_flat, P_flat, self.device, self.dtype
        )
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
            n_viable, offspring_probs, self.device, self.dtype
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
        recruited = self._sample_multinomial_rows(
            desired, probs, self.device, self.dtype
        )

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
        """Stochastic migration using Binomial outbound + sparse Multinomial split."""
        D = self.state.shape[0]
        C = self.state.shape[1] * self.state.shape[3]  # sex * ztype per age
        max_nnz = self.neighbor_probs.shape[1]

        for age_idx, rate in enumerate(self.migration_rate_by_age):
            if rate <= 0.0:
                continue
            value = self.state[:, :, age_idx, :]  # (D, sex, G)
            value_flat = value.reshape(D, C)      # (D, categories)

            outbound = self._sample_binomial(
                value_flat,
                rate,
                self.device,
                self.dtype,
            )  # (D, C)
            stay = value_flat - outbound

            # Flatten rows as (source, category).  Each source deme uses its
            # own sparse neighbour probability vector.
            totals = outbound.reshape(-1)  # D*C
            probs = self.neighbor_probs.repeat_interleave(C, dim=0)  # (D*C, max_nnz)
            sampled = self._sample_multinomial_rows(
                totals, probs, self.device, self.dtype
            )  # (D*C, max_nnz)

            # Map sampled neighbour counts back to destination demes.
            dst = self.neighbor_indices.repeat_interleave(C, dim=0)  # (D*C, max_nnz)
            cat = (
                torch.arange(C, device=self.device)
                .unsqueeze(0)
                .expand(D, -1)
                .reshape(-1)
            )  # (D*C,)
            cat = cat.unsqueeze(1).expand(-1, max_nnz)  # (D*C, max_nnz)

            valid = dst >= 0
            index_flat = (dst * C + cat).clamp_min(0).reshape(-1)
            counts_flat = torch.where(
                valid.reshape(-1), sampled.reshape(-1), torch.zeros_like(sampled.reshape(-1))
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
        history = [self.state.detach().cpu().numpy().copy()]
        for _ in range(self.n_ticks):
            self.step()
            history.append(self.state.detach().cpu().numpy().copy())
        return np.stack(history, axis=0)


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
    topology = reference_cpu.SquareGrid(
        rows=reference_cpu.N_ROWS, cols=reference_cpu.N_COLS
    )
    adjacency = reference_cpu.build_adjacency_matrix(
        topology, row_normalize=True
    )

    model = SpatialDiscreteXPU(
        state=state,
        config=cfg,
        adjacency=adjacency,
        migration_rate=reference_cpu.MIGRATION_RATE,
        n_ticks=reference_cpu.N_TICKS,
        device=device,
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
