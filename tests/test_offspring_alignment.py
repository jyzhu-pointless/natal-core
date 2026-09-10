"""Alignment tests: the folded ``offspring_tensor`` is exactly the
meiosis-by-fertilization contraction.

The inheritance stage has two equivalent representations:

- the two-step view: ``zygotes_to_gametes_map`` (per-sex meiosis
  probabilities, ``(2, Z, G)``) plus ``gametes_to_zygotes_map``
  (gamete-pair to zygote fusion, ``(G, G, Z)``) — the representation
  presets and modifiers actually read and rewrite;
- the folded view: ``offspring_tensor`` (``(Z, Z, Z)``) indexed by
  ``(mother, father, offspring)`` — the precomputed table the
  reproduction kernels consume.

The contract under test::

    P[i, j, k] = sum_{a, b} meiosis_f[i, a] * meiosis_m[j, b] * g2z[a, b, k]

is checked with an independent einsum oracle (which never touches
``offspring_tensor``) at four layers:

1. the production fold (``recompute_offspring_tensor``, the Rust-backed
   single owner of the derivation) on hand-built maps, including
   asymmetric (drive-biased) meiosis and zygote-lethal fusion tables
   whose rows sum to less than one;
2. the same identity on a PopulationBuilder-built draft's maps;
3. the Wright-Fisher fused kernel (Rust, deterministic mode 3) must
   reproduce the map-direct contraction weighted by the replicated pair
   weights — i.e. consuming the folded tensor is semantically identical
   to consuming the two maps directly;
4. after a Toxin-Antidote drive preset (with a Cas9 deposition gamete
   label) rewrites both stages, the fold identity and the kernel
   alignment must still hold — the regression guard behind "edit the
   maps, the fold follows".
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from natal.backends.rust.rust_backend import (
    RustDiscreteLifecycleBackend,
    rust_backend_available,
)
from natal.frontend.model import ModelDraft
from natal.frontend.data import DiscretePopulationState
from natal.frontend.genetics.matrices import recompute_offspring_tensor
from natal.frontend.genetics import Species
from natal.frontend.hooks.types import HookProgram
from natal.frontend.population.discrete_generation import (
    DiscreteGenerationPopulation,
)
from natal.frontend.presets import ToxinAntidoteDrive

_requires_rust = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


def _empty_hook_program() -> HookProgram:
    return HookProgram(
        n_events=np.int32(4),
        n_hooks=np.int32(0),
        hook_offsets=np.zeros(5, dtype=np.int64),
        n_ops_list=np.zeros(0, dtype=np.int64),
        op_offsets=np.zeros(1, dtype=np.int64),
        op_types_data=np.zeros(0, dtype=np.int64),
        zidx_offsets_data=np.zeros(1, dtype=np.int64),
        zidx_data=np.zeros(0, dtype=np.int64),
        age_offsets_data=np.zeros(1, dtype=np.int64),
        age_data=np.zeros(0, dtype=np.int64),
        sex_masks_data=np.zeros(0, dtype=np.float64),
        params_data=np.zeros(0, dtype=np.float64),
        condition_offsets_data=np.zeros(1, dtype=np.int64),
        condition_types_data=np.zeros(0, dtype=np.int64),
        condition_params_data=np.zeros(0, dtype=np.int64),
        deme_selector_types=np.zeros(0, dtype=np.int64),
        deme_selector_offsets=np.zeros(1, dtype=np.int64),
        deme_selector_data=np.zeros(0, dtype=np.int64),
    )


# ----------------------------------------------------------- oracle ------
def _two_step_tensor(
    meiosis_f: NDArray[np.float64],
    meiosis_m: NDArray[np.float64],
    g2z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Independent contraction oracle: P[i, j, k] = Σ_ab mf·mm·g2z."""
    spread: NDArray[np.float64] = np.einsum(
        "ia,abk->ibk", meiosis_f, g2z, optimize=True
    )
    return np.einsum("jb,ibk->ijk", meiosis_m, spread, optimize=True)


def _two_step_row(
    meiosis_f_row: NDArray[np.float64],
    meiosis_m_row: NDArray[np.float64],
    g2z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Single-pair oracle row: P[i, j, :] for one (mother, father) pair."""
    return np.einsum(
        "a,b,abk->k", meiosis_f_row, meiosis_m_row, g2z, optimize=True
    )


def _mendelian_meiosis(n_haplotypes: int) -> NDArray[np.float64]:
    """Mendelian segregation rows for unordered diploid pairs of *n* haplotypes."""
    n = n_haplotypes
    z = n * (n + 1) // 2
    ia, ib = np.triu_indices(n)
    rows = np.zeros((z, n), dtype=np.float64)
    for pair, (a, b) in enumerate(zip(ia.tolist(), ib.tolist(), strict=True)):
        if a == b:
            rows[pair, a] = 1.0
        else:
            rows[pair, a] = 0.5
            rows[pair, b] = 0.5
    return rows


def _onehot_fusion(
    n_haplotypes: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """One-hot fusion table plus the haplotype-pair index lookup."""
    n = n_haplotypes
    z = n * (n + 1) // 2
    ia, ib = np.triu_indices(n)
    pair_index = np.full((n, n), -1, dtype=np.int64)
    pair_index[ia, ib] = np.arange(z)
    pair_index = np.maximum(pair_index, pair_index.T)
    g2z = np.zeros((n, n, z), dtype=np.float64)
    g2z[np.arange(n)[:, None], np.arange(n)[None, :], pair_index] = 1.0
    return g2z, pair_index


def _wf_expected_from_maps(cfg: ModelDraft) -> NDArray[np.float64]:
    """Deterministic WF tick output computed from the maps only.

    Replicates the fused Wright-Fisher tick (``extreme_speed_mode=3``):
    pair weights from fecundity / adult mating rates / sexual selection,
    gamete-pool contraction through the fusion table, sex split, age-0
    viability.  The new generation is placed directly at the adult age,
    matching the kernel.  Never reads ``cfg.offspring_tensor``.
    """
    assert cfg.juvenile_growth_mode == 0, "oracle assumes no density regulation"
    z = cfg.n_ztypes
    adult_f = cfg.initial_individual_count[0, 1, :]
    adult_m = cfg.initial_individual_count[1, 1, :]
    nf = adult_f * cfg.fecundity_fitness[0] * cfg.age_based_mating_rates[0, 1]
    nm_eff = adult_m * cfg.age_based_mating_rates[1, 1]
    ss = cfg.sexual_selection_fitness
    rowsum = ss @ nm_eff
    weights = (nf[:, None] * ss * (nm_eff * cfg.fecundity_fitness[1])[None, :]) / rowsum[:, None]
    meiosis_f = cfg.zygotes_to_gametes_map[0]
    meiosis_m = cfg.zygotes_to_gametes_map[1]
    pair_weights: NDArray[np.float64] = meiosis_f.T @ (weights @ meiosis_m)
    total: NDArray[np.float64] = (
        cfg.eggs_per_female
        * cfg.age_based_reproduction_rates[1]
        * np.einsum(
            "ab,abk->k", pair_weights, cfg.gametes_to_zygotes_map, optimize=True
        )
    )
    out = np.zeros((2, 2, z), dtype=np.float64)
    out[0, 1, :] = total * cfg.sex_ratio * cfg.viability_fitness[0, 0, :]
    out[1, 1, :] = total * (1.0 - cfg.sex_ratio) * cfg.viability_fitness[1, 0, :]
    return out


def _ztype_index(cfg: ModelDraft, genotype: str) -> int:
    """Name-directory lookup: index of the first ztype with this genotype."""
    for idx, name in enumerate(cfg.ztype_names):
        if name.split(":")[0] == genotype:
            return idx
    raise AssertionError(f"genotype {genotype!r} not in the name directory")


def _single_pair_state(cfg: ModelDraft) -> NDArray[np.float64]:
    """Adult population consisting of one A|B female and A|B male group."""
    idx = _ztype_index(cfg, "A|B")
    ind = np.zeros((2, 2, cfg.n_ztypes), dtype=np.float64)
    ind[0, 1, idx] = 40.0
    ind[1, 1, idx] = 40.0
    return ind


# --------------------------------------------------------- fixtures ------
def _build_plain_population(name: str) -> DiscreteGenerationPopulation:
    species = Species.from_dict(
        name="OffspringAlignmentSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return (
        DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"A|A": 40.0, "A|B": 20.0},
                "male": {"A|B": 30.0, "B|B": 10.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(
            eggs_per_female=20.0,
            female_adult_mating_rate=1.0,
            male_adult_mating_rate=1.0,
        )
        .competition(growth_mode="no_competition")
        .build()
    )


def _build_ta_population(
    name: str, with_preset: bool
) -> DiscreteGenerationPopulation:
    species = Species.from_dict(
        name="OffspringAlignmentTASpecies",
        structure={"chr1": {"loc": ["WT", "Drive", "Disrupted"]}},
        gamete_labels=["default", "cas9_deposited"],
    )
    chain = DiscreteGenerationPopulation.setup(
        species=species, name=name, stochastic=False
    )
    if with_preset:
        chain = chain.presets(
            ToxinAntidoteDrive(
                name="TA_Alignment",
                drive_allele="Drive",
                target_allele="WT",
                disrupted_allele="Disrupted",
                conversion_rate=0.9,
                embryo_disruption_rate=0.6,
                cas9_deposition_glab="cas9_deposited",
            )
        )
    return (
        chain.initial_state(
            individual_count={
                "female": {"WT|WT": 40.0, "WT|Drive": 20.0},
                "male": {"WT|WT": 30.0, "Drive|Drive": 10.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(
            eggs_per_female=20.0,
            female_adult_mating_rate=1.0,
            male_adult_mating_rate=1.0,
        )
        .competition(growth_mode="no_competition")
        .build()
    )


@pytest.fixture(scope="module")
def plain_config() -> ModelDraft:
    return _build_plain_population("align_plain").config


# ------------------------------------------------- layer 1: hand maps ----
def test_fold_matches_independent_contraction() -> None:
    meiosis = _mendelian_meiosis(2)
    g2z, _ = _onehot_fusion(2)
    folded = recompute_offspring_tensor(np.stack([meiosis, meiosis]), g2z)
    np.testing.assert_allclose(
        folded, _two_step_tensor(meiosis, meiosis, g2z), rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(folded.sum(axis=2), 1.0, rtol=0.0, atol=1e-15)
    # The classic Mendelian cross: Aa x Aa -> 1/4, 1/2, 1/4.
    np.testing.assert_allclose(folded[1, 1], [0.25, 0.5, 0.25], rtol=0.0, atol=1e-15)


def test_fold_matches_contraction_with_asymmetric_meiosis() -> None:
    meiosis_f = _mendelian_meiosis(2)
    meiosis_f[1, :] = np.array([0.9, 0.1])  # drive-biased female segregation
    meiosis_m = _mendelian_meiosis(2)
    g2z, _ = _onehot_fusion(2)
    folded = recompute_offspring_tensor(
        np.stack([meiosis_f, meiosis_m]), g2z
    )
    np.testing.assert_allclose(
        folded, _two_step_tensor(meiosis_f, meiosis_m, g2z), rtol=0.0, atol=1e-15
    )


def test_fold_preserves_zygote_lethality_semantics() -> None:
    meiosis = _mendelian_meiosis(2)
    g2z, _ = _onehot_fusion(2)
    g2z = g2z.copy()
    g2z[1, 1, 2] = 0.4  # a x a cross: aa zygotes 60% lethal
    folded = recompute_offspring_tensor(np.stack([meiosis, meiosis]), g2z)
    np.testing.assert_allclose(
        folded, _two_step_tensor(meiosis, meiosis, g2z), rtol=0.0, atol=1e-15
    )
    # Row sums drop below one exactly where the fusion table is lethal;
    # the kernels read that deficit as the per-pair zygote survival rate.
    expected_row_sums: NDArray[np.float64] = np.einsum(
        "ia,jb,ab->ij", meiosis, meiosis, g2z.sum(axis=2), optimize=True
    )
    np.testing.assert_allclose(
        folded.sum(axis=2), expected_row_sums, rtol=0.0, atol=1e-15
    )
    assert folded[2, 2, 2] == pytest.approx(0.4)


# --------------------------------------- layer 2: production draft maps --
def test_production_draft_fold_matches_contraction(
    plain_config: ModelDraft,
) -> None:
    cfg = plain_config
    oracle = _two_step_tensor(
        cfg.zygotes_to_gametes_map[0],
        cfg.zygotes_to_gametes_map[1],
        cfg.gametes_to_zygotes_map,
    )
    np.testing.assert_allclose(
        cfg.offspring_tensor, oracle, rtol=1e-12, atol=1e-15
    )


# ------------------------------- layer 3: WF kernel consumes the maps ----
@_requires_rust
def test_rust_wf_tick_aligns_with_map_direct_oracle(
    plain_config: ModelDraft,
) -> None:
    cfg = plain_config._replace(extreme_speed_mode=3)
    state = DiscretePopulationState(
        n_tick=0, individual_count=cfg.initial_individual_count.copy()
    )
    expected = _wf_expected_from_maps(cfg)
    backend = RustDiscreteLifecycleBackend(cfg, _empty_hook_program(), seed=0)
    backend.set_state(state)
    backend.run(n_steps=1, record_every=0)
    tick, ind_flat = backend.state_snapshot()
    rust_state = DiscretePopulationState(
        n_tick=tick, individual_count=ind_flat.reshape(state.individual_count.shape)
    )
    np.testing.assert_allclose(
        rust_state.individual_count, expected, rtol=1e-10, atol=1e-8
    )


def _standard_tick_expected_from_maps(
    cfg: ModelDraft, female: str, male: str
) -> NDArray[np.float64]:
    """Deterministic standard discrete tick output from the maps only.

    Single-genotype mating pair: every female of genotype *female* mates
    (adult mating rate 1.0) with the only male genotype *male*; the fold
    row supplies the offspring distribution; survival and age-0 viability
    default to the configured (all-one) tables; juveniles become adults.
    """
    i = _ztype_index(cfg, female)
    j = _ztype_index(cfg, male)
    n_pairs = 40.0 * cfg.age_based_mating_rates[0, 1]
    n_total = (
        n_pairs
        * cfg.age_based_reproduction_rates[1]
        * cfg.eggs_per_female
        * cfg.fecundity_fitness[0, i]
        * cfg.fecundity_fitness[1, j]
    )
    row = _two_step_row(
        cfg.zygotes_to_gametes_map[0][i],
        cfg.zygotes_to_gametes_map[1][j],
        cfg.gametes_to_zygotes_map,
    )
    total = row * n_total
    out = np.zeros((2, 2, cfg.n_ztypes), dtype=np.float64)
    out[0, 1, :] = (
        total
        * cfg.sex_ratio
        * cfg.viability_fitness[0, 0, :]
        * cfg.age_based_survival_rates[0, 0]
    )
    out[1, 1, :] = (
        total
        * (1.0 - cfg.sex_ratio)
        * cfg.viability_fitness[1, 0, :]
        * cfg.age_based_survival_rates[1, 0]
    )
    return out


@_requires_rust
def test_rust_standard_tick_aligns_with_map_direct_oracle(
    plain_config: ModelDraft,
) -> None:
    cfg = plain_config._replace(
        initial_individual_count=_single_pair_state(plain_config)
    )
    state = DiscretePopulationState(
        n_tick=0, individual_count=cfg.initial_individual_count.copy()
    )
    expected = _standard_tick_expected_from_maps(cfg, "A|B", "A|B")
    backend = RustDiscreteLifecycleBackend(cfg, _empty_hook_program(), seed=0)
    backend.set_state(state)
    backend.run(n_steps=1, record_every=0)
    tick, ind_flat = backend.state_snapshot()
    rust_state = DiscretePopulationState(
        n_tick=tick, individual_count=ind_flat.reshape(state.individual_count.shape)
    )
    np.testing.assert_allclose(
        rust_state.individual_count, expected, rtol=1e-10, atol=1e-8
    )


# --------------------------- layer 4: drive preset rewrites both stages --
def test_drive_preset_rewrites_both_stages_and_keeps_fold_identity() -> None:
    cfg = _build_ta_population("align_ta", with_preset=True).config
    plain = _build_ta_population("align_ta_plain", with_preset=False).config

    # The Cas9 deposition gamete label is present, and both stages were
    # actually rewritten relative to the preset-free baseline.
    assert cfg.n_glabs == 2
    assert not np.allclose(
        cfg.zygotes_to_gametes_map, plain.zygotes_to_gametes_map
    )
    assert not np.allclose(cfg.gametes_to_zygotes_map, plain.gametes_to_zygotes_map)

    oracle = _two_step_tensor(
        cfg.zygotes_to_gametes_map[0],
        cfg.zygotes_to_gametes_map[1],
        cfg.gametes_to_zygotes_map,
    )
    np.testing.assert_allclose(
        cfg.offspring_tensor, oracle, rtol=1e-10, atol=1e-12
    )
    # Embryo disruption redistributes probability (WT offspring become
    # Disrupted) rather than removing mass, so the folded rows stay
    # normalized here; the lethal row-deficit semantics are covered by
    # the hand-built fusion table in the lethality test above.
    row_sums = cfg.offspring_tensor.sum(axis=2)
    assert (row_sums <= 1.0 + 1e-12).all()


@_requires_rust
def test_rust_wf_tick_aligns_after_drive_preset() -> None:
    cfg = _build_ta_population("align_ta_wf_rust", with_preset=True).config
    cfg = cfg._replace(extreme_speed_mode=3)
    state = DiscretePopulationState(
        n_tick=0, individual_count=cfg.initial_individual_count.copy()
    )
    expected = _wf_expected_from_maps(cfg)
    backend = RustDiscreteLifecycleBackend(cfg, _empty_hook_program(), seed=0)
    backend.set_state(state)
    backend.run(n_steps=1, record_every=0)
    tick, ind_flat = backend.state_snapshot()
    rust_state = DiscretePopulationState(
        n_tick=tick, individual_count=ind_flat.reshape(state.individual_count.shape)
    )
    np.testing.assert_allclose(
        rust_state.individual_count, expected, rtol=1e-10, atol=1e-8
    )
