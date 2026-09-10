"""Materialization: build-time draft -> contract data.

``build()`` calls :func:`materialize` once to turn the build-time draft
(:class:`~natal.frontend.data.config.ModelDraft`) into the two runtime
artefacts:

- a :class:`~natal.contracts.blueprint.Blueprint` holding the frozen
  model specification (dimensions, flags, name directory, masks,
  initial state);
- a :class:`~natal.contracts.params.Params` owning every
  runtime-mutable value, with arrays **copied** out of the draft so the
  contract owns its data outright and the draft can retire.

Ownership discipline: every contract array is a fresh copy (or a
freshly derived array); nothing aliases the draft.  Mutating the draft
after materialization therefore cannot leak into a live population —
the reverse of the legacy share-by-reference behavior, which existed
only because the draft *was* the runtime carrier.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from natal.contracts.blueprint import Blueprint, format_type_name, frozen
from natal.contracts.params import CustomValue, Params
from natal.frontend.data.config import ModelDraft

__all__ = ["Materialized", "SpatialMigration", "materialize", "materialize_params"]


class Materialized(NamedTuple):
    """Pair of (blueprint, params) with named access."""

    blueprint: Blueprint
    params: Params


class SpatialMigration(NamedTuple):
    """Build-time migration payload folded into the contracts.

    Everything spatial about migration — topology resolution, adjacency,
    kernel selection (including per-deme kernel banks), kernel-center
    handling, and boundary normalization — is resolved by the frontend
    into three CSR arrays before this point.  ``rate`` is the runtime
    (n_demes, n_sexes, n_ages) migration-rate column consumed by the
    engines together with the frozen CSR.

    Attributes:
        indptr: ``(n_demes + 1,)`` int64 CSR row pointer.
        dest_idx: ``(nnz,)`` int64 destination deme per entry.
        weights: ``(nnz,)`` float64 normalized outbound weight per entry.
        rate: ``(n_demes, n_sexes, n_ages)`` float64 migration rate.
    """

    indptr: NDArray[np.int64]
    dest_idx: NDArray[np.int64]
    weights: NDArray[np.float64]
    rate: NDArray[np.float64]


def _custom_slots(draft: ModelDraft) -> dict[str, CustomValue]:
    """Copy user custom slots out of the draft's slot dict.

    Args:
        draft: The built draft whose ``custom`` field carries the
            registered custom slots (empty dict when none).

    Returns:
        A fresh ``{name: value}`` mapping owning its storage: scalars
        are native Python values, arrays come back as float64 copies.
    """
    slots: dict[str, CustomValue] = {}
    for name, value in draft.custom.items():
        if isinstance(value, np.generic):
            # Normalize NumPy scalars (e.g. from set_param writes) to
            # native Python values.
            slots[str(name)] = value.item()
        elif isinstance(value, np.ndarray):
            slots[str(name)] = np.array(value, dtype=np.float64, order="C")
        else:
            slots[str(name)] = value
    return slots


def _params(draft: ModelDraft, migration_rate: NDArray[np.float64]) -> Params:
    """Build the Params contract, copying every array out of the draft.

    Args:
        draft: The built draft.
        migration_rate: Owned ``(n_demes, n_sexes, n_ages)`` rate column
            (a fresh copy is made; panmictic callers pass the all-zero
            single-deme column).

    Returns:
        A fully owned ``Params`` instance.
    """
    equilibrium = getattr(draft, "equilibrium_individual_distribution", None)
    if equilibrium is None:
        equilibrium = np.zeros((0, 0), dtype=np.float64)
    else:
        equilibrium = np.array(equilibrium, dtype=np.float64, order="C")
    external_eggs = draft.external_expected_eggs
    return Params(
        carrying_capacity=float(draft.carrying_capacity),
        eggs_per_female=float(draft.eggs_per_female),
        sex_ratio=float(draft.sex_ratio),
        sperm_displacement_rate=float(draft.sperm_displacement_rate),
        low_density_growth_rate=float(draft.low_density_growth_rate),
        growth_mode=int(draft.juvenile_growth_mode),
        external_expected_eggs=(
            float(external_eggs) if external_eggs is not None else -1.0
        ),
        survival_rates=np.array(draft.age_based_survival_rates, dtype=np.float64, order="C"),
        mating_rates=np.array(draft.age_based_mating_rates, dtype=np.float64, order="C"),
        reproduction_rates=np.array(
            draft.age_based_reproduction_rates, dtype=np.float64, order="C"
        ),
        fertility=np.array(draft.female_age_based_fertility, dtype=np.float64, order="C"),
        competition_weights=np.array(
            draft.age_based_relative_competition_strength, dtype=np.float64, order="C"
        ),
        equilibrium_distribution=equilibrium,
        migration_rate=np.array(migration_rate, dtype=np.float64, order="C"),
        custom_slots=_custom_slots(draft),
        viability_fitness=np.array(draft.viability_fitness, dtype=np.float64, order="C"),
        fecundity_fitness=np.array(draft.fecundity_fitness, dtype=np.float64, order="C"),
        sexual_selection_fitness=np.array(
            draft.sexual_selection_fitness, dtype=np.float64, order="C"
        ),
        zygote_viability_fitness=np.array(
            draft.zygote_viability_fitness, dtype=np.float64, order="C"
        ),
        offspring_tensor=np.array(draft.offspring_tensor, dtype=np.float64, order="C"),
        meiosis_map=np.array(draft.zygotes_to_gametes_map, dtype=np.float64, order="C"),
        female_ztype_compatibility=np.array(
            draft.female_ztype_compatibility, dtype=np.float64, order="C"
        ),
        male_ztype_compatibility=np.array(
            draft.male_ztype_compatibility, dtype=np.float64, order="C"
        ),
    )


def _blueprint(
    draft: ModelDraft,
    migration: SpatialMigration | None,
) -> Blueprint:
    """Build the frozen Blueprint from the draft.

    Args:
        draft: The built draft.
        migration: Spatial CSR payload, or ``None`` for a panmictic
            blueprint (one deme, empty CSR).

    Returns:
        A ``Blueprint`` with copied frozen arrays and the symbolic name
        directory.
    """
    sperm = draft.initial_sperm_storage
    if migration is None:
        n_demes = 1
        indptr = np.zeros(0, dtype=np.int64)
        dest_idx = np.zeros(0, dtype=np.int64)
        weights = np.zeros(0, dtype=np.float64)
    else:
        n_demes = int(migration.rate.shape[0])
        indptr = np.array(migration.indptr, dtype=np.int64, order="C")
        dest_idx = np.array(migration.dest_idx, dtype=np.int64, order="C")
        weights = np.array(migration.weights, dtype=np.float64, order="C")
    return Blueprint(
        n_sexes=int(draft.n_sexes),
        n_ages=int(draft.n_ages),
        n_ztypes=int(draft.n_ztypes),
        n_gtypes=int(draft.n_gtypes),
        n_glabs=int(draft.n_glabs),
        new_adult_age=int(draft.new_adult_age),
        adult_ages=frozen(np.array(draft.adult_ages, dtype=np.int64, order="C")),
        stochastic=bool(draft.stochastic),
        continuous_sampling=bool(draft.continuous_sampling),
        fixed_egg_count=bool(draft.fixed_egg_count),
        has_sex_chromosomes=bool(draft.has_sex_chromosomes),
        extreme_speed_mode=int(draft.extreme_speed_mode),
        ztype_names=tuple(draft.ztype_names),
        gtype_names=tuple(draft.gtype_names),
        female_only_by_sex_chrom=frozen(
            np.array(draft.female_only_by_sex_chrom, dtype=np.bool_, order="C")
        ),
        male_only_by_sex_chrom=frozen(
            np.array(draft.male_only_by_sex_chrom, dtype=np.bool_, order="C")
        ),
        initial_individual_count=frozen(
            np.array(draft.initial_individual_count, dtype=np.float64, order="C")
        ),
        initial_sperm_storage=(
            frozen(np.array(sperm, dtype=np.float64, order="C"))
            if sperm.size
            else frozen(np.zeros((0,)))
        ),
        n_demes=n_demes,
        migration_indptr=frozen(indptr),
        migration_dest_idx=frozen(dest_idx),
        migration_weights=frozen(weights),
    )


def materialize(
    draft: ModelDraft,
    migration: SpatialMigration | None = None,
) -> Materialized:
    """Turn a built draft into (blueprint, params).

    Args:
        draft: A fully built :class:`ModelDraft` (the build-time
            scratch produced by the builder factories).
        migration: Optional spatial migration payload (CSR + rate
            column).  ``None`` materializes a panmictic contract: one
            deme, empty CSR, an all-zero (1, n_sexes, n_ages) rate.

    Returns:
        A :class:`Materialized` pair owning fresh copies of every
        array; the draft is safe to discard afterwards.
    """
    rate = _copy_migration_rate(draft, None if migration is None else migration.rate)
    return Materialized(
        blueprint=_blueprint(draft, migration),
        params=_params(draft, rate),
    )


def materialize_params(
    draft: ModelDraft,
    migration_rate: NDArray[np.float64] | None = None,
) -> Params:
    """Build an owned runtime ``Params`` projection from a draft.

    This entry point is for runtime parameter refreshes that do not need a
    rebuilt ``Blueprint``.  It keeps the same fresh-copy ownership contract
    as :func:`materialize` while avoiding construction of frozen metadata.

    Args:
        draft: The current build-time configuration draft.
        migration_rate: Optional owned migration-rate column.  When omitted,
            a zero single-deme column is used.

    Returns:
        A fully owned ``Params`` instance.
    """
    return _params(draft, _copy_migration_rate(draft, migration_rate))


def _copy_migration_rate(
    draft: ModelDraft, migration_rate: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Return an owned migration-rate column for Params-only materialization."""
    if migration_rate is None:
        return np.zeros((1, int(draft.n_sexes), int(draft.n_ages)), dtype=np.float64)
    return np.array(migration_rate, dtype=np.float64, order="C")


def ztype_names_from_registry(
    index_to_ztype: Sequence[tuple[object, str]],
) -> tuple[str, ...]:
    """Build the ztype name directory from registry index tuples.

    Args:
        index_to_ztype: ``(genotype, slab_label)`` pairs in index order.

    Returns:
        Canonical ``"<genotype>:<slab>"`` strings.
    """
    return tuple(format_type_name(gt, slab) for gt, slab in index_to_ztype)


def gtype_names_from_registry(
    index_to_gtype: Sequence[tuple[object, str]],
) -> tuple[str, ...]:
    """Build the gtype name directory from registry index tuples.

    Args:
        index_to_gtype: ``(haploid_genotype, glab_label)`` pairs in
            index order.

    Returns:
        Canonical ``"<haplotype>:<glab>"`` strings.
    """
    return tuple(format_type_name(hg, glab) for hg, glab in index_to_gtype)
