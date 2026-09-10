"""Spatial-dashboard serialization: landscape, deme detail, migration.

Every function is a pure read over the :class:`SpatialPopulation`; handlers
in ``rest.py`` hold the engine mutex so these views stay consistent with the
running tick loop.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import HexGrid, SquareGrid

from .serialization import (
    GenotypeStateRow,
    HistorySeries,
    compute_allele_frequencies,
    genotype_rows_builder,
    known_allele_names,
)

# ---------------------------------------------------------------------------
# Landscape
# ---------------------------------------------------------------------------


class SpatialTopologyInfo(TypedDict):
    """Grid layout metadata for the landscape map."""

    kind: str  # "hex" | "square" | "none"
    rows: int
    cols: int
    wrap: bool
    grid_ij: list[list[int]] | None  # per-deme (row, col)
    xy: list[list[float]] | None  # per-deme geometric (x, y)


class SpatialLandscapePayload(TypedDict):
    """One snapshot of per-deme metrics over the landscape layout."""

    n_demes: int
    topology: SpatialTopologyInfo
    deme_names: list[str]
    totals: list[float]
    females: list[float]
    males: list[float]
    genotype_labels: list[str]
    genotype_counts: list[list[float]]  # [genotype][deme]
    allele_names: list[str]
    allele_frequencies: list[list[float]]  # [allele][deme]


def spatial_landscape(population: SpatialPopulation) -> SpatialLandscapePayload:
    """Serialize the landscape layout plus per-deme metrics."""
    topology = population.topology
    if isinstance(topology, HexGrid):
        info = SpatialTopologyInfo(
            kind="hex",
            rows=topology.rows,
            cols=topology.cols,
            wrap=topology.wrap,
            grid_ij=[],
            xy=[],
        )
    elif isinstance(topology, SquareGrid):
        info = SpatialTopologyInfo(
            kind="square",
            rows=topology.rows,
            cols=topology.cols,
            wrap=topology.wrap,
            grid_ij=[],
            xy=[],
        )
    else:
        info = SpatialTopologyInfo(
            kind="none", rows=0, cols=0, wrap=False, grid_ij=None, xy=None
        )

    if topology is not None:
        grid_ij_all: list[list[int]] = []
        xy_all: list[list[float]] = []
        for index in range(population.n_demes):
            grid_ij = topology.from_index(index)
            xy = topology.to_xy(grid_ij)
            grid_ij_all.append([int(grid_ij[0]), int(grid_ij[1])])
            xy_all.append([float(xy[0]), float(xy[1])])
        info["grid_ij"] = grid_ij_all
        info["xy"] = xy_all

    totals: list[float] = []
    females: list[float] = []
    males: list[float] = []
    genotype_counts: list[list[float]] = []
    allele_frequencies: list[list[float]] = []
    deme_names: list[str] = []

    species = population.species
    allele_names = known_allele_names(species)

    for deme in population.demes:
        counts = deme.state.individual_count
        totals.append(float(counts.sum()))
        females.append(float(counts[0].sum()))
        males.append(float(counts[1].sum()))
        deme_names.append(str(deme.name))

        if not genotype_counts:
            genotype_counts = [
                [] for _ in range(len(deme.registry.index_to_genotype))
            ]
        for g_idx, gt in enumerate(deme.registry.index_to_genotype):
            z_indices = deme.registry.ztype_indices_for(gt)
            genotype_counts[g_idx].append(float(counts[:, :, z_indices].sum()))

        freqs = compute_allele_frequencies(deme.registry, species, counts)
        for a_idx, allele in enumerate(allele_names):
            if a_idx >= len(allele_frequencies):
                allele_frequencies.append([])
            allele_frequencies[a_idx].append(freqs.get(allele, 0.0))

    genotype_labels = (
        [str(g) for g in population.demes[0].registry.index_to_genotype]
        if population.n_demes
        else []
    )

    return SpatialLandscapePayload(
        n_demes=population.n_demes,
        topology=info,
        deme_names=deme_names,
        totals=totals,
        females=females,
        males=males,
        genotype_labels=genotype_labels,
        genotype_counts=genotype_counts,
        allele_names=allele_names,
        allele_frequencies=allele_frequencies,
    )


# ---------------------------------------------------------------------------
# Deme detail
# ---------------------------------------------------------------------------


class SpatialDemeDetail(TypedDict):
    """Full inspection view of one deme."""

    index: int
    name: str
    grid_ij: list[int] | None
    total: float
    female: float
    male: float
    is_age_structured: bool
    female_per_age: list[float]
    male_per_age: list[float]
    genotypes: list[GenotypeStateRow]


def spatial_deme_detail(population: SpatialPopulation, index: int) -> SpatialDemeDetail:
    """Serialize one deme's full inspection detail.

    Raises:
        IndexError: If *index* is outside the deme range.
    """
    deme = population.demes[index]
    counts = deme.state.individual_count
    grid_ij: list[int] | None = None
    if population.topology is not None:
        grid_ij = list(population.topology.from_index(index))

    return SpatialDemeDetail(
        index=index,
        name=str(deme.name),
        grid_ij=grid_ij,
        total=float(counts.sum()),
        female=float(counts[0].sum()),
        male=float(counts[1].sum()),
        is_age_structured=counts.shape[1] > 2,
        female_per_age=counts[0].sum(axis=1).tolist(),
        male_per_age=counts[1].sum(axis=1).tolist(),
        genotypes=genotype_rows_builder(deme.registry, deme.config, counts),
    )


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


class SpatialMigrationEntry(TypedDict):
    """One outbound migration edge of a source deme."""

    dest: int
    dest_name: str
    weight: float
    share: float  # weight normalized by the source row sum
    dest_total: float


class SpatialMigrationDetail(TypedDict):
    """Outbound migration view of one source deme."""

    source: int
    source_name: str
    rate_mean: float  # mean migration rate over (sex, age)
    entries: list[SpatialMigrationEntry]


def spatial_migration_detail(
    population: SpatialPopulation, index: int
) -> SpatialMigrationDetail:
    """Serialize the outbound migration edges of one source deme.

    Raises:
        IndexError: If *index* is outside the deme range.
    """
    deme_names = [str(deme.name) for deme in population.demes]
    deme_totals = [
        float(deme.state.individual_count.sum()) for deme in population.demes
    ]

    csr = population.migration_csr
    start = int(csr.indptr[index])
    end = int(csr.indptr[index + 1])
    raw_weights = [
        float(csr.weights[k]) for k in range(start, end)
    ]
    row_sum = sum(raw_weights)

    entries: list[SpatialMigrationEntry] = []
    offset = 0
    for k in range(start, end):
        dest = int(csr.dest_idx[k])
        weight = raw_weights[offset]
        entries.append(
            SpatialMigrationEntry(
                dest=dest,
                dest_name=deme_names[dest] if dest < len(deme_names) else str(dest),
                weight=weight,
                share=(weight / row_sum) if row_sum > 0 else 0.0,
                dest_total=deme_totals[dest],
            )
        )
        offset += 1

    rate_row: NDArray[np.float64] = np.asarray(population.params.migration_rate[index])
    rate_mean = float(rate_row.mean()) if rate_row.size else 0.0

    return SpatialMigrationDetail(
        source=index,
        source_name=deme_names[index] if index < len(deme_names) else str(index),
        rate_mean=rate_mean,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Global history series (aggregated over demes)
# ---------------------------------------------------------------------------


def spatial_history_series(
    population: SpatialPopulation,
    *,
    max_points: int = 500,
) -> HistorySeries:
    """Build the global chart series aggregated over all demes.

    Returns:
        The same :class:`HistorySeries` shape as the panmictic endpoint, so
        the frontend chart components are reused unchanged.
    """
    history = population.history
    ticks: list[int] = []
    aggregates: list[NDArray[np.float64]] = []  # (sex, age, ztype) per point

    if history.schema.mode == "raw":
        counts_all = history.individual_count  # (record, deme, sex, age, ztype)
        for index, tick in enumerate(history.ticks):
            ticks.append(int(tick))
            aggregated: NDArray[np.float64] = counts_all[index].sum(axis=0)
            aggregates.append(aggregated)

    if population.tick not in ticks:
        ticks.append(int(population.tick))
        aggregates.append(population.aggregate_individual_count())

    n = len(ticks)
    stride = max(1, n // max(1, max_points))
    kept_ticks = ticks[::stride]

    totals: list[float] = []
    females: list[float] = []
    males: list[float] = []
    freq_by_allele: dict[str, list[float]] = {
        name: [] for name in known_allele_names(population.species)
    }
    allele_names = known_allele_names(population.species)

    registry = population.demes[0].registry if population.n_demes else None
    for kept_index in range(len(kept_ticks)):
        aggregate = aggregates[kept_index * stride]
        totals.append(float(aggregate.sum()))
        females.append(float(aggregate[0].sum()))
        males.append(float(aggregate[1].sum()))
        if registry is not None:
            freqs = compute_allele_frequencies(registry, population.species, aggregate)
        else:
            freqs = {}
        for allele in allele_names:
            freq_by_allele[allele].append(freqs.get(allele, 0.0))

    return HistorySeries(
        ticks=kept_ticks,
        total=totals,
        female=females,
        male=males,
        known_alleles=allele_names,
        allele_frequencies=freq_by_allele,
        truncated=stride > 1,
    )


# ---------------------------------------------------------------------------
# Debug: params log, raw dump, per-deme totals diff
# ---------------------------------------------------------------------------


class SpatialParamChangeRow(TypedDict):
    """One parameter write aggregated from a deme's audit log."""

    tick: int
    name: str  # "deme{i}:{param}" to identify the owning deme
    old: float
    new: float


def spatial_params_log_rows(
    population: SpatialPopulation,
) -> list[SpatialParamChangeRow]:
    """Aggregate per-deme parameter audit logs into one tick-ordered list.

    Spatial populations split the journal across demes (each deme absorbs
    its own plain-name rows), so the dashboard view re-attaches the deme
    prefix that the Rust journal originally carried.
    """
    rows: list[SpatialParamChangeRow] = []
    for index, deme in enumerate(population.demes):
        prefix = f"deme{index}:"
        for tick, name, old, new in deme.params_log:
            rows.append(
                SpatialParamChangeRow(
                    tick=int(tick), name=prefix + name, old=float(old), new=float(new)
                )
            )
    rows.sort(key=lambda row: (row["tick"], row["name"]))
    return rows


class SpatialDiffEntry(TypedDict):
    deme: int
    name: str
    total_a: float
    total_b: float
    delta: float


class SpatialDiffPayload(TypedDict):
    tick_a: int
    tick_b: int
    found_a: bool
    found_b: bool
    delta_total: float
    demes: list[SpatialDiffEntry]


def _spatial_counts_at(
    population: SpatialPopulation, tick: int
) -> tuple[bool, NDArray[np.float64]]:
    """Per-deme stacked counts ``(deme, sex, age, ztype)`` at *tick*.

    Falls back to the live stacked state when the tick is not recorded.
    """
    history = population.history
    if history.schema.mode == "raw" and tick in history.ticks:
        index = history.ticks.index(tick)
        stacked: NDArray[np.float64] = history.individual_count[index]
        return True, stacked
    stacked = np.stack(
        [deme.state.individual_count for deme in population.demes], axis=0
    )
    return False, stacked  # type: ignore[return-value]  # np.stack loses the dtype parameter


def spatial_state_diff(
    population: SpatialPopulation, tick_a: int, tick_b: int
) -> SpatialDiffPayload:
    """Diff per-deme totals between two ticks (pure history reads)."""
    found_a, counts_a = _spatial_counts_at(population, tick_a)
    found_b, counts_b = _spatial_counts_at(population, tick_b)

    totals_a = counts_a.sum(axis=(1, 2, 3))
    totals_b = counts_b.sum(axis=(1, 2, 3))
    entries: list[SpatialDiffEntry] = []
    for index, deme in enumerate(population.demes):
        total_a = float(totals_a[index])
        total_b = float(totals_b[index])
        entries.append(
            SpatialDiffEntry(
                deme=index,
                name=str(deme.name),
                total_a=total_a,
                total_b=total_b,
                delta=total_b - total_a,
            )
        )

    return SpatialDiffPayload(
        tick_a=tick_a,
        tick_b=tick_b,
        found_a=found_a,
        found_b=found_b,
        delta_total=float(totals_b.sum() - totals_a.sum()),
        demes=entries,
    )


class SpatialRawDump(TypedDict):
    """Unrounded array dump of one deme at one tick."""

    tick: int
    deme: int
    mode: str
    found: bool
    individual_count: list[list[list[float]]]


def spatial_raw_dump(
    population: SpatialPopulation, tick: int | None, deme_index: int
) -> SpatialRawDump:
    """Dump one deme's ``(sex, age, ztype)`` tensor.

    Raises:
        IndexError: If *deme_index* is outside the deme range.
    """
    deme = population.demes[deme_index]
    if tick is None or tick == population.tick:
        counts = deme.state.individual_count
        return SpatialRawDump(
            tick=int(deme.state.n_tick),
            deme=deme_index,
            mode="live",
            found=True,
            individual_count=counts.tolist(),
        )
    history = population.history
    if (
        history.schema.mode == "raw"
        and tick in history.ticks
        and deme_index < population.n_demes
    ):
        index = history.ticks.index(tick)
        counts = history.individual_count[index, deme_index]
        return SpatialRawDump(
            tick=tick,
            deme=deme_index,
            mode="history",
            found=True,
            individual_count=counts.tolist(),
        )
    live_counts = deme.state.individual_count
    return SpatialRawDump(
        tick=int(deme.state.n_tick),
        deme=deme_index,
        mode="live",
        found=False,
        individual_count=live_counts.tolist(),
    )
