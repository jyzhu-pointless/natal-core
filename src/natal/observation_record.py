"""Observation record builders for simulation kernels (Numba-accelerated).

Provides independent njit functions that construct flat observation rows
from individual count arrays and pre-compiled observation masks.  Called
from both the Numba simulation kernels and Python-layer manual recording.

Panmictic vs spatial
--------------------

**Panmictic** has a single population without a deme dimension.  The
recorded row always has shape ``(n_groups, n_sexes, n_ages)`` — perfectly
uniform across groups.  :func:`build_observation_row_panmictic` uses
``sum(mask * ind, axis=-1).ravel()`` and no offset metadata is needed
(the layout is simply ``arange(n_groups) * sex_ages``).

**Spatial** adds a deme axis and per-group deme selection modes
(mask / expand / aggregate), so each group may occupy a different number
of float64 chunks.  The row is no longer uniform; :class:`CompactMeta`
precomputes per-group start positions and lookup tables.
:func:`build_observation_row_spatial` iterates groups via
``compact.offsets`` and writes only the required chunks, using ``-1.0``
as a sentinel for unselected demes in mask mode so that "genuinely zero"
and "masked out" are distinguishable.

The three per-group demean modes:

  - ``"mask"`` (default): record every deme.  Selected demes carry real
    counts; unselected demes are filled with ``-1.0``.
  - ``"aggregate"``: sum across selected demes into a single chunk.
  - ``"expand"``: record only the selected demes, no padding.

In other words, panmictic is the degenerate case where every group is
``"mask"`` with one "deme" (the whole population).  ``CompactMeta`` is
unnecessary there because the layout reduces to uniform spacing.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from natal.numba_utils import njit_switch

__all__ = [
    "CompactMeta",
    "build_compact_metadata",
    "build_observation_row_panmictic",
    "build_observation_row_spatial",
]


# ---------------------------------------------------------------------------
# Compact metadata — precomputed per-group layout for spatial observation rows
# ---------------------------------------------------------------------------

class CompactMeta(NamedTuple):
    """Precomputed per-group layout for a compact spatial observation row.

    Attributes:
        offsets: ``(n_groups,)`` int64 — start position of each group in the
            flat compact row.
        deme_map: ``(n_groups, n_demes)`` int64 — flat deme indices per
            group; selected indices come first, then unselected.
        n_demes_per_group: ``(n_groups,)`` int64 — total slots per group
            (equals ``n_demes`` for mask mode, ``len(selected)`` for expand,
            ``1`` for aggregate).
        selected_n: ``(n_groups,)`` int64 — how many of the first entries in
            *deme_map* are actually selected (the rest get the -1.0 sentinel).
        mode_aggregate: ``(n_groups,)`` bool — whether each group uses
            aggregate mode.
        row_size: Total float64 count for one compact snapshot row.
    """

    offsets: NDArray[np.int64]
    deme_map: NDArray[np.int64]
    n_demes_per_group: NDArray[np.int64]
    selected_n: NDArray[np.int64]
    mode_aggregate: NDArray[np.bool_]
    row_size: int


# ---------------------------------------------------------------------------
# Compact metadata builder (pure Python, called once at setup time)
# ---------------------------------------------------------------------------

def build_compact_metadata(
    n_demes: int,
    n_groups: int,
    n_sexes: int,
    n_ages: int,
    demean_modes: Dict[int, Tuple[str, List[int]]],
) -> CompactMeta:
    """Precompute offsets and lookup tables for compact spatial observation rows.

    Each group's demean mode determines how many float64 slots it occupies
    in the compact row::

        "mask"      → ``n_demes`` chunks (selected: real data; unselected: -1.0)
        "expand"    → ``len(selected)`` chunks
        "aggregate" → 1 chunk (sum of selected)

    A chunk is always ``n_sexes * n_ages`` floats.

    Args:
        n_demes: Total number of demes.
        n_groups: Number of observation groups.
        n_sexes: Number of sexes in the population.
        n_ages: Number of age classes.
        demean_modes: Per-group ``(mode, [flat_deme_indices, ...])`` dict.
            Groups not in the dict get ``("mask", list(range(n_demes)))``.

    Returns:
        A ``CompactMeta`` instance with precomputed offsets, lookup tables,
        and total row size.
    """
    sex_ages = n_sexes * n_ages
    all_deme_indices = list(range(n_demes))

    offsets = np.zeros(n_groups, dtype=np.int64)
    n_demes_per_group = np.zeros(n_groups, dtype=np.int64)
    selected_n = np.zeros(n_groups, dtype=np.int64)
    mode_aggregate = np.zeros(n_groups, dtype=np.bool_)
    demean_map = np.full((n_groups, n_demes), -1, dtype=np.int64)

    offset = 0
    for gi in range(n_groups):
        mode_info = demean_modes.get(gi)
        if mode_info is None:
            mode = "mask"
            selected = all_deme_indices
        else:
            mode, selected = mode_info

        if mode == "aggregate":
            nd = 1
            selected_n[gi] = 1
            mode_aggregate[gi] = True
            for di, d in enumerate(selected[:1]):
                demean_map[gi, di] = d
        elif mode == "expand":
            nd = len(selected)
            selected_n[gi] = nd
            for di, d in enumerate(selected):
                demean_map[gi, di] = d
        else:  # "mask"
            nd = n_demes
            n_sel = len(selected)
            selected_n[gi] = n_sel
            for di, d in enumerate(selected):
                demean_map[gi, di] = d
            unselected = [d for d in all_deme_indices if d not in set(selected)]
            for ui, d in enumerate(unselected):
                demean_map[gi, n_sel + ui] = d

        offsets[gi] = offset
        n_demes_per_group[gi] = nd
        offset += nd * sex_ages

    return CompactMeta(
        offsets=offsets,
        deme_map=demean_map,
        n_demes_per_group=n_demes_per_group,
        selected_n=selected_n,
        mode_aggregate=mode_aggregate,
        row_size=offset,
    )


# ---------------------------------------------------------------------------
# Numba-accelerated row builders
# ---------------------------------------------------------------------------

@njit_switch(cache=True)
def build_observation_row_panmictic(
    individual_count: NDArray[np.float64],
    observation_mask: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build a flat observation row for a single panmictic population.

    Args:
        individual_count: Count array of shape ``(n_sexes, n_ages, n_genotypes)``
            or ``(n_sexes, n_genotypes)`` for non-age-structured.
        observation_mask: 4-D or 3-D binary mask matching *individual_count*.

    Returns:
        1-D float64 array of shape ``(n_groups * n_sexes * n_ages,)``.
    """
    observed = np.sum(observation_mask * individual_count[None, :, :, :], axis=-1)
    return observed.ravel()


_SENTINEL_MASKED: float = -1.0


@njit_switch(cache=True)
def build_observation_row_spatial(
    individual_count: NDArray[np.float64],
    observation_mask: NDArray[np.float64],
    compact: CompactMeta,
) -> NDArray[np.float64]:
    """Build a compact flat observation row for a spatial population.

    Iterates over groups and writes the required chunks.  For mask-mode
    groups, selected demes carry real counts while unselected demes are
    filled with ``-1.0`` as a sentinel value (distinguishable from genuine
    zero-count entries).

    Args:
        individual_count: Stacked count array of shape
            ``(n_demes, n_sexes, n_ages, n_genotypes)``.
        observation_mask: 4-D binary mask ``(n_groups, n_sexes, n_ages, n_genotypes)``.
        compact: Precomputed ``CompactMeta`` for this population.

    Returns:
        1-D float64 array of shape ``(compact.row_size,)``.
    """
    result = np.zeros(compact.row_size, dtype=np.float64)
    n_sexes = observation_mask.shape[1]
    n_ages = observation_mask.shape[2]
    sex_ages = n_sexes * n_ages
    sentinel_vec = np.full(sex_ages, _SENTINEL_MASKED, dtype=np.float64)

    for gi in range(len(compact.offsets)):
        offset = compact.offsets[gi]
        nd = compact.n_demes_per_group[gi]
        sel_n = compact.selected_n[gi]

        if compact.mode_aggregate[gi]:
            agg = np.zeros((n_sexes, n_ages), dtype=np.float64)
            for di in range(sel_n):
                d = compact.deme_map[gi, di]
                agg += np.sum(observation_mask[gi] * individual_count[d], axis=-1)
            result[offset : offset + sex_ages] = agg.ravel()
        else:
            for di in range(nd):
                chunk_start = offset + di * sex_ages
                if di < sel_n:
                    d = compact.deme_map[gi, di]
                    observed = np.sum(observation_mask[gi] * individual_count[d], axis=-1)
                    result[chunk_start : chunk_start + sex_ages] = observed.ravel()
                else:
                    result[chunk_start : chunk_start + sex_ages] = sentinel_vec

    return result
