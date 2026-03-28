"""Spatial grid topology and migration utilities.

This module provides foundational building blocks for multi-deme spatial
simulation without coupling to a specific population implementation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np

__all__ = [
    "GridTopology",
    "SquareGrid",
    "HexGrid",
    "build_adjacency_matrix",
    "apply_migration_adjacency",
    "apply_migration_convolution",
]


Coord = Tuple[int, int]


@dataclass(frozen=True)
class GridTopology:
    """Base topology over a 2D grid of demes."""

    rows: int
    cols: int
    wrap: bool = False

    @property
    def n_demes(self) -> int:
        return self.rows * self.cols

    def to_index(self, coord: Coord) -> int:
        row, col = coord
        return row * self.cols + col

    def from_index(self, index: int) -> Coord:
        row = index // self.cols
        col = index % self.cols
        return (row, col)

    def normalize_coord(self, row: int, col: int) -> Coord | None:
        if self.wrap:
            return (row % self.rows, col % self.cols)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        return (row, col)

    def neighbor_coords(self, coord: Coord) -> List[Coord]:
        raise NotImplementedError

    def to_xy(self, coord: Coord) -> Tuple[float, float]:
        """Map one grid coord to a geometric embedding coordinate."""
        row, col = coord
        return (float(col), float(row))

    def neighbor_vectors(self, coord: Coord) -> List[Tuple[float, float]]:
        """Return geometric displacement vectors from coord to each neighbor."""
        x0, y0 = self.to_xy(coord)
        vectors: List[Tuple[float, float]] = []
        for n_coord in self.neighbor_coords(coord):
            x1, y1 = self.to_xy(n_coord)
            vectors.append((x1 - x0, y1 - y0))
        return vectors

    def neighbors(self, index: int) -> List[int]:
        return [self.to_index(coord) for coord in self.neighbor_coords(self.from_index(index))]


@dataclass(frozen=True)
class SquareGrid(GridTopology):
    """Square grid with Von Neumann or Moore neighborhood."""

    neighborhood: Literal["von_neumann", "moore"] = "moore"

    def neighbor_coords(self, coord: Coord) -> List[Coord]:
        row, col = coord
        if self.neighborhood == "von_neumann":
            offsets: Iterable[Coord] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif self.neighborhood == "moore":
            offsets = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        else:
            raise ValueError(f"Unsupported neighborhood: {self.neighborhood}")

        result: List[Coord] = []
        for dr, dc in offsets:
            normalized = self.normalize_coord(row + dr, col + dc)
            if normalized is not None:
                result.append(normalized)
        return result


@dataclass(frozen=True)
class HexGrid(GridTopology):
    """Hex grid using odd-r offset indices with pointy-top geometric embedding.

    Index-space neighbor offsets depend on row parity (odd-r), but geometric
    neighbor vectors are parity-invariant and all have unit length.
    """

    _SQRT3_OVER_2 = math.sqrt(3.0) / 2.0

    @staticmethod
    def _offset_to_axial(coord: Coord) -> Tuple[int, int]:
        """Convert odd-r offset (row, col) to axial (q, r)."""
        row, col = coord
        q = col - ((row - (row & 1)) // 2)
        r = row
        return (q, r)

    @staticmethod
    def _axial_to_offset(q: int, r: int) -> Coord:
        """Convert axial (q, r) to odd-r offset (row, col)."""
        row = r
        col = q + ((r - (r & 1)) // 2)
        return (row, col)

    def to_xy(self, coord: Coord) -> Tuple[float, float]:
        """Map odd-r offset coord to pointy-top Cartesian coordinates.

        Positive x points right, positive y points downward.
        With side length = 1, adjacent hex centers are distance 1 apart.
        """
        q, r = self._offset_to_axial(coord)
        x = q + 0.5 * r
        y = self._SQRT3_OVER_2 * r
        return (float(x), float(y))

    def neighbor_direction_vectors(self) -> Tuple[Tuple[float, float], ...]:
        """Canonical geometric neighbor vectors for pointy-top hexes.

        Includes the right-down vector (1/2, sqrt(3)/2).
        """
        s = self._SQRT3_OVER_2
        return (
            (1.0, 0.0),
            (0.5, s),
            (-0.5, s),
            (-1.0, 0.0),
            (-0.5, -s),
            (0.5, -s),
        )

    def neighbor_coords(self, coord: Coord) -> List[Coord]:
        q, r = self._offset_to_axial(coord)
        axial_offsets: Sequence[Tuple[int, int]] = (
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
        )
        result: List[Coord] = []
        for dq, dr in axial_offsets:
            cand_row, cand_col = self._axial_to_offset(q + dq, r + dr)
            normalized = self.normalize_coord(cand_row, cand_col)
            if normalized is not None:
                result.append(normalized)
        return result


def build_adjacency_matrix(
    topology: GridTopology,
    include_self: bool = False,
    row_normalize: bool = False,
) -> np.ndarray:
    """Build adjacency matrix A where A[i, j] means i can migrate to j."""
    n = topology.n_demes
    adj = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        neighbors = topology.neighbors(i)
        if include_self:
            neighbors = neighbors + [i]
        for j in neighbors:
            adj[i, j] = 1.0

    if row_normalize:
        row_sum = adj.sum(axis=1, keepdims=True)
        nonzero = row_sum[:, 0] > 0.0
        adj[nonzero] = adj[nonzero] / row_sum[nonzero]

    return adj


def apply_migration_adjacency(state: np.ndarray, adjacency: np.ndarray, rate: float) -> np.ndarray:
    """Apply one migration step using an outbound row-stochastic adjacency matrix.

    Args:
        state: Array with shape (n_demes, ...).
        adjacency: Matrix with shape (n_demes, n_demes). Rows represent outbound
            migration probabilities and should sum to 1 for active rows.
        rate: Migration share in [0, 1].
    """
    if state.ndim < 1:
        raise ValueError("state must have at least one dimension")
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"rate must be in [0, 1], got {rate}")

    n_demes = state.shape[0]
    if adjacency.shape != (n_demes, n_demes):
        raise ValueError(
            f"adjacency shape mismatch: expected ({n_demes}, {n_demes}), got {adjacency.shape}"
        )

    flat = state.reshape(n_demes, -1)
    migrated_in = adjacency.T @ flat
    out = (1.0 - rate) * flat + rate * migrated_in
    return out.reshape(state.shape)


def apply_migration_convolution(
    state: np.ndarray,
    topology: GridTopology,
    kernel: np.ndarray,
    rate: float,
    include_center: bool = False,
) -> np.ndarray:
    """Apply one migration step by distributing each source deme via a local kernel.

    The kernel is interpreted as outbound weights from each source deme to nearby
    destination demes around the source's coordinate. At borders, invalid targets
    are ignored and remaining weights are renormalized.
    """
    if state.ndim < 1:
        raise ValueError("state must have at least one dimension")
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel must be a 2D array with odd dimensions")

    n_demes = topology.n_demes
    if state.shape[0] != n_demes:
        raise ValueError(
            f"state first dimension mismatch: expected {n_demes}, got {state.shape[0]}"
        )

    flat = state.reshape(n_demes, -1)
    out = (1.0 - rate) * flat.copy()

    kr = kernel.shape[0] // 2
    kc = kernel.shape[1] // 2

    for src in range(n_demes):
        src_coord = topology.from_index(src)
        targets: List[int] = []
        weights: List[float] = []

        for r in range(kernel.shape[0]):
            for c in range(kernel.shape[1]):
                if not include_center and r == kr and c == kc:
                    continue
                w = float(kernel[r, c])
                if w <= 0.0:
                    continue
                dr = r - kr
                dc = c - kc
                mapped = topology.normalize_coord(src_coord[0] + dr, src_coord[1] + dc)
                if mapped is None:
                    continue
                targets.append(topology.to_index(mapped))
                weights.append(w)

        if not targets:
            out[src] += rate * flat[src]
            continue

        total = float(sum(weights))
        if total <= 0.0:
            out[src] += rate * flat[src]
            continue

        for dst, w in zip(targets, weights):
            out[dst] += rate * flat[src] * (w / total)

    return out.reshape(state.shape)
