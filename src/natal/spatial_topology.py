"""Spatial grid topology and migration utilities.

This module defines topology objects for arranging demes on 2D grids and
provides helper functions for building adjacency matrices and applying one
migration step to an array whose first axis indexes demes.

Attributes:
    Coord (type[tuple[int, int]]): Coordinate alias used for grid positions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GridTopology",
    "HeterogeneousKernelParams",
    "MigrationParams",
    "SpatialTopology",
    "SquareGrid",
    "HexGrid",
    "build_adjacency_matrix",
    "build_gaussian_kernel",
    "apply_migration_adjacency",
    "apply_migration_convolution",
]


# ---------------------------------------------------------------------------
# Frozen parameter bundles for spatial kernel dispatch
# ---------------------------------------------------------------------------


class SpatialTopology(NamedTuple):
    """Resolved topology dimensions for kernel routing.

    Built once from ``GridTopology`` at construction time. When no
    topology is set, all fields default to zero / ``False``.

    Attributes:
        rows: Number of grid rows (0 if no topology).
        cols: Number of grid columns (0 if no topology).
        wrap: Whether periodic boundary conditions apply.
    """

    rows: int
    cols: int
    wrap: bool


class MigrationParams(NamedTuple):
    """Fixed migration configuration for spatial kernels.

    All fields are determined at construction time and never change
    during a simulation.

    Attributes:
        kernel: Migration kernel weight matrix.  A ``(1, 1)`` zero array
            when no kernel is set (adjacency mode).
        include_center: Whether the kernel centre contributes outbound mass.
        rate: Fraction of each deme that migrates per tick.
        adjust_on_edge: Whether boundary demes normalise to match internal
            total outbound rate.
        adjacency: Dense ``(n_demes, n_demes)`` outbound migration matrix.
        mode_code: Backend selector (``0`` = adjacency, ``1`` = kernel).
    """

    kernel: NDArray[np.float64]
    include_center: bool
    rate: float
    adjust_on_edge: bool
    adjacency: NDArray[np.float64]
    mode_code: int


class HeterogeneousKernelParams(NamedTuple):
    """Per-kernel offset tables for heterogeneous kernel routing.

    Only populated when ``kernel_bank`` and ``deme_kernel_ids`` are both
    set on the spatial population.

    Attributes:
        deme_kernel_ids: ``(n_demes,)`` int64 — kernel index per source deme.
        d_row: ``(n_kernels, max_nnz)`` int64 — row offsets per kernel.
        d_col: ``(n_kernels, max_nnz)`` int64 — column offsets per kernel.
        weights: ``(n_kernels, max_nnz)`` float64 — per-offset weights.
        nnzs: ``(n_kernels,)`` int64 — number of valid entries per kernel.
        total_sums: ``(n_kernels,)`` float64 — sum of all weights per kernel.
        max_nnz: Maximum valid entries across all kernels.
    """

    deme_kernel_ids: NDArray[np.int64]
    d_row: NDArray[np.int64]
    d_col: NDArray[np.int64]
    weights: NDArray[np.float64]
    nnzs: NDArray[np.int64]
    total_sums: NDArray[np.float64]
    max_nnz: int


Coord = Tuple[int, int]


@dataclass(frozen=True)
class GridTopology:
    """Base topology over a 2D grid of demes.

    Attributes:
        rows (int): Number of grid rows.
        cols (int): Number of grid columns.
        wrap (bool): Whether coordinates use periodic boundary conditions.
            When ``True``, coordinates that move beyond one edge are wrapped to
            the opposite edge by modular arithmetic. When ``False``,
            out-of-bounds coordinates are discarded.

    Examples:
        With ``wrap=False``, out-of-bounds coordinates are rejected::

            grid = SquareGrid(rows=3, cols=3, wrap=False)
            grid.normalize_coord(-1, 1) is None

        With ``wrap=True``, opposite edges are connected::

            grid = SquareGrid(rows=3, cols=3, wrap=True)
            grid.normalize_coord(-1, 1) == (2, 1)
            grid.normalize_coord(1, -1) == (1, 2)
    """

    rows: int
    cols: int
    wrap: bool = False

    # cos(θ) where θ is the angle opposite the resultant of the two
    # basis vectors. For a square grid basis vectors are 90° apart →
    # θ = 90°, cos 90° = 0, so the cross-term in the law of cosines
    # vanishes: dist² = dr² + dc² (Cartesian).
    COS_OPPOSITE_ANGLE: float = 0.0

    @property
    def n_demes(self) -> int:
        """int: Total number of demes in the grid."""
        return self.rows * self.cols

    def to_index(self, coord: Coord) -> int:
        """Convert one grid coordinate to a flattened deme index.

        Args:
            coord: Grid coordinate as ``(row, col)``.

        Returns:
            Flattened deme index in row-major order.
        """
        row, col = coord
        return row * self.cols + col

    def from_index(self, index: int) -> Coord:
        """Convert one flattened deme index to a grid coordinate.

        Args:
            index: Flattened deme index in row-major order.

        Returns:
            Grid coordinate as ``(row, col)``.
        """
        row = index // self.cols
        col = index % self.cols
        return (row, col)

    def normalize_coord(self, row: int, col: int) -> Coord | None:
        """Normalize one coordinate according to the topology boundary rule.

        Args:
            row: Candidate row index.
            col: Candidate column index.

        Returns:
            A valid grid coordinate if the location is in bounds or can be
            wrapped. Returns ``None`` when the location lies outside a
            non-wrapping topology. With ``wrap=True``, the result is computed as
            ``(row % rows, col % cols)``, so opposite edges are connected.
        """
        if self.wrap:
            return (row % self.rows, col % self.cols)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        return (row, col)

    def neighbor_coords(self, coord: Coord) -> List[Coord]:
        """Return neighboring grid coordinates for one source coordinate.

        Args:
            coord: Source grid coordinate.

        Returns:
            Neighbor coordinates in the topology-specific neighborhood order.

        Raises:
            NotImplementedError: If a subclass does not implement neighbor
                generation.
        """
        raise NotImplementedError

    def to_xy(self, coord: Coord) -> Tuple[float, float]:
        """Map one grid coordinate to a geometric embedding coordinate.

        Args:
            coord: Source grid coordinate.

        Returns:
            Cartesian coordinate used for geometry-aware utilities.
        """
        row, col = coord
        return (float(col), float(row))

    def neighbor_vectors(self, coord: Coord) -> List[Tuple[float, float]]:
        """Return geometric displacement vectors from one coord to neighbors.

        Args:
            coord: Source grid coordinate.

        Returns:
            Displacement vectors from ``coord`` to each neighboring coordinate
            in embedding space.
        """
        x0, y0 = self.to_xy(coord)
        vectors: List[Tuple[float, float]] = []
        for n_coord in self.neighbor_coords(coord):
            x1, y1 = self.to_xy(n_coord)
            vectors.append((x1 - x0, y1 - y0))
        return vectors

    def offset_dist_sq(self, dr: NDArray[np.float64], dc: NDArray[np.float64]) -> NDArray[np.float64]:
        """Squared distance between grid coords offset by ``(dr, dc)``.

        Uses the law of cosines with ``_COS_OPPOSITE_ANGLE``::

            dist² = dr² + dc² - 2·dr·dc·cos(θ)

        For square grids cos(90°) = 0 → Cartesian distance. Subclasses
        override the class attribute ``_COS_OPPOSITE_ANGLE`` to change the
        metric (e.g. hex grids set cos(120°) = -0.5).
        """
        return dr**2 + dc**2 - 2.0 * self.COS_OPPOSITE_ANGLE * dr * dc

    def neighbors(self, index: int) -> List[int]:
        """Return neighboring deme indices for one flattened deme index.

        Args:
            index: Source deme index in row-major order.

        Returns:
            Neighbor deme indices in the same order as ``neighbor_coords``.
        """
        return [self.to_index(coord) for coord in self.neighbor_coords(self.from_index(index))]


@dataclass(frozen=True)
class SquareGrid(GridTopology):
    """Square grid with Von Neumann or Moore neighborhood.

    Attributes:
        neighborhood (Literal["von_neumann", "moore"]): Neighborhood rule used
            to enumerate adjacent demes.
        wrap (bool): Inherited periodic-boundary flag. When enabled, neighbors
            that would cross one edge of the rectangular grid re-enter from the
            opposite edge.

    Examples:
        Compare neighbors of the corner cell ``(0, 0)``::

            grid = SquareGrid(rows=3, cols=3, neighborhood="von_neumann", wrap=False)
            grid.neighbor_coords((0, 0)) == [(1, 0), (0, 1)]

            wrapped = SquareGrid(rows=3, cols=3, neighborhood="von_neumann", wrap=True)
            wrapped.neighbor_coords((0, 0)) == [(2, 0), (1, 0), (0, 2), (0, 1)]
    """

    neighborhood: Literal["von_neumann", "moore"] = "moore"

    def neighbor_coords(self, coord: Coord) -> List[Coord]:
        """Return neighboring coordinates for one square-grid location.

        Args:
            coord: Source grid coordinate.

        Returns:
            Neighbor coordinates after applying boundary normalization.

        Raises:
            ValueError: If ``neighborhood`` is not supported.
        """
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
    """Hex grid using parallelogram coordinates with pointy-top geometric embedding.

    This implementation uses a parallelogram grid where each cell has six
    neighbors in a hexagonal pattern. The grid is defined using two basis
    vectors that form a 60-degree angle, creating a natural hexagonal topology.

    Examples:
        On a non-wrapping grid, a corner cell loses out-of-bounds neighbors::

            grid = HexGrid(rows=3, cols=4, wrap=False)
            len(grid.neighbor_coords((0, 0))) < 6

        On a wrapping grid, the same corner cell keeps six neighbors because
        out-of-bounds coordinates are folded back into the opposite edge::

            wrapped = HexGrid(rows=3, cols=4, wrap=True)
            len(wrapped.neighbor_coords((0, 0))) == 6
    """

    _SQRT3_OVER_2 = math.sqrt(3.0) / 2.0

    # Basis vectors are 60° apart → angle opposite the resultant is
    # 180° - 60° = 120°. cos(120°) = -0.5 introduces the cross-term
    # dr·dc in the distance formula.
    COS_OPPOSITE_ANGLE: float = -0.5

    def to_xy(self, coord: Coord) -> Tuple[float, float]:
        """Map parallelogram coordinate to pointy-top Cartesian coordinates.

        Positive x points right, positive y points downward.
        With side length = 1, adjacent hex centers are distance 1 apart.
        """
        i, j = coord
        # Parallelogram basis vectors for pointy-top hexes:
        # moving +i shifts x by 1; moving +j shifts x by 1/2 and y by sqrt(3)/2.
        x = i + 0.5 * j
        y = self._SQRT3_OVER_2 * j
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
        """Return neighboring coordinates for one hex-grid location.

        Args:
            coord: Source grid coordinate in parallelogram form.

        Returns:
            Neighbor coordinates after boundary normalization.

        Examples:
            A center cell has six neighbors. Under periodic boundaries, edge
            cells also keep six neighbors because wrapped coordinates are folded
            back into the valid row/col range::

                HexGrid(rows=4, cols=4, wrap=False).neighbor_coords((1, 1))
                HexGrid(rows=4, cols=4, wrap=True).neighbor_coords((0, 0))
        """
        i, j = coord
        # Direct neighbor offsets in parallelogram coordinates
        parallelogram_offsets: Sequence[Tuple[int, int]] = (
            (1, 0),   # right
            (0, 1),   # down-right
            (-1, 1),  # down-left
            (-1, 0),  # left
            (0, -1),  # up-left
            (1, -1),  # up-right
        )
        result: List[Coord] = []
        for di, dj in parallelogram_offsets:
            # Compute neighbors directly in parallelogram space
            normalized = self.normalize_coord(i + di, j + dj)
            if normalized is not None:
                result.append(normalized)
        return result


def build_gaussian_kernel(
    topology_cls: Literal["square", "hex"] | type[GridTopology] = "hex",
    size: int = 5,
    sigma: float | None = None,
    mean_dispersal: float | None = None,
) -> np.ndarray:
    """Build a normalized Gaussian migration kernel for a grid topology.

    The kernel is a ``(size, size)`` matrix where each entry ``[r, c]`` is
    the outbound migration weight from a virtual centre cell to the grid
    cell at offset ``(r - centre, c - centre)``. Distances are computed
    via the topology's ``COS_OPPOSITE_ANGLE``, so the correct metric is
    used for square grids (Cartesian) vs hex grids (oblique /
    law-of-cosines).

    At runtime the spatial migration kernel slides this matrix over every
    source deme and re-normalises valid destinations at boundaries.

    Args:
        topology_cls: Grid topology class (e.g. ``SquareGrid``, ``HexGrid``)
            or a string shorthand ``"hex"`` / ``"square"``.
        size: Odd integer kernel size (default 5). Larger kernels capture
            longer-range dispersal but increase the non-zero offset table.
        sigma: Gaussian width parameter. Defaults to 1.0 when neither
            ``sigma`` nor ``mean_dispersal`` is given.
        mean_dispersal: Target mean dispersal distance. When provided,
            ``sigma`` is derived via the 2D Rayleigh mean formula::

                sigma = mean_dispersal / sqrt(π / 2)

            Mutually exclusive with ``sigma``.

    Returns:
        Normalised ``(size, size)`` float64 kernel summing to 1.

    Raises:
        ValueError: If ``size`` is even, both ``sigma`` and
            ``mean_dispersal`` are given, or ``topology_cls`` is an unknown
            string.

    Examples:
        Via sigma::

            kernel = build_gaussian_kernel("hex", size=11, sigma=1.5)

        Via mean dispersal::

            kernel = build_gaussian_kernel("hex", size=11, mean_dispersal=2.0)
    """
    if sigma is not None and mean_dispersal is not None:
        raise ValueError(
            "sigma and mean_dispersal are mutually exclusive; "
            "specify one or neither, not both"
        )
    if mean_dispersal is not None:
        sigma = mean_dispersal / math.sqrt(math.pi / 2.0)
    elif sigma is None:
        sigma = 1.0

    if isinstance(topology_cls, str):
        if topology_cls == "hex":
            topology_cls = HexGrid
        elif topology_cls == "square":
            topology_cls = SquareGrid
        else:
            raise ValueError(
                f"Unknown topology shorthand {topology_cls!r}, expected 'hex' or 'square'"
            )

    if size % 2 == 0:
        raise ValueError(f"kernel size must be odd, got {size}")

    center = (size - 1) / 2.0
    y_idx, x_idx = np.indices((size, size), dtype=np.float64)
    dr = y_idx - center
    dc = x_idx - center
    cos_opposite = topology_cls.COS_OPPOSITE_ANGLE
    dist_sq = dr**2 + dc**2 - 2.0 * cos_opposite * dr * dc
    kernel = np.exp(-dist_sq / (2.0 * sigma**2)).astype(np.float64, copy=False)
    kernel /= np.sum(kernel)
    return kernel


def build_adjacency_matrix(
    topology: GridTopology,
    include_self: bool = False,
    row_normalize: bool = False,
) -> np.ndarray:
    """Build an adjacency matrix from one topology.

    The returned matrix uses the convention ``A[i, j]`` = weight from source
    deme ``i`` to destination deme ``j``.

    Args:
        topology: Grid topology used to define neighborhood relations.
        include_self: Whether each deme should include itself as an outbound
            target.
        row_normalize: Whether to normalize each non-zero row to sum to 1.

    Returns:
        A dense adjacency matrix with shape ``(n_demes, n_demes)``.
    """
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

    Returns:
        One migrated state array with the same shape as ``state``.

    Raises:
        ValueError: If ``state`` has no deme axis, ``rate`` lies outside
            ``[0, 1]``, or ``adjacency`` has an incompatible shape.
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

    Args:
        state: Array with shape ``(n_demes, ...)``.
        topology: Grid topology used to map deme indices to coordinates.
        kernel: Odd-sized 2D kernel of outbound migration weights.
        rate: Migration share in ``[0, 1]``.
        include_center: Whether the kernel center contributes outbound weight to
            the source deme itself.

    Returns:
        One migrated state array with the same shape as ``state``.

    Raises:
        ValueError: If ``state`` has no deme axis, ``rate`` lies outside
            ``[0, 1]``, ``kernel`` is not an odd-sized 2D array, or the first
            dimension of ``state`` does not match ``topology.n_demes``.
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
                # Translate kernel coordinates into offsets around the current
                # source deme, then let the topology decide how boundaries are
                # handled (drop vs wrap).
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
            # Re-normalize valid targets only, so truncated kernels at borders
            # still conserve total mass.
            out[dst] += rate * flat[src] * (w / total)

    return out.reshape(state.shape)
