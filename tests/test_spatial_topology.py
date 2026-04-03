#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.spatial_topology import (  # noqa: E402
    HexGrid,
    SquareGrid,
    apply_migration_adjacency,
    apply_migration_convolution,
    build_adjacency_matrix,
)


def test_square_grid_neighbors_moore_center_has_8():
    grid = SquareGrid(rows=3, cols=3, neighborhood="moore", wrap=False)
    center = grid.to_index((1, 1))
    assert len(grid.neighbors(center)) == 8


def test_hex_grid_neighbors_center_has_6():
    grid = HexGrid(rows=5, cols=5, wrap=False)
    center = grid.to_index((2, 2))
    assert len(grid.neighbors(center)) == 6


def test_hex_grid_has_expected_right_down_direction_vector():
    grid = HexGrid(rows=3, cols=3, wrap=False)
    vectors = grid.neighbor_direction_vectors()
    expected = (0.5, math.sqrt(3.0) / 2.0)
    assert any(np.allclose(v, expected) for v in vectors)


def test_hex_grid_neighbor_vectors_are_unit_length_for_center_cell():
    grid = HexGrid(rows=5, cols=5, wrap=False)
    vectors = grid.neighbor_vectors((2, 2))
    assert len(vectors) == 6
    for dx, dy in vectors:
        assert np.isclose(math.hypot(dx, dy), 1.0)


def test_adjacency_row_normalize():
    grid = SquareGrid(rows=3, cols=3, neighborhood="von_neumann", wrap=False)
    adj = build_adjacency_matrix(grid, include_self=False, row_normalize=True)

    row_sums = adj.sum(axis=1)
    positive = row_sums > 0.0
    assert np.allclose(row_sums[positive], 1.0)


def test_migration_adjacency_conserves_mass():
    grid = SquareGrid(rows=3, cols=3, neighborhood="von_neumann", wrap=False)
    adj = build_adjacency_matrix(grid, include_self=False, row_normalize=True)

    state = np.zeros((grid.n_demes, 2, 1, 1), dtype=np.float64)
    state[grid.to_index((1, 1)), 0, 0, 0] = 100.0

    out = apply_migration_adjacency(state, adj, rate=0.3)
    assert out.shape == state.shape
    assert np.isclose(out.sum(), state.sum())


def test_migration_adjacency_matches_expected_split_on_line_grid():
    grid = SquareGrid(rows=1, cols=3, neighborhood="von_neumann", wrap=False)
    adj = build_adjacency_matrix(grid, include_self=False, row_normalize=True)

    state = np.zeros((grid.n_demes, 1), dtype=np.float64)
    state[grid.to_index((0, 1)), 0] = 100.0

    out = apply_migration_adjacency(state, adj, rate=0.5)

    expected = np.array([[25.0], [50.0], [25.0]], dtype=np.float64)
    assert np.allclose(out, expected)


def test_migration_convolution_conserves_mass_for_square_and_hex():
    kernel = np.array(
        [
            [0.05, 0.1, 0.05],
            [0.1, 0.0, 0.1],
            [0.05, 0.1, 0.05],
        ],
        dtype=np.float64,
    )

    for topology in (SquareGrid(rows=4, cols=4), HexGrid(rows=4, cols=4)):
        state = np.zeros((topology.n_demes, 2, 1, 1), dtype=np.float64)
        state[topology.to_index((1, 1)), 0, 0, 0] = 40.0
        state[topology.to_index((2, 2)), 1, 0, 0] = 60.0

        out = apply_migration_convolution(state, topology, kernel, rate=0.25)
        assert out.shape == state.shape
        assert np.isclose(out.sum(), state.sum())


def test_migration_convolution_renormalizes_kernel_at_border():
    topology = SquareGrid(rows=1, cols=3, neighborhood="von_neumann", wrap=False)
    kernel = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)

    state = np.zeros((topology.n_demes, 1), dtype=np.float64)
    state[topology.to_index((0, 0)), 0] = 10.0

    out = apply_migration_convolution(state, topology, kernel, rate=1.0)

    expected = np.array([[0.0], [10.0], [0.0]], dtype=np.float64)
    assert np.allclose(out, expected)


def test_migration_convolution_hex_border_renormalizes_valid_offsets():
    topology = HexGrid(rows=3, cols=3, wrap=False)
    kernel = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    state = np.zeros((topology.n_demes, 1), dtype=np.float64)
    state[topology.to_index((0, 0)), 0] = 12.0

    out = apply_migration_convolution(state, topology, kernel, rate=1.0)

    expected = np.zeros_like(state)
    expected[topology.to_index((0, 1)), 0] = 4.0
    expected[topology.to_index((1, 0)), 0] = 4.0
    expected[topology.to_index((1, 1)), 0] = 4.0
    assert np.allclose(out, expected)


def test_migration_convolution_hex_wrap_preserves_wrapped_offsets():
    topology = HexGrid(rows=3, cols=3, wrap=True)
    kernel = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    state = np.zeros((topology.n_demes, 1), dtype=np.float64)
    state[topology.to_index((0, 0)), 0] = 8.0

    out = apply_migration_convolution(state, topology, kernel, rate=1.0)

    expected = np.zeros_like(state)
    for coord in [
        (2, 2),
        (2, 0),
        (2, 1),
        (0, 2),
        (0, 1),
        (1, 2),
        (1, 0),
        (1, 1),
    ]:
        expected[topology.to_index(coord), 0] = 1.0
    assert np.allclose(out, expected)
