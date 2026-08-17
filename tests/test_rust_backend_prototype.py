"""Prototype parity tests for the Rust aging kernels.

These tests compare the PyO3 kernels against explicit reference
implementations of the aging semantics.  They are skipped automatically when
the optional ``natal._engine_rs`` extension has not been built with maturin.
"""

from __future__ import annotations

import numpy as np
import pytest

from natal.configurator import Configurator
from natal.data import (
    DiscretePopulationConfig,
    DiscretePopulationState,
    PopulationConfig,
    PopulationState,
)
from natal.engine.backends.rust_backend import (
    rust_backend_available,
    rust_run_age_structured_aging,
    rust_run_discrete_aging,
)
from natal.genetics import Species

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


@pytest.fixture(scope="module")
def age_config() -> PopulationConfig:
    """Build a small age-structured config without touching the Rust backend."""
    species = Species.from_dict(
        name="RustPrototypeAgeSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return Configurator.from_species(species).age_structure(5, 2).build().config


@pytest.fixture(scope="module")
def discrete_config() -> DiscretePopulationConfig:
    """Build a minimal discrete-generation config without touching the Rust backend."""
    species = Species.from_dict(
        name="RustPrototypeDiscreteSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return Configurator.from_species(species, discrete=True).build().config


def _reference_age_structured_aging(
    ind_count: np.ndarray,
    sperm_store: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference arrays after one age-advance step."""
    expected_ind = ind_count.copy()
    expected_sperm = sperm_store.copy()
    n_ages = ind_count.shape[1]
    for age in range(n_ages - 1, 0, -1):
        expected_ind[:, age, :] = expected_ind[:, age - 1, :]
        expected_sperm[age, :, :] = expected_sperm[age - 1, :, :]
    expected_ind[:, 0, :] = 0.0
    expected_sperm[0, :, :] = 0.0
    return expected_ind, expected_sperm


def _reference_discrete_aging(ind_count: np.ndarray) -> np.ndarray:
    """Return reference arrays after one discrete-generation age step."""
    expected = ind_count.copy()
    for sex in (0, 1):
        expected[sex, 1, :] = expected[sex, 0, :]
        expected[sex, 0, :] = 0.0
    return expected


def test_age_structured_aging_matches_reference(age_config: PopulationConfig) -> None:
    """Rust aging must reproduce the age-shift semantics exactly."""
    rng = np.random.default_rng(2026_07_17)
    ind_count = rng.normal(size=(2, 5, 3))
    sperm_store = rng.normal(size=(5, 3, 3))
    state = PopulationState(
        n_tick=7,
        individual_count=ind_count,
        sperm_storage=sperm_store,
    )

    result = rust_run_age_structured_aging(state, age_config)

    expected_ind, expected_sperm = _reference_age_structured_aging(
        ind_count, sperm_store
    )
    assert result.n_tick == state.n_tick
    assert np.array_equal(result.individual_count, expected_ind)
    assert np.array_equal(result.sperm_storage, expected_sperm)


def test_discrete_aging_matches_reference(
    discrete_config: DiscretePopulationConfig,
) -> None:
    """Rust discrete aging must reproduce the juvenile-to-adult shift exactly."""
    rng = np.random.default_rng(2026_07_18)
    ind_count = rng.normal(size=(2, 2, 4))
    state = DiscretePopulationState(n_tick=3, individual_count=ind_count)

    result = rust_run_discrete_aging(state, discrete_config)

    expected = _reference_discrete_aging(ind_count)
    assert result.n_tick == state.n_tick
    assert np.array_equal(result.individual_count, expected)


def test_aging_wrappers_do_not_mutate_inputs(age_config: PopulationConfig) -> None:
    """The Python adapter copies state before Rust mutates its views."""
    rng = np.random.default_rng(2026_07_19)
    ind_count = rng.normal(size=(2, 4, 3))
    sperm_store = rng.normal(size=(4, 3, 3))
    original_ind = ind_count.copy()
    original_sperm = sperm_store.copy()
    state = PopulationState(
        n_tick=1,
        individual_count=ind_count,
        sperm_storage=sperm_store,
    )

    rust_run_age_structured_aging(state, age_config)

    assert np.array_equal(ind_count, original_ind)
    assert np.array_equal(sperm_store, original_sperm)


def test_age_structured_aging_rejects_bad_sperm_shape(
    age_config: PopulationConfig,
) -> None:
    """Rust must reject mismatched sperm shape before mutating anything."""
    ind_count = np.ones((2, 3, 2), dtype=np.float64)
    bad_sperm = np.ones((3, 2, 4), dtype=np.float64)
    state = PopulationState(
        n_tick=0,
        individual_count=ind_count,
        sperm_storage=bad_sperm,
    )

    with pytest.raises(ValueError, match="sperm_storage shape"):
        rust_run_age_structured_aging(state, age_config)

    assert np.all(ind_count == 1.0)
    assert np.all(bad_sperm == 1.0)


def test_discrete_aging_rejects_too_few_ages(
    discrete_config: DiscretePopulationConfig,
) -> None:
    """Rust must reject one-age discrete states instead of indexing out of bounds."""
    ind_count = np.ones((2, 1, 2), dtype=np.float64)
    state = DiscretePopulationState(n_tick=0, individual_count=ind_count)

    with pytest.raises(ValueError, match="n_ages >= 2"):
        rust_run_discrete_aging(state, discrete_config)

    assert np.all(ind_count == 1.0)
