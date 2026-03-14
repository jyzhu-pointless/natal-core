"""PopulationState data container.

This module implements a lightweight NumPy-backed PopulationState used
to store per-sex, per-age (optional), per-genotype individual counts, as
well as optional sperm storage and female occupancy arrays for age-structured
models.

Key points:
- The jitclass version (PopulationState) always uses 3D shape 
    (n_sexes, n_ages, n_genotypes) for Numba compatibility.
    Non-age-structured mode is represented with n_ages=1.
- The dataclass version (PopulationStateDataclass) supports optional 2D 
    or 3D individual_count (kept for backward compatibility).
- Age-related methods check n_ages (for jitclass) or dimension check 
    (for dataclass).

Example:
    from natal.population_state import PopulationState

    state = PopulationState(n_sexes=2, n_ages=4, n_genotypes=5, n_haploid_genotypes=10)
    # Use state.individual_count, state.sperm_storage, state.female_occupancy
        
Note:
        This module is a data container only; mating, selection and inheritance
        logic belong to higher-level components (for example the population
        implementations and simulation loop).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union
from numpy.typing import NDArray
from numba import types as nb_types

from natal.numba_utils import jitclass_switch, njit_switch
from natal.index_core import compress_hg_glab, decompress_hg_glab

__all__ = [
    # No user-facing API for now
]

# ============================================================================
# Numba jitclass Version (Numba-compatible)
# ============================================================================

_popstate_spec = [
    ('n_tick', nb_types.int32),
    ('individual_count', nb_types.float64[:, :, :]),  # (sex, age, genotype)
    ('sperm_storage', nb_types.float64[:, :, :]),     # (age, genotype (female), genotype (male))
]


@njit_switch
def _validate_or_default_array(arr: Optional[NDArray[np.float64]], expected_shape: tuple, name: str):
    """Validate an array's shape or return a default zero array.

    Args:
        arr: Input array, or None to use default.
        expected_shape: Expected shape tuple.
        name: Array name used in assertion messages.

    Returns:
        The validated array cast to float64, or a new zero array with the expected shape.
    """
    if arr is not None:
        assert arr.shape == expected_shape, f"Invalid shape for {name}: expected {expected_shape}, got {arr.shape}"
        return arr.astype(np.float64)
    else:
        return np.zeros(expected_shape, dtype=np.float64)


@jitclass_switch(_popstate_spec)
class PopulationState:
    """Numba-compatible population state container.
    
    Always stores individual_count in 3D format (sex, age, genotype).
    For non-age-structured mode, use n_ages=2 (0: offspring, 1: adult).
    """
    
    def __init__(
        self,
        n_genotypes: int = 0,
        n_sexes: int = None,
        n_ages: int = 2,
        n_tick: int = 0,
        individual_count: Optional[NDArray[np.float64]] = None,
        sperm_storage: Optional[NDArray[np.float64]] = None
    ):
        """Construct PopulationState with allocated/validated arrays.

        This __init__ mirrors the validation and defaulting behaviour used by
        `PopulationConfig` and allocates Numba-compatible numpy arrays.
        """
        if n_sexes is None:
            n_sexes = 2

        # validate small values
        assert n_genotypes > 0, "n_genotypes must be positive"
        assert n_ages > 0, "n_ages must be positive"
        assert n_tick >= 0, "n_tick must be non-negative"

        self.n_tick = np.int32(n_tick)

        # individual_count: (sex, age, genotype)
        self.individual_count = _validate_or_default_array(
            individual_count, (n_sexes, n_ages, n_genotypes), "individual_count"
        )

        # sperm_storage: (age, genotype (female), genotype (male))
        self.sperm_storage = _validate_or_default_array(
            sperm_storage, (n_ages, n_genotypes, n_genotypes), "sperm_storage"
        )
    
    def get_count(self, sex: int, age: int, genotype_index: int) -> float:
        """Get individual count for sex/age/genotype."""
        return self.individual_count[sex, age, genotype_index]
    
    def add_count(self, sex: int, age: int, genotype_index: int, count: float) -> None:
        """Add count for sex/age/genotype."""
        self.individual_count[sex, age, genotype_index] += count
    
    def set_count(self, sex: int, age: int, genotype_index: int, count: float) -> None:
        """Set count for sex/age/genotype."""
        self.individual_count[sex, age, genotype_index] = count
    
    def get_stored_sperm(self, age: int, female_genotype_index: int, male_genotype_index: int, ) -> float:
        """Get stored sperm count."""
        return self.sperm_storage[age, female_genotype_index, male_genotype_index]
    
    def set_stored_sperm(self, age: int, female_genotype_index: int, male_genotype_index: int, count: float) -> None:
        """Add stored sperm count."""
        self.sperm_storage[age, female_genotype_index, male_genotype_index] += count

    def flatten_all(self) -> NDArray[np.float64]:
        """Flatten n_ticks, individual_count, and sperm_storage into a single 1D array."""
        # Use tuple instead of list for concatenate, Numba prefers tuple for heterogeneous types or known size
        # However, concatenate usually takes a sequence.
        # Issue might be list of standard arrays being passed to concatenate inside jitclass
        tick_arr = np.array([float(self.n_tick)], dtype=np.float64)
        ind_flat = self.individual_count.flatten()
        sperm_flat = self.sperm_storage.flatten()
        return np.concatenate((tick_arr, ind_flat, sperm_flat))


# ============================================================================
# parse_flattened_state: Numba-compiled
# ============================================================================

@njit_switch
def parse_flattened_state(
    flat_array: NDArray[np.float64],
    n_sexes: Union[int, np.integer],
    n_ages: Union[int, np.integer],
    n_genotypes: Union[int, np.integer],
    copy: bool = True
) -> PopulationState:
    """Parse flattened state from array with automatic type handling.
    
    Works in both pure Python and Numba-compiled contexts.
    
    Args:
        flat_array: Flattened 1D array [n_tick, ind_count.ravel(), sperm.ravel()]
        n_sexes: Number of sexes
        n_ages: Number of age classes
        n_genotypes: Number of genotypes
        copy: Whether to copy arrays. If True, arrays are copied; if False, arrays are used in-place (default True)
        
    Returns:
        PopulationState: Reconstructed state object
        
    Example:
        >>> flat = np.array([1.0, 2.0, 3.0, ...])
        >>> state = parse_flattened_state(flat, 2, 8, 9)
    """
    n_tick = np.int32(flat_array[0])
    individual_count = flat_array[1:1+n_sexes*n_ages*n_genotypes].reshape((n_sexes, n_ages, n_genotypes))
    sperm_storage = flat_array[1+n_sexes*n_ages*n_genotypes:].reshape((n_ages, n_genotypes, n_genotypes))

    if copy:
        individual_count = individual_count.copy()
        sperm_storage = sperm_storage.copy()
    
    # Create PopulationState with position parameters
    return PopulationState(
        n_genotypes,
        n_sexes,
        n_ages,
        n_tick,
        individual_count,
        sperm_storage
    )


