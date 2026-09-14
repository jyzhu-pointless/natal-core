"""Population state containers based on NamedTuple.

These containers keep scalar metadata immutable while allowing in-place mutation
of NumPy array contents, which remains compatible with the engines.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PopulationState",
    "DiscretePopulationState",
]


def state_axes(individual_count: NDArray[np.float64]) -> Tuple[int, int, int]:
    """Derive ``(n_sexes, n_ages, n_ztypes)`` from a count tensor.

    Every count tensor carries an age axis; a rank-2 ``(sex, ztype)`` tensor
    simply has a degenerate one, so it is read as a single age class — the same
    rule the projection inputs use.

    Args:
        individual_count: Count tensor of rank 2 ``(sex, ztype)`` or rank 3
            ``(sex, age, ztype)``.

    Returns:
        ``(n_sexes, n_ages, n_ztypes)`` with ``n_ages == 1`` for a rank-2 input.
    """
    n_sexes = int(individual_count.shape[0])
    n_ages = int(individual_count.shape[1]) if individual_count.ndim == 3 else 1
    n_ztypes = int(individual_count.shape[-1])
    return n_sexes, n_ages, n_ztypes


class PopulationState(NamedTuple):
    """Age-structured state container.

    Scalars are immutable (use ``_replace`` to rebuild); array values remain
    mutable in-place.

    Attributes:
        n_tick: Current simulation time step.
        individual_count: Array of shape (n_sexes, n_ages, n_ztypes) – counts
            of individuals per sex, age, and zygote type.
        sperm_storage: Array of shape (n_ages, n_ztypes, n_ztypes) – stored
            sperm counts per female age, female zygote type, and male zygote type.
    """

    n_tick: int
    individual_count: NDArray[np.float64]
    sperm_storage: NDArray[np.float64]

    @classmethod
    def create(
        cls,
        n_ztypes: int,
        n_sexes: Optional[int] = None,
        n_ages: int = 2,
        n_tick: int = 0,
        individual_count: Optional[NDArray[np.float64]] = None,
        sperm_storage: Optional[NDArray[np.float64]] = None,
    ) -> PopulationState:
        """Create a PopulationState with optionally provided arrays.

        If arrays are not provided, they are initialized to zeros.

        Args:
            n_ztypes: Number of zygote types (diploid genotype types after slab expansion).
            n_sexes: Number of sexes (defaults to 2 if not given).
            n_ages: Number of age classes (default 2).
            n_tick: Initial tick value (default 0).
            individual_count: Optional array (n_sexes, n_ages, n_ztypes).
            sperm_storage: Optional array (n_ages, n_ztypes, n_ztypes).

        Returns:
            A new PopulationState instance.

        Raises:
            AssertionError: If dimensions are invalid or provided arrays have wrong shape.
        """
        if n_sexes is None:
            n_sexes = 2
        # Validate dimensions before allocating so bad declarations fail fast.
        assert n_ztypes > 0, "n_ztypes must be positive"
        assert n_ages > 0, "n_ages must be positive"
        assert n_tick >= 0, "n_tick must be non-negative"

        if individual_count is None:
            ind = np.zeros((n_sexes, n_ages, n_ztypes), dtype=np.float64)
        else:
            expected_shape = (n_sexes, n_ages, n_ztypes)
            assert individual_count.shape == expected_shape, (
                f"Invalid shape for individual_count: expected {expected_shape}, got {individual_count.shape}"
            )
            # astype always copies, so the container never aliases caller memory.
            ind = individual_count.astype(np.float64)

        if sperm_storage is None:
            sperm = np.zeros((n_ages, n_ztypes, n_ztypes), dtype=np.float64)
        else:
            expected_shape = (n_ages, n_ztypes, n_ztypes)
            assert sperm_storage.shape == expected_shape, (
                f"Invalid shape for sperm_storage: expected {expected_shape}, got {sperm_storage.shape}"
            )
            # Same copy-on-ingest rule for the (age, female ztype, male ztype) plane.
            sperm = sperm_storage.astype(np.float64)

        return cls(n_tick=int(n_tick), individual_count=ind, sperm_storage=sperm)

    def get_count(self, sex: int, age: int, ztype_idx: int) -> float:
        """Retrieve the count of individuals for a specific category.

        Args:
            sex: Sex index.
            age: Age class index.
            ztype_idx: ZType (genotype) index.

        Returns:
            The count (float).
        """
        return self.individual_count[sex, age, ztype_idx]

    def add_count(self, sex: int, age: int, ztype_idx: int, count: float) -> None:
        """Add to the count of individuals for a specific category.

        Args:
            sex: Sex index.
            age: Age class index.
            ztype_idx: ZType (genotype) index.
            count: Amount to add (can be negative).
        """
        self.individual_count[sex, age, ztype_idx] += count

    def set_count(self, sex: int, age: int, ztype_idx: int, count: float) -> None:
        """Set the count of individuals for a specific category.

        Args:
            sex: Sex index.
            age: Age class index.
            ztype_idx: ZType (genotype) index.
            count: New count.
        """
        self.individual_count[sex, age, ztype_idx] = count

    def get_stored_sperm(self, age: int, female_ztype_idx: int, male_ztype_idx: int) -> float:
        """Retrieve stored sperm count for a given combination.

        Args:
            age: Age class of the female.
            female_ztype_idx: Female ZType (genotype) index.
            male_ztype_idx: Male ZType (genotype) index.

        Returns:
            Stored sperm count.
        """
        return self.sperm_storage[age, female_ztype_idx, male_ztype_idx]

    def set_stored_sperm(self, age: int, female_ztype_idx: int, male_ztype_idx: int, count: float) -> None:
        """Add to stored sperm count (in‑place addition).

        Args:
            age: Age class of the female.
            female_ztype_idx: Female ZType (genotype) index.
            male_ztype_idx: Male ZType (genotype) index.
            count: Amount to add (can be negative).
        """
        self.sperm_storage[age, female_ztype_idx, male_ztype_idx] += count

    def flatten_all(self) -> NDArray[np.float64]:
        """Flatten the entire state into a single 1D array.

        The order is: tick, then individual_count flattened (row‑major),
        then sperm_storage flattened.

        Returns:
            1D array of floats.
        """
        # Flat format [tick, counts.ravel(), sperm.ravel()] is shared with
        # parse_flattened_state and the recorded history rows; C-order must match.
        tick_arr = np.array([float(self.n_tick)], dtype=np.float64)
        return np.concatenate((tick_arr, self.individual_count.flatten(), self.sperm_storage.flatten()))


class DiscretePopulationState(NamedTuple):
    """Discrete‑generation state container (no sperm storage).

    Attributes:
        n_tick: Current simulation time step.
        individual_count: Array of shape (n_sexes, n_ages, n_ztypes) – counts
            of individuals per sex, age, and zygote type.
    """

    n_tick: int
    individual_count: NDArray[np.float64]

    @classmethod
    def create(
        cls,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
        n_tick: int = 0,
        individual_count: Optional[NDArray[np.float64]] = None,
    ) -> DiscretePopulationState:
        """Create a DiscretePopulationState with optionally provided array.

        Args:
            n_sexes: Number of sexes.
            n_ages: Number of age classes.
            n_ztypes: Number of zygote types (diploid genotypes after slab expansion).
            n_tick: Initial tick value (default 0).
            individual_count: Optional array (n_sexes, n_ages, n_ztypes);
                if None, filled with zeros.

        Returns:
            A new DiscretePopulationState instance.

        Raises:
            AssertionError: If dimensions are invalid or array shape mismatch.
        """
        assert n_sexes > 0, "n_sexes must be positive"
        assert n_ages > 0, "n_ages must be positive"
        assert n_ztypes > 0, "n_ztypes must be positive"
        assert n_tick >= 0, "n_tick must be non-negative"

        if individual_count is None:
            ind = np.zeros((n_sexes, n_ages, n_ztypes), dtype=np.float64)
        else:
            expected_shape = (n_sexes, n_ages, n_ztypes)
            assert individual_count.shape == expected_shape, (
                f"Invalid shape for individual_count: expected {expected_shape}, got {individual_count.shape}"
            )
            # astype copies, keeping the container detached from the caller's array.
            ind = individual_count.astype(np.float64)

        return cls(n_tick=int(n_tick), individual_count=ind)

    def flatten_all(self) -> NDArray[np.float64]:
        """Flatten the entire state into a single 1D array.

        The order is: tick, then individual_count flattened (row‑major).

        Returns:
            1D array of floats.
        """
        # Same layout minus the sperm block: [tick, counts.ravel()].
        tick_arr = np.array([float(self.n_tick)], dtype=np.float64)
        return np.concatenate((tick_arr, self.individual_count.flatten()))


def _validate_flat_length(
    flat_array: NDArray[np.float64],
    expected: int,
    *,
    label: str,
) -> None:
    """Reject a flattened state whose length cannot hold the declared layout.

    Checked before any slicing so a truncated or oversized buffer fails with a
    named length instead of a NumPy reshape error raised somewhere inside the
    parse.

    Args:
        flat_array: Candidate flattened state.
        expected: Exact number of values the layout requires.
        label: Layout name used in the error message.

    Raises:
        ValueError: If the array is not 1-D or does not hold exactly
            *expected* values.
    """
    array = np.asarray(flat_array)
    if array.ndim != 1:
        raise ValueError(f"{label} must be 1-D, got shape {array.shape}")
    if array.size != expected:
        raise ValueError(
            f"{label} must hold {expected} values "
            f"(tick plus the declared state), got {array.size}"
        )


def parse_flattened_state(
    flat_array: NDArray[np.float64],
    n_sexes: Union[int, np.integer],
    n_ages: Union[int, np.integer],
    n_ztypes: Union[int, np.integer],
    copy: bool = True,
) -> PopulationState:
    """Reconstruct a PopulationState from a flattened array.

    The flattened array must be in the format produced by ``flatten_all()``.

    Args:
        flat_array: 1D array containing tick, individual_count, sperm_storage.
        n_sexes: Number of sexes.
        n_ages: Number of age classes.
        n_ztypes: Number of zygote types (diploid genotypes after slab expansion).
        copy: If True, arrays are deep‑copied; otherwise they are viewed.

    Returns:
        A PopulationState instance.

    Raises:
        ValueError: If *flat_array* is not 1-D or its length does not match
            ``1 + n_sexes*n_ages*n_ztypes + n_ages*n_ztypes**2``.
    """
    # Fixed layout [tick | counts | sperm]; end marks the sperm block's start offset.
    _validate_flat_length(
        flat_array,
        1 + int(n_sexes) * int(n_ages) * int(n_ztypes)
        + int(n_ages) * int(n_ztypes) * int(n_ztypes),
        label="flattened state",
    )
    n_tick = int(flat_array[0])
    end = 1 + n_sexes * n_ages * n_ztypes
    individual_count = flat_array[1:end].reshape((n_sexes, n_ages, n_ztypes))
    sperm_storage = flat_array[end:].reshape((n_ages, n_ztypes, n_ztypes))

    # copy=False leaves these as views into flat_array, so the caller must keep
    # that buffer alive; the default detaches both arrays.
    if copy:
        individual_count = individual_count.copy()
        sperm_storage = sperm_storage.copy()

    return PopulationState(
        n_tick=n_tick,
        individual_count=individual_count,
        sperm_storage=sperm_storage,
    )


def parse_flattened_discrete_state(
    flat_array: NDArray[np.float64],
    n_sexes: Union[int, np.integer],
    n_ages: Union[int, np.integer],
    n_ztypes: Union[int, np.integer],
    copy: bool = True,
) -> DiscretePopulationState:
    """Reconstruct a DiscretePopulationState from a flattened array.

    The flattened array must be in the format produced by ``flatten_all()``.

    Args:
        flat_array: 1D array containing tick and individual_count.
        n_sexes: Number of sexes.
        n_ages: Number of age classes.
        n_ztypes: Number of zygote types (diploid genotypes after slab expansion).
        copy: If True, the array is deep‑copied; otherwise it is viewed.

    Returns:
        A DiscretePopulationState instance.

    Raises:
        ValueError: If *flat_array* is not 1-D or its length does not match
            ``1 + n_sexes*n_ages*n_ztypes``.
    """
    # Same fixed layout minus the sperm block: tick then the counts.
    _validate_flat_length(
        flat_array,
        1 + int(n_sexes) * int(n_ages) * int(n_ztypes),
        label="flattened discrete state",
    )
    n_tick = int(flat_array[0])
    individual_count = flat_array[1:].reshape((n_sexes, n_ages, n_ztypes))

    # copy=False views flat_array; copy=True detaches the state from the source row.
    if copy:
        individual_count = individual_count.copy()

    return DiscretePopulationState(
        n_tick=n_tick,
        individual_count=individual_count,
    )
