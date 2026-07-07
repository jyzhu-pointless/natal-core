"""RecombinationMap — stores recombination rates between adjacent loci on a chromosome.

Provides :class:`RecombinationMap`, a 1D container that stores and
retrieves recombination rates between adjacent loci.  Supports single-locus
lookups, pairwise cumulative-rate queries, and slice/array indexing.

Originally defined as an inner class of Chromosome, factored out for module size.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

if TYPE_CHECKING:
    from .locus import Locus

import numpy as np
from numpy.typing import NDArray


class RecombinationMap:
    """
    A 1D container storing recombination rates between adjacent loci.

    For n loci, the map has n-1 entries where entry i is the recombination
    rate between locus i and locus i+1 (in sorted order by position).

    Attributes:
        loci_names (List[str]): Locus names in sorted positional order.
        _rates (np.ndarray): Adjacent-locus recombination rates with length n-1.

    Examples:
        For loci [A, B, C, D], the map is [r(A,B), r(B,C), r(C,D)]
        where index i = rate between locus i and locus i+1.
    """
    loci_names: List[str]
    _rates: np.ndarray
    _KeyType = Union[int, str, 'Locus']

    def __init__(
        self,
        loci: Optional[List[Locus]] = None,
        rates: Optional[np.ndarray] = None
    ) -> None:
        """Initialize a RecombinationMap.

        Args:
            loci: Ordered list of Locus instances (at least 2).
            rates: Optional array of initial recombination rates.  Must
                have length ``len(loci) - 1``.

        Raises:
            ValueError: If fewer than 2 loci are provided, or if *rates*
                length does not match.
        """
        size = len(loci) - 1 if loci and len(loci) > 1 else 0
        if size <= 0:
            raise ValueError("RecombinationMap requires at least 2 loci.")

        self._rates = np.zeros(size, dtype=np.float64)
        self.loci_names = [locus.name for locus in loci] if loci else []

        if rates is not None:
            if len(rates) != size:
                raise ValueError(f"Expected {size} rates, got {len(rates)}")
            self._rates[:] = rates

    # ---------- Name conversion ----------
    def _name_to_index(self, name: str) -> int:
        """Convert locus name to index in the loci list.

        Args:
            name: Locus name.

        Returns:
            Integer index in the loci list.

        Raises:
            KeyError: If the name is not found.
        """
        if not self.loci_names:
            raise ValueError("No loci names defined in map.")
        try:
            return self.loci_names.index(name)
        except ValueError:
            raise KeyError(f"Locus name '{name}' not found.") from ValueError

    def name_to_index(self, name: str) -> int:
        """Public wrapper for converting locus name to locus index."""
        return self._name_to_index(name)

    def _normalize_single_key(self, key: _KeyType) -> int:
        """Normalize a single key (int, str, or Locus) to integer index.

        Args:
            key: A locus index, name, or Locus instance.

        Returns:
            Integer index in the loci list.

        Raises:
            KeyError: If the key does not correspond to a registered locus.
        """
        if isinstance(key, str):
            return self._name_to_index(key)
        elif isinstance(key, self._get_locus_class()):
            return self._name_to_index(key.name)
        else:
            return key

    @staticmethod
    def _get_locus_class():
        """Return the Locus class (lazy import to avoid circular deps).

        Returns:
            The :class:`~natal.genetics.structures.locus.Locus` class.
        """
        from .locus import Locus
        return Locus

    # ---------- Reading ----------
    @overload
    def __getitem__(self, key: _KeyType) -> float: ...

    @overload
    def __getitem__(self, key: Tuple[_KeyType, _KeyType]) -> float: ...

    @overload
    def __getitem__(self, key: Union[slice, NDArray[np.integer], List[int]]) -> NDArray[np.float64]: ...

    def __getitem__(
        self, key: Union[
            _KeyType,
            Tuple[_KeyType, _KeyType],
            Union[slice, NDArray[np.integer], List[int]]
        ]
    ) -> Union[float, NDArray[np.float64]]:
        """Retrieve recombination rate(s) from the map.

        Three indexing modes are supported:

        1. Single specifier: Returns the rate between the specified locus and the
        next locus. The specifier can be an integer index, locus name (str), or
        a `Locus` object.

        2. Tuple of two specifiers: Returns the cumulative recombination rate
        between the two loci (sum of all adjacent rates in the interval),
        capped at 0.5 (independent assortment).

        3. Slice or list of indices: Returns an array of rates for the specified
        positions.

        Args:
            key: A single specifier, a pair of specifiers, or a slice/array/list
                of integer indices.

        Returns:
            A single float for single-locus or pair requests, otherwise a NumPy
            array of rates.

        Raises:
            KeyError: If a specifier does not correspond to a registered locus.
        """
        # Case 1: tuple of two specifiers -> cumulative rate
        if isinstance(key, tuple) and len(key) == 2:
            a, b = key
            idx_a = self._normalize_single_key(a)
            idx_b = self._normalize_single_key(b)
            # Ensure correct order
            if idx_a > idx_b:
                idx_a, idx_b = idx_b, idx_a
            # Sum rates in the interval [idx_a, idx_b)
            total_rate = 0.0
            for i in range(idx_a, idx_b):
                total_rate += float(self._rates[i])
            return float(min(total_rate, 0.5))

        # Case 2: slice or array-like -> return array
        if isinstance(key, (slice, np.ndarray, list)):
            result = self._rates[key]
            return result

        # Case 3: single specifier (int, str, Locus)
        idx = self._normalize_single_key(key)
        return float(self._rates[idx])

    # ---------- Writing ----------
    def __setitem__(
        self,
        key: Union[
            _KeyType,
            Tuple[_KeyType, _KeyType],
            slice,
            List[int],
            NDArray[np.integer]
        ],
        value: Union[float, np.ndarray]
    ) -> None:
        """
        Set recombination rate(s).

        Two forms are supported:

        - Single specifier: Sets the rate between the locus and the next locus.
        - Tuple of two specifiers: Sets the rate between two adjacent loci.
        - Slice or array indexing: Sets rates for multiple adjacent intervals.

        The specifier can be an integer index, a locus name (str), or a `Locus`
        object.

        Args:
            key: Single specifier, pair of adjacent specifiers, or slice/array index.
            value: New rate(s). A scalar value is broadcast to all selected
                positions; an array must have the correct shape.

        Raises:
            ValueError: If any new rate is outside the [0, 0.5] range.
            KeyError: If the specifier does not correspond to a registered locus,
                or if a pair of specifiers does not represent adjacent loci.

        Note:
            Modifying recombination rates after `Genotype.produce_gametes()`
            has been called will **not** invalidate the gamete cache. You must
            manually clear the cache: `genotype._gamete_cache = None`.
        """
        arr_val = np.asarray(value, dtype=self._rates.dtype)

        if np.any((arr_val < 0) | (arr_val > 0.5)):
            raise ValueError("Recombination rates must be in [0, 0.5]")

        if isinstance(key, tuple) and len(key) == 2:
            # Set rate between adjacent loci
            a, b = key
            idx_a = self._normalize_single_key(a)
            idx_b = self._normalize_single_key(b)
            if abs(idx_a - idx_b) != 1:
                raise KeyError(
                    f"Loci {a!r} and {b!r} are not adjacent. "
                    f"RecombinationMap only stores rates between adjacent loci."
                )
            self._rates[min(idx_a, idx_b)] = arr_val
        else:
            # For other keys (single specifier, slice, list, array), convert
            # specifiers to integer indices first, then assign.
            # Note: slice, list, array are passed directly to _rates; they must
            # already contain integer indices (no str/Locus conversion needed).
            idx = (self._normalize_single_key(key) if not isinstance(key, (slice, list, np.ndarray))
                   else key)
            self._rates[idx] = arr_val

    # ---------- Visualization ----------
    def __repr__(self) -> str:
        """Return a formatted representation of the recombination map."""
        return self._formatted_repr()

    def __str__(self) -> str:
        """Return a formatted string of the recombination map."""
        return self._formatted_repr()

    def _formatted_repr(self) -> str:
        if self.loci_names and len(self.loci_names) > 1:
            pairs: List[str] = []
            for i in range(len(self)):
                pairs.append(f"r({self.loci_names[i]},{self.loci_names[i+1]})={self[i]:.3f}")
            return f"RecombinationMap([{', '.join(pairs)}])"
        else:
            return f"RecombinationMap({np.array2string(self._rates, precision=3)})"

    # ---------- Utility methods ----------
    def validate(self) -> Tuple[bool, str]:
        """Validate the recombination map."""
        if np.any(self._rates < 0) or np.any(self._rates > 0.5):
            return False, "Values out of range [0, 0.5]."
        return True, "Map is valid."

    def get_adjacent_loci(self, index: int) -> Tuple[str, str]:
        """Get the names of the two loci at the given rate index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for map of size {len(self)}")
        return (self.loci_names[index], self.loci_names[index + 1])

    def __len__(self) -> int:
        """Return the number of adjacent-locus intervals."""
        return len(self._rates)

    def __iter__(self):
        """Iterate over recombination rates."""
        return iter(self._rates)

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        if dtype is not None:
            return np.asarray(self._rates, dtype=dtype)
        arr = np.asarray(self._rates)
        if copy:
            arr = arr.copy()
        return arr

    @property
    def dtype(self):
        """Return the NumPy dtype of the underlying rates array."""
        return self._rates.dtype
