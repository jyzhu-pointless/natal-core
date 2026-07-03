"""Chromosome structure — groups linked loci with recombination rates."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from ._base import GeneticStructure
from ._types import SexChromosomeType
from .chromosome_map import RecombinationMap
from .locus import Locus

if TYPE_CHECKING:
    from ._registry import ChildStructureRegistry
    from .species import Species


# Chromosome (structure-level) -> Haplotype (entity-level)
class Chromosome(GeneticStructure['Haplotype']):  # pyright: ignore[reportUndefinedVariable]
    """
    Represents a chromosome structure with linkage information among loci.

    A Chromosome groups multiple Loci that are physically linked on the same
    chromosome. It also stores the recombination rates between loci.

    Attributes:
        sex_type (SexChromosomeType): Sex chromosome type.
            - None or 'autosome': Autosome (default)
            - 'X': X chromosome in XY system
            - 'Y': Y chromosome in XY system (paternal only)
            - 'Z': Z chromosome in ZW system
            - 'W': W chromosome in ZW system (maternal only)
        loci (List[Locus]): Child loci sorted by position.
        recombination_map (Chromosome.RecombinationMap): Adjacent-locus recombination map.
        recombination_matrix (Chromosome.RecombinationMap): Backward-compatible alias
            of recombination_map.
        is_sex_chromosome (bool): Whether this chromosome participates in sex determination.
        is_autosome (bool): Whether this chromosome is an autosome.
        sex_system (Optional[str]): Sex system inferred from sex_type ('XY', 'ZW', or None).

    Examples:
        >>> chr_x = Chromosome('X', sex_type='X')
        >>> chr_y = Chromosome('Y', sex_type='Y')
        >>> print(chr_x.is_sex_chromosome)  # True
        >>> print(chr_y.sex_type.paternal_only)  # True

    This class is also exported as Linkage for backward compatibility.
    """
    child_structure_type = Locus  # Chromosome contains Loci as children
    child_structures: ChildStructureRegistry[Locus]

    def __init__(
        self,
        name: str,
        loci: Optional[List[Locus]] = None,
        species: Optional[Species] = None,
        parent: Optional[Species] = None,
        recombination_rates: Optional[Union[List[float], np.ndarray]] = None,
        sex_type: Optional[Union[SexChromosomeType, str]] = None,
    ):
        # Initialize placeholders BEFORE super().__init__
        # because __iter__ may be called during parent registration
        if not hasattr(self, '_recombination_map'):
            self._recombination_map: Optional[RecombinationMap] = None
            self._sorted_loci_cache: Optional[List[Locus]] = None  # Cache for sorted loci
            self._sex_type: SexChromosomeType = SexChromosomeType.AUTOSOME

        # Check if already initialized (cached instance)
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Parent is alias for species
        if species is None:
            species = parent

        # Set sex chromosome type
        self._set_sex_type(sex_type)

        # Chromosome's species is automatically inherited from parent Species
        super().__init__(name, parent=species)

        if loci:
            for locus in loci:
                self.add_locus(locus)

        # Initialize recombination map
        self._invalidate_recombination_map_cache()

        # Set recombination rates if provided
        if recombination_rates is not None:
            if len(self.loci) < 2:
                raise ValueError("Cannot set recombination rates with less than 2 loci.")
            if len(recombination_rates) != len(self.loci) - 1:
                raise ValueError(
                    f"Expected {len(self.loci) - 1} recombination rates for {len(self.loci)} loci, "
                    f"got {len(recombination_rates)} instead."
                )
            for i, rate in enumerate(recombination_rates):
                self.recombination_map[i] = rate

    def _set_sex_type(self, sex_type: Optional[Union[SexChromosomeType, str]]) -> None:
        """Set sex chromosome type (internal method)"""
        assert isinstance(sex_type, (SexChromosomeType, str, type(None))), f"Expected SexChromosomeType or str, got {type(sex_type).__name__}"
        if sex_type is None:
            self._sex_type = SexChromosomeType.AUTOSOME
        elif isinstance(sex_type, SexChromosomeType):
            self._sex_type = sex_type
        else:
            sex_type_upper = sex_type.upper()
            if sex_type_upper in ('AUTOSOME', 'AUTO', 'A', ''):
                self._sex_type = SexChromosomeType.AUTOSOME
            elif sex_type_upper == 'X':
                self._sex_type = SexChromosomeType.X
            elif sex_type_upper == 'Y':
                self._sex_type = SexChromosomeType.Y
            elif sex_type_upper == 'Z':
                self._sex_type = SexChromosomeType.Z
            elif sex_type_upper == 'W':
                self._sex_type = SexChromosomeType.W
            else:
                raise ValueError(
                    f"Unknown sex_type: {sex_type!r}. "
                    f"Valid values: 'X', 'Y', 'Z', 'W', 'autosome', or SexChromosomeType enum."
                )

    @property
    def sex_type(self) -> SexChromosomeType:
        """Returns the sex chromosome type"""
        return self._sex_type

    @sex_type.setter
    def sex_type(self, value: Optional[Union[SexChromosomeType, str]]) -> None:
        """Set the sex chromosome type"""
        self._set_sex_type(value)

    @property
    def is_sex_chromosome(self) -> bool:
        """Whether this is a sex chromosome"""
        return self._sex_type.is_sex_chromosome

    @property
    def is_autosome(self) -> bool:
        """Whether this is an autosome"""
        return not self.is_sex_chromosome

    @property
    def sex_system(self) -> Optional[str]:
        """Returns the sex determination system this chromosome belongs to ('XY', 'ZW', or None)"""
        return self._sex_type.sex_system

    @property
    def entity_type(self):
        """Lazy import to avoid circular dependency."""
        from ..entities.haplotype import Haplotype
        return Haplotype

    @property
    def loci(self) -> List[Locus]:
        """Returns the list of loci in this chromosome, sorted by position (cached)."""
        if self._sorted_loci_cache is None:
            self._sorted_loci_cache = sorted(
                self.child_structures.all,
                key=lambda loc: loc.position
            )
        return self._sorted_loci_cache

    def _invalidate_recombination_map_cache(self) -> None:
        """Invalidate sorted loci cache and update recombination map."""
        self._sorted_loci_cache = None
        self._update_recombination_map()

    @property
    def recombination_map(self) -> RecombinationMap:
        """Returns the recombination map for this chromosome.

        The recombination map stores recombination rates between adjacent loci.
        For n loci, the map has n-1 entries where entry i is the recombination
        rate between locus i and locus i+1.
        """
        if self._recombination_map is None:
            self._update_recombination_map()
        if self._recombination_map is None:
            raise ValueError("Recombination map is unavailable for chromosomes with fewer than 2 loci.")
        return self._recombination_map

    # Backward compatibility alias
    @property
    def recombination_matrix(self) -> RecombinationMap:
        """Deprecated: Use recombination_map instead."""
        return self.recombination_map

    def add_locus(
        self,
        locus_or_name: Union[Locus, str],
        position: Optional[Union[int, float]] = None,
        recombination_rate_with_previous: float = 0.0,
        **kwargs: Any
    ) -> Locus:
        """
        Add a locus to this chromosome.

        When inserting a new locus:
        - If it's the first locus of the chromosome: the recombination rate parameter
          sets the rate with the next (second) locus.
        - Otherwise: the recombination rate parameter sets the rate with the previous
          locus, and the rate with the next locus is inherited from the old rate
          between the previous and next loci.

        Args:
            locus_or_name: Either a Locus instance or a name to create a new Locus.
            position: Optional position (only used when creating new Locus by name).
                If not specified, defaults to max(position) + 1 among existing loci.
            recombination_rate_with_previous: Recombination rate with the adjacent locus.
                Defaults to 0 (complete linkage). If the first locus of the chromosome,
                sets the rate with the second locus; otherwise sets the rate with the
                previous locus.
            **kwargs: Additional custom parameters to pass to the Locus constructor.

        Returns:
            The added Locus instance.
        """
        # Get current sorted loci and old map before adding
        old_sorted_loci = self.loci.copy() if self._sorted_loci_cache else []
        old_map = self._recombination_map

        assert isinstance(locus_or_name, (Locus, str)), \
            f"Expected Locus instance or str, got {type(locus_or_name).__name__}"
        if isinstance(locus_or_name, str):
            # Create new Locus via base class add method with kwargs
            created = self.add(locus_or_name, position=position, **kwargs)
            assert isinstance(created, Locus), \
                f"Expected add() to return Locus, got {type(created).__name__}"
            locus = created
        else:
            locus = locus_or_name
            # Register existing Locus if not already in registry
            if locus.name not in self.child_structures:
                self.child_structures.register(locus)

        # Invalidate cache and update recombination map with insertion handling
        self._sorted_loci_cache = None
        self._update_recombination_map_on_insert(
            locus, old_sorted_loci, old_map, recombination_rate_with_previous
        )
        if self._species is not None:
            self._species.invalidate_gene_index_cache()
        return locus

    def remove_locus(self, locus_or_name: Union[Locus, str]) -> None:
        """
        Remove a locus from this chromosome.

        When removing a locus, the recombination rates are adjusted to maintain
        connectivity between the remaining loci.

        Args:
            locus_or_name: Either a Locus instance or a name.
        """
        if isinstance(locus_or_name, str):
            name = locus_or_name
        else:
            name = locus_or_name.name

        if name in self.child_structures:
            # Get old state
            old_sorted_loci = self.loci.copy()
            old_map = self._recombination_map

            # Find the index of the locus to remove
            locus_to_remove = self.child_structures.get(name)
            remove_idx = old_sorted_loci.index(locus_to_remove)

            # Unregister the locus
            self.child_structures.unregister(name)
            self._sorted_loci_cache = None

            # Update recombination map
            self._update_recombination_map_on_remove(remove_idx, old_map)
            if self._species is not None:
                self._species.invalidate_gene_index_cache()

    def get_locus(self, name: str) -> Optional[Locus]:
        """
        Get a locus by name.

        Args:
            name: Name of the locus.

        Returns:
            The Locus instance or None if not found.
        """
        if name in self.child_structures:
            return self.child_structures.get(name)
        return None

    def _update_recombination_map(self, old_sorted_loci: Optional[List[Locus]] = None,
                                old_map: Optional[RecombinationMap] = None) -> None:
        """Create a recombination map.

        If order remains the same, preserve recombination rates.
        If order changes, simulate removal and reinsertion of moved loci.
        """
        if len(self.child_structures) <= 1:
            self._recombination_map = None
            return

        new_sorted_loci = self.loci

        # If we have old state information and order remains the same, preserve rates
        if (old_sorted_loci and old_map and len(old_sorted_loci) == len(new_sorted_loci) and
            all(a == b for a, b in zip(old_sorted_loci, new_sorted_loci))):
            # Order unchanged, preserve all recombination rates by extracting them from old map
            old_rates = np.array([old_map[i] for i in range(len(old_sorted_loci) - 1)], dtype=np.float64)
            self._recombination_map = RecombinationMap(loci=new_sorted_loci, rates=old_rates)
        elif old_sorted_loci and old_map and len(old_sorted_loci) == len(new_sorted_loci):
            # Order changed, simulate removal and reinsertion of moved loci
            # Find which loci have changed position
            differences = 0
            moved_locus = None
            old_idx = -1

            for old_locus, new_locus in zip(old_sorted_loci, new_sorted_loci):
                if old_locus != new_locus:
                    differences += 1
                    moved_locus = new_locus
                    # Find old index of the moved locus
                    old_idx = old_sorted_loci.index(moved_locus)

            if differences == 1 and moved_locus is not None:
                # Single locus moved - use the encapsulated method
                self._update_recombination_map_on_move(moved_locus, old_sorted_loci, old_map, old_idx)
            else:
                # Multiple loci moved, create fresh map
                self._recombination_map = RecombinationMap(loci=new_sorted_loci)
        else:
            # No old state information or different lengths, create fresh map
            self._recombination_map = RecombinationMap(loci=new_sorted_loci)

    def invalidate_recombination_map_cache(self) -> None:
        """Public wrapper for recombination-map cache invalidation."""
        self._invalidate_recombination_map_cache()

    def _update_recombination_map_on_insert(
        self,
        new_locus: Locus,
        old_sorted_loci: List[Locus],
        old_map: Optional[RecombinationMap],
        recombination_rate_with_previous: float
    ) -> None:
        """Update recombination map when a new locus is inserted.

        When inserting a new locus:
        - If it's the first locus of the chromosome: uses recombination_rate_with_previous
          for the rate with the next (second) locus.
        - Otherwise: uses recombination_rate_with_previous for the rate with the previous
          locus, and inherits the rate with the next locus from the old rate between
          the previous and next loci.
        """
        new_sorted_loci = self.loci
        n = len(new_sorted_loci)

        if n <= 1:
            self._recombination_map = None
            return

        # Find the new position of the inserted locus
        new_idx = new_sorted_loci.index(new_locus)

        # Create new map
        new_rates = np.zeros(n - 1)

        if old_map is not None and len(old_sorted_loci) > 1:
            # Copy old rates, adjusting for insertion
            for new_i in range(n - 1):
                if new_i == new_idx - 1:
                    # Rate between previous locus and new locus
                    new_rates[new_i] = recombination_rate_with_previous
                elif new_idx == 0 and new_i == 0:
                    # Rate between new locus (first) and next locus
                    new_rates[new_i] = recombination_rate_with_previous
                elif new_i >= new_idx:
                    # Rate between new locus and next locus
                    # Inherit from the old rate between prev and next
                    old_idx = new_i - 1  # Account for insertion
                    new_rates[new_i] = old_map[old_idx] if old_idx < len(old_map) else 0.0
                else:  # new_i < new_idx
                    # Copy from old map
                    new_rates[new_i] = old_map[new_i] if new_i < len(old_map) else 0.0
        else:
            # First locus of the chromosome, set the rate with the next locus (second locus) instead
            new_rates[0] = recombination_rate_with_previous

        self._recombination_map = RecombinationMap(
            loci=new_sorted_loci,
            rates=new_rates
        )

    def _update_recombination_map_on_remove(
        self,
        remove_idx: int,
        old_map: Optional[RecombinationMap]
    ) -> None:
        """Update recombination map when a locus is removed.

        When removing locus C from [A, C, B], the new rate r(A,B) = r(A,C) + r(C,B).
        This follows the additive property of genetic distances for small rates.
        """
        new_sorted_loci = self.loci
        n = len(new_sorted_loci)

        if n <= 1:
            self._recombination_map = None
            return

        new_rates = np.zeros(n - 1)

        if old_map is not None:
            for new_i in range(n - 1):
                if new_i < remove_idx - 1:
                    # Before the pair affected by removal
                    new_rates[new_i] = old_map[new_i] if new_i < len(old_map) else 0.0
                elif new_i == remove_idx - 1:
                    # This is the new adjacent pair created by removal
                    # r(A,B) = r(A,C) + r(C,B) where C was removed
                    rate_before = old_map[remove_idx - 1] if remove_idx - 1 < len(old_map) else 0.0
                    rate_after = old_map[remove_idx] if remove_idx < len(old_map) else 0.0
                    new_rates[new_i] = min(rate_before + rate_after, 0.5)  # Cap at 0.5
                else:
                    # After the affected pair - shift by 1
                    old_idx = new_i + 1
                    new_rates[new_i] = old_map[old_idx] if old_idx < len(old_map) else 0.0

        self._recombination_map = RecombinationMap(
            loci=new_sorted_loci,
            rates=new_rates
        )

    def _update_recombination_map_on_move(
        self,
        moved_locus: Locus,
        old_sorted_loci: List[Locus],
        old_map: RecombinationMap,
        old_idx: int
    ) -> None:
        """Update recombination map when a locus is moved.

        Simulates removal and reinsertion of the moved locus.
        """
        # Step 1: Remove the locus from old position
        temp_loci = [loc for loc in old_sorted_loci if loc != moved_locus]

        # Apply removal logic using existing method
        if len(temp_loci) > 1:
            self._update_recombination_map_on_remove(old_idx, old_map)
            temp_map = self._recombination_map

            # Step 2: Reinsert the locus at new position with rate=0
            # Temporarily set the recombination map to the post-removal state
            self._recombination_map = temp_map
            self._update_recombination_map_on_insert(moved_locus, temp_loci, temp_map, recombination_rate_with_previous=0.0)
        else:
            # Only one locus remains after removal, create fresh map
            self._recombination_map = RecombinationMap(loci=self.loci)

    def get_locus_index(self, name: str) -> int:
        """Get the index of a locus by name in the sorted loci list."""
        return self.recombination_map.name_to_index(name)

    def set_recombination(self, locus_a: Union[Locus, str], locus_b: Union[Locus, str], rate: float):
        """
        Set the recombination rate between two adjacent loci.

        Args:
            locus_a: First locus (by name or Locus object)
            locus_b: Second locus (by name or Locus object)
            rate: Recombination rate (must be in [0, 0.5])

        Raises:
            KeyError: If the loci are not adjacent
            ValueError: If rate is out of range or fewer than 2 loci
        """
        if self._recombination_map is None:
            raise ValueError("Cannot set recombination rate with fewer than 2 loci.")
        self.recombination_map[locus_a, locus_b] = rate

    def set_recombination_bulk(self, settings: Dict[Tuple[Union[Locus, str], Union[Locus, str]], float]):
        """
        Bulk set recombination rates between adjacent loci.

        Args:
            settings: Dictionary of {(locus_a, locus_b): rate}
        """
        for (a, b), rate in settings.items():
            self.set_recombination(a, b, rate)

    def set_recombination_all(self, value: float):
        """
        Set all recombination rates to the same value.

        Args:
            value: Recombination rate (must be in [0, 0.5])
        """
        if self._recombination_map is not None:
            self._recombination_map[:] = value

    # Backward compatibility alias
    def set_recombination_default(self, value: float):
        """Deprecated: Use set_recombination_all instead."""
        self.set_recombination_all(value)

    def set_recombination_rate(self, locus_a: Union[Locus, str], locus_b: Union[Locus, str], rate: float):
        """
        Deprecated: Use set_recombination instead.
        """
        self.set_recombination(locus_a, locus_b, rate)

    def set_recombination_rates(self, settings: Dict[Tuple[Union[Locus, str], Union[Locus, str]], float]):
        """
        Deprecated: Use set_recombination_bulk instead.
        """
        self.set_recombination_bulk(settings)

    def __repr__(self):
        return f"Chromosome({self.name!r}, loci={[loc.name for loc in self.loci]})"

    def __iter__(self):
        return iter(self.loci)

    def __len__(self):
        return len(self.loci)


# Assign RecombinationMap as inner class of Chromosome (was originally defined inline)
Chromosome.RecombinationMap = RecombinationMap  # pyright: ignore[reportAttributeAccessIssue]
Chromosome.RecombinationMatrix = RecombinationMap  # Backward compatibility alias  # pyright: ignore[reportAttributeAccessIssue]
