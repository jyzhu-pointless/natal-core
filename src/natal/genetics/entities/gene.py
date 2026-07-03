"""Gene (allele) entity bound to a Locus structure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ._base import GeneticEntity

if TYPE_CHECKING:
    from ..structures.locus import Locus

from ..structures._base import GeneticStructure


# Gene (entity-level) <- Locus (structure-level)
class Gene(GeneticEntity['Locus']):
    """
    Represents a single allele at a genetic locus.

    A `Gene` must be bound to a `Locus` and is automatically registered
    upon creation. Same name under same `Locus` returns the same instance.
    (Alias: `Allele`)

    Attributes:
        name (str): The name of the gene.
        locus (Locus): The locus structure this gene is bound to.

    Examples:
        >>> locus = Locus("A")
        >>> gene1 = Gene("A1", locus=locus)
        >>> gene2 = Gene("A1", locus=locus)
        >>> assert gene1 is gene2
    """
    structure_type: type[GeneticStructure[Any]]  # Set to Locus at module init time

    def __new__(cls, name: str, locus: Optional[Locus] = None, **kwargs: Any) -> Gene:
        # Pass locus to parent __new__ via kwargs
        return super().__new__(cls, name, locus=locus, **kwargs)

    def __init__(
        self,
        name: str,
        locus: Optional[Locus] = None,
        **kwargs: Any
    ):
        # Prevent re-initialization of cached instances
        if hasattr(self, "_initialized") and self._initialized:
            return

        if locus is None:
            raise TypeError("Gene must be bound to a Locus. Please provide locus parameter.")

        # Set locus alias
        self.locus = locus  # type: ignore[assignment]

        # Store custom parameters as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate name format
        from natal.utils.helpers import validate_name
        if not validate_name(name):
            raise ValueError(f"Invalid gene name format: '{name}'. "
                             f"Gene names must contain only letters, numbers, and underscores.")

        # Check for duplicate gene names in the species
        species = locus.species
        if species is not None:
            # Use the public has_gene method to check for existing gene
            if hasattr(species, 'has_gene') and species.has_gene(name):
                # If has_gene returns True, the gene exists
                existing_gene = species.get_gene(name)
                if existing_gene:
                    raise ValueError(
                        f"Duplicate gene name '{name}' found in species. "
                        f"Gene names must be unique for string-based lookups. "
                        f"Found at locus '{existing_gene.locus.name}' and '{locus.name}'."
                    )

        # Call parent constructor which handles registration
        super().__init__(name, structure=locus)

    def __repr__(self):
        return f"Gene({self.name!r}, locus={self.locus.name!r})"


# Resolve the circular import: Gene.structure_type must be Locus,
# but Locus safely imports Gene only inside method bodies (lazy).
from ..structures.locus import Locus  # noqa: E402

Gene.structure_type = Locus
