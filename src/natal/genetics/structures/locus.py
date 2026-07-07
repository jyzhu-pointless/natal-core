"""Locus structure — a genetic position on a chromosome.

This module provides :class:`Locus`, the structural blueprint for a single
genetic locus.  A Locus can have multiple :class:`~natal.genetics.entities.gene.Gene`
entities (alleles) registered to it and maintains a positional coordinate
used for recombination rate calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Union

from ._base import GeneticStructure

if TYPE_CHECKING:
    from ..entities.gene import Gene
    from .chromosome import Chromosome


# Locus (model-level) -> Gene (entity-level)
class Locus(GeneticStructure['Gene']):
    """
    Represents a genetic locus with its name.

    A Locus is a blueprint for a genetic position. Multiple Gene entities
    (alleles) can be bound to a single Locus.

    Attributes:
        position (Union[int, float]): The linear position on the chromosome. Used for defining
            recombination rates. If not specified, defaults to max(position) + 1
            among existing loci in the parent Linkage, or 0 if no parent.
        alleles (List[Gene]): Registered allele entities bound to this locus.
    """
    child_structure_type = None  # Locus has no child structures

    def __init__(
        self,
        name: str,
        position: Optional[Union[int, float]] = None,
        chromosome: Optional[Chromosome] = None,
        parent: Optional[Chromosome] = None,
        **kwargs: Any  # extra parameters stored as custom locus attributes
    ):
        """Initialize a Locus structure.

        Computes a default position (max position among siblings + 1) when
        *position* is not given, and registers with the parent Chromosome.

        Args:
            name: Locus name.
            position: Linear position on the chromosome (used for
                recombination ordering).  Defaults to ``max(siblings) + 1``.
            chromosome: Parent Chromosome (alternative to *parent*).
            parent: Parent Chromosome (alias for *chromosome*).
            **kwargs: Extra attributes stored as instance attributes.
        """
        # Check if already initialized (cached instance)
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Parent is alias for chromosome
        if chromosome is None:
            chromosome = parent

        # Save parent reference for cache invalidation
        self._parent_chromosome = chromosome  # type: ignore[assignment]

        # Compute default position before super().__init__
        # (since parent.register may be called)
        if position is None:
            if chromosome is not None and hasattr(chromosome, 'child_structures') and len(chromosome.child_structures) > 0:
                # Default: max position in parent + 1
                max_pos = max(
                    (loc.position for loc in chromosome.child_structures),
                    default=-1
                )
                position = max_pos + 1
            else:
                position = 0

        self._position = position

        # Store custom parameters as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Locus's species is automatically inherited from parent Chromosome
        super().__init__(name, parent=chromosome)

    @property
    def position(self) -> Union[int, float]:
        """The linear position on the chromosome."""
        return self._position

    @position.setter
    def position(self, value: Union[int, float]) -> None:
        """Set the position. Triggers cache invalidation in parent Linkage."""
        self._position = value
        # Invalidate parent's cache if exists
        if hasattr(self, '_parent_chromosome') and self._parent_chromosome is not None:
            self._parent_chromosome.invalidate_recombination_map_cache()

    @property
    def entity_type(self):
        """Return the entity type for this structure.

        Uses a lazy import to avoid circular dependencies.

        Returns:
            The :class:`~natal.genetics.entities.gene.Gene` class.
        """
        from ..entities.gene import Gene
        return Gene

    def register(
        self,
        entity_or_entities: Union[Gene, List[Gene], Tuple[Gene, ...], Set[Gene]]
    ) -> Locus:
        """
        Register gene entities and invalidate species gene index cache.
        """
        super().register(entity_or_entities)
        if self._species is not None:
            self._species.invalidate_gene_index_cache()
        return self

    def unregister(
        self,
        entity_or_entities: Union[Gene, str, List[Union[Gene, str]], Tuple[Union[Gene, str], ...], Set[Union[Gene, str]]]
    ) -> Locus:
        """
        Unregister gene entities and invalidate species gene index cache.
        """
        super().unregister(entity_or_entities)
        if self._species is not None:
            self._species.invalidate_gene_index_cache()
        return self

    @property
    def alleles(self) -> List[Gene]:
        """Alias for all_entities - returns all registered alleles (genes)."""
        return self.all_entities

    def allele_index(self, gene_name: str) -> int:
        """Return the positional index of *gene_name* in this locus's allele list.

        Alleles are stored in registration order, giving each a deterministic
        index used for canonical genotype ordering (e.g. :meth:`Species.unordered_genotype`).

        Raises:
            ValueError: If *gene_name* is not a registered allele at this locus.
        """
        for i, g in enumerate(self.alleles):
            if g.name == gene_name:
                return i
        raise ValueError(
            f"Allele {gene_name!r} not found at locus {self.name!r}"
        )

    def register_allele(self, gene: Gene) -> Locus:
        """Alias for register - register a single allele."""
        return self.register(gene)

    def unregister_allele(self, gene: Gene) -> Locus:
        """Alias for unregister - unregister a single allele."""
        return self.unregister(gene)

    def add_alleles(
        self,
        alleles_or_allele_names: Union[List[Union[Gene, str]], Gene, str],
    ) -> Locus:
        """
        Add one or more alleles (genes) to this locus.

        Args:
            alleles_or_allele_names: Single Gene instance, single allele name (str),
                or list of Gene instances and/or allele names (str).
        Returns:
            The Locus instance (for chaining). Note that for other structure-level add methods,
            the return type is the child structure(s) added. But here we return self for consistency
            with the register_allele/unregister_allele methods.
        """
        from ..entities.gene import Gene

        if isinstance(alleles_or_allele_names, (Gene, str)):
            alleles_or_allele_names = [alleles_or_allele_names]

        for item in alleles_or_allele_names:
            assert isinstance(item, (Gene, str)), f"Expected Gene or str, got {type(item).__name__} instead."
            if isinstance(item, Gene):
                self.register(item)
            else:
                Gene(item, locus=self)  # Auto-registers via Gene.__init__

        return self

    @classmethod
    def with_alleles(
        cls,
        name: str,
        alleles_or_allele_names: Union[List[Union[Gene, str]], Gene, str],
        position: Optional[Union[int, float]] = None
    ) -> Locus:
        """
        Factory method to create a Locus and register alleles (genes) by names.

        Args:
            name: Name of the locus.
            alleles_or_allele_names: Single Gene instance, single allele name (str),
                or list of Gene instances and/or allele names (str).
            position: Optional position on the chromosome.

        Returns:
            Locus instance with registered alleles.

        Examples:
            >>> locus = Locus.with_alleles("A", ["A1", "A2", "A3"])
            >>> locus.alleles  # -> [Gene("A1"), Gene("A2"), Gene("A3")]
        """
        return cls(name, position=position).add_alleles(alleles_or_allele_names)

    def __repr__(self) -> str:
        """Return a string representation of this Locus."""
        allele_names = [g.name for g in self.alleles]
        return f"Locus({self.name!r}, position={self.position}, alleles={allele_names})"
