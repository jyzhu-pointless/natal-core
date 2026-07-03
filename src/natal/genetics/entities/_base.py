"""Base class for all genetic entities (Gene, Haplotype, HaploidGenotype, Genotype)."""

from __future__ import annotations

from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, cast

from ..structures._base import GeneticStructure

S = TypeVar("S", bound="GeneticStructure[Any]")  # Genetic Structure Type
E = TypeVar("E", bound="GeneticEntity[Any]")  # Concrete entity type for __new__


class GeneticEntity(Generic[S]):
    """
    Base class for genetic entities bound to genetic structures.

    Entities follow three invariants:
    1. An entity must be bound to a structure.
    2. An entity auto-registers to its structure at creation time.
    3. The same entity name under the same structure resolves to one cached instance.

    Attributes:
        structure_type (type[GeneticStructure[Any]]): Required bound structure type
            for subclasses.
        name (str): Entity identifier within its bound structure.
        structure (GeneticStructure[Any]): Bound structure instance.

    Examples:
            gene = Gene("A1", locus=locus_A)  # ✅ Required locus
            assert gene in locus_A.all_entities  # ✅ Auto-registered
            gene2 = Gene("A1", locus=locus_A)  # ✅ Returns same instance
            assert gene is gene2
    """
    structure_type: type[GeneticStructure[Any]] = GeneticStructure  # Override in subclass
    # Cache: {(species_id, structure_type, structure_name, entity_class, entity_name): entity_instance}
    _instance_cache: Dict[Tuple[int, type, str, type, str], object] = {}
    # Late-bound during __new__/__init__. Annotations only (no defaults) so hasattr checks keep working.
    _pending_cache_key: Tuple[int, type, str, type, str]
    _initialized: bool
    structure: GeneticStructure[Any]

    def __new__(
        cls: type[E],
        name: str,
        structure: Any = None,
        **kwargs: Any
    ) -> E:
        # For subclasses that use different parameter names (e.g., locus, chromosome, species)
        # We need to extract the structure from kwargs
        # If structure is not provided as positional arg, check kwargs
        if structure is None:
            structure = kwargs.pop('structure', None)
        actual_structure: Optional[GeneticStructure[Any]] = None
        if isinstance(structure, GeneticStructure):
            actual_structure = cast(GeneticStructure[Any], structure)
        if actual_structure is None:
            # Check common parameter names (new and old names)
            for key in ('locus', 'chromosome', 'species', 'linkage', 'genome'):
                if key in kwargs:
                    candidate = kwargs[key]
                    if isinstance(candidate, GeneticStructure):
                        actual_structure = cast(GeneticStructure[Any], candidate)
                    break

        if actual_structure is None:
            # Will be caught in __init__
            return object.__new__(cls)

        # Get the Species from the structure
        species = getattr(actual_structure, '_species', None)

        if species is None:
            # No species context - create without caching (for backward compatibility)
            return object.__new__(cls)

        # Use Species-level entity cache
        # Cache key: (species id, structure type, structure name, entity class, entity name)
        # This ensures uniqueness within a Species
        cache_key: Tuple[int, type[GeneticStructure[Any]], str, type[E], str] = (
            id(species),
            type(actual_structure),
            str(actual_structure.name),
            cls,
            name,
        )

        cached = GeneticEntity._instance_cache.get(cache_key)
        if cached is not None:
            # Return cached instance
            if isinstance(cached, cls):
                return cached
            raise TypeError(
                f"Cache type mismatch: expected {cls.__name__}, got {type(cached).__name__}."
            )

        # Create new instance (do NOT cache here - cache in __init__ after success)
        instance = object.__new__(cls)
        # Store cache_key for use in __init__
        instance._pending_cache_key = cache_key
        return instance

    def __init__(
        self,
        name: str,
        structure: Any = None,
        **kwargs: Any
    ):
        # Prevent re-initialization of cached instances
        if hasattr(self, "_initialized") and self._initialized:
            return

        if name.strip() == "":
            raise ValueError("Entity name cannot be empty.")

        if structure is None:
            raise TypeError(
                f"{self.__class__.__name__} must be bound to a structure. "
                f"Please provide a valid structure parameter."
            )
        structure = cast(GeneticStructure[Any], structure)
        _ = kwargs  # keep constructor signature aligned with __new__


        # Validate structure type using class attribute
        expected_type = self.__class__.structure_type
        if expected_type != GeneticStructure and not isinstance(structure, expected_type):
            raise TypeError(
                f"structure must be of type {expected_type.__name__}, "
                f"got {type(structure).__name__}."
            )

        self.name = name
        self.structure = structure

        # Auto-register with the structure ("register upon creation")
        register_owner = cast(Any, structure)
        register_owner.register(self)

        # Mark as initialized
        self._initialized = True

        # Cache the instance AFTER successful initialization
        if hasattr(self, '_pending_cache_key'):
            GeneticEntity._instance_cache[self._pending_cache_key] = self
            del self._pending_cache_key

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the instance cache for this entity class.
        Useful for testing or resetting the global state.
        """
        keys_to_remove = [k for k in GeneticEntity._instance_cache if k[3] == cls]
        for key in keys_to_remove:
            del GeneticEntity._instance_cache[key]

    @classmethod
    def clear_all_caches(cls) -> None:
        """
        Clear all entity instance caches.
        """
        GeneticEntity._instance_cache.clear()

    @classmethod
    def clear_species_cache(cls, species_id: int) -> None:
        """Clear entity cache entries that belong to one species id."""
        keys_to_remove = [k for k in GeneticEntity._instance_cache if k[0] == species_id]
        for key in keys_to_remove:
            del GeneticEntity._instance_cache[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, structure={self.structure.name!r})"
