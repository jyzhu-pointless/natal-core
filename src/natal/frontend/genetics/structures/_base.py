"""Base class for all genetic structures (Species, Chromosome, Locus).

Provides :class:`GeneticStructure`, the abstract base for all structural
blueprints in the genetics package.  Structures define the hierarchical
architecture of a species (Species → Chromosome → Locus) and manage
child-structure registries and entity bindings.
"""

from __future__ import annotations

import importlib
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ._registry import (
    GLOBAL_STRUCTURE_CACHE,
    ChildStructureRegistry,
    EntityRegistry,
)
from ._types import E

if TYPE_CHECKING:
    from .species import Species


def ensure_type(obj: Any, expected_type: type) -> None:
    """
    Ensures that an object is an instance of a given class, with lazy import.

    Args:
        obj (any): The object to check
        expected_type (type): The expected class type.

    Raises:
        TypeError: If obj is not an instance of the specified class.
    """
    module = importlib.import_module(expected_type.__module__)  # Lazy import
    cls = getattr(module, expected_type.__name__)
    if not isinstance(obj, cls):
        raise TypeError(
            f"Expected {expected_type.__name__} from {expected_type.__module__}, got {type(obj).__name__} instead."
        )


class GeneticStructure(Generic[E]):
    """
    Base class for genetic structures.

    Structure uniqueness is now scoped to a Species, not globally.
    Within the same Species, structures of the same type must have unique names.

    Attributes:
        child_structure_type (Optional[type[GeneticStructure[Any]]]): Child structure
            class used by subclasses, or None when no child structures are supported.
        name (str): Structure identifier unique within the same structure type and species.
        species (Optional[Species]): Species this structure is currently bound to.
        all_entities (List[E]): Snapshot of currently registered runtime entities.

    Examples:
        >>> species1 = Species("Species1")
        >>> locus1 = Locus("A", species=species1)
        >>> locus2 = Locus("A", species=species1)
        >>> assert locus1 is locus2  # Same object within species1
        >>>
        >>> species2 = Species("Species2")
        >>> locus3 = Locus("A", species=species2)
        >>> assert locus1 is not locus3  # Different speciess allow same name
    """
    child_structure_type: Optional[type[GeneticStructure[Any]]] = None  # Child structure type per subclass
    _species: Optional[Species]

    @property
    def species(self) -> Optional[Species]:
        """Public accessor for the bound Species."""
        return self._species

    def __new__(
        cls,
        name: str,
        *args: Any,
        **kwargs: Any
    ):
        """Create or retrieve a cached GeneticStructure instance.

        When a Species context is available the instance is cached under
        the species-scoped cache; otherwise the global fallback cache is
        used.  Returns the existing cached instance if one with the same
        name already exists.

        Args:
            name: Structure name (must be unique within type and species).
            *args: Positional arguments forwarded to ``__init__``.
            **kwargs: Keyword arguments forwarded to ``__init__``.
                ``species`` and ``parent`` are intercepted for cache
                resolution.

        Returns:
            A new or cached structure instance.

        Raises:
            AssertionError: If *name* is not a string or is empty.
        """
        # Extract species and parent from kwargs
        species = kwargs.get('species')
        parent = kwargs.get('parent')

        assert isinstance(name, str), "Structure name must be a string."
        if name.strip() == "":
            raise ValueError("Structure name cannot be empty.")

        # Determine which cache to use
        target_species = None
        if species is not None:
            assert isinstance(species, Species), "species must be a Species instance."
            target_species = species
        elif parent is not None:
            assert isinstance(parent, GeneticStructure), "parent must be a GeneticStructure instance."
            target_species = parent.species

        # Get the appropriate cache
        if target_species is not None:
            # Use Species-scoped cache
            if cls not in target_species.structure_cache:
                target_species.structure_cache[cls] = {}
            cache = target_species.structure_cache[cls]
        else:
            # Use global fallback cache for structures without species
            if cls not in GLOBAL_STRUCTURE_CACHE:
                GLOBAL_STRUCTURE_CACHE[cls] = {}
            cache = GLOBAL_STRUCTURE_CACHE[cls]

        # Check if instance already exists in cache
        if name in cache:
            warnings.warn(
                f"Structure '{name}' of type {cls.__name__} already exists in cache. "
                f"Returning cached instance. "
                f"Note: Structure names must be unique within the same type.",
                UserWarning,
                stacklevel=2
            )
            return cache[name]

        # Create new instance (do NOT cache here - cache in __init__ after success)
        instance = super().__new__(cls)

        return instance

    def __init__(
        self,
        name: str,
        parent: Optional[GeneticStructure[Any]] = None,
        species: Optional[Species] = None
    ):
        """Initialize a GeneticStructure.

        Sets up the entity registry, resolves species binding (from
        *species*, *parent*, or self for a top-level Species), initialises
        the child-structure registry if applicable, and registers with the
        parent.

        Args:
            name: Structure name (must be non-empty).
            parent: Parent structure, or ``None`` for top-level structures.
            species: Explicit Species reference, or ``None`` to inherit
                from *parent*.

        Raises:
            AssertionError: If *name* is empty or *parent* validation fails.
        """
        # Prevent re-initialization of cached instances
        if hasattr(self, "_initialized") and self._initialized:
            return

        assert isinstance(name, str), "Structure name must be a string."
        if name.strip() == "":
            raise ValueError("Structure name cannot be empty.")

        # Registry wiring:
        # - _entities tracks runtime-bound entity instances (Gene/Haplotype/...)
        # - child_structures (if enabled by subclass) tracks structural children
        #   (Locus under Chromosome, Chromosome under Species, etc.)
        #
        # entity_type remains a subclass property to support lazy import and avoid
        # circular imports with natal.genetic_entities.
        self.name = name
        self._entities: EntityRegistry[E] = EntityRegistry()

        # Track the root Species for this structure
        if species is not None:
            self._species = species
        elif parent is not None:
            # Inherit species from parent
            self._species = parent.species
        else:
            # This is a Species itself
            self._species = None

        # Initialize child structures registry if applicable
        cls = self.__class__
        if cls.child_structure_type:
            self.child_structures = ChildStructureRegistry[cls.child_structure_type](
                owner=self,
                expected_type=cls.child_structure_type
            )

        # Strict constraint: must be added to a parent unless top-level
        if parent is not None:
            assert isinstance(parent, GeneticStructure), \
                "parent must be a GeneticStructure instance."
            # Register this structure as a child of the parent
            assert parent.child_structures is not None, \
                f"Parent {parent.__class__.__name__} does not support child structures."
            parent.child_structures.register(self)

        # Mark as initialized, avoiding re-initialization when created from cache
        self._initialized = True

        # Cache the instance AFTER successful initialization
        self._add_to_cache(self._species)

    def _get_cache_for_species(self, species: Optional[Species]) -> Dict[str, GeneticStructure[E]]:
        """Get the appropriate cache for the given species.

        Args:
            species: Target species, or ``None`` for the global fallback cache.

        Returns:
            The species-scoped or global cache dict.
        """
        cls = self.__class__
        if species is not None:
            if cls not in species.structure_cache:
                species.structure_cache[cls] = {}
            return species.structure_cache[cls]
        else:
            if cls not in GLOBAL_STRUCTURE_CACHE:
                GLOBAL_STRUCTURE_CACHE[cls] = {}
            return GLOBAL_STRUCTURE_CACHE[cls]

    def _remove_from_cache(self, species: Optional[Species]) -> None:
        """Remove this structure from the specified species's cache (or global cache).

        Args:
            species: Target species, or ``None`` for the global fallback cache.
        """
        cache = self._get_cache_for_species(species)
        cache.pop(self.name, None)

    def _add_to_cache(self, species: Optional[Species]) -> None:
        """Add this structure to the specified species's cache (or global cache).

        Args:
            species: Target species, or ``None`` for the global fallback cache.
        """
        cache = self._get_cache_for_species(species)
        cache[self.name] = self

    def _bind_to_species(self, new_species: Optional[Species]) -> None:
        """Change the species binding and update caches accordingly.

        Args:
            new_species: The new Species to bind to, or None to unbind.

        This method:
        1. Removes the structure from its current cache (old species or global)
        2. Updates _species reference
        3. Adds the structure to the new cache (new species or global)
        """
        if not hasattr(self, '_species'):
            # Not yet initialized, skip cache management
            return

        old_species = self._species

        # No change, do nothing
        if old_species is new_species:
            return

        # Remove from old cache
        self._remove_from_cache(old_species)

        # Update species reference
        self._species = new_species

        # Add to new cache
        self._add_to_cache(new_species)

    @property
    def entity_type(self) -> Optional[type]:
        """
        Override in subclass to specify the entity type.
        Using property allows lazy import to avoid circular dependencies.
        """
        return None

    @classmethod
    def clear_cache(cls) -> None:
        """
        Deprecated: Caching is now managed by Species.
        This method does nothing but is kept for backward compatibility.
        """
        pass

    def clear_all_caches(self) -> None:
        """
        Clear all caches including:
        - Global fallback cache (for structures without Species)
        - All Species-specific caches are cleared via Species.clear_all_caches()

        This method is primarily for testing and cleanup.
        """
        GLOBAL_STRUCTURE_CACHE.clear()

    def add(
        self,
        name_or_specs: Union[str, List[str], List[Tuple[str, Dict[str, Any]]]],
        **kwargs: Any,
    ) -> Union[GeneticStructure[Any], List[GeneticStructure[Any]]]:
        """
        Add child structure(s) to this structure.

        Args:
            name_or_specs: Can be:
                - str: Single child name
                - List[str]: List of child names
                - List[Tuple[str, Dict]]: List of (name, kwargs) tuples
            **kwargs: Additional keyword arguments for single child creation.

        Returns:
            Single child structure or list of child structures.

        Examples:
            >>> linkage.add("LocusA", location=100)  # Single child
            >>> linkage.add(["LocusA", "LocusB"])    # Multiple children
            >>> linkage.add([("LocusA", {"location": 100}), ("LocusB", {"location": 200})])
        """
        child_registry = self._requirechild_structures_registry()

        assert isinstance(name_or_specs, (str, list)), \
            f"Expected str, List[str], or List[Tuple[str, Dict]], got {type(name_or_specs).__name__}"

        # Single name
        if isinstance(name_or_specs, str):
            return child_registry.add(name_or_specs, **kwargs)

        # List of names or (name, kwargs) tuples
        else:
            results: List[GeneticStructure[Any]] = []
            for item in name_or_specs:
                if isinstance(item, str):
                    results.append(child_registry.add(item, **kwargs))
                elif len(item) == 2:
                    name, child_kwargs = item
                    merged_kwargs = {**kwargs, **child_kwargs}
                    results.append(child_registry.add(name, **merged_kwargs))
                else:
                    raise TypeError(f"Invalid item in list: {item}. Expected str or (str, dict) tuple.")
            return results

    def remove(
        self,
        name_or_child: Union[str, GeneticStructure[Any], List[Union[str, GeneticStructure[Any]]]],
    ) -> None:
        """
        Remove child structure(s) from this structure.

        Args:
            name_or_child: Can be:
                - str: Child name to remove
                - GeneticStructure: Child instance to remove
                - List: List of names or instances to remove

        Examples:
            >>> linkage.remove("LocusA")           # Remove by name
            >>> linkage.remove(locus_a)            # Remove by instance
            >>> linkage.remove(["LocusA", "LocusB"])  # Remove multiple
        """
        child_registry = self._requirechild_structures_registry()

        # Delegate to registry - it handles both str and object
        child_registry.unregister(name_or_child)

    def get_child(self, name: str) -> GeneticStructure[Any]:
        """
        Get a child structure by name.

        Args:
            name: Name of the child structure.

        Returns:
            The child structure instance.

        Raises:
            KeyError: If no child with that name exists.
        """
        child_registry = self._requirechild_structures_registry()
        return child_registry.get(name)

    @property
    def children(self) -> List[GeneticStructure[Any]]:
        """Returns all child structures."""
        if not hasattr(self, "child_structures"):
            return []
        child_registry = self.child_structures
        return child_registry.all

    def _requirechild_structures_registry(self) -> ChildStructureRegistry[GeneticStructure[Any]]:
        """Return the child-structures registry or raise.

        Returns:
            The child-structures registry.

        Raises:
            AttributeError: If this structure type does not support
                child structures.
        """
        if not hasattr(self, "child_structures"):
            raise AttributeError(f"{self.__class__.__name__} does not support child structures.")
        return self.child_structures

    def register(
        self,
        entity_or_entities: Union[E, List[E], Tuple[E, ...], Set[E]]
    ) -> GeneticStructure[E]:
        """
        Register a single entity or an iterable of entities with this structure.

        EntityRegistry performs runtime type validation based on the expected type provided at construction.

        Args:
            entity_or_entities: Single entity or iterable of entities to register.

        Returns:
            The GeneticStructure instance (for chaining).
        """
        # Delegate single/batch normalization and strict type-checking to EntityRegistry.
        self._entities.register(entity_or_entities)
        return self

    def unregister(
        self,
        entity_or_entities: Union[E, str, List[Union[E, str]], Tuple[Union[E, str], ...], Set[Union[E, str]]]
    ) -> GeneticStructure[E]:
        """
        Unregister a single entity or an iterable of entities from this structure.

        Args:
            entity_or_entities: Single entity or iterable of entities to unregister.

        Returns:
            The GeneticStructure instance (for chaining).
        """
        self._entities.unregister(entity_or_entities)
        return self

    @property
    def all_entities(self) -> List[E]:
        """Return a list of all entities currently registered to this structure.

        Returns:
            A snapshot list of registered entity instances.
        """
        return self._entities.all

    @classmethod
    def with_entities(
        cls,
        name: str,
        entity_ids: Union[str, Iterable[str]],
        **entity_kwargs: Any
    ) -> GeneticStructure[E]:
        """
        Factory method to create a GeneticStructure instance and register entities by their identifiers.

        Args:
            name (str): Name of the genetic structure.
            entity_ids (str | Iterable[str]): Single identifier or iterable of identifiers for entities to register.
            **entity_kwargs: Additional keyword arguments to pass to the entity constructor.
        """
        structure = cls(name)
        entity_type = structure.entity_type
        if entity_type is None:
            raise TypeError(f"{cls.__name__} has no entity type defined.")

        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]

        entities = [entity_type(name=en, **entity_kwargs) for en in entity_ids]
        structure.register(entities)
        return structure

    def __repr__(self) -> str:
        """Return a string representation of this GeneticStructure."""
        return f"{self.__class__.__name__}({self.name}, {self.entity_type}={self.all_entities})"
