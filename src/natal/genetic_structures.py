"""
genetic_structures
==================
Defines the immutable genetic architecture of the simulation.

Responsibilities
----------------
- Represent static, model-level genetic elements (loci, chromosomes, speciess).
- Store configuration and rules such as locus order, recombination rates, and chromosome IDs.
- Serve as the authoritative blueprint for creating and validating genetic entities.
- Optionally track bound entities via internal registration mechanisms.

Design Notes
------------
- No runtime dependency on `genetic_entities` to avoid circular imports.
- Modifications to a structure after binding entities are discouraged.
"""

from __future__ import annotations
import numpy as np
from typing import (
    Generic, TypeVar, Union, Optional, Any,
    Dict, List, Set, Tuple, Iterable, Callable, 
    get_args, TYPE_CHECKING, Literal
)
from enum import Enum
import itertools
import importlib
import inspect
import logging

if TYPE_CHECKING:
    from natal.genetic_entities import GeneticEntity, Gene, Haplotype, HaploidGenome, Genotype

__all__ = [
    "Locus",
    "Chromosome", "Linkage",
    "Species", "GenomeTemplate", "Karyotype"
]

T = TypeVar("T")  # Generic type
E = TypeVar("E")  # Generic type for entities (bound at runtime)
S = TypeVar("S", bound='GeneticStructure')  # Generic type for structures

logger = logging.getLogger(__name__)  # temp logger

# Global fallback cache for structures created without a Species (backward compatibility)
# Format: {structure_type: {name: instance}}
_GLOBAL_STRUCTURE_CACHE: Dict[type, Dict[str, 'GeneticStructure']] = {}

def ensure_type(obj, expected_type: type) -> None:
    """
    Ensures that an object is an instance of a given class, with lazy import.

    Args:
        obj (any): The object to check
        expected_type (type): The expected class type.

    Raises:
        TypeError: If obj is not an instance of the specified class.
    """
    module = importlib.import_module(expected_type.__module__) # Lazy import
    cls = getattr(module, expected_type.__name__)
    if not isinstance(obj, cls):
        raise TypeError(
            f"Expected {expected_type.__name__} from {expected_type.__module__}, got {type(obj).__name__} instead."
        )

class SexChromosomeType(Enum):
    """
    性染色体类型枚举。
    
    定义了常见的性染色体类型和它们的遗传特性：
    - AUTOSOME: 常染色体，不参与性别决定
    - X: 哺乳动物 X 染色体，可来自任意亲本
    - Y: 哺乳动物 Y 染色体，只能来自 paternal
    - Z: 鸟类/蛾类 Z 染色体，可来自任意亲本
    - W: 鸟类/蛾类 W 染色体，只能来自 maternal
    """
    AUTOSOME = "autosome"  # 常染色体
    X = "X"                # XY系统中的X染色体
    Y = "Y"                # XY系统中的Y染色体，只从父本遗传
    Z = "Z"                # ZW系统中的Z染色体
    W = "W"                # ZW系统中的W染色体，只从母本遗传
    
    @property
    def is_sex_chromosome(self) -> bool:
        """是否为性染色体"""
        return self != SexChromosomeType.AUTOSOME
    
    @property
    def sex_system(self) -> Optional[str]:
        """返回所属的性别决定系统"""
        if self in (SexChromosomeType.X, SexChromosomeType.Y):
            return "XY"
        elif self in (SexChromosomeType.Z, SexChromosomeType.W):
            return "ZW"
        return None
    
    @property
    def maternal_only(self) -> bool:
        """是否只能从母本遗传"""
        return self == SexChromosomeType.W
    
    @property
    def paternal_only(self) -> bool:
        """是否只能从父本遗传"""
        return self == SexChromosomeType.Y
    
class RegistryBase(Generic[T]):
    """
    Base class for registries. Provides common interface for register/unregister operations.
    
    Subclasses must implement:
        - _get_key(item): Extract the key used for deduplication
        - _single_register(item): Register a single item
        - _single_unregister(item): Unregister a single item
    """
    def __init__(self, expected_type: Optional[type] = None):
        self._expected_type = expected_type

    def _check_type(self, item: T) -> None:
        if self._expected_type and not isinstance(item, self._expected_type):
            raise TypeError(f"Expected type {self._expected_type.__name__}, got {type(item).__name__}")

    def _get_key(self, item: T):
        """Extract the key for deduplication. Override in subclass."""
        raise NotImplementedError
    
    def _single_register(self, item: T) -> None:
        """Register a single item. Override in subclass."""
        raise NotImplementedError
    
    def _single_unregister(self, key_or_item) -> None:
        """Unregister a single item by key or item. Override in subclass."""
        raise NotImplementedError

    def register(self, item_or_items: Union[T, Iterable[T]]) -> None:
        """Register one or more items."""
        # GeneticStructure is iterable (yields children) but should be registered as single item
        # Use explicit list/tuple/set check instead of Iterable to avoid this
        if isinstance(item_or_items, (list, tuple, set)):
            for item in item_or_items:
                self._single_register(item)
        else:
            self._single_register(item_or_items)

    def unregister(self, item_or_items) -> None:
        """Unregister one or more items (by key or item object)."""
        if isinstance(item_or_items, (list, tuple, set)):
            for item in item_or_items:
                self._single_unregister(item)
        else:
            self._single_unregister(item_or_items)

    def __len__(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class EntityRegistry(RegistryBase[E]):
    """
    Registry for entity objects. Deduplication by object identity.
    """
    def __init__(self, expected_type: Optional[type] = None):
        super().__init__(expected_type)
        self._storage: List[E] = []
        self._set: Set[E] = set()

    def _get_key(self, item: E) -> E:
        return item  # Use object identity
    
    def _single_register(self, item: E) -> None:
        self._check_type(item)
        if item not in self._set:
            self._storage.append(item)
            self._set.add(item)

    def _single_unregister(self, item: E) -> None:
        if item in self._set:
            self._storage.remove(item)
            self._set.remove(item)

    def __iter__(self):
        return iter(self._storage)

    def __contains__(self, item: E) -> bool:
        return item in self._set

    def __len__(self) -> int:
        return len(self._storage)

    def clear(self) -> None:
        self._storage.clear()
        self._set.clear()

    @property
    def all(self) -> List[E]:
        """Returns all registered entities."""
        return list(self._storage)


class ChildStructureRegistry(RegistryBase[S]):
    """
    Registry for child structures. Keyed by name, preserves insertion order.
    Supports both register (existing) and add (create + register).
    """
    def __init__(self, owner: 'GeneticStructure', expected_type: Optional[type] = None):
        super().__init__(expected_type)
        self._owner = owner  # The parent structure that owns this registry
        self._storage: Dict[str, S] = {}

    def _get_key(self, item: S) -> str:
        return item.name
    
    def _single_register(self, item: S) -> None:
        """Register an existing child structure."""
        self._check_type(item)
        if not hasattr(item, 'name'):
            raise TypeError("Child must have a 'name' attribute.")
        if item.name not in self._storage:
            self._storage[item.name] = item

    def _single_unregister(self, key_or_item) -> None:
        """Unregister by name (str) or by item."""
        name = key_or_item if isinstance(key_or_item, str) else key_or_item.name
        self._storage.pop(name, None)

    def add(self, name: str, **kwargs) -> S:
        """
        Create a new child structure and register it.
        This is a convenience method: create + register.
        
        Uses Species-level caching to ensure uniqueness within the same Species.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Child structure name must be a non-empty string.")
        if name in self._storage:
            raise ValueError(f"Child structure '{name}' already exists.")
        if not self._expected_type:
            raise ValueError("expected_type not set, cannot construct child structure.")
        
        # Get the Species from the owner
        species = getattr(self._owner, '_species', None)
        
        # Check if structure already exists in cache (Species-scoped or global)
        if species is not None and hasattr(species, '_structure_cache'):
            # Use Species-scoped cache
            if self._expected_type not in species._structure_cache:
                species._structure_cache[self._expected_type] = {}
            
            cache = species._structure_cache[self._expected_type]
            if name in cache:
                # Return cached instance
                child = cache[name]
                # Still register it in this owner's registry if not already there
                if name not in self._storage:
                    self._storage[name] = child
                return child
            
            # Create new child with parent (species is inherited automatically)
            child = self._expected_type(name, parent=self._owner, **kwargs)
            
            # Cache it in the Species
            cache[name] = child
        else:
            # Use global fallback cache for backward compatibility
            if self._expected_type not in _GLOBAL_STRUCTURE_CACHE:
                _GLOBAL_STRUCTURE_CACHE[self._expected_type] = {}
            
            cache = _GLOBAL_STRUCTURE_CACHE[self._expected_type]
            if name in cache:
                # Return cached instance
                child = cache[name]
                # Still register it in this owner's registry if not already there
                if name not in self._storage:
                    self._storage[name] = child
                return child
            
            # Create new child with parent (no species means orphan)
            child = self._expected_type(name, parent=self._owner, **kwargs)
            
            # Cache it globally
            cache[name] = child
        
        return child

    def get(self, name: str) -> S:
        """Get a child structure by name."""
        if name not in self._storage:
            raise KeyError(f"No child structure named '{name}' found.")
        return self._storage[name]

    def __iter__(self):
        return iter(self._storage.values())

    def __contains__(self, name_or_item) -> bool:
        name = name_or_item if isinstance(name_or_item, str) else name_or_item.name
        return name in self._storage

    def __len__(self) -> int:
        return len(self._storage)

    def clear(self) -> None:
        self._storage.clear()

    @property
    def all(self) -> List[S]:
        """Returns all registered child structures."""
        return list(self._storage.values())


class GeneticStructure(Generic[E]):
    """
    Base class for genetic structures.

    Structure uniqueness is now scoped to a Species, not globally.
    Within the same Species, structures of the same type must have unique names.
    
    Example:
        >>> species1 = Species("Species1")
        >>> locus1 = Locus("A", species=species1)
        >>> locus2 = Locus("A", species=species1)
        >>> assert locus1 is locus2  # Same object within species1
        >>>
        >>> species2 = Species("Species2")
        >>> locus3 = Locus("A", species=species2)
        >>> assert locus1 is not locus3  # Different speciess allow same name
    """
    child_structure_type: Optional[type] = None  # Child structure type per subclass

    def __new__(
        cls,
        name: str,
        *args, 
        **kwargs
    ):
        # Extract species and parent from kwargs
        species = kwargs.get('species')
        parent = kwargs.get('parent')
        
        # Determine which cache to use
        target_species = None
        if species is not None and hasattr(species, '_structure_cache'):
            target_species = species
        elif parent is not None and hasattr(parent, '_species'):
            target_species = parent._species
        
        # Get the appropriate cache
        if target_species is not None and hasattr(target_species, '_structure_cache'):
            # Use Species-scoped cache
            if cls not in target_species._structure_cache:
                target_species._structure_cache[cls] = {}
            cache = target_species._structure_cache[cls]
        else:
            # Use global fallback cache for structures without species
            if cls not in _GLOBAL_STRUCTURE_CACHE:
                _GLOBAL_STRUCTURE_CACHE[cls] = {}
            cache = _GLOBAL_STRUCTURE_CACHE[cls]
        
        # Check if instance already exists in cache
        if name in cache:
            return cache[name]
        
        # Create new instance (do NOT cache here - cache in __init__ after success)
        instance = super().__new__(cls)
        
        return instance

    def __init__(
        self, 
        name: str,
        parent: Optional['GeneticStructure'] = None,
        species: Optional['Species'] = None
    ):
        # Prevent re-initialization of cached instances
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        if not isinstance(name, str):
            raise TypeError("Structure name must be a string.")
        if name.strip() == "":
            raise ValueError("Structure name cannot be empty.")
        
        # Create registry (entity_type is now a property in subclasses)
        self.name = name
        self._entities: EntityRegistry = EntityRegistry()
        
        # Track the root Species for this structure
        if species is not None:
            self._species = species
        elif parent is not None:
            # Inherit species from parent
            self._species = getattr(parent, '_species', None)
        else:
            # This is a Species itself
            self._species = None
        
        # Initialize child structures registry if applicable
        cls = self.__class__
        if cls.child_structure_type:
            self._child_structures = ChildStructureRegistry(owner=self, expected_type=cls.child_structure_type)
        
        # Strict constraint: must be added to a parent unless top-level
        if parent is not None:
            if not isinstance(parent, GeneticStructure):
                raise TypeError(f"Parent must be a GeneticStructure instance, got {type(parent).__name__} instead.")
            # Register this structure as a child of the parent
            if hasattr(parent, '_child_structures') and parent._child_structures is not None:
                parent._child_structures.register(self)
            else:
                raise TypeError(f"Parent {parent.__class__.__name__} does not support child structures.")
        
        # Mark as initialized, avoiding re-initialization when created from cache
        self._initialized = True
        
        # Cache the instance AFTER successful initialization
        self._add_to_cache(self._species)
    
    def _get_cache_for_species(self, species: Optional['Species']) -> Dict[str, 'GeneticStructure']:
        """Get the appropriate cache for the given species."""
        cls = self.__class__
        if species is not None and hasattr(species, '_structure_cache'):
            if cls not in species._structure_cache:
                species._structure_cache[cls] = {}
            return species._structure_cache[cls]
        else:
            if cls not in _GLOBAL_STRUCTURE_CACHE:
                _GLOBAL_STRUCTURE_CACHE[cls] = {}
            return _GLOBAL_STRUCTURE_CACHE[cls]
    
    def _remove_from_cache(self, species: Optional['Species']) -> None:
        """Remove this structure from the specified species's cache (or global cache)."""
        cache = self._get_cache_for_species(species)
        cache.pop(self.name, None)
    
    def _add_to_cache(self, species: Optional['Species']) -> None:
        """Add this structure to the specified species's cache (or global cache)."""
        cache = self._get_cache_for_species(species)
        cache[self.name] = self
    
    def _bind_to_species(self, new_species: Optional['Species']) -> None:
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

    @classmethod
    def clear_all_caches(cls) -> None:
        """
        Clear all caches including:
        - Global fallback cache (for structures without Species)
        - All Species-specific caches are cleared via Species.clear_all_caches()
        
        This method is primarily for testing and cleanup.
        """
        global _GLOBAL_STRUCTURE_CACHE
        _GLOBAL_STRUCTURE_CACHE.clear()
    
    def add(
        self,
        name_or_specs: Union[str, List[str], List[Tuple[str, Dict]]],
        **kwargs
    ) -> Union['GeneticStructure', List['GeneticStructure']]:
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
        
        Example:
            >>> linkage.add("LocusA", location=100)  # Single child
            >>> linkage.add(["LocusA", "LocusB"])    # Multiple children
            >>> linkage.add([("LocusA", {"location": 100}), ("LocusB", {"location": 200})])
        """
        if not hasattr(self, "_child_structures"):
            raise AttributeError(f"{self.__class__.__name__} does not support child structures.")
        
        # Single name
        if isinstance(name_or_specs, str):
            return self._child_structures.add(name_or_specs, **kwargs)
        
        # List of names or (name, kwargs) tuples
        if isinstance(name_or_specs, list):
            results = []
            for item in name_or_specs:
                if isinstance(item, str):
                    results.append(self._child_structures.add(item, **kwargs))
                elif isinstance(item, tuple) and len(item) == 2:
                    name, child_kwargs = item
                    merged_kwargs = {**kwargs, **child_kwargs}
                    results.append(self._child_structures.add(name, **merged_kwargs))
                else:
                    raise TypeError(f"Invalid item in list: {item}. Expected str or (str, dict) tuple.")
            return results
        
        raise TypeError(f"Expected str, List[str], or List[Tuple[str, Dict]], got {type(name_or_specs).__name__}")

    def remove(
        self,
        name_or_child: Union[str, 'GeneticStructure', List[Union[str, 'GeneticStructure']]]
    ) -> None:
        """
        Remove child structure(s) from this structure.

        Args:
            name_or_child: Can be:
                - str: Child name to remove
                - GeneticStructure: Child instance to remove  
                - List: List of names or instances to remove

        Example:
            >>> linkage.remove("LocusA")           # Remove by name
            >>> linkage.remove(locus_a)            # Remove by instance
            >>> linkage.remove(["LocusA", "LocusB"])  # Remove multiple
        """
        if not hasattr(self, "_child_structures"):
            raise AttributeError(f"{self.__class__.__name__} does not support child structures.")
        
        # Delegate to registry - it handles both str and object
        self._child_structures.unregister(name_or_child)

    def get_child(self, name: str) -> 'GeneticStructure':
        """
        Get a child structure by name.
        
        Args:
            name: Name of the child structure.
            
        Returns:
            The child structure instance.
            
        Raises:
            KeyError: If no child with that name exists.
        """
        if not hasattr(self, "_child_structures"):
            raise AttributeError(f"{self.__class__.__name__} does not support child structures.")
        return self._child_structures.get(name)

    @property
    def children(self) -> List['GeneticStructure']:
        """Returns all child structures."""
        if not hasattr(self, "_child_structures"):
            return []
        return self._child_structures.all

    
    def register(
        self,
        entity_or_entities: E | Iterable[E]
    ) -> 'GeneticStructure[E]':
        """
        Register a single entity or an iterable of entities with this structure.
        
        EntityRegistry performs runtime type validation based on the expected type provided at construction.

        Args:
            entity_or_entities: Single entity or iterable of entities to register.
        
        Returns:
            The GeneticStructure instance (for chaining).
        """
        # Delegate to the EntityRegistry which handles single/bulk inputs and type checks.
        self._entities.register(entity_or_entities)
        return self
    
    def unregister(
        self,
        entity_or_entities: E | Iterable[E]
    ) -> 'GeneticStructure[E]':
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
        """
        Returns a list of all entities currently registered to this structure.
        """
        return self._entities.all
    
    @classmethod
    def with_entities(
        cls,
        name: str,
        entity_ids: str | Iterable[str],
        **entity_kwargs
    ) -> GeneticStructure[E]:
        """
        Factory method to create a GeneticStructure instance and register entities by their identifiers.

        Args:
            name (str): Name of the genetic structure.
            entity_ids (str | Iterable[str]): Single identifier or iterable of identifiers for entities to register.
            **entity_kwargs: Additional keyword arguments to pass to the entity constructor.
        """
        structure = cls(name)
        entity_type = cls.entity_type
        if entity_type is None:
            raise TypeError(f"{cls.__name__} has no entity type defined.")
        
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]

        entities = [entity_type(name=en, **entity_kwargs) for en in entity_ids]
        structure.register(entities)
        return structure
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.entity_type}={self.all_entities})"
    
# Locus (model-level) -> Gene (entity-level)
class Locus(GeneticStructure['Gene']):
    """
    Represents a genetic locus with its name.
    
    A Locus is a blueprint for a genetic position. Multiple Gene entities
    (alleles) can be bound to a single Locus.
    
    Attributes:
        position: The linear position on the chromosome. Used for defining
            recombination rates. If not specified, defaults to max(position) + 1
            among existing loci in the parent Linkage, or 0 if no parent.
    """
    child_structure_type = None  # Locus has no child structures

    def __init__(
        self, 
        name: str, 
        position: Optional[Union[int, float]] = None,
        chromosome: Optional['Chromosome'] = None,
        parent: Optional['Chromosome'] = None,
        **kwargs
    ):
        # Check if already initialized (cached instance)
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Parent is alias for chromosome
        if chromosome is None:
            chromosome = parent
        
        # Save parent reference for cache invalidation
        self._parent_chromosome = chromosome
        
        # Compute default position before super().__init__ 
        # (since parent.register may be called)
        if position is None:
            if chromosome is not None and hasattr(chromosome, '_child_structures') and len(chromosome._child_structures) > 0:
                # Default: max position in parent + 1
                max_pos = max(
                    (l.position for l in chromosome._child_structures if l.position is not None),
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
            self._parent_chromosome._invalidate_recombination_map_cache()

    @property
    def entity_type(self):
        """Lazy import to avoid circular dependency."""
        from natal.genetic_entities import Gene
        return Gene

    def register(
        self,
        entity_or_entities: E | Iterable[E]
    ) -> 'GeneticStructure[E]':
        """
        Register gene entities and invalidate species gene index cache.
        """
        result = super().register(entity_or_entities)
        if self._species is not None and hasattr(self._species, '_invalidate_gene_index_cache'):
            self._species._invalidate_gene_index_cache()
        return result

    def unregister(
        self,
        entity_or_entities: E | Iterable[E]
    ) -> 'GeneticStructure[E]':
        """
        Unregister gene entities and invalidate species gene index cache.
        """
        result = super().unregister(entity_or_entities)
        if self._species is not None and hasattr(self._species, '_invalidate_gene_index_cache'):
            self._species._invalidate_gene_index_cache()
        return result

    @property
    def alleles(self) -> List['Gene']:
        """Alias for all_entities - returns all registered alleles (genes)."""
        return self.all_entities

    def register_allele(self, gene: 'Gene') -> 'Locus':
        """Alias for register - register a single allele."""
        return self.register(gene)

    def unregister_allele(self, gene: 'Gene') -> 'Locus':
        """Alias for unregister - unregister a single allele."""
        return self.unregister(gene)
    
    def add_alleles(
        self,
        alleles_or_allele_names: Union[List[Union['Gene', str]], 'Gene', str],
    ) -> 'Locus':
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
        from natal.genetic_entities import Gene
        
        if isinstance(alleles_or_allele_names, (Gene, str)):
            alleles_or_allele_names = [alleles_or_allele_names]
        
        for item in alleles_or_allele_names:
            if isinstance(item, Gene):
                self.register(item)
            elif isinstance(item, str):
                Gene(item, locus=self)  # Auto-registers via Gene.__init__
            else:
                raise TypeError(f"Expected Gene or str, got {type(item).__name__} instead.")
        
        return self

    @classmethod
    def with_alleles(
        cls,
        name: str,
        alleles_or_allele_names: Union[List[Union['Gene', str]], 'Gene', str],
        position: Optional[Union[int, float]] = None
    ) -> 'Locus':
        """
        Factory method to create a Locus and register alleles (genes) by names.

        Args:
            name: Name of the locus.
            alleles_or_allele_names: Single Gene instance, single allele name (str),
                or list of Gene instances and/or allele names (str).
            position: Optional position on the chromosome.

        Returns:
            Locus instance with registered alleles.
        
        Example:
            >>> locus = Locus.with_alleles("A", ["A1", "A2", "A3"])
            >>> locus.alleles  # → [Gene("A1"), Gene("A2"), Gene("A3")]
        """
        return cls(name, position=position).add_alleles(alleles_or_allele_names)
    
    def __repr__(self) -> str:
        allele_names = [g.name for g in self.alleles]
        return f"Locus({self.name!r}, position={self.position}, alleles={allele_names})"

# Chromosome (structure-level) -> Haplotype (entity-level)
class Chromosome(GeneticStructure['Haplotype']):
    """
    Represents a chromosome structure with linkage information among loci.
    
    A Chromosome groups multiple Loci that are physically linked on the same
    chromosome. It also stores the recombination rates between loci.
    
    Attributes:
        sex_type: 性染色体类型 (SexChromosomeType 或字符串)。
            - None 或 'autosome': 常染色体（默认）
            - 'X': XY系统中的X染色体
            - 'Y': XY系统中的Y染色体（只能从父本遗传）
            - 'Z': ZW系统中的Z染色体
            - 'W': ZW系统中的W染色体（只能从母本遗传）
    
    Example:
        >>> chr_x = Chromosome('X', sex_type='X')
        >>> chr_y = Chromosome('Y', sex_type='Y')
        >>> print(chr_x.is_sex_chromosome)  # True
        >>> print(chr_y.sex_type.paternal_only)  # True
    
    Aliases: Linkage
    """
    child_structure_type = Locus  # Chromosome contains Loci as children

    def __init__(
        self, 
        name: str, 
        loci: Optional[List[Locus]] = None,
        species: Optional['Species'] = None,
        parent: Optional['Chromosome'] = None,
        recombination_rates: Optional[Union[List[float], np.ndarray]] = None,
        sex_type: Optional[Union[SexChromosomeType, str]] = None,
    ):
        # Initialize placeholders BEFORE super().__init__
        # because __iter__ may be called during parent registration
        if not hasattr(self, '_recombination_map'):
            self._recombination_map: Optional[Chromosome.RecombinationMap] = None
            self._sorted_loci_cache: Optional[List[Locus]] = None  # Cache for sorted loci
            self._sex_type: SexChromosomeType = SexChromosomeType.AUTOSOME
        
        # Check if already initialized (cached instance)
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Parent is alias for species
        if species is None:
            species = parent
        
        # 设置性染色体类型
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
        """设置性染色体类型（内部方法）"""
        if sex_type is None:
            self._sex_type = SexChromosomeType.AUTOSOME
        elif isinstance(sex_type, SexChromosomeType):
            self._sex_type = sex_type
        elif isinstance(sex_type, str):
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
        else:
            raise TypeError(f"sex_type must be SexChromosomeType or str, got {type(sex_type).__name__}")
    
    @property
    def sex_type(self) -> SexChromosomeType:
        """返回性染色体类型"""
        return self._sex_type
    
    @sex_type.setter
    def sex_type(self, value: Optional[Union[SexChromosomeType, str]]) -> None:
        """设置性染色体类型"""
        self._set_sex_type(value)
    
    @property
    def is_sex_chromosome(self) -> bool:
        """是否为性染色体"""
        return self._sex_type.is_sex_chromosome
    
    @property
    def is_autosome(self) -> bool:
        """是否为常染色体"""
        return not self.is_sex_chromosome
    
    @property
    def sex_system(self) -> Optional[str]:
        """返回所属的性别决定系统 ('XY', 'ZW', 或 None)"""
        return self._sex_type.sex_system

    @property
    def entity_type(self):
        """Lazy import to avoid circular dependency."""
        from natal.genetic_entities import Haplotype
        return Haplotype

    @property
    def loci(self) -> List[Locus]:
        """Returns the list of loci in this chromosome, sorted by position (cached)."""
        if self._sorted_loci_cache is None:
            self._sorted_loci_cache = sorted(
                self._child_structures.all, 
                key=lambda l: l.position
            )
        return self._sorted_loci_cache
    
    def _invalidate_recombination_map_cache(self) -> None:
        """Invalidate sorted loci cache and update recombination map."""
        self._sorted_loci_cache = None
        self._update_recombination_map()
    
    @property
    def recombination_map(self) -> 'Chromosome.RecombinationMap':
        """Returns the recombination map for this chromosome.
        
        The recombination map stores recombination rates between adjacent loci.
        For n loci, the map has n-1 entries where entry i is the recombination
        rate between locus i and locus i+1.
        """
        if self._recombination_map is None:
            self._update_recombination_map()
        return self._recombination_map
    
    # Backward compatibility alias
    @property
    def recombination_matrix(self) -> 'Chromosome.RecombinationMap':
        """Deprecated: Use recombination_map instead."""
        return self.recombination_map
    
    def add_locus(
        self, 
        locus_or_name: Union[Locus, str],
        position: Optional[Union[int, float]] = None,
        recombination_rate_with_previous: float = 0.0,
        **kwargs
    ) -> Locus:
        """
        Add a locus to this chromosome.
        
        When inserting a new locus, it defaults to complete linkage with the previous
        locus (recombination rate = 0), and inherits the recombination rate with the
        next locus from the previous locus's rate with that next locus.
        
        Args:
            locus_or_name: Either a Locus instance or a name to create a new Locus.
            position: Optional position (only used when creating new Locus by name).
                If not specified, defaults to max(position) + 1 among existing loci.
            recombination_rate_with_previous: Recombination rate with the previous locus.
                Defaults to 0 (complete linkage).
            **kwargs: Additional custom parameters to pass to the Locus constructor.
            
        Returns:
            The added Locus instance.
        """
        # Get current sorted loci and old map before adding
        old_sorted_loci = self.loci.copy() if self._sorted_loci_cache else []
        old_map = self._recombination_map
        
        if isinstance(locus_or_name, str):
            # Create new Locus via base class add method with kwargs
            locus = self.add(locus_or_name, position=position, **kwargs)
        elif isinstance(locus_or_name, Locus):
            locus = locus_or_name
            # Register existing Locus if not already in registry
            if locus.name not in self._child_structures:
                self._child_structures._storage[locus.name] = locus
        else:
            raise TypeError("locus_or_name must be a Locus instance or string.")
        
        # Invalidate cache and update recombination map with insertion handling
        self._sorted_loci_cache = None
        self._update_recombination_map_on_insert(
            locus, old_sorted_loci, old_map, recombination_rate_with_previous
        )
        if self._species is not None and hasattr(self._species, '_invalidate_gene_index_cache'):
            self._species._invalidate_gene_index_cache()
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
        
        if name in self._child_structures:
            # Get old state
            old_sorted_loci = self.loci.copy()
            old_map = self._recombination_map
            
            # Find the index of the locus to remove
            locus_to_remove = self._child_structures.get(name)
            remove_idx = old_sorted_loci.index(locus_to_remove)
            
            # Unregister the locus
            self._child_structures.unregister(name)
            self._sorted_loci_cache = None
            
            # Update recombination map
            self._update_recombination_map_on_remove(remove_idx, old_map)
            if self._species is not None and hasattr(self._species, '_invalidate_gene_index_cache'):
                self._species._invalidate_gene_index_cache()
    
    def _update_recombination_map(self) -> None:
        """Create a fresh recombination map (all rates = 0)."""
        if len(self._child_structures) > 1:
            self._recombination_map = Chromosome.RecombinationMap(loci=self.loci)
        else:
            self._recombination_map = None
    
    def _update_recombination_map_on_insert(
        self, 
        new_locus: Locus,
        old_sorted_loci: List[Locus],
        old_map: Optional['Chromosome.RecombinationMap'],
        recombination_rate_with_previous: float
    ) -> None:
        """Update recombination map when a new locus is inserted."""
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
            old_i = 0
            for new_i in range(n - 1):
                if new_i == new_idx - 1:
                    # Rate between previous locus and new locus
                    new_rates[new_i] = recombination_rate_with_previous
                elif new_i == new_idx:
                    # Rate between new locus and next locus
                    # Inherit from the old rate between prev and next
                    if new_idx > 0 and new_idx - 1 < len(old_map):
                        new_rates[new_i] = old_map[new_idx - 1]
                    else:
                        new_rates[new_i] = 0.0
                else:
                    # Copy from old map
                    if new_i < new_idx:
                        new_rates[new_i] = old_map[new_i] if new_i < len(old_map) else 0.0
                    else:  # new_i > new_idx
                        old_idx = new_i - 1  # Account for insertion
                        new_rates[new_i] = old_map[old_idx] if old_idx < len(old_map) else 0.0
        else:
            # First pair of loci, set the rate
            new_rates[0] = recombination_rate_with_previous
        
        self._recombination_map = Chromosome.RecombinationMap(
            loci=new_sorted_loci, 
            rates=new_rates
        )
    
    def _update_recombination_map_on_remove(
        self,
        remove_idx: int,
        old_map: Optional['Chromosome.RecombinationMap']
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
        
        self._recombination_map = Chromosome.RecombinationMap(
            loci=new_sorted_loci,
            rates=new_rates
        )
    
    def get_locus_index(self, name: str) -> int:
        """Get the index of a locus by name in the sorted loci list."""
        return self.recombination_map._name_to_index(name)

    class RecombinationMap(np.ndarray):
        """
        A 1D array storing recombination rates between adjacent loci.
        
        For n loci, the map has n-1 entries where entry i is the recombination
        rate between locus i and locus i+1 (in sorted order by position).
        
        Example:
            For loci [A, B, C, D], the map is [r(A,B), r(B,C), r(C,D)]
            where index i = rate between locus i and locus i+1.
        """
        def __new__(
            cls,
            loci: Optional[List[Locus]] = None,
            rates: Optional[np.ndarray] = None,
            dtype=float
        ):
            size = len(loci) - 1 if loci and len(loci) > 1 else 0
            if size <= 0:
                raise ValueError("RecombinationMap requires at least 2 loci.")
            
            obj = super().__new__(cls, (size,), dtype)
            obj.loci_names = [locus.name for locus in loci] if loci else []
            
            if rates is not None:
                if len(rates) != size:
                    raise ValueError(f"Expected {size} rates, got {len(rates)}")
                obj[:] = rates
            else:
                obj[:] = 0.0  # Default: complete linkage
            
            return obj

        # ---------- Name conversion ----------
        def _name_to_index(self, name):
            """Convert locus name to index in loci list."""
            if not self.loci_names:
                raise ValueError("No loci names defined in map.")
            try:
                return self.loci_names.index(name)
            except ValueError:
                raise KeyError(f"Locus name '{name}' not found.")

        def _normalize_single_key(self, key):
            """Normalize a single key to integer index."""
            if isinstance(key, str):
                return self._name_to_index(key)
            elif isinstance(key, Locus):
                return self._name_to_index(key.name)
            else:
                return key

        # ---------- Reading ----------
        def __getitem__(self, key):
            """
            Get recombination rate(s).
            
            Usage:
                map[i] -> rate between locus i and locus i+1
                map[locus_a, locus_b] -> rate between locus_a and locus_b
                map['A', 'B'] -> rate between loci named 'A' and 'B'
                
            For non-adjacent loci, returns the sum of rates in the interval.
            For example, map['A', 'C'] returns r(A,B) + r(B,C).
            """
            if isinstance(key, tuple) and len(key) == 2:
                # Access by pair of loci
                a, b = key
                idx_a = self._normalize_single_key(a)
                idx_b = self._normalize_single_key(b)
                
                # Ensure correct order
                if idx_a > idx_b:
                    idx_a, idx_b = idx_b, idx_a
                
                # Sum rates in the interval [idx_a, idx_b)
                total_rate = 0.0
                for i in range(idx_a, idx_b):
                    total_rate += super().__getitem__(i)
                
                return min(total_rate, 0.5)  # Cap at 0.5
            else:
                # Direct index access
                return super().__getitem__(key)

        # ---------- Writing ----------
        def __setitem__(self, key, value):
            """
            Set recombination rate(s).
            
            Usage:
                map[i] = rate  # Set rate between locus i and locus i+1
                map[locus_a, locus_b] = rate  # Set rate between adjacent loci
                map['A', 'B'] = rate  # Set rate between adjacent loci by name
            
            Warning:
                Modifying recombination rates after Genotype.produce_gametes()
                has been called will NOT invalidate the gamete cache. You must
                manually clear the cache: genotype._gamete_cache = None
            """
            arr_val = np.asarray(value, dtype=self.dtype)
            
            if np.any((arr_val < 0) | (arr_val > 0.5)):
                raise ValueError("Recombination rates must be in [0, 0.5]")
            
            if isinstance(key, tuple) and len(key) == 2:
                # Access by pair of loci
                a, b = key
                idx_a = self._normalize_single_key(a)
                idx_b = self._normalize_single_key(b)
                
                # Ensure they are adjacent
                if abs(idx_a - idx_b) != 1:
                    raise KeyError(
                        f"Loci {a!r} and {b!r} are not adjacent. "
                        f"RecombinationMap only stores rates between adjacent loci."
                    )
                super().__setitem__(min(idx_a, idx_b), arr_val)
            else:
                # Direct index access
                super().__setitem__(key, arr_val)

        # ---------- Visualization ----------
        def __repr__(self):
            return self._formatted_repr()

        def __str__(self):
            return self._formatted_repr()

        def _formatted_repr(self):
            if self.loci_names and len(self.loci_names) > 1:
                pairs = []
                for i in range(len(self)):
                    pairs.append(f"r({self.loci_names[i]},{self.loci_names[i+1]})={self[i]:.3f}")
                return f"RecombinationMap([{', '.join(pairs)}])"
            else:
                return f"RecombinationMap({np.array2string(self, precision=3)})"

        # ---------- Utility methods ----------
        def validate(self):
            """Validate the recombination map."""
            if np.any(self < 0) or np.any(self > 0.5):
                return False, "Values out of range [0, 0.5]."
            return True, "Map is valid."
        
        def get_adjacent_loci(self, index: int) -> tuple:
            """Get the names of the two loci at the given rate index."""
            if index < 0 or index >= len(self):
                raise IndexError(f"Index {index} out of range for map of size {len(self)}")
            return (self.loci_names[index], self.loci_names[index + 1])
    
    # Backward compatibility alias
    RecombinationMatrix = RecombinationMap
    
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

    def set_recombination_bulk(self, settings: dict):
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
    
    def set_recombination_rates(self, settings: dict):
        """
        Deprecated: Use set_recombination_bulk instead.
        """
        self.set_recombination_bulk(settings)

    def __repr__(self):
        return f"Chromosome({self.name!r}, loci={[l.name for l in self.loci]})"
    
    def __iter__(self):
        return iter(self.loci)
    
    def __len__(self):
        return len(self.loci)

# Species (structure-level) -> HaploidGenome (entity-level)
class Species(GeneticStructure['HaploidGenome']):
    """
    Represents the complete genetic architecture defined by chromosomes.
    
    A Species is the top-level structure that contains multiple Chromosomes,
    each representing a chromosome with its loci and recombination rates.
    
    Aliases: GenomeTemplate
    """
    child_structure_type = Chromosome  # Species contains Chromosomes as children

    def __init__(
        self, 
        name: str, 
        chromosomes: Optional[List['Chromosome']] = None,
        gamete_labels: Optional[list] = None
    ):
        # Initialize structure caches for this Species
        # Format: {structure_type: {name: instance}}
        self._structure_cache: Dict[type, Dict[str, 'GeneticStructure']] = {}
        self._gene_index_cache: Optional[Dict[str, 'Gene']] = None
        
        super().__init__(name, parent=None, species=None)  # Species is top-level, no parent
        
        # Set self as the species
        self._species = self
        
        # Add initial chromosomes if provided
        if chromosomes:
            for chrom in chromosomes:
                self.add_chromosome(chrom)

        # Gamete labels for this species
        if gamete_labels is not None:
            self._gamete_labels = list(gamete_labels)
        else:
            self._gamete_labels = []

    @property
    def gamete_labels(self) -> list:
        """Return the list of gamete labels for this species."""
        return self._gamete_labels

    @gamete_labels.setter
    def gamete_labels(self, labels: list) -> None:
        self._gamete_labels = list(labels)

    @property
    def entity_type(self):
        """Lazy import to avoid circular dependency."""
        from natal.genetic_entities import HaploidGenome
        return HaploidGenome

    def clear_structure_cache(self) -> None:
        """
        Clear all structure caches for this Species.
        This removes all cached Structure instances (Locus, Chromosome) within this Species.
        """
        self._structure_cache.clear()
        self._invalidate_gene_index_cache()
    
    def clear_entity_cache(self) -> None:
        """
        Clear all entity caches for this Species.
        This removes all cached Entity instances (Gene, Haplotype, etc.) within this Species.
        """
        from natal.genetic_entities import GeneticEntity
        species_id = id(self)
        keys_to_remove = [k for k in GeneticEntity._instance_cache if k[0] == species_id]
        for key in keys_to_remove:
            del GeneticEntity._instance_cache[key]
    
    def clear_all_caches(self) -> None:
        """
        Clear both structure and entity caches for this Species.
        """
        self.clear_structure_cache()
        self.clear_entity_cache()

    def _invalidate_gene_index_cache(self) -> None:
        """Invalidate species-level gene name lookup cache."""
        self._gene_index_cache = None

    @property
    def chromosomes(self) -> List['Chromosome']:
        """Returns the list of chromosomes in this species."""
        return self._child_structures.all
    
    # Alias for backward compatibility
    @property
    def linkages(self) -> List['Chromosome']:
        """Alias for chromosomes (backward compatibility)."""
        return self.chromosomes
    
    @property
    def sex_chromosomes(self) -> List['Chromosome']:
        """返回所有性染色体"""
        return [c for c in self.chromosomes if c.is_sex_chromosome]
    
    @property
    def autosomes(self) -> List['Chromosome']:
        """返回所有常染色体"""
        return [c for c in self.chromosomes if c.is_autosome]
    
    @property
    def sex_system(self) -> Optional[str]:
        """
        返回性别决定系统 ('XY', 'ZW', 或 None)。
        
        根据 Chromosome.sex_type 自动推断。如果有多个系统会抛出错误。
        """
        systems = set()
        for chrom in self.chromosomes:
            if chrom.sex_system:
                systems.add(chrom.sex_system)
        
        if len(systems) == 0:
            return None
        elif len(systems) == 1:
            return systems.pop()
        else:
            raise ValueError(
                f"Multiple sex chromosome systems detected: {systems}. "
                f"A species should only have one sex determination system."
            )
    
    @property
    def gene_index(self) -> Dict[str, 'Gene']:
        """返回基因名称到基因实例的映射。"""
        return self._build_gene_index()

    def _build_sex_chromosome_groups(self) -> Dict[str, List['Chromosome']]:
        """
        从 Chromosome.sex_type 自动构建 _sex_chromosome_groups。
        
        Returns:
            性染色体组字典，key 是系统名如 'XY' 或 'ZW'
        """
        groups: Dict[str, List['Chromosome']] = {}
        
        for chrom in self.chromosomes:
            system = chrom.sex_system
            if system:
                if system not in groups:
                    groups[system] = []
                groups[system].append(chrom)
        
        return groups
    
    def _build_valid_sex_genotypes(self) -> List[Tuple['Chromosome', 'Chromosome']]:
        """
        从 Chromosome.sex_type 自动推断有效的性染色体基因型组合。
        
        规则：
        - XY 系统: X 可来自任意亲本，Y 只能来自 paternal
          → 有效组合: (X, X), (X, Y)
        - ZW 系统: Z 可来自任意亲本，W 只能来自 maternal
          → 有效组合: (Z, Z), (W, Z)
        
        Returns:
            有效的 (maternal_chrom, paternal_chrom) 组合列表
        """
        valid_combos: List[Tuple['Chromosome', 'Chromosome']] = []
        
        # 按性别决定系统分组染色体
        system_chroms: Dict[str, Dict[str, 'Chromosome']] = {}
        
        for chrom in self.chromosomes:
            if not chrom.is_sex_chromosome:
                continue
            system = chrom.sex_system
            if system not in system_chroms:
                system_chroms[system] = {}
            # 用性染色体类型名称作为 key
            system_chroms[system][chrom.sex_type.value] = chrom
        
        for system, chroms in system_chroms.items():
            if system == 'XY':
                x_chrom = chroms.get('X')
                y_chrom = chroms.get('Y')
                if x_chrom:
                    # XX (female) - X from both parents
                    valid_combos.append((x_chrom, x_chrom))
                    if y_chrom:
                        # XY (male) - X from maternal, Y from paternal
                        valid_combos.append((x_chrom, y_chrom))
            
            elif system == 'ZW':
                z_chrom = chroms.get('Z')
                w_chrom = chroms.get('W')
                if z_chrom:
                    # ZZ (male) - Z from both parents
                    valid_combos.append((z_chrom, z_chrom))
                    if w_chrom:
                        # ZW (female) - W from maternal, Z from paternal
                        valid_combos.append((w_chrom, z_chrom))
        
        return valid_combos
    
    def get_chromosome(self, name: str) -> 'Chromosome':
        """根据名称获取染色体"""
        return self._child_structures.get(name)
    
    def add_chromosome(
        self, 
        chrom_or_name: Union['Chromosome', str],
        loci: Optional[List[Locus]] = None,
        sex_type: Optional[Union[SexChromosomeType, str]] = None
    ) -> 'Chromosome':
        """
        Add a chromosome to this species.
        
        Args:
            chrom_or_name: Either a Chromosome instance or a name to create a new one.
            loci: Optional list of loci (only used when creating new Chromosome by name).
            sex_type: 性染色体类型 ('X', 'Y', 'Z', 'W' 或 None 表示常染色体)
            
        Returns:
            The added Chromosome instance.
        """
        if isinstance(chrom_or_name, str):
            # Create new Chromosome via base class add method
            chrom = self.add(chrom_or_name, loci=loci, sex_type=sex_type)
        elif isinstance(chrom_or_name, Chromosome):
            chrom = chrom_or_name
            # Update sex_type if provided
            if sex_type is not None:
                chrom.sex_type = sex_type
            # Register existing Chromosome if not already in registry
            if chrom.name not in self._child_structures:
                self._child_structures._storage[chrom.name] = chrom
        else:
            raise TypeError("chrom_or_name must be a Chromosome instance or string.")
        
        self._invalidate_gene_index_cache()
        return chrom
    
    # Alias for backward compatibility
    def add_linkage(
        self, 
        linkage_or_name: Union['Chromosome', str],
        loci: Optional[List[Locus]] = None
    ) -> 'Chromosome':
        """Alias for add_chromosome (backward compatibility)."""
        return self.add_chromosome(linkage_or_name, loci=loci)
    
    def remove_chromosome(self, chrom_or_name: Union['Chromosome', str]) -> None:
        """
        Remove a chromosome from this species.
        
        Args:
            chrom_or_name: Either a Chromosome instance or a name.
        """
        if isinstance(chrom_or_name, str):
            name = chrom_or_name
        else:
            name = chrom_or_name.name
        
        if name in self._child_structures:
            self._child_structures.unregister(name)
            self._invalidate_gene_index_cache()
    
    # Alias for backward compatibility
    def remove_linkage(self, linkage_or_name: Union['Chromosome', str]) -> None:
        """Alias for remove_chromosome (backward compatibility)."""
        return self.remove_chromosome(linkage_or_name)
    
    def get_all_loci(self) -> List[Locus]:
        """Returns all loci across all chromosomes."""
        all_loci = []
        for chrom in self.chromosomes:
            all_loci.extend(chrom.loci)
        return all_loci
    
    @classmethod
    def from_dict(
        cls,
        name: str,
        structure: Dict[str, Union[List[str], Dict[str, List[str]]]],
        gamete_labels: Optional[list] = None
    ) -> 'Species':
        """
        Create a Species with complete hierarchy from a dictionary specification.
        
        Args:
            name: Name of the species.
            structure: Dictionary defining the structure. Format:
                {
                    'ChromName': ['Locus1', 'Locus2', ...],  # Simple: locus names only
                    # OR
                    'ChromName': {
                        'Locus1': ['allele1', 'allele2'],  # With alleles
                        'Locus2': ['allele1', 'allele2'],
                    }
                }
        
        Returns:
            Species instance with all Chromosomes and Loci created.
        
        Example:
            >>> # Simple: just loci names
            >>> species = Species.from_dict('Species', {
            ...     'Chr1': ['LocusA', 'LocusB'],
            ...     'Chr2': ['LocusC']
            ... })
            >>> 
            >>> # With alleles
            >>> species = Species.from_dict('Species', {
            ...     'Chr1': {
            ...         'LocusA': ['A1', 'A2'],
            ...         'LocusB': ['B1', 'B2', 'B3']
            ...     },
            ...     'Chr2': {
            ...         'LocusC': ['C1', 'C2']
            ...     }
            ... })
        """
        from natal.genetic_entities import Gene
        
        species = cls(name, gamete_labels=gamete_labels)
        
        for chrom_name, loci_spec in structure.items():
            chrom = species.add_chromosome(chrom_name)
            
            if isinstance(loci_spec, list):
                # Simple format: list of locus names
                for locus_name in loci_spec:
                    chrom.add_locus(locus_name)
            
            elif isinstance(loci_spec, dict):
                # Detailed format: {locus_name: [allele_names]}
                for locus_name, allele_names in loci_spec.items():
                    locus = chrom.add_locus(locus_name)
                    # Create alleles (genes)
                    for allele_name in allele_names:
                        Gene(allele_name, locus=locus)
            else:
                raise TypeError(
                    f"Invalid loci specification for chromosome '{chrom_name}'. "
                    f"Expected list or dict, got {type(loci_spec).__name__}"
                )
        
        return species
    
    def get_locus(self, name: str) -> Optional[Locus]:
        """
        Get a locus by name across all chromosomes.
        
        Args:
            name: Name of the locus.
            
        Returns:
            The Locus instance or None if not found.
        """
        for chrom in self.chromosomes:
            for locus in chrom.loci:
                if locus.name == name:
                    return locus
        return None
    
    def get_chromosome(self, name: str) -> Optional['Chromosome']:
        """
        Get a chromosome by name.
        
        Args:
            name: Name of the chromosome.
            
        Returns:
            The Chromosome instance or None if not found.
        """
        if name in self._child_structures:
            return self._child_structures.get(name)
        return None
    
    # Alias for backward compatibility
    def get_linkage(self, name: str) -> Optional['Chromosome']:
        """Alias for get_chromosome (backward compatibility)."""
        return self.get_chromosome(name)
    
    def _build_gene_index(self) -> Dict[str, 'Gene']:
        """
        Build a lookup index from gene name to Gene object.
        
        Returns:
            Dict mapping gene name to Gene instance.
            
        Raises:
            ValueError: If duplicate gene names exist in the species.
        """
        if self._gene_index_cache is not None:
            return self._gene_index_cache

        gene_index = {}
        for chrom in self.chromosomes:
            for locus in chrom.loci:
                for gene in locus.alleles:
                    if gene.name in gene_index:
                        # TODO: 目前不支持重复基因名，后续可考虑支持
                        raise ValueError(
                            f"Duplicate gene name '{gene.name}' found in species. "
                            f"Gene names must be unique for string-based lookups. "
                            f"Found at locus '{gene.locus.name}' and '{gene_index[gene.name].locus.name}'."
                        )
                    gene_index[gene.name] = gene
        self._gene_index_cache = gene_index
        return gene_index
    
    def _parse_haplotype_segment_str(
        self, 
        hap_str: str, 
        gene_index: Dict[str, 'Gene']
    ) -> Tuple['Chromosome', List['Gene']]:
        """
        Parse a haplotype segment string into (Chromosome, [Genes]).
        
        Args:
            hap_str: String like "ABC" or "a1/b1/c1" or "Allele1"
            gene_index: Gene name to Gene lookup
            
        Returns:
            Tuple of (Chromosome, list of Genes)
        """
        hap_str = hap_str.strip()
        if not hap_str:
            raise ValueError("Empty haplotype segment string")
        
        # Parse gene names using intelligent detection:
        # 1. If contains '/', split by it
        # 2. If entire string is a valid gene name, treat as single gene
        # 3. Otherwise, try single characters
        if '/' in hap_str:
            gene_names = [g.strip() for g in hap_str.split('/')]
        elif hap_str in gene_index:
            # Entire string is a single gene name
            gene_names = [hap_str]
        else:
            # Try single characters first
            gene_names = list(hap_str)
            # Verify all chars are valid genes
            if not all(c in gene_index for c in gene_names):
                raise ValueError(
                    f"Cannot parse haplotype segment string '{hap_str}'. "
                    f"Use '/' to separate multi-character gene names. "
                    f"Available genes: {list(gene_index.keys())}"
                )
        
        # Lookup genes
        genes = []
        for gname in gene_names:
            if gname not in gene_index:
                raise ValueError(
                    f"Gene '{gname}' not found in species '{self.name}'. "
                    f"Available genes: {list(gene_index.keys())}"
                )
            genes.append(gene_index[gname])
        
        # Resolve chromosome by intersecting all candidate chromosomes of each gene locus.
        locus_to_chroms = {}
        for chrom in self.chromosomes:
            for locus in chrom.loci:
                locus_to_chroms.setdefault(locus, []).append(chrom)

        candidate_chroms = None
        for gene in genes:
            chroms_for_locus = locus_to_chroms.get(gene.locus, [])
            if not chroms_for_locus:
                raise ValueError(
                    f"Gene '{gene.name}' at locus '{gene.locus.name}' is not assigned to any chromosome "
                    f"in species '{self.name}'."
                )
            if candidate_chroms is None:
                candidate_chroms = set(chroms_for_locus)
            else:
                candidate_chroms.intersection_update(chroms_for_locus)

            if not candidate_chroms:
                raise ValueError(
                    f"No common chromosome found for genes {[g.name for g in genes]} in species '{self.name}'."
                )

        if len(candidate_chroms) > 1:
            chrom_names = [c.name for c in self.chromosomes if c in candidate_chroms]
            raise ValueError(
                f"Multiple chromosomes match genes {[g.name for g in genes]} in species '{self.name}': {chrom_names}. "
                f"Please ensure gene names are unique across chromosomes."
            )
            logger.warning(
                "Multiple chromosomes match genes %s in species '%s': %s. Use the first one: %s",
                [g.name for g in genes],
                self.name,
                chrom_names,
                chrom_names[0],
            )

        chrom = next(c for c in self.chromosomes if c in candidate_chroms)
        
        # Verify we have one gene per locus in this chromosome
        loci_with_genes = set(gene.locus for gene in genes)
        expected_loci = set(chrom.loci)
        
        if loci_with_genes != expected_loci:
            missing = expected_loci - loci_with_genes
            if missing:
                raise ValueError(
                    f"Missing genes for loci: {[l.name for l in missing]} in chromosome '{chrom.name}'"
                )
        
        # Sort genes by locus order in chromosome
        locus_order = {locus: i for i, locus in enumerate(chrom.loci)}
        genes_sorted = sorted(genes, key=lambda g: locus_order[g.locus])
        
        return chrom, genes_sorted
    
    def get_haploid_genome_from_str(self, haploid_str: str) -> 'HaploidGenome':
        """
        Create or retrieve a HaploidGenome from a string representation.
        
        Syntax:
            - Semicolon (;) separates different chromosomes
            - Slash (/) separates genes within a chromosome
            - If all genes are single characters, slash can be omitted
        
        Args:
            haploid_str: String like "ABC;XY" or "a1/b1/c1;x1/y1"
        
        Returns:
            HaploidGenome instance
        
        Example:
            >>> species = Species.from_dict("Test", {
            ...     "Chr1": {"A": ["A", "a"], "B": ["B", "b"], "C": ["C", "c"]},
            ...     "Chr2": {"X": ["X", "x"], "Y": ["Y", "y"]}
            ... })
            >>> hap = species.get_haploid_genome_from_str("ABC;XY")
            >>> hap = species.get_haploid_genome_from_str("a/b/c;x/y")  # equivalent
        """
        from natal.genetic_entities import Haplotype, HaploidGenome
        
        gene_index = self._build_gene_index()
        
        # Split by semicolon for different chromosomes
        hap_strs = [s.strip() for s in haploid_str.split(';') if s.strip()]

        # Allow sex chromosome groups: exactly one chromosome per group is required
        sex_chr_groups = getattr(self, '_sex_chromosome_groups', None)
        if sex_chr_groups:
            # Expected segments = autosomes count + number of sex groups
            autosome_count = 0
            for chrom in self.chromosomes:
                # A chromosome is considered part of a sex group if it appears in any group list
                in_sex_group = any(chrom in group for group in sex_chr_groups.values())
                if not in_sex_group:
                    autosome_count += 1
            expected_segments = autosome_count + len(sex_chr_groups)
        else:
            expected_segments = len(self.chromosomes)

        if len(hap_strs) != expected_segments:
            raise ValueError(
                f"Expected {expected_segments} haplotype segments (one per chromosome; "
                f"for sex groups: one per group), got {len(hap_strs)}. "
                f"Chromosomes: {[c.name for c in self.chromosomes]}"
            )
        
        # Parse each haplotype segment
        haplotypes = []
        chroms_used = set()
        
        for hap_str in hap_strs:
            chrom, genes = self._parse_haplotype_segment_str(hap_str, gene_index)
            
            if chrom in chroms_used:
                raise ValueError(
                    f"Chromosome '{chrom.name}' appears multiple times in haploid genome string"
                )
            chroms_used.add(chrom)
            
            hap = Haplotype(chromosome=chrom, genes=genes)
            haplotypes.append(hap)
        
        # Sort haplotypes by chromosome order in species
        chrom_order = {chrom: i for i, chrom in enumerate(self.chromosomes)}
        haplotypes_sorted = sorted(haplotypes, key=lambda h: chrom_order[h.chromosome])
        
        return HaploidGenome(species=self, haplotypes=haplotypes_sorted)
    
    def get_haploid_genotype_from_str(self, haplotype_str: str) -> 'HaploidGenome':
        """Alias for get_haploid_genome_from_str."""
        return self.get_haploid_genome_from_str(haplotype_str)
    
    def get_genotype_from_str(self, genotype_str: str) -> 'Genotype':
        """
        Create or retrieve a Genotype from a string representation.
        
        Syntax:
            - Pipe (|) separates maternal (left) and paternal (right) haploid genomes
            - Semicolon (;) separates different chromosomes
            - Slash (/) separates genes within a chromosome  
            - If all genes are single characters, slash can be omitted
        
        The order of chromosomes in the string does not need to match
        the internal chromosome order - matching is done by gene names.
        
        Args:
            genotype_str: String like "ABC|abc" or "a1/b1/c1|a2/b2/c2;X/Y|x/y"
        
        Returns:
            Genotype instance
        
        Examples:
            >>> species = Species.from_dict("Test", {
            ...     "Chr1": {"A": ["A", "a"], "B": ["B", "b"], "C": ["C", "c"]},
            ...     "Chr2": {"X": ["X", "x"], "Y": ["Y", "y"]}
            ... })
            >>> 
            >>> # Simple single-char genes
            >>> gt = species.get_genotype_from_str("ABC|abc;XY|xy")
            >>> 
            >>> # Multi-char genes with slash separator  
            >>> gt = species.get_genotype_from_str("A1/B1/C1|A2/B2/C2;X1/Y1|X2/Y2")
            >>>
            >>> # Order doesn't matter (unordered matching)
            >>> gt = species.get_genotype_from_str("XY|xy;ABC|abc")  # Same result
        """
        from natal.genetic_entities import Genotype
        
        genotype_str = genotype_str.strip()
        
        # Split by semicolon first (different chromosomes)
        chrom_segments = [s.strip() for s in genotype_str.split(';') if s.strip()]

        # Allow sex chromosome groups: exactly one chromosome per group is required
        sex_chr_groups = getattr(self, '_sex_chromosome_groups', None)
        if sex_chr_groups:
            autosome_count = 0
            for chrom in self.chromosomes:
                in_sex_group = any(chrom in group for group in sex_chr_groups.values())
                if not in_sex_group:
                    autosome_count += 1
            expected_segments = autosome_count + len(sex_chr_groups)
        else:
            expected_segments = len(self.chromosomes)

        if len(chrom_segments) != expected_segments:
            raise ValueError(
                f"Expected {expected_segments} chromosome segments (separated by ;, and one per sex group when defined), "
                f"got {len(chrom_segments)}. Chromosomes: {[c.name for c in self.chromosomes]}"
            )
        
        # For each chromosome segment, split by | to get maternal/paternal
        maternal_hap_strs = []
        paternal_hap_strs = []
        
        for segment in chrom_segments:
            parts = segment.split('|')
            if len(parts) != 2:
                raise ValueError(
                    f"Each chromosome segment must have exactly 2 parts separated by '|'. "
                    f"Got: '{segment}'"
                )
            maternal_hap_strs.append(parts[0].strip())
            paternal_hap_strs.append(parts[1].strip())
        
        # Build haploid genome strings and parse
        maternal_str = ';'.join(maternal_hap_strs)
        paternal_str = ';'.join(paternal_hap_strs)
        
        maternal = self.get_haploid_genome_from_str(maternal_str)
        paternal = self.get_haploid_genome_from_str(paternal_str)
        
        return Genotype(species=self, maternal=maternal, paternal=paternal)
    
    # ========================================================================
    # GENOTYPE PATTERN MATCHING
    # ========================================================================

    def _resolve_single_genotype_selector(
        self,
        selector: Union['Genotype', str],
        all_genotypes: Optional[Iterable['Genotype']] = None,
        context: str = 'selector'
    ) -> List['Genotype']:
        """Resolve a single genotype selector atom.

        Supports:
            - Genotype object: exact match
            - String exact genotype syntax
            - String genotype pattern syntax
        """
        from natal.genetic_entities import Genotype

        if all_genotypes is None:
            all_genotypes = self.get_all_genotypes()

        if isinstance(selector, Genotype):
            return [selector]

        if not isinstance(selector, str):
            raise TypeError(
                f"{context} selector must be Genotype or str, got {type(selector).__name__}"
            )

        # Keep backward compatibility: exact parser first, pattern parser fallback.
        try:
            return [self.get_genotype_from_str(selector)]
        except Exception as exact_err:
            try:
                pattern_filter = self.parse_genotype_pattern(selector)
            except Exception as pattern_err:
                raise ValueError(
                    f"Invalid {context} selector '{selector}'. "
                    f"Not an exact genotype string and not a valid genotype pattern. "
                    f"exact_error={exact_err}; pattern_error={pattern_err}"
                ) from pattern_err

        matched = [gt for gt in all_genotypes if pattern_filter(gt)]
        if not matched:
            raise ValueError(
                f"{context} pattern '{selector}' matched no genotypes in species '{self.name}'."
            )
        return matched

    def resolve_genotype_selectors(
        self,
        selector: Union['Genotype', str, Tuple[Union['Genotype', str], ...]],
        all_genotypes: Optional[Iterable['Genotype']] = None,
        context: str = 'selector'
    ) -> List['Genotype']:
        """Resolve one or more genotype selectors into concrete ``Genotype`` objects.

        Args:
            selector: Selector expression to resolve. Supported forms:
                - ``Genotype``: treated as an exact selector.
                - ``str``: resolved with exact-genotype parsing first; if exact
                  parsing fails, falls back to genotype-pattern parsing.
                - ``tuple`` of ``Genotype``/``str``: union semantics. Each atom
                  is resolved independently, then merged with de-duplication
                  while preserving first-seen order.
            all_genotypes: Optional candidate genotype iterable used by pattern
                matching. If ``None``, all genotypes of the species are used.
            context: Human-readable context label used in error messages (for
                example ``"viability"`` or ``"sexual_selection"``).

        Returns:
            A list of resolved ``Genotype`` objects.

            - For a single selector atom, returns all matches from that atom.
            - For tuple selectors, returns the de-duplicated union of all atom
              matches.

        Raises:
            TypeError: If a selector atom is neither ``Genotype`` nor ``str``.
            ValueError: If the selector is invalid, if pattern parsing fails, if
                a pattern matches no genotypes, or if a tuple selector is empty.
        """
        if all_genotypes is None:
            all_genotypes = self.get_all_genotypes()

        if isinstance(selector, tuple):
            if len(selector) == 0:
                raise ValueError(f"{context} selector tuple cannot be empty")

            merged: List['Genotype'] = []
            for atom in selector:
                matches = self._resolve_single_genotype_selector(
                    selector=atom,
                    all_genotypes=all_genotypes,
                    context=context,
                )
                for gt in matches:
                    if gt not in merged:
                        merged.append(gt)
            return merged

        return self._resolve_single_genotype_selector(
            selector=selector,
            all_genotypes=all_genotypes,
            context=context,
        )
    
    def parse_genotype_pattern(self, pattern: str) -> Callable[['Genotype'], bool]:
        """
        Parse a genotype pattern string and return a filter function.
        
        Supports regex-like syntax for flexible pattern matching:
            - ; separates chromosomes
            - | separates maternal (left) and paternal (right)
            - / separates loci within a chromosome
            - * matches any allele
            - {A,B,C} matches any allele in the set
            - !A matches any allele except A
            - :: matches unordered pair (A::B matches A|B or B|A)
            - () explicitly groups chromosome loci
            - Omitted chromosomes default to wildcard matching
        
        Args:
            pattern: Pattern string, e.g. "A1/B1|A2/B2; C1/C2"
        
        Returns:
            A filter function that takes a Genotype and returns bool.
        
        Examples:
            >>> filter_func = species.parse_genotype_pattern("A1/B1|A2/B2; C1::*")
            >>> genotypes = [gt for gt in pop.genotypes if filter_func(gt)]
        
        Raises:
            PatternParseError: If the pattern is invalid.
        """
        from natal.genetic_patterns import GenotypePatternParser
        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse(pattern)
        return pattern_obj.to_filter()
    
    def filter_genotypes_by_pattern(
        self,
        genotypes: Iterable['Genotype'],
        pattern: str
    ) -> List['Genotype']:
        """
        Filter a collection of genotypes by a pattern string.
        
        Args:
            genotypes: Iterable of Genotype objects to filter.
            pattern: Pattern string (see parse_genotype_pattern for syntax).
        
        Returns:
            List of genotypes that match the pattern.
        
        Examples:
            >>> matched = species.filter_genotypes_by_pattern(pop.genotypes, "A1/*|A2/B2")
        """
        pattern_filter = self.parse_genotype_pattern(pattern)
        return [gt for gt in genotypes if pattern_filter(gt)]
    
    def enumerate_genotypes_matching_pattern(
        self,
        pattern: str,
        max_count: Optional[int] = None
    ):
        """
        Enumerate all genotypes matching a pattern.
        
        Generates all possible genotype combinations that satisfy the pattern.
        If a pattern element is a wildcard (*) or set ({}), all possible
        alleles are explored. For single alleles, only that allele is used.
        
        Args:
            pattern: Pattern string (see parse_genotype_pattern for syntax).
            max_count: Maximum number of genotypes to yield (prevents explosion).
                      If None, yields all possible genotypes.
        
        Yields:
            Genotype objects matching the pattern.
        
        Examples:
            >>> for gt in species.enumerate_genotypes_matching_pattern("A1/*|A2/*; C1/C1"):
            ...     print(gt)
        
        Raises:
            PatternParseError: If the pattern is invalid.
        """
        from natal.genetic_patterns import GenotypePatternParser
        from itertools import islice, product as iterproduct
        
        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse(pattern)
        
        # For each chromosome pattern, extract allowed alleles
        count = 0
        
        try:
            for genotype_combo in self._generate_genotype_combinations(
                pattern_obj, parser
            ):
                if max_count is not None and count >= max_count:
                    return
                
                try:
                    # Convert combination back to genotype
                    genotype_str = self._convert_combo_to_genotype_str(genotype_combo)
                    genotype = self.get_genotype_from_str(genotype_str)
                    yield genotype
                    count += 1
                except Exception:
                    # Skip invalid combinations
                    continue
        except Exception:
            # Handle parsing or other errors gracefully
            return
    
    def _generate_genotype_combinations(self, pattern_obj, parser):
        """Generate all chromosome combinations from pattern.
        
        Each chromosome pattern is a ChromosomePairPattern with maternal 
        and paternal HaplotypePath objects.
        """
        from itertools import product as iterproduct
        
        chromosome_combos = []
        
        for chr_idx, chr_pattern in enumerate(pattern_obj.chromosome_patterns):
            if chr_pattern is None:
                # Omitted chromosome - skip it
                continue
            
            # Generate maternal haplotype combinations (for all loci on this chromosome)
            mat_locus_combos = []
            for locus_pattern in chr_pattern.maternal_pattern.locus_patterns:
                mat_alleles = parser.get_allowed_alleles(locus_pattern)
                mat_locus_combos.append(mat_alleles)
            mat_hap_combos = list(iterproduct(*mat_locus_combos))
            
            # Generate paternal haplotype combinations (for all loci on this chromosome)
            pat_locus_combos = []
            for locus_pattern in chr_pattern.paternal_pattern.locus_patterns:
                pat_alleles = parser.get_allowed_alleles(locus_pattern)
                pat_locus_combos.append(pat_alleles)
            pat_hap_combos = list(iterproduct(*pat_locus_combos))
            
            # All combinations for this chromosome (maternal × paternal)
            chr_combos = [
                (mat_hap_combo, pat_hap_combo)
                for mat_hap_combo in mat_hap_combos
                for pat_hap_combo in pat_hap_combos
            ]
            chromosome_combos.append((chr_idx, chr_combos))
        
        # Generate all combinations across chromosomes
        if chromosome_combos:
            chr_indices, chr_combo_lists = zip(*chromosome_combos)
            for combo in iterproduct(*chr_combo_lists):
                yield dict(zip(chr_indices, combo))
        else:
            # No specified chromosomes - yield empty
            yield {}
    
    def _convert_combo_to_genotype_str(self, combo: Dict) -> str:
        """Convert a chromosome combination back to genotype string format.
        
        combo format: {chr_idx: (mat_alleles_tuple, pat_alleles_tuple), ...}
        Output format: "A1/B1|A2/B2; C1/C1|..."
        """
        genotype_parts = []
        
        for chr_idx in range(len(self.chromosomes)):
            if chr_idx not in combo:
                # Omitted chromosome - use wildcards
                chromosome = self.chromosomes[chr_idx]
                locus_strs = ["*" for _ in chromosome.loci]
                genotype_parts.append("/".join(locus_strs) + "|" + "/".join(locus_strs))
            else:
                # Specified chromosome: combo[chr_idx] = (mat_alleles_tuple, pat_alleles_tuple)
                mat_alleles_tuple, pat_alleles_tuple = combo[chr_idx]
                
                # Convert tuples to lists of strings
                mat_str = "/".join(str(allele) for allele in mat_alleles_tuple)
                pat_str = "/".join(str(allele) for allele in pat_alleles_tuple)
                genotype_parts.append(f"{mat_str}|{pat_str}")
        
        return ";".join(genotype_parts)
    
    # ========================================================================
    # HAPLOIDGENOME PATTERN MATCHING
    # ========================================================================
    
    def parse_haploid_genome_pattern(self, pattern: str) -> Callable[['HaploidGenome'], bool]:
        """
        Parse a haploid genome pattern string and return a filter function.
        
        Supports regex-like syntax for flexible pattern matching of haploid genomes.
        A HaploidGenome represents one complete DNA strand (all haplotypes).
        Uses same syntax as Genotype patterns but applies to single haplotypes:
            - ; separates chromosomes
            - / separates loci within a chromosome
            - * matches any allele
            - {A,B,C} matches any allele in the set
            - !A matches any allele except A
            - () explicitly groups chromosome loci
            - Omitted chromosomes default to wildcard matching
        
        Args:
            pattern: Pattern string, e.g. "A1/B1; C1/C2"
        
        Returns:
            A filter function that takes a HaploidGenome and returns bool.
        
        Examples:
            >>> filter_func = species.parse_haploid_genome_pattern("A1/B1; C1")
            >>> haploid_genomes = [hg for hg in pop.haploid_genomes if filter_func(hg)]
        
        Raises:
            PatternParseError: If the pattern is invalid.
        """
        from natal.genetic_patterns import GenotypePatternParser
        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse_haploid_genome_pattern(pattern)
        return pattern_obj.to_filter()
    
    def filter_haploid_genomes_by_pattern(
        self,
        haploid_genomes: Iterable['HaploidGenome'],
        pattern: str
    ) -> List['HaploidGenome']:
        """
        Filter a collection of haploid genomes by a pattern string.
        
        Args:
            haploid_genomes: Iterable of HaploidGenome objects to filter.
            pattern: Pattern string (see parse_haploid_genome_pattern for syntax).
        
        Returns:
            List of haploid genomes that match the pattern.
        
        Examples:
            >>> matched = species.filter_haploid_genomes_by_pattern(pop.haploid_genomes, "A1/*; C1")
        """
        pattern_filter = self.parse_haploid_genome_pattern(pattern)
        return [hg for hg in haploid_genomes if pattern_filter(hg)]
    
    def enumerate_haploid_genomes_matching_pattern(
        self,
        pattern: str,
        max_count: Optional[int] = None
    ):
        """
        Enumerate all haploid genomes matching a pattern.
        
        Generates all possible haploid genome combinations that satisfy the pattern.
        If a pattern element is a wildcard (*) or set ({}), all possible
        alleles are explored. For single alleles, only that allele is used.
        
        Args:
            pattern: Pattern string (see parse_haploid_genome_pattern for syntax).
            max_count: Maximum number of haploid genomes to yield (prevents explosion).
                      If None, yields all possible haploid genomes.
        
        Yields:
            HaploidGenome objects matching the pattern.
        
        Examples:
            >>> for hg in species.enumerate_haploid_genomes_matching_pattern("A1/*; C1"):
            ...     print(hg)
        
        Raises:
            PatternParseError: If the pattern is invalid.
        """
        from natal.genetic_patterns import GenotypePatternParser
        from itertools import islice, product as iterproduct
        
        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse_haploid_genome_pattern(pattern)
        
        # For each haplotype pattern, extract allowed alleles
        count = 0
        
        try:
            for haploid_combo in self._generate_haploid_genome_combinations(
                pattern_obj, parser
            ):
                if max_count is not None and count >= max_count:
                    return
                
                try:
                    # Convert combination back to haploid genome
                    haploid_str = self._convert_haploid_combo_to_haploid_genome_str(haploid_combo)
                    haploid_genome = self.get_haploid_genome_from_str(haploid_str)
                    yield haploid_genome
                    count += 1
                except Exception:
                    # Skip invalid combinations
                    continue
        except Exception:
            # Handle parsing or other errors gracefully
            return
    
    def _generate_haploid_genome_combinations(self, pattern_obj, parser):
        """Generate all haplotype combinations from a HaploidGenomePattern.
        
        Each haplotype pattern is a HaplotypePath with a list of locus
        patterns for one DNA strand.
        """
        from itertools import product as iterproduct
        
        chromosome_combos = []
        
        for chr_idx, haplotype_pattern in enumerate(pattern_obj.haplotype_patterns):
            if haplotype_pattern is None:
                # Omitted chromosome - skip it
                continue
            
            # Generate locus combinations for this haplotype
            locus_combos = []
            for locus_pattern in haplotype_pattern.locus_patterns:
                alleles = parser.get_allowed_alleles(locus_pattern)
                locus_combos.append(alleles)
            
            # Cartesian product of all loci for this chromosome
            hap_combos = list(iterproduct(*locus_combos))
            chromosome_combos.append((chr_idx, hap_combos))
        
        # Generate all combinations across chromosomes
        if chromosome_combos:
            chr_indices, chr_combo_lists = zip(*chromosome_combos)
            for combo in iterproduct(*chr_combo_lists):
                yield dict(zip(chr_indices, combo))
        else:
            # No specified chromosomes - yield empty
            yield {}
    
    def _convert_haploid_combo_to_haploid_genome_str(self, combo: Dict) -> str:
        """Convert a haplotype combination back to haploid genome string format.
        
        combo format: {chr_idx: alleles_tuple, ...}
        Output format: "A1/B1; C1"
        """
        haploid_parts = []
        
        for chr_idx in range(len(self.chromosomes)):
            if chr_idx not in combo:
                # Omitted chromosome - use wildcards
                chromosome = self.chromosomes[chr_idx]
                locus_strs = ["*" for _ in chromosome.loci]
                haploid_parts.append("/".join(locus_strs))
            else:
                # Specified chromosome: combo[chr_idx] = alleles_tuple
                alleles_tuple = combo[chr_idx]
                
                # Convert tuple to string
                allele_str = "/".join(str(allele) for allele in alleles_tuple)
                haploid_parts.append(allele_str)
        
        return ";".join(haploid_parts)
    
    def __repr__(self):
        chrom_strs = []
        for chrom in self.chromosomes:
            loci_names = [locus.name for locus in chrom.loci]
            chrom_strs.append(f"'{chrom.name}': {loci_names}")
        return f"Species({self.name!r}, {{{', '.join(chrom_strs)}}})"

    def __iter__(self):
        return iter(self.chromosomes)
    
    def __len__(self):
        return len(self.chromosomes)
    
    # ========================================================================
    # 基因型枚举和计数
    # ========================================================================
    
    def _get_sex_chromosome_groups(self) -> Optional[Dict[str, List['Chromosome']]]:
        """
        获取性染色体组配置。
        
        优先使用显式设置的 _sex_chromosome_groups 属性，
        否则从 Chromosome.sex_type 自动推断。
        
        Returns:
            性染色体组字典，或 None（如果没有性染色体）
        """
        # 优先使用显式设置
        if hasattr(self, '_sex_chromosome_groups') and self._sex_chromosome_groups:
            return self._sex_chromosome_groups
        
        # 自动推断
        groups = self._build_sex_chromosome_groups()
        return groups if groups else None
    
    def _get_valid_sex_genotypes(self) -> Optional[List[Tuple['Chromosome', 'Chromosome']]]:
        """
        获取有效的性染色体基因型组合。
        
        优先使用显式设置的 _valid_sex_genotypes 属性，
        否则从 Chromosome.sex_type 自动推断。
        
        Returns:
            有效的 (maternal_chrom, paternal_chrom) 组合列表，或 None
        """
        # 优先使用显式设置
        if hasattr(self, '_valid_sex_genotypes') and self._valid_sex_genotypes:
            return self._valid_sex_genotypes
        
        # 自动推断
        valid = self._build_valid_sex_genotypes()
        return valid if valid else None
    
    def count_alleles(self) -> int:
        """
        计算所有位点的等位基因总数。
        
        Returns:
            等位基因总数
        """
        total = 0
        for chrom in self.chromosomes:
            for locus in chrom.loci:
                total += len(locus.alleles)
        return total
    
    def count_haploid_genotypes(self) -> int:
        """
        计算所有可能的单倍体基因组数量。
        
        对于每个位点有 n 个等位基因，单倍体基因组数 = 各位点等位基因数的乘积。
        如果存在性染色体组，每个组内只选一个染色体。
        
        Returns:
            可能的单倍体基因组总数
        """
        # 获取性染色体配置（优先使用显式设置，否则自动推断）
        sex_chr_groups = self._get_sex_chromosome_groups()
        
        # 识别性染色体组中的所有染色体
        sex_chroms = set()
        if sex_chr_groups:
            for group_chroms in sex_chr_groups.values():
                sex_chroms.update(group_chroms)
        
        total = 1
        
        # 常染色体的等位基因数乘积
        for chrom in self.chromosomes:
            if chrom in sex_chroms:
                continue  # 跳过性染色体，后面单独处理
            for locus in chrom.loci:
                n_alleles = len(locus.alleles)
                if n_alleles > 0:
                    total *= n_alleles
        
        # 对于性染色体组，每个组可以选择组内任一染色体
        # 每个染色体的 haplotype 数量 = 其 loci 等位基因数的乘积
        if sex_chr_groups:
            for group_chroms in sex_chr_groups.values():
                group_total = 0
                for chrom in group_chroms:
                    chrom_total = 1
                    for locus in chrom.loci:
                        n_alleles = len(locus.alleles)
                        if n_alleles > 0:
                            chrom_total *= n_alleles
                    group_total += chrom_total
                total *= group_total
        
        return total
    
    def count_genotypes(self) -> int:
        """
        计算所有可能的二倍体基因型数量。
        
        如果定义了 _valid_sex_genotypes，只计算有效的性染色体组合。
        
        性染色体系统配置：
        - 可以通过设置 Chromosome.sex_type 自动推断
        - 也可以手动设置 _sex_chromosome_groups 和 _valid_sex_genotypes
        
        Returns:
            可能的基因型总数
        """
        # 使用辅助方法获取配置（优先显式设置，否则自动推断）
        sex_chr_groups = self._get_sex_chromosome_groups()
        valid_sex_gts = self._get_valid_sex_genotypes()
        
        if not sex_chr_groups:
            # 没有性染色体，简单的 n^2
            n_haploid = self.count_haploid_genotypes()
            return n_haploid * n_haploid
        
        # 有性染色体时需要特殊处理
        # 先计算常染色体部分的组合数
        sex_chroms = set()
        for group_chroms in sex_chr_groups.values():
            sex_chroms.update(group_chroms)
        
        autosome_haploid_count = 1
        for chrom in self.chromosomes:
            if chrom in sex_chroms:
                continue
            for locus in chrom.loci:
                n_alleles = len(locus.alleles)
                if n_alleles > 0:
                    autosome_haploid_count *= n_alleles
        
        # 常染色体的基因型数 = autosome_haploid_count^2
        autosome_genotype_count = autosome_haploid_count * autosome_haploid_count
        
        # 计算每个染色体的 haplotype 数
        def count_chrom_haplotypes(chrom):
            count = 1
            for locus in chrom.loci:
                n_alleles = len(locus.alleles)
                if n_alleles > 0:
                    count *= n_alleles
            return count
        
        # 对于性染色体组，计算有效组合数
        if valid_sex_gts:
            # 使用显式定义的有效基因型
            sex_genotype_count = 0
            for mat_chrom, pat_chrom in valid_sex_gts:
                n_mat = count_chrom_haplotypes(mat_chrom)
                n_pat = count_chrom_haplotypes(pat_chrom)
                sex_genotype_count += n_mat * n_pat
        else:
            # 没有定义有效基因型，默认所有组合都有效
            sex_genotype_count = 1
            for group_chroms in sex_chr_groups.values():
                group_total = 0
                for chrom in group_chroms:
                    group_total += count_chrom_haplotypes(chrom)
                # 每个组的 maternal × paternal
                sex_genotype_count *= group_total * group_total
        
        return autosome_genotype_count * sex_genotype_count
    
    def iter_haploid_genotypes(self) -> Iterable['HaploidGenome']:
        """
        迭代所有可能的单倍体基因组 (HaploidGenome)。
        
        如果存在性染色体组，每个组只选择一个染色体。
        注意：此方法返回所有可能的 haploid genotypes，不区分 maternal/paternal。
        对于需要区分的场景，请使用 iter_maternal_haploid_genotypes() 和 
        iter_paternal_haploid_genotypes()。
        
        Yields:
            HaploidGenome 实例
        
        Example:
            >>> for hg in species.iter_haploid_genotypes():
            ...     print(hg)
        """
        from natal.genetic_entities import Haplotype, HaploidGenome
        
        sex_chr_groups = self._get_sex_chromosome_groups()
        
        # 识别性染色体组中的所有染色体
        sex_chroms = set()
        if sex_chr_groups:
            for group_chroms in sex_chr_groups.values():
                sex_chroms.update(group_chroms)
        
        # 为常染色体收集所有可能的 Haplotype
        autosome_haplotypes: List[List['Haplotype']] = []
        for chrom in self.chromosomes:
            if chrom in sex_chroms:
                continue  # 性染色体单独处理
            
            locus_alleles = [list(locus.alleles) for locus in chrom.loci]
            if not locus_alleles or any(len(a) == 0 for a in locus_alleles):
                continue
            
            haps_for_chrom = []
            for genes in itertools.product(*locus_alleles):
                hap = Haplotype(chromosome=chrom, genes=list(genes))
                haps_for_chrom.append(hap)
            autosome_haplotypes.append(haps_for_chrom)
        
        # 为性染色体组收集可能的 Haplotype 选项
        # 每个组内的所有染色体的 haplotype 放在一个列表中（选其一）
        sex_group_haplotypes: List[List['Haplotype']] = []
        if sex_chr_groups:
            for group_chroms in sex_chr_groups.values():
                group_haps = []
                for chrom in group_chroms:
                    locus_alleles = [list(locus.alleles) for locus in chrom.loci]
                    if not locus_alleles or any(len(a) == 0 for a in locus_alleles):
                        continue
                    for genes in itertools.product(*locus_alleles):
                        hap = Haplotype(chromosome=chrom, genes=list(genes))
                        group_haps.append(hap)
                if group_haps:
                    sex_group_haplotypes.append(group_haps)
        
        # 合并常染色体和性染色体组的 haplotype 列表
        all_haplotype_options = autosome_haplotypes + sex_group_haplotypes
        
        if not all_haplotype_options:
            return
        
        # 所有组合 -> HaploidGenome
        for haplotype_combo in itertools.product(*all_haplotype_options):
            yield HaploidGenome(species=self, haplotypes=list(haplotype_combo))
    
    def _iter_haploid_genotypes_for_parent(
        self, 
        is_paternal: bool
    ) -> Iterable['HaploidGenome']:
        """
        迭代指定亲本（maternal 或 paternal）可用的单倍体基因组。
        
        根据 _valid_sex_genotypes 确定每个亲本可用的染色体。
        
        Args:
            is_paternal: True 表示 paternal，False 表示 maternal
            
        Yields:
            HaploidGenome 实例
        """
        from natal.genetic_entities import Haplotype, HaploidGenome
        
        sex_chr_groups = self._get_sex_chromosome_groups()
        valid_sex_gts = self._get_valid_sex_genotypes()
        
        # 识别性染色体组中的所有染色体
        sex_chroms = set()
        if sex_chr_groups:
            for group_chroms in sex_chr_groups.values():
                sex_chroms.update(group_chroms)
        
        # 确定该亲本可用的性染色体
        available_sex_chroms = set()
        if sex_chr_groups:
            if valid_sex_gts:
                # 从有效基因型中提取该亲本可用的染色体
                for mat_chrom, pat_chrom in valid_sex_gts:
                    if is_paternal:
                        available_sex_chroms.add(pat_chrom)
                    else:
                        available_sex_chroms.add(mat_chrom)
            else:
                # 没有限制，所有性染色体都可用
                available_sex_chroms = sex_chroms
        
        # 为常染色体收集所有可能的 Haplotype
        autosome_haplotypes: List[List['Haplotype']] = []
        for chrom in self.chromosomes:
            if chrom in sex_chroms:
                continue
            
            locus_alleles = [list(locus.alleles) for locus in chrom.loci]
            if not locus_alleles or any(len(a) == 0 for a in locus_alleles):
                continue
            
            haps_for_chrom = []
            for genes in itertools.product(*locus_alleles):
                hap = Haplotype(chromosome=chrom, genes=list(genes))
                haps_for_chrom.append(hap)
            autosome_haplotypes.append(haps_for_chrom)
        
        # 为性染色体组收集可能的 Haplotype 选项
        sex_group_haplotypes: List[List['Haplotype']] = []
        if sex_chr_groups:
            for group_chroms in sex_chr_groups.values():
                group_haps = []
                for chrom in group_chroms:
                    # 只包含该亲本可用的染色体
                    if chrom not in available_sex_chroms:
                        continue
                    
                    locus_alleles = [list(locus.alleles) for locus in chrom.loci]
                    if not locus_alleles or any(len(a) == 0 for a in locus_alleles):
                        continue
                    for genes in itertools.product(*locus_alleles):
                        hap = Haplotype(chromosome=chrom, genes=list(genes))
                        group_haps.append(hap)
                if group_haps:
                    sex_group_haplotypes.append(group_haps)
        
        # 合并常染色体和性染色体组的 haplotype 列表
        all_haplotype_options = autosome_haplotypes + sex_group_haplotypes
        
        if not all_haplotype_options:
            return
        
        # 所有组合 -> HaploidGenome
        for haplotype_combo in itertools.product(*all_haplotype_options):
            yield HaploidGenome(species=self, haplotypes=list(haplotype_combo))
    
    def iter_maternal_haploid_genotypes(self) -> Iterable['HaploidGenome']:
        """
        迭代 maternal（母本）可遗传的单倍体基因组。
        
        根据 _valid_sex_genotypes 确定可用的性染色体。
        
        Yields:
            HaploidGenome 实例
        """
        return self._iter_haploid_genotypes_for_parent(is_paternal=False)
    
    def iter_paternal_haploid_genotypes(self) -> Iterable['HaploidGenome']:
        """
        迭代 paternal（父本）可遗传的单倍体基因组。
        
        根据 _valid_sex_genotypes 确定可用的性染色体。
        
        Yields:
            HaploidGenome 实例
        """
        return self._iter_haploid_genotypes_for_parent(is_paternal=True)
    
    def iter_genotypes(self) -> Iterable['Genotype']:
        """
        迭代所有可能的基因型 (Genotype)。
        
        区分 maternal 和 paternal，所以 (A|B) 和 (B|A) 是不同的基因型。
        如果定义了 _valid_sex_genotypes 或 Chromosome.sex_type，只生成有效的性染色体组合。
        
        Yields:
            Genotype 实例
        
        Example:
            >>> for gt in species.iter_genotypes():
            ...     print(gt)
        """
        from natal.genetic_entities import Genotype
        
        sex_chr_groups = self._get_sex_chromosome_groups()
        valid_sex_gts = self._get_valid_sex_genotypes()
        
        if not sex_chr_groups:
            # 没有性染色体，简单的笛卡尔积
            all_haploid_genotypes = list(self.iter_haploid_genotypes())
            for maternal, paternal in itertools.product(all_haploid_genotypes, repeat=2):
                yield Genotype(species=self, maternal=maternal, paternal=paternal)
        elif valid_sex_gts:
            # 有性染色体且定义了有效基因型，需要验证组合
            maternal_hgs = list(self.iter_maternal_haploid_genotypes())
            paternal_hgs = list(self.iter_paternal_haploid_genotypes())
            
            # 构建有效组合的 set 用于快速查找
            valid_chrom_pairs = set(valid_sex_gts)
            
            for maternal, paternal in itertools.product(maternal_hgs, paternal_hgs):
                # 获取 maternal 和 paternal 的性染色体
                mat_sex_chrom = self._get_sex_chromosome(maternal, sex_chr_groups)
                pat_sex_chrom = self._get_sex_chromosome(paternal, sex_chr_groups)
                
                # 检查是否是有效组合
                if (mat_sex_chrom, pat_sex_chrom) in valid_chrom_pairs:
                    yield Genotype(species=self, maternal=maternal, paternal=paternal)
        else:
            # 有性染色体但没有定义有效基因型，所有组合都有效
            maternal_hgs = list(self.iter_maternal_haploid_genotypes())
            paternal_hgs = list(self.iter_paternal_haploid_genotypes())
            
            for maternal, paternal in itertools.product(maternal_hgs, paternal_hgs):
                yield Genotype(species=self, maternal=maternal, paternal=paternal)
    
    def _get_sex_chromosome(
        self, 
        haploid_genome: 'HaploidGenome',
        sex_chr_groups: Dict[str, List['Chromosome']]
    ) -> Optional['Chromosome']:
        """
        获取 HaploidGenome 中的性染色体。
        
        假设每个性染色体组只选择一个染色体。
        
        Args:
            haploid_genome: 单倍体基因组
            sex_chr_groups: 性染色体组定义
            
        Returns:
            性染色体，如果没有则返回 None
        """
        sex_chroms = set()
        for group_chroms in sex_chr_groups.values():
            sex_chroms.update(group_chroms)
        
        for hap in haploid_genome.haplotypes:
            if hap.chromosome in sex_chroms:
                return hap.chromosome
        return None
    
    def get_all_haploid_genotypes(self) -> List['HaploidGenome']:
        """
        获取所有可能的单倍体基因组列表。
        
        Returns:
            所有 HaploidGenome 实例的列表
        """
        return list(self.iter_haploid_genotypes())
    
    def get_all_genotypes(self) -> List['Genotype']:
        """
        获取所有可能的基因型列表。
        
        Returns:
            所有 Genotype 实例的列表
        """
        return list(self.iter_genotypes())


# Aliases for backward compatibility
Linkage = Chromosome
GenomeTemplate = Species
Karyotype = Species
