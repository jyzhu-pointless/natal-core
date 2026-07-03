"""Base registry classes for genetic structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeGuard,
    Union,
    cast,
)

from ._types import E, S, T

if TYPE_CHECKING:
    from ._base import GeneticStructure
    from .species import Species

# Global fallback cache for structures created without a Species (backward compatibility)
# Format: {structure_type: {name: instance}}
_GLOBAL_STRUCTURE_CACHE: Dict[type, Dict[str, GeneticStructure[Any]]] = {}


class RegistryBase(ABC, Generic[T]):
    """
    Base class for registries.

    Provides the common interface for register/unregister operations while
    delegating storage semantics to subclass hooks.

    Attributes:
        _expected_type (Optional[type[GeneticStructure[E]]]): Runtime type used to
            validate registry items when provided.
    """
    # RegistryBase intentionally separates:
    # - "single item" primitives (_single_register/_single_unregister/_single_unregister_by_key)
    # - "batch input" orchestration (register/unregister)
    #
    # This keeps subclass logic focused on storage semantics while centralizing
    # input normalization and strict runtime type checks.
    def __init__(self, expected_type: Optional[type[GeneticStructure[E]]] = None):
        self._expected_type = expected_type

    def _check_type(self, item: T) -> None:
        if self._expected_type and not isinstance(item, self._expected_type):
            raise TypeError(f"Expected type {self._expected_type.__name__}, got {type(item).__name__}")

    def _is_valid_item_type(self, item: object) -> TypeGuard[T]:
        """Runtime validator + static type narrower for generic T.

        Why this exists:
        - Pylance/Pyright often cannot narrow Union/object to T in generic code.
        - TypeGuard[T] makes the narrowing explicit after a runtime check.
        """
        if self._expected_type is None:
            return True
        return isinstance(item, self._expected_type)

    @abstractmethod
    def _get_key(self, item: T) -> Hashable:
        """Extract the key for deduplication. Override in subclass."""
        pass

    @abstractmethod
    def _single_register(self, item: T) -> None:
        """Register a single item. Override in subclass."""
        pass

    @abstractmethod
    def _single_unregister(self, item: T) -> None:
        """Unregister a single item. Override in subclass."""
        pass

    @abstractmethod
    def _single_unregister_by_key(self, key: str) -> None:
        """Unregister a single item by key. Override in subclass."""
        pass

    def register(self, item_or_items: Union[T, List[T], Tuple[T, ...], Set[T]]) -> None:
        """Register one or more items."""
        # GeneticStructure is iterable (yields children) but should be registered as single item
        # Use explicit list/tuple/set check instead of Iterable to avoid this.
        # (str / custom iterable objects should not be silently treated as batch input.)
        if isinstance(item_or_items, list):
            item_or_items = cast(List[T], item_or_items)
            for item in item_or_items:
                if self._is_valid_item_type(item):
                    self._single_register(item)
                else:
                    raise TypeError(f"Expected registry item type, got {type(item).__name__}")
        elif isinstance(item_or_items, tuple):
            item_or_items = cast(Tuple[T, ...], item_or_items)
            for item in item_or_items:
                if self._is_valid_item_type(item):
                    self._single_register(item)
                else:
                    raise TypeError(f"Expected registry item type, got {type(item).__name__}")
        elif isinstance(item_or_items, set):
            item_or_items = cast(Set[T], item_or_items)
            for item in item_or_items:
                if self._is_valid_item_type(item):
                    self._single_register(item)
                else:
                    raise TypeError(f"Expected registry item type, got {type(item).__name__}")
        else:
            if self._is_valid_item_type(item_or_items):
                self._single_register(item_or_items)
            else:
                raise TypeError(f"Expected registry item type, got {type(item_or_items).__name__}")

    def unregister(
        self,
        item_or_items: Union[T, str, List[Union[T, str]], Tuple[Union[T, str], ...], Set[Union[T, str]]]
    ) -> None:
        """Unregister one or more items (by key or item object)."""
        # unregister supports mixed batch input: [item, "name", item, ...]
        # where str means key-based removal and non-str means object removal.
        if isinstance(item_or_items, list):
            item_or_items = cast(List[Union[T, str]], item_or_items)
            for item in item_or_items:
                if isinstance(item, str):
                    self._single_unregister_by_key(item)
                else:
                    if self._is_valid_item_type(item):
                        self._single_unregister(item)
                    else:
                        raise TypeError(f"Expected registry item type, got {type(item).__name__}")
        elif isinstance(item_or_items, tuple):
            item_or_items = cast(Tuple[Union[T, str], ...], item_or_items)
            for item in item_or_items:
                if isinstance(item, str):
                    self._single_unregister_by_key(item)
                else:
                    if self._is_valid_item_type(item):
                        self._single_unregister(item)
                    else:
                        raise TypeError(f"Expected registry item type, got {type(item).__name__}")
        elif isinstance(item_or_items, set):
            item_or_items = cast(Set[Union[T, str]], item_or_items)
            for item in item_or_items:
                if isinstance(item, str):
                    self._single_unregister_by_key(item)
                else:
                    if self._is_valid_item_type(item):
                        self._single_unregister(item)
                    else:
                        raise TypeError(f"Expected registry item type, got {type(item).__name__}")
        elif isinstance(item_or_items, str):
            self._single_unregister_by_key(item_or_items)
        else:
            if self._is_valid_item_type(item_or_items):
                self._single_unregister(item_or_items)
            else:
                raise TypeError(f"Expected registry item type, got {type(item_or_items).__name__}")

    def __len__(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class EntityRegistry(RegistryBase[E]):
    """
    Registry for entity objects. Deduplication by object identity.

    Attributes:
        _storage (List[E]): Ordered storage for deterministic iteration.
        _set (Set[E]): Identity-based membership set for O(1) deduplication checks.
    """
    # EntityRegistry is identity-based:
    # - _set provides O(1) membership uniqueness checks
    # - _storage preserves insertion order for deterministic iteration
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

    def _single_unregister_by_key(self, key: str) -> None:
        # Entities do not have a globally unique name key in this registry layer.
        # Key-based unregister belongs to ChildStructureRegistry.
        raise TypeError(
            "EntityRegistry does not support unregistering by string key; pass entity instance(s) instead."
        )

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

    Attributes:
        _owner (GeneticStructure[Any]): Parent structure that owns this registry.
        _storage (Dict[str, S]): Name-to-child mapping for registered structures.
    """
    def __init__(
        self,
        owner: GeneticStructure[Any],
        expected_type: Optional[type[GeneticStructure[E]]] = None
    ):
        super().__init__(expected_type)
        self._owner = owner  # The parent structure that owns this registry
        self._storage: Dict[str, S] = {}

    def _is_expected_child(self, item: object) -> TypeGuard[S]:
        # Additional explicit narrower used when reading from heterogeneous caches.
        # Cache values are stored in broader Dict[str, GeneticStructure] maps; this
        # function safely narrows back to S for assignment into Dict[str, S].
        expected_type = self._expected_type
        return expected_type is not None and isinstance(item, expected_type)

    def _get_key(self, item: S) -> str:
        return item.name

    def _single_register(self, item: S) -> None:
        """Register an existing child structure."""
        self._check_type(item)
        if not hasattr(item, 'name'):
            raise TypeError("Child must have a 'name' attribute.")
        if item.name not in self._storage:
            self._storage[item.name] = item

    def _single_unregister(self, item: S) -> None:
        """Unregister by item."""
        self._storage.pop(item.name, None)

    def _single_unregister_by_key(self, key: str) -> None:
        """Unregister by name."""
        self._storage.pop(key, None)

    def add(self, name: str, **kwargs: Any) -> S:  # forwarded to expected_type child constructor
        """
        Create a new child structure and register it.
        This is a convenience method: create + register.

        If a child with the same *name* already exists in this registry, the
        cached instance is returned immediately.  This makes ``add`` idempotent
        and consistent with ``GeneticStructure.__new__``, which also returns
        cached instances rather than creating duplicates.

        Uses Species-level caching to ensure uniqueness within the same Species.
        """
        assert isinstance(name, str), "Child structure name must be a string."
        if not name.strip():
            raise ValueError("Child structure name must be a non-empty string.")
        if name in self._storage:
            return self._storage[name]
        expected_type = self._expected_type
        if expected_type is None:
            raise ValueError("expected_type not set, cannot construct child structure.")

        # Get the Species from the owner
        species: Optional[Species] = self._owner.species

        # Check if structure already exists in cache (Species-scoped or global)
        if species is not None:
            # Preferred path: Species-scoped cache (isolation between species).
            if expected_type not in species.structure_cache:
                species.structure_cache[expected_type] = {}

            cache = species.structure_cache[expected_type]
            if name in cache:
                # Return cached instance
                cached_child = cache[name]
                if self._is_expected_child(cached_child):
                    # Still register it in this owner's registry if not already there
                    if name not in self._storage:
                        self._storage[name] = cached_child
                    return cached_child
                raise TypeError(
                    f"Cached structure for '{name}' has wrong type: "
                    f"expected {expected_type.__name__}, got {type(cached_child).__name__}"
                )

            # Create new child with parent (species is inherited automatically)
            created_child = expected_type(name, parent=self._owner, **kwargs)
            if not self._is_expected_child(created_child):
                raise TypeError(
                    f"Created structure for '{name}' has wrong type: "
                    f"expected {expected_type.__name__}, got {type(created_child).__name__}"
                )
            child = created_child

            # Cache it in the Species
            cache[name] = child
        else:
            # Fallback path for backward compatibility with structures not yet
            # bound into a Species context.
            if expected_type not in _GLOBAL_STRUCTURE_CACHE:
                _GLOBAL_STRUCTURE_CACHE[expected_type] = {}

            cache = _GLOBAL_STRUCTURE_CACHE[expected_type]
            if name in cache:
                # Return cached instance
                cached_child = cache[name]
                if self._is_expected_child(cached_child):
                    # Still register it in this owner's registry if not already there
                    if name not in self._storage:
                        self._storage[name] = cached_child
                    return cached_child
                raise TypeError(
                    f"Cached structure for '{name}' has wrong type: "
                    f"expected {expected_type.__name__}, got {type(cached_child).__name__}"
                )

            # Create new child with parent (no species means orphan)
            created_child = expected_type(name, parent=self._owner, **kwargs)
            if not self._is_expected_child(created_child):
                raise TypeError(
                    f"Created structure for '{name}' has wrong type: "
                    f"expected {expected_type.__name__}, got {type(created_child).__name__}"
                )
            child = created_child

            # Cache it globally
            cache[name] = child

        return child

    def get(self, name: str) -> S:
        """Get a child structure by name."""
        if name not in self._storage:
            raise KeyError(f"No child structure named '{name}' found.")
        return self._storage[name]

    def __iter__(self):
        """Iterate over child structures."""
        return iter(self._storage.values())

    def __contains__(self, name_or_item: Union[str, GeneticStructure[Any]]) -> bool:
        """Check if a child structure exists by name or instance."""
        name = name_or_item if isinstance(name_or_item, str) else name_or_item.name
        return name in self._storage

    def __len__(self) -> int:
        """Return the number of registered structures."""
        return len(self._storage)

    def clear(self) -> None:
        """Clear all registered child structures."""
        self._storage.clear()

    @property
    def all(self) -> List[S]:
        """Returns all registered child structures."""
        return list(self._storage.values())
