"""
Genotype selector for observation and filtering, plus zygote type resolution.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Set

from natal.genetics import Species
from natal.registry.index import IndexRegistry

from .elements.diploid import GenotypePattern, ZygoteTypePattern
from .parser import GenotypePatternParser


class GenotypeSelector:
    """Unified genotype selector for observation and filtering.

    This class provides a unified interface for selecting genotypes using various
    input formats, leveraging the existing pattern matching system.
    """

    def __init__(self, species: Species):
        """Initialize genotype selector for a specific species.

        Args:
            species: The Species object to use for pattern parsing.
        """
        self.species = species
        self.parser = GenotypePatternParser(species)

    def resolve_genotype_indices(
        self,
        gen_spec: Optional[Iterable[Any]],
        diploid_genotypes: Optional[Sequence[Any]],
        unordered: bool = False,
    ) -> List[int]:
        """Resolve genotype selectors into a list of indices.

        This method provides the same functionality as observation.py's
        _resolve_genotype_list() but uses the pattern matching system.

        Args:
            gen_spec: Genotype selector specification. Can be:
                - None: select all genotypes
                - int: genotype index
                - str: genotype pattern string
                - Genotype: genotype object
                - Iterable of any of the above
            diploid_genotypes: Sequence of diploid genotypes for resolution.
            unordered: Whether to treat genotypes as unordered (A|a == a|A).

        Returns:
            List of resolved genotype indices.

        Raises:
            ValueError: If diploid_genotypes is required but missing.
        """
        if gen_spec is None:
            if diploid_genotypes is None:
                raise ValueError("diploid_genotypes required to enumerate genotypes")
            return list(range(len(diploid_genotypes)))

        # Handle single item vs iterable
        if not isinstance(gen_spec, (list, tuple, set)):
            gen_spec = [gen_spec]

        resolved_indices: Set[int] = set()

        for selector in gen_spec:
            if isinstance(selector, int):
                # Direct index
                resolved_indices.add(selector)
            elif isinstance(selector, str):
                # Pattern string - use pattern matching system
                pattern = self.parser.parse(selector)
                if diploid_genotypes is None:
                    raise ValueError("diploid_genotypes required for pattern matching")

                for i, genotype in enumerate(diploid_genotypes):
                    if pattern.matches(genotype):
                        resolved_indices.add(i)
            else:
                # Assume it's a Genotype object or similar
                if diploid_genotypes is None:
                    raise ValueError("diploid_genotypes required for genotype matching")

                for i, genotype in enumerate(diploid_genotypes):
                    if self._genotypes_equal(selector, genotype, unordered):
                        resolved_indices.add(i)

        return sorted(resolved_indices)

    def _genotypes_equal(
        self,
        gen1: Any,
        gen2: Any,
        unordered: bool = False
    ) -> bool:
        """Check if two genotypes are equal, with optional unordered matching.

        NOTE: This method and its callers (``resolve_genotype_indices``,
        ``create_filter_function``) are **legacy/dead code** as of 2026-06.
        The active code path uses ``ZygoteTypePattern.parse()`` + ``::`` syntax
        directly. ``unordered=True`` is never passed from any caller, so the
        ``|`` → ``::`` fallback below never triggers. Keep for reference.

        Args:
            gen1: First genotype.
            gen2: Second genotype.
            unordered: If True, consider genotypes equal regardless of maternal/paternal order.

        Returns:
            True if genotypes are equal.
        """
        if not unordered:
            return gen1 == gen2

        # For unordered matching, check both orderings
        try:
            # Try direct equality first
            if gen1 == gen2:
                return True

            # Try reversed ordering if genotypes support it
            if hasattr(gen1, 'reversed') and hasattr(gen2, 'reversed'):
                return gen1.reversed() == gen2 or gen1 == gen2.reversed()

            # Fallback: use string representation comparison
            gen1_str = str(gen1)
            gen2_str = str(gen2)

            # Check if strings are equal when normalized for unordered matching
            if "::" in gen1_str or "::" in gen2_str:
                # Already using unordered notation
                return gen1_str == gen2_str
            else:
                # Convert to unordered notation and compare
                gen1_unordered = gen1_str.replace("|", "::")
                gen2_unordered = gen2_str.replace("|", "::")
                return gen1_unordered == gen2_unordered

        except Exception:
            # If any comparison fails, fall back to direct equality
            return gen1 == gen2

    def create_filter_function(
        self,
        gen_spec: Optional[Iterable[Any]],
        unordered: bool = False
    ) -> Callable[[Any], bool]:
        """Create a filter function for genotype selection.

        NOTE: **Dead code** as of 2026-06 — zero callers in the codebase.
        The active path uses ``ZygoteTypePattern`` and ``resolve_zygote_type``
        directly. Keep for reference.

        Args:
            gen_spec: Genotype selector specification.
            unordered: Whether to use unordered matching.

        Returns:
            A callable that takes a genotype and returns True if it matches.
        """
        if gen_spec is None:
            # Match all genotypes
            return lambda genotype: True

        # Handle single item vs iterable
        if not isinstance(gen_spec, (list, tuple, set)):
            gen_spec = [gen_spec]

        # Create pattern-based filters for string selectors
        pattern_filters: List[Callable[[Any], bool]] = []
        other_selectors: List[Any] = []

        for selector in gen_spec:
            if isinstance(selector, str):
                pattern = self.parser.parse(selector)
                pattern_filters.append(pattern.to_filter())
            else:
                other_selectors.append(selector)

        def filter_func(genotype: Any) -> bool:
            # Check pattern filters
            for pattern_filter in pattern_filters:
                if pattern_filter(genotype):
                    return True

            # Check other selectors
            for selector in other_selectors:
                if self._genotypes_equal(selector, genotype, unordered):
                    return True

            return False

        return filter_func

    def get_pattern_for_selector(self, selector: Any) -> Optional[GenotypePattern]:
        """Convert a selector to a GenotypePattern if possible.

        Args:
            selector: Genotype selector.

        Returns:
            GenotypePattern if selector can be converted, None otherwise.
        """
        if isinstance(selector, str):
            return self.parser.parse(selector)
        elif isinstance(selector, GenotypePattern):
            return selector
        else:
            return None


def resolve_zygote_type(
    spec: str,
    species: Species,
    index_registry: IndexRegistry,
) -> list[int]:
    """Resolve a genotype string to ZType indices, with species-appropriate matching.

    For unordered species, auto-promotes ``|`` to ``::`` so that ``"A|a"``
    matches both ordered and unordered (canonicalized) registrations.  This
    mirrors the canonicalization logic in
    :meth:`genetic_structures.Species.resolve_genotype_selectors`.

    For ordered species (e.g. sex chromosomes), ``|`` is treated strictly —
    ``"a|A"`` and ``"A|a"`` are distinct genotypes and will each only match
    their exact ordering.

    Does NOT perform the reversed-maternal/paternal fallback (that would be
    a bug for ordered species).

    Args:
        spec: Genotype selector string (e.g. ``"A|A"``, ``"Drive|WT"``,
            ``"*"``, ``"A@exposed"``).
        species: Species for genotype-resolution context.
        index_registry: Registry for ZType index resolution.

    Returns:
        List of matching ZType indices (may be empty if nothing matches).
    """
    # Canonicalize | → :: for unordered species only (same pattern as
    # Species._resolve_single_genotype_selector in genetic_structures.py).
    # The \x00 trick preserves any :: the user already wrote.
    if species.unordered:
        spec = spec.replace("::", "\x00").replace("|", "::").replace("\x00", "::")

    pattern = ZygoteTypePattern.parse(spec, species)
    return index_registry.resolve_ztype_indices(pattern)
