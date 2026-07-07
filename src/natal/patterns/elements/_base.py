"""Base classes for genetic pattern matching elements.

Provides :class:`PatternElement` (abstract base for allele-level pattern
matching) and :class:`PatternParseError` (exception raised on invalid
pattern syntax).
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod as abstract_method
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from natal.genetics import Gene


class PatternParseError(Exception):
    """Error raised during genotype pattern parsing.

    Indicates that a pattern string could not be parsed due to invalid
    syntax, unsupported constructs, or species context mismatches.
    """
    pass


class PatternElement(ABC):
    """Base class for all pattern elements representing allele-level matching."""

    @abstract_method
    def matches(self, gene: Optional[Gene]) -> bool:
        """Check if a single allele matches this pattern element.

        Args:
            gene: The Gene object to match, or None.

        Returns:
            True if the gene matches this pattern element.
        """
        pass

    @abstract_method
    def __repr__(self) -> str:
        """Return a string representation of this pattern element."""
