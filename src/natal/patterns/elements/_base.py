"""
Base classes for genetic pattern matching elements.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod as abstract_method
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from natal.genetics import Gene


class PatternParseError(Exception):
    """Error raised during genotype pattern parsing."""
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
        pass
