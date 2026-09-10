"""Type aliases and enumerations shared by genetic structures."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Tuple, TypeVar

if TYPE_CHECKING:
    from ._base import GeneticStructure

T = TypeVar("T")  # Generic type
E = TypeVar("E")  # Generic type for entities (bound at runtime)
S = TypeVar("S", bound='GeneticStructure[Any]')  # Generic type for structures

# Pattern-combination helper aliases used by Species pattern enumeration.
AlleleTuple = Tuple[str, ...]
GenotypeChromosomeCombo = Tuple[AlleleTuple, AlleleTuple]
GenotypeComboMap = Dict[int, GenotypeChromosomeCombo]
HaploidComboMap = Dict[int, AlleleTuple]


class SexChromosomeType(Enum):
    """
    Sex chromosome type enumeration.

    Defines common sex chromosome categories and inheritance constraints used
    by chromosome-level sex-system logic.

    Attributes:
        AUTOSOME (SexChromosomeType): Autosome not involved in sex determination.
        X (SexChromosomeType): X chromosome in the XY system; can come from either parent.
        Y (SexChromosomeType): Y chromosome in the XY system; paternal only.
        Z (SexChromosomeType): Z chromosome in the ZW system; can come from either parent.
        W (SexChromosomeType): W chromosome in the ZW system; maternal only.
    """
    AUTOSOME = "autosome"  # Autosome
    X = "X"                # X chromosome in XY system
    Y = "Y"                # Y chromosome in XY system, paternal only
    Z = "Z"                # Z chromosome in ZW system
    W = "W"                # W chromosome in ZW system, maternal only

    @property
    def is_sex_chromosome(self) -> bool:
        """Whether this is a sex chromosome"""
        return self != SexChromosomeType.AUTOSOME

    @property
    def sex_system(self) -> str | None:
        """Returns the sex determination system this chromosome belongs to"""
        if self in (SexChromosomeType.X, SexChromosomeType.Y):
            return "XY"
        elif self in (SexChromosomeType.Z, SexChromosomeType.W):
            return "ZW"
        return None

    @property
    def maternal_only(self) -> bool:
        """Whether it can only be inherited from mother"""
        return self == SexChromosomeType.W

    @property
    def paternal_only(self) -> bool:
        """Whether it can only be inherited from father"""
        return self == SexChromosomeType.Y
