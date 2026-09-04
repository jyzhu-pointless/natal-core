"""Type definitions for individuals and gametes.

This module centralizes lightweight type aliases and small helpers used to
represent individuals and gametes in the simulation. Types are deliberately
simple (tuples and ints) so they are easily indexable and efficient to use
with the numeric backends.
"""

from enum import IntEnum
from typing import TypeAlias

__all__ = ["Sex", "Age", "GameteLabel"]

class Sex(IntEnum):
    """Sex enum backed by integers.

    Using :class:`IntEnum` makes values directly usable as array indices and
    compatible across all engine backends.
    """
    FEMALE = 0
    MALE = 1
    # HERMAPHRODITE = 2

    def __repr__(self):
        return f"Sex.{self.name}"

Age: TypeAlias = int  # Age represented as an integer
GameteLabel: TypeAlias = str  # Gamete label represented as a string
