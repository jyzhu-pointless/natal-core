"""Type definitions for individuals and gametes.

This module centralizes lightweight type aliases and small helpers used to
represent individuals and gametes in the simulation. Types are deliberately
simple (tuples and ints) so they are easily indexable and efficient to use
with numeric backends and Numba-accelerated code.
"""

from enum import IntEnum
from typing import Tuple, TypeAlias

__all__ = ["Sex", "Age", "GameteLabel"]

class Sex(IntEnum):
    """Sex enum backed by integers.

    Using :class:`IntEnum` makes values directly usable as array indices and
    compatible with Numba-friendly code.
    """
    FEMALE = 0
    MALE = 1
    # HERMAPHRODITE = 2

    def __repr__(self):
        return f"Sex.{self.name}"

Age: TypeAlias = int  # Age represented as an integer
GenotypeIndex: TypeAlias = int  # Diploid genotype represented by an integer index
IndividualType: TypeAlias = Tuple[Sex, Age, GenotypeIndex]  # (sex, age, genotype_index)

HaploidGenotypeIndex: TypeAlias = int  # Haploid genotype represented by an integer index
GameteLabel: TypeAlias = str  # Gamete label represented as a string
GlabIndex: TypeAlias = int  # Gamete-label index represented as an integer
GameteType: TypeAlias = Tuple[Sex, HaploidGenotypeIndex, GlabIndex]  # (sex, haplotype_index, glab_index)

def make_individual_type(sex: Sex, age: Age, genotype_index: GenotypeIndex) -> IndividualType:
    """Create and normalize an :data:`IndividualType` tuple.

    This helper accepts flexible ``sex`` inputs (either a ``Sex`` enum or an
    integer-like value) and ensures all tuple elements are concrete Python
    types (``Sex`` and ``int``).

    Args:
        sex: Sex enum or integer-like value.
        age: Age as an integer.
        genotype_index: Diploid genotype index.

    Returns:
        IndividualType: Normalized tuple ``(Sex, int(age), int(genotype_index))``.
    """
    # Normalize sex input to a `Sex` instance.
    if isinstance(sex, Sex):
        sex_val = sex
    else:
        # Accept ints or objects that can be cast to int (e.g., numpy integers)
        try:
            sex_val = Sex(int(sex))
        except Exception as e:
            raise TypeError(f"invalid sex value: {sex!r}") from e
    return (sex_val, int(age), int(genotype_index))

def make_gamete_type(sex: Sex, haplo_idx: HaploidGenotypeIndex, glab_idx: GlabIndex) -> GameteType:
    """Create and normalize a :data:`GameteType` tuple.

    Ensures ``sex`` is a :class:`Sex` enum and that numeric fields are plain
    Python ``int`` values.

    Args:
        sex: Sex enum or integer-like value.
        haplo_idx: Haploid genotype index.
        glab_idx: Gamete-label index.

    Returns:
        GameteType: Normalized tuple ``(Sex, int(haplo_idx), int(glab_idx))``.
    """
    # Normalize sex input to a `Sex` instance.
    if isinstance(sex, Sex):
        sex_val = sex
    else:
        # Accept ints or objects that can be cast to int (e.g., numpy integers)
        try:
            sex_val = Sex(int(sex))
        except Exception as e:
            raise TypeError(f"invalid sex value: {sex!r}") from e
    return (sex_val, int(haplo_idx), int(glab_idx))


def get_sex(ind: IndividualType) -> Sex:
    """Return the ``Sex`` of an :data:`IndividualType`.

    Args:
        ind: An :data:`IndividualType` tuple.

    Returns:
        Sex: The sex value stored in the tuple.
    """
    return ind[0]


def get_age(ind: IndividualType) -> Age:
    """Return the age component from an :data:`IndividualType`.

    Args:
        ind: An :data:`IndividualType` tuple.

    Returns:
        Age: The age value.
    """
    return ind[1]


def get_genotype_index(ind: IndividualType) -> GenotypeIndex:
    """Return the genotype index from an :data:`IndividualType`.

    Args:
        ind: An :data:`IndividualType` tuple.

    Returns:
        GenotypeIndex: The diploid genotype index.
    """
    return ind[2]
