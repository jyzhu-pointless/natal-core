"""Helper functions for various utilities."""

from typing import Union

from natal.type_def import Sex

def resolve_sex_label(sex_label: Union[Sex, int, str]) -> int:
    """Convert sex label to PopulationConfig sex index.

    Convention follows ``Sex`` enum and ``PopulationConfig``:
    - female/f -> 0
    - male/m -> 1
    """
    if isinstance(sex_label, int):
        if sex_label in (0, 1):
            return sex_label
        raise ValueError(f"Invalid sex index '{sex_label}'. Expected 0/1.")
    elif isinstance(sex_label, Sex):
        return sex_label.value
    elif isinstance(sex_label, str):
        normalized = sex_label.lower()
        if normalized in ('female', 'f'):
            return 0
        if normalized in ('male', 'm'):
            return 1
        raise ValueError(f"Invalid sex label '{sex_label}'. Expected female/male/f/m.")
    raise TypeError(f"Invalid sex label type '{type(sex_label).__name__}'. Expected Sex, int, or str.")

def validate_name(name: str) -> bool:
    """Validate if a name is valid.

    A valid name consists of only letters, numbers, and underscores.

    Args:
        name (str): The name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    import re

    pattern = r'^[A-Za-z0-9_]+$'
    return bool(re.match(pattern, name))
