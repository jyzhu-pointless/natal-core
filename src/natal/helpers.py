"""Helper functions for various utilities."""

from natal.type_def import Sex


def resolve_sex_label(sex_label: object) -> int:
    """Convert sex label to PopulationConfig sex index.

    Convention follows ``Sex`` enum and ``PopulationConfig``:
    - female/f -> 0
    - male/m -> 1
    """
    assert isinstance(sex_label, (Sex, int, str)), (
        f"Invalid sex label type '{type(sex_label).__name__}'. Expected Sex, int, or str."
    )

    sex_value: object = sex_label
    if isinstance(sex_value, int):
        if sex_value in (0, 1):
            return sex_value
        raise ValueError(f"Invalid sex index '{sex_value}'. Expected 0/1.")
    if isinstance(sex_value, Sex):
        return sex_value.value
    normalized = sex_value.lower()
    if normalized in ('female', 'f'):
        return 0
    if normalized in ('male', 'm'):
        return 1
    raise ValueError(f"Invalid sex label '{sex_value}'. Expected female/male/f/m.")

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
