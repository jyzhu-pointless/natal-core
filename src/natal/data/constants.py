"""Growth mode constants shared across configuration and simulation modules.

These constants must stay in sync with the simulation engine's growth mode
handling in ``algorithms.py``.
"""

__all__ = [
    'NO_COMPETITION', 'FIXED', 'LOGISTIC', 'LINEAR', 'CONCAVE', 'BEVERTON_HOLT',
]

# Growth mode constants (keep in sync with algorithms.py)
NO_COMPETITION = 0
FIXED = 1
LOGISTIC = LINEAR = 2
CONCAVE = BEVERTON_HOLT = 3
