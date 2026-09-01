"""Population models: base, age-structured, and discrete-generation."""

from .age_structured import AgeStructuredPopulation
from .base import BasePopulation
from .discrete_generation import DiscreteGenerationPopulation

__all__ = [
    "BasePopulation",
    "AgeStructuredPopulation",
    "DiscreteGenerationPopulation",
]
