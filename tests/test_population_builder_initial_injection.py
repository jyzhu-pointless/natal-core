from __future__ import annotations

import unittest
import uuid

import numpy as np

from natal.configurator import (
    PopulationConfigBuilder,
)
from natal.genetics import Species


def _make_species(prefix: str = "BuilderInjectionSpecies") -> Species:
    return Species.from_dict(
        f"{prefix}_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT", "Drive"],
            }
        },
    )


class TestPopulationBuilderInitialInjection(unittest.TestCase):
    def setUp(self) -> None:
        self.species = _make_species()

    def test_survival_parser_supports_legacy_formats(self) -> None:
        seq_none = PopulationConfigBuilder.resolve_age_param([1.0, 0.5, None], 5, [0.0])
        self.assertTrue(np.allclose(seq_none, np.array([1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)))

        from_dict = PopulationConfigBuilder.resolve_age_param({1: 0.7, 3: 0.2}, 4, [0.0])
        self.assertTrue(np.allclose(from_dict, np.array([1.0, 0.7, 1.0, 0.2], dtype=np.float64)))

        from_callable = PopulationConfigBuilder.resolve_age_param(lambda age: 1.0 - 0.1 * age, 3, [0.0])
        self.assertTrue(np.allclose(from_callable, np.array([1.0, 0.9, 0.8], dtype=np.float64)))

        from_scalar = PopulationConfigBuilder.resolve_age_param(0.3, 4, [0.0])
        self.assertTrue(np.allclose(from_scalar, np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float64)))


if __name__ == "__main__":
    unittest.main()
