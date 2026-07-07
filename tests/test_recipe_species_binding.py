from __future__ import annotations

import unittest
import uuid

from natal.presets import HomingDrive
from natal.genetics import Species
import natal as nt


def _make_species(prefix: str = "PresetBindingSpecies") -> Species:
    return Species.from_dict(
        f"{prefix}_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT", "Drive", "R2", "R1"],
            }
        },
    )


def _make_preset(species: Species | None = None) -> HomingDrive:
    return HomingDrive(
        name="TestHoming",
        drive_allele="Drive",
        target_allele="WT",
        resistance_allele="R2",
        functional_resistance_allele="R1",
        drive_conversion_rate=0.9,
        species=species,
    )


class TestPresetSpeciesBinding(unittest.TestCase):
    def test_preset_without_species_is_injected_at_apply_time(self) -> None:
        species = _make_species("Injected")
        preset = _make_preset(species=None)

        pop = (nt.DiscreteGenerationPopulation
            .setup(species, stochastic=False)
            .setup(name="PopInjected")
            .initial_state(
                {
                    "female": {"WT|WT": 20},
                    "male": {"Drive|Drive": 20},
                }
            )
            .presets(preset)
            .build())

        self.assertIsNotNone(pop)
        self.assertIs(preset._bound_species, species)

    def test_preset_with_matching_species_is_allowed(self) -> None:
        species = _make_species("Matched")
        preset = _make_preset(species=species)

        pop = (nt.DiscreteGenerationPopulation
            .setup(species, stochastic=False)
            .setup(name="PopMatched")
            .initial_state(
                {
                    "female": {"WT|WT": 20},
                    "male": {"Drive|Drive": 20},
                }
            )
            .presets(preset)
            .build())

        self.assertIsNotNone(pop)
        self.assertIs(preset._bound_species, species)

    def test_preset_with_mismatched_species_raises(self) -> None:
        preset_species = _make_species("PresetSpecies")
        pop_species = _make_species("PopulationSpecies")
        preset = _make_preset(species=preset_species)

        cfg = (nt.DiscreteGenerationPopulation
            .setup(pop_species, stochastic=False)
            .setup(name="PopMismatch")
            .initial_state(
                {
                    "female": {"WT|WT": 20},
                    "male": {"Drive|Drive": 20},
                }
            )
        )

        with self.assertRaises(ValueError) as ctx:
            cfg.presets(preset)

        self.assertIn("already bound to species", str(ctx.exception))
        self.assertIn("cannot be applied to population species", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
