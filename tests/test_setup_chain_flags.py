"""Build-chain flag coverage retired from the public low-level entry.

The ``extreme_speed_mode`` parameter moved to the ``setup()`` chain entry
(finalized decision #1 in ``ARCHITECTURE_SIMPLIFICATION_PLAN.md``); these
tests pin the chain spelling and its validation.
"""

import pytest

import natal as nt


def _species() -> nt.Species:
    return nt.Species.from_dict(
        name="setup_flag_species", structure={"auto": {"A": ["WT"]}}
    )


def test_setup_chain_accepts_extreme_speed_mode() -> None:
    """The chain entry stores the flag into the draft verbatim."""
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            _species(), stochastic=False, extreme_speed_mode=3
        )
        .initial_state(
            individual_count={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}}
        )
        .reproduction(eggs_per_female=4)
        .build()
    )
    assert pop.config.extreme_speed_mode == 3
    pop.run(1)
    assert pop.tick == 1


def test_setup_chain_extreme_speed_mode_age_structured() -> None:
    """The age-structured entry forwards the same parameter."""
    builder = nt.AgeStructuredPopulation.setup(
        _species(), stochastic=False, extreme_speed_mode=1
    )
    assert builder.config.extreme_speed_mode == 1


def test_setup_chain_rejects_unknown_extreme_speed_mode() -> None:
    """Out-of-range modes fail at the chain boundary, not at build time."""
    with pytest.raises(ValueError, match="extreme_speed_mode"):
        nt.DiscreteGenerationPopulation.setup(
            _species(), stochastic=False, extreme_speed_mode=7
        )
