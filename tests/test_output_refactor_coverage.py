"""Coverage contracts for typed output configuration on the build chain."""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.patterns import IndividualSelector
from natal.frontend.spatial.builder import SpatialPopulationBuilder


def _species(name: str) -> nt.Species:
    """Create a two-allele species with three active ZTypes.

    Args:
        name: Species identifier.

    Returns:
        A species with one biallelic locus.
    """
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )


def _discrete_population(
    name: str,
    *,
    history_mode: str = "raw",
) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic discrete population for typed History tests.

    Args:
        name: Base identifier for species and population.
        history_mode: ``"raw"`` (default) or ``"observation"``.

    Returns:
        A built discrete-generation population.
    """
    builder = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(f"{name}_species"),
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 30.0, "WT|Dr": 20.0},
                "male": {"WT|WT": 10.0, "Dr|Dr": 40.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10.0)
        .competition(
            juvenile_growth_mode="beverton_holt",
            low_density_growth_rate=2.0,
            carrying_capacity=100,
        )
    )
    if history_mode == "observation":
        builder.record_history(mode="observation")
    return builder.build()


def test_base_population_requires_installed_history_and_observation() -> None:
    """Uninitialized public output properties fail with explicit errors."""
    population = object.__new__(nt.AgeStructuredPopulation)
    population._history_obj = None  # type: ignore[reportPrivateUsage]  # construct pre-build state
    population._observation = None  # type: ignore[reportPrivateUsage]  # construct pre-build state

    with pytest.raises(RuntimeError, match="History has not been initialized"):
        _ = population.history
    with pytest.raises(RuntimeError, match="Observation has not been initialized"):
        _ = population.observation


def test_spatial_builder_has_no_runtime_binding() -> None:
    """Output policies are frozen because no runtime SpatialPopulationBuilder exists.

    The runtime-binding seam (``_pop_ref`` / ``for_population``) was removed
    with the P5 three-split; spatial output policies can therefore only be
    declared during the build chain.
    """
    builder = SpatialPopulationBuilder(_species("spatial_runtime"), n_demes=1)

    assert not hasattr(builder, "_pop_ref")
    assert not hasattr(SpatialPopulationBuilder, "for_population")


def test_spatial_builder_rejects_invalid_groups_and_mode() -> None:
    """Spatial output configuration validates public boundary values."""
    builder = SpatialPopulationBuilder(_species("spatial_invalid"), n_demes=1)

    with pytest.raises(TypeError, match="mapping"):
        builder.with_observation(groups=[])  # type: ignore[arg-type]  # invalid runtime input
    with pytest.raises(ValueError, match="non-empty"):
        builder.with_observation(groups={})
    with pytest.raises(ValueError, match="mode must be"):
        builder.record_history(mode="invalid")  # type: ignore[arg-type]  # invalid runtime input


@pytest.mark.parametrize("max_rows", [0, -1])
def test_spatial_builder_rejects_invalid_max_rows(max_rows: int) -> None:
    """History capacity must be None or a positive row count.

    Args:
        max_rows: An invalid capacity value supplied via parametrize.
    """
    builder = SpatialPopulationBuilder(_species(f"spatial_rows_{max_rows}"), n_demes=1)
    with pytest.raises(ValueError, match="max_rows"):
        builder.record_history(max_rows=max_rows)


def test_output_capacity_validation_and_spatial_success_path() -> None:
    """Both updaters validate capacity and retain valid frozen settings."""
    species = _species("capacity_validation")
    with pytest.raises(ValueError, match="max_rows"):
        nt.DiscreteGenerationPopulation.setup(species).record_history(max_rows=0)

    builder = SpatialPopulationBuilder(species, n_demes=1)
    groups = {"wild": IndividualSelector(ztype="WT|WT")}
    result = builder.with_observation(groups=groups).record_history(
        mode="observation", max_rows=3
    )
    assert result is builder
    assert builder._observation_groups == groups  # type: ignore[reportPrivateUsage]  # verify frozen builder input
    assert builder._record_history_mode == "observation"  # type: ignore[reportPrivateUsage]  # verify frozen builder input
    assert builder._record_history_max_rows == 3  # type: ignore[reportPrivateUsage]  # verify frozen builder input
