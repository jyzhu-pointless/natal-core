"""Numerical contracts for conversion refresh and runtime reconfiguration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.modifiers.module import GameteModifier, ZygoteModifier
from natal.numba.utils import numba_disabled
from natal.population.base import BasePopulation

PopulationKind: TypeAlias = Literal["age", "discrete"]
ConversionKind: TypeAlias = Literal["gamete", "zygote"]
Rate: TypeAlias = float | tuple[float, float]
Population: TypeAlias = nt.AgeStructuredPopulation | nt.DiscreteGenerationPopulation
GroupLabel: TypeAlias = Literal["A", "B"]
BuildFailureScenario: TypeAlias = Literal["same-call", "append"]


class _DeferredFailurePreset(nt.GeneticPreset):
    """Preset whose valid recipe can fail only when its closure is invoked."""

    def __init__(
        self,
        name: str,
        *,
        fail_during_rebuild: bool = False,
        failure_stage: Literal["gamete", "zygote", "fitness"] = "gamete",
        fail_capacity: float | None = None,
    ) -> None:
        """Initialize the delayed-failure switch.

        Args:
            name: Unique preset name.
            fail_during_rebuild: Whether the returned modifier should fail.
            failure_stage: Rebuild stage that raises when failure is enabled.
            fail_capacity: Optional capacity restricting the failure to one
                config-sharing group.
        """
        super().__init__(name=name)
        self.fail_during_rebuild = fail_during_rebuild
        self.failure_stage = failure_stage
        self.fail_capacity = fail_capacity

    def _should_fail(self, population: BasePopulation[Any]) -> bool:  # Any: inherited preset contract accepts either population-state model.
        """Return whether this population should fail at the selected stage."""
        if not self.fail_during_rebuild:
            return False
        if self.fail_capacity is None:
            return True
        return float(population.config.carrying_capacity[()]) == self.fail_capacity

    def gamete_modifier(
        self,
        population: BasePopulation[Any],  # Any: inherited preset contract accepts either population-state model.
    ) -> GameteModifier:
        """Return a modifier that defers validation until map rebuilding.

        Args:
            population: Population receiving the modifier.

        Returns:
            A valid modifier callable that may raise when invoked.
        """
        should_fail = self._should_fail(population)

        def deferred_modifier(
            *args: object,
            **kwargs: object,
        ) -> dict[tuple[int, int], dict[int, float]]:
            """Raise only when the wrapper invokes this modifier."""
            del args, kwargs
            if self.failure_stage == "gamete" and should_fail:
                raise ValueError("deferred modifier failure")
            return {}

        return deferred_modifier

    def zygote_modifier(
        self,
        population: BasePopulation[Any],  # Any: inherited preset contract accepts either population-state model.
    ) -> ZygoteModifier | None:
        """Return a zygote modifier only when that stage is under test.

        Args:
            population: Population receiving the preset.
        """
        if self.failure_stage != "zygote":
            return None
        should_fail = self._should_fail(population)

        def deferred_modifier(
            *args: object,
            **kwargs: object,
        ) -> dict[tuple[int, int], dict[int, float]]:
            """Raise only when the wrapper invokes this modifier."""
            del args, kwargs
            if should_fail:
                raise ValueError("deferred modifier failure")
            return {}

        return deferred_modifier

    def fitness_patch(self) -> nt.PresetFitnessPatch | None:
        """Raise when the fitness-application stage is under test."""
        if self.failure_stage == "fitness" and self.fail_during_rebuild:
            raise ValueError("deferred fitness failure")
        return None


class _ConfigSensitivePreset(nt.GeneticPreset):
    """Preset whose gamete and zygote rates depend on config identity group."""

    def __init__(self, name: str, *, fail_during_rebuild: bool = False) -> None:
        """Initialize the group-sensitive preset.

        Args:
            name: Unique preset name.
            fail_during_rebuild: Whether returned modifiers should fail.
        """
        super().__init__(name=name)
        self.fail_during_rebuild = fail_during_rebuild

    def _rate(self, population: BasePopulation[Any]) -> float:  # Any: inherited preset contract accepts either state model.
        """Select the exact conversion rate from the group's capacity."""
        return 0.8 if float(population.config.carrying_capacity[()]) == 700.0 else 0.2

    def gamete_modifier(
        self,
        population: BasePopulation[Any],  # Any: inherited preset contract accepts either population-state model.
    ) -> GameteModifier:
        """Return a group-specific Drive|WT gamete distribution.

        Args:
            population: Population whose config determines the rate.

        Returns:
            Bulk gamete modifier with exact probabilities.
        """
        rate = self._rate(population)
        registry = population.index_registry
        species = population.species
        parent = registry.ztype_index(
            species.get_genotype_from_str("Drive|WT"),
            "default",
        )
        drive = registry.gtype_index(
            species.get_haploid_genotype_from_str("Drive"),
            "default",
        )
        wt = registry.gtype_index(
            species.get_haploid_genotype_from_str("WT"),
            "default",
        )
        disrupted = registry.gtype_index(
            species.get_haploid_genotype_from_str("Disrupted"),
            "default",
        )

        def modifier(
            *args: object,
            **kwargs: object,
        ) -> dict[tuple[int, int], dict[int, float]]:
            """Return the captured group-specific gamete probabilities."""
            del args, kwargs
            if self.fail_during_rebuild:
                raise ValueError("group modifier failure")
            distribution = {
                drive: 0.5,
                wt: 0.5 * (1.0 - rate),
                disrupted: 0.5 * rate,
            }
            return {(0, parent): distribution, (1, parent): distribution}

        return modifier

    def zygote_modifier(
        self,
        population: BasePopulation[Any],  # Any: inherited preset contract accepts either population-state model.
    ) -> ZygoteModifier:
        """Return a group-specific Drive×WT zygote distribution.

        Args:
            population: Population whose config determines the rate.

        Returns:
            Bulk zygote modifier with exact probabilities.
        """
        rate = self._rate(population)
        registry = population.index_registry
        species = population.species
        drive = registry.gtype_index(
            species.get_haploid_genotype_from_str("Drive"),
            "default",
        )
        wt = registry.gtype_index(
            species.get_haploid_genotype_from_str("WT"),
            "default",
        )
        unchanged = registry.ztype_index(
            species.get_genotype_from_str("Drive|WT"),
            "default",
        )
        disrupted = registry.ztype_index(
            species.get_genotype_from_str("Drive|Disrupted"),
            "default",
        )
        c1, c2 = sorted((drive, wt))
        pair = (c1, c2)

        def modifier(
            *args: object,
            **kwargs: object,
        ) -> dict[tuple[int, int], dict[int, float]]:
            """Return the captured group-specific zygote probabilities."""
            del args, kwargs
            if self.fail_during_rebuild:
                raise ValueError("group modifier failure")
            return {pair: {unchanged: 1.0 - rate, disrupted: rate}}

        return modifier


@pytest.fixture(autouse=True)
def _use_python_path() -> Iterator[None]:
    """Keep this configuration-contract matrix independent of JIT codegen."""
    with numba_disabled():
        yield


def _make_species(name: str, *, extra_glab: bool = False) -> nt.Species:
    """Create the three-allele species used by conversion contracts."""
    glabs = ["default", "unused"] if extra_glab else ["default"]
    return nt.Species.from_dict(
        name=name,
        structure={"Chr1": {"L1": ["WT", "Drive", "Disrupted"]}},
        gamete_labels=glabs,
    )


def _make_toxin_preset(
    conversion: ConversionKind,
    rate: Rate,
    *,
    name: str,
) -> nt.ToxinAntidoteDrive:
    """Create a preset with exactly one active conversion stage."""
    conversion_rate: Rate = rate if conversion == "gamete" else 0.0
    embryo_rate: Rate = rate if conversion == "zygote" else 0.0
    return nt.ToxinAntidoteDrive(
        name=name,
        drive_allele="Drive",
        target_allele="WT",
        disrupted_allele="Disrupted",
        conversion_rate=conversion_rate,
        embryo_disruption_rate=embryo_rate,
    )


def _build_population(
    species: nt.Species,
    preset: nt.GeneticPreset,
    *,
    kind: PopulationKind,
    compress: bool,
    name: str,
    carrying_capacity: float = 200.0,
) -> Population:
    """Build a deterministic population with a build-time preset."""
    if kind == "age":
        return (
            nt.AgeStructuredPopulation.setup(
                species=species,
                name=name,
                stochastic=False,
                compress=compress,
            )
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({
                "female": {"Drive|WT": {1: 50}},
                "male": {"Drive|WT": {1: 50}},
            })
            .reproduction(eggs_per_female=2.0)
            .competition(
                carrying_capacity=carrying_capacity,
                juvenile_growth_mode=nt.NO_COMPETITION,
            )
            .presets(preset)
            .build()
        )
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
            compress=compress,
        )
        .initial_state({
            "female": {"Drive|WT": 50},
            "male": {"Drive|WT": 50},
        })
        .reproduction(eggs_per_female=2.0)
        .competition(
            carrying_capacity=carrying_capacity,
            juvenile_growth_mode=nt.NO_COMPETITION,
        )
        .presets(preset)
        .build()
    )


def _build_spatial_population(
    species: nt.Species,
    preset: nt.GeneticPreset,
    *,
    kind: PopulationKind,
    compress: bool,
    name: str,
    n_demes: int = 2,
) -> nt.SpatialPopulation:
    """Build two homogeneous demes with a shared build-time preset."""
    pop_type = "age_structured" if kind == "age" else "discrete_generation"
    builder = nt.SpatialPopulation.builder(
        species=species,
        n_demes=n_demes,
        pop_type=pop_type,
    ).setup(name=name, stochastic=False, compress=compress)
    if kind == "age":
        builder = builder.age_structure(n_ages=2, new_adult_age=1).initial_state({
            "female": {"Drive|WT": {1: 50}},
            "male": {"Drive|WT": {1: 50}},
        })
    else:
        builder = builder.initial_state({
            "female": {"Drive|WT": 50},
            "male": {"Drive|WT": 50},
        })
    return (
        builder
        .reproduction(eggs_per_female=2.0)
        .competition(
            carrying_capacity=200.0,
            juvenile_growth_mode=nt.NO_COMPETITION,
        )
        .presets(preset)
        .build()
    )


def _build_configurator(
    species: nt.Species,
    *,
    kind: PopulationKind,
    name: str,
) -> nt.Configurator:
    """Create an unbuilt configurator for build-time transaction tests."""
    if kind == "age":
        return (
            nt.AgeStructuredPopulation.setup(
                species=species,
                name=name,
                stochastic=False,
            )
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({
                "female": {"Drive|WT": {1: 50}},
                "male": {"Drive|WT": {1: 50}},
            })
            .reproduction(eggs_per_female=2.0)
            .competition(
                carrying_capacity=200.0,
                juvenile_growth_mode=nt.NO_COMPETITION,
            )
        )
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name=name,
            stochastic=False,
        )
        .initial_state({
            "female": {"Drive|WT": 50},
            "male": {"Drive|WT": 50},
        })
        .reproduction(eggs_per_female=2.0)
        .competition(
            carrying_capacity=200.0,
            juvenile_growth_mode=nt.NO_COMPETITION,
        )
    )


def _make_fitness_drive(name: str) -> nt.HomingDrive:
    """Create a preset that changes modifiers and every fitness tensor."""
    return nt.HomingDrive(
        name=name,
        drive_allele="Drive",
        target_allele="WT",
        resistance_allele="Disrupted",
        drive_conversion_rate=0.4,
        viability_scaling=0.8,
        fecundity_scaling=0.7,
        sexual_selection_scaling=0.6,
        zygote_viability_scaling=0.5,
    )


def _copy_config_arrays(
    config: nt.PopulationConfig | nt.DiscretePopulationConfig,
) -> dict[str, NDArray[np.generic]]:
    """Copy every ndarray field so rollback checks cover the whole config."""
    copies: dict[str, NDArray[np.generic]] = {}
    for field in config._fields:
        value = getattr(config, field)
        if isinstance(value, np.ndarray):
            copies[field] = value.copy()
    return copies


def _assert_config_arrays_equal(
    config: nt.PopulationConfig | nt.DiscretePopulationConfig,
    expected: dict[str, NDArray[np.generic]],
) -> None:
    """Assert exact values for every snapshotted config array."""
    assert set(expected) == {
        field for field in config._fields
        if isinstance(getattr(config, field), np.ndarray)
    }
    for field, expected_value in expected.items():
        np.testing.assert_array_equal(getattr(config, field), expected_value)


def _assert_fitness_drive_values(pop: Population) -> None:
    """Assert exact one-copy effects from ``_make_fitness_drive``."""
    drive_wt = pop.index_registry.ztype_index(
        pop.species.get_genotype_from_str("Drive|WT"),
        "default",
    )
    wt_wt = pop.index_registry.ztype_index(
        pop.species.get_genotype_from_str("WT|WT"),
        "default",
    )
    viability_age = int(pop.config.new_adult_age) - 1
    for sex in (0, 1):
        assert pop.config.viability_fitness[sex, viability_age, drive_wt] == pytest.approx(0.8)
        assert pop.config.fecundity_fitness[sex, drive_wt] == pytest.approx(0.7)
        assert pop.config.zygote_viability_fitness[sex, drive_wt] == pytest.approx(0.5)
        assert pop.config.viability_fitness[sex, viability_age, wt_wt] == pytest.approx(1.0)
        assert pop.config.fecundity_fitness[sex, wt_wt] == pytest.approx(1.0)
        assert pop.config.zygote_viability_fitness[sex, wt_wt] == pytest.approx(1.0)
    assert pop.config.sexual_selection_fitness[wt_wt, drive_wt] == pytest.approx(0.6)
    assert pop.config.sexual_selection_fitness[wt_wt, wt_wt] == pytest.approx(1.0)


def _sex_rates(rate: Rate) -> tuple[float, float]:
    """Normalize a scalar test rate to female and male rates."""
    return (rate, rate) if isinstance(rate, float) else rate


def _assert_conversion_probabilities(
    pop: Population,
    *,
    conversion: ConversionKind,
    rate: Rate,
) -> None:
    """Assert exact conversion probabilities and row normalization."""
    species = pop.species
    registry = pop.index_registry
    drive_haplo = species.get_haploid_genotype_from_str("Drive")
    wt_haplo = species.get_haploid_genotype_from_str("WT")
    disrupted_haplo = species.get_haploid_genotype_from_str("Disrupted")
    drive_wt = species.get_genotype_from_str("Drive|WT")
    drive_disrupted = species.get_genotype_from_str("Drive|Disrupted")

    if conversion == "gamete":
        parent = registry.ztype_index(drive_wt, "default")
        drive = registry.gtype_index(drive_haplo, "default")
        wt = registry.gtype_index(wt_haplo, "default")
        disrupted = registry.gtype_index(disrupted_haplo, "default")
        for sex, sex_rate in enumerate(_sex_rates(rate)):
            row = pop.config.zygotes_to_gametes_map[sex, parent]
            assert row[drive] == pytest.approx(0.5)
            assert row[wt] == pytest.approx(0.5 * (1.0 - sex_rate))
            assert row[disrupted] == pytest.approx(0.5 * sex_rate)
            assert row.sum() == pytest.approx(1.0)
        return

    drive = registry.gtype_index(drive_haplo, "default")
    wt = registry.gtype_index(wt_haplo, "default")
    unchanged = registry.ztype_index(drive_wt, "default")
    disrupted = registry.ztype_index(drive_disrupted, "default")
    c1, c2 = sorted((drive, wt))
    row = pop.config.gametes_to_zygotes_map[c1, c2]
    female_rate, male_rate = _sex_rates(rate)
    expected_unchanged = (1.0 - female_rate) * (1.0 - male_rate)
    assert row[unchanged] == pytest.approx(expected_unchanged)
    assert row[disrupted] == pytest.approx(1.0 - expected_unchanged)
    assert row.sum() == pytest.approx(1.0)


def _assert_maps_equal(actual: Population, expected: Population) -> None:
    """Assert exact equality of all derived inheritance maps."""
    assert actual.config.zygotes_to_gametes_map.flags.c_contiguous
    assert actual.config.gametes_to_zygotes_map.flags.c_contiguous
    assert actual.config.offspring_tensor.flags.c_contiguous
    np.testing.assert_array_equal(
        actual.config.zygotes_to_gametes_map,
        expected.config.zygotes_to_gametes_map,
    )
    np.testing.assert_array_equal(
        actual.config.gametes_to_zygotes_map,
        expected.config.gametes_to_zygotes_map,
    )
    np.testing.assert_array_equal(
        actual.config.offspring_tensor,
        expected.config.offspring_tensor,
    )
    np.testing.assert_allclose(actual.config.offspring_tensor.sum(axis=-1), 1.0)


def _arrange_noncontiguous_config_groups(
    pop: nt.SpatialPopulation,
    layout: tuple[GroupLabel, ...],
) -> tuple[nt.PopulationConfig | nt.DiscretePopulationConfig, nt.PopulationConfig | nt.DiscretePopulationConfig]:
    """Arrange shared config identities according to an A/B layout.

    Args:
        pop: Homogeneous spatial population to rearrange.
        layout: Identity labels, with both A and B present.

    Returns:
        The original A config and detached B config.
    """
    assert len(pop.demes) == len(layout)
    config_a = pop.deme(0).config
    pop.update(deme=1).competition(carrying_capacity=700.0)
    config_b = pop.deme(1).config
    assert config_b is not config_a
    for i, label in enumerate(layout):
        pop.deme(i).set_config(config_a if label == "A" else config_b)
        expected_capacity = 200.0 if label == "A" else 700.0
        assert pop.deme(i).config.carrying_capacity[()] == expected_capacity
    return config_a, config_b


def _assert_group_sensitive_probabilities(
    pop: Population,
    rate: float,
) -> None:
    """Assert exact gamete and zygote distributions for one config group."""
    species = pop.species
    registry = pop.index_registry
    parent = registry.ztype_index(
        species.get_genotype_from_str("Drive|WT"),
        "default",
    )
    drive = registry.gtype_index(
        species.get_haploid_genotype_from_str("Drive"),
        "default",
    )
    wt = registry.gtype_index(
        species.get_haploid_genotype_from_str("WT"),
        "default",
    )
    disrupted = registry.gtype_index(
        species.get_haploid_genotype_from_str("Disrupted"),
        "default",
    )
    drive_disrupted = registry.ztype_index(
        species.get_genotype_from_str("Drive|Disrupted"),
        "default",
    )
    for sex in (0, 1):
        gamete_row = pop.config.zygotes_to_gametes_map[sex, parent]
        assert gamete_row[drive] == pytest.approx(0.5)
        assert gamete_row[wt] == pytest.approx(0.5 * (1.0 - rate))
        assert gamete_row[disrupted] == pytest.approx(0.5 * rate)
        assert gamete_row.sum() == pytest.approx(1.0)
    c1, c2 = sorted((drive, wt))
    zygote_row = pop.config.gametes_to_zygotes_map[c1, c2]
    assert zygote_row[parent] == pytest.approx(1.0 - rate)
    assert zygote_row[drive_disrupted] == pytest.approx(rate)
    assert zygote_row.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pop.config.offspring_tensor.sum(axis=-1), 1.0)


@pytest.mark.parametrize("kind", ["age", "discrete"])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("conversion", ["gamete", "zygote"])
@pytest.mark.parametrize("new_rate", [0.4, (0.2, 0.6)], ids=["scalar", "tuple"])
def test_nonspatial_reconfigure_refresh_matches_fresh_build(
    kind: PopulationKind,
    compress: bool,
    conversion: ConversionKind,
    new_rate: Rate,
) -> None:
    """Reconfigure and repeated refresh equal a fresh build on every axis."""
    axis = f"{kind}_{compress}_{conversion}_{type(new_rate).__name__}"
    species = _make_species(f"conversion_nonspatial_{axis}")
    old_preset = _make_toxin_preset(conversion, 0.9, name=f"old_{axis}")
    candidate = _build_population(
        species,
        old_preset,
        kind=kind,
        compress=compress,
        name=f"candidate_{axis}",
    )
    change = (
        {"conversion_rate": new_rate}
        if conversion == "gamete"
        else {"embryo_disruption_rate": new_rate}
    )

    candidate.update().reconfigure_preset(old_preset, **change)
    candidate.refresh_modifiers()
    candidate.refresh_modifiers()

    fresh_preset = _make_toxin_preset(conversion, new_rate, name=f"fresh_{axis}")
    fresh = _build_population(
        species,
        fresh_preset,
        kind=kind,
        compress=compress,
        name=f"fresh_{axis}",
    )
    _assert_conversion_probabilities(candidate, conversion=conversion, rate=new_rate)
    _assert_maps_equal(candidate, fresh)

    candidate.run(2)
    fresh.run(2)
    assert candidate.tick == fresh.tick == 2
    np.testing.assert_array_equal(
        candidate.state.individual_count,
        fresh.state.individual_count,
    )
    assert np.all(candidate.state.individual_count >= 0.0)


@pytest.mark.parametrize("kind", ["age", "discrete"])
@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("conversion", ["gamete", "zygote"])
@pytest.mark.parametrize("new_rate", [0.4, (0.2, 0.6)], ids=["scalar", "tuple"])
def test_spatial_all_deme_reconfigure_refresh_matches_fresh_build(
    kind: PopulationKind,
    compress: bool,
    conversion: ConversionKind,
    new_rate: Rate,
) -> None:
    """Spatial all-deme reconfigure remains exact through refresh and run."""
    axis = f"{kind}_{compress}_{conversion}_{type(new_rate).__name__}"
    species = _make_species(f"conversion_spatial_{axis}")
    old_preset = _make_toxin_preset(conversion, 0.9, name=f"old_spatial_{axis}")
    candidate = _build_spatial_population(
        species,
        old_preset,
        kind=kind,
        compress=compress,
        name=f"candidate_spatial_{axis}",
    )
    change = (
        {"conversion_rate": new_rate}
        if conversion == "gamete"
        else {"embryo_disruption_rate": new_rate}
    )
    candidate.update().reconfigure_preset(old_preset, **change)

    fresh_preset = _make_toxin_preset(
        conversion,
        new_rate,
        name=f"fresh_spatial_{axis}",
    )
    fresh = _build_spatial_population(
        species,
        fresh_preset,
        kind=kind,
        compress=compress,
        name=f"fresh_spatial_{axis}",
    )

    for candidate_deme, fresh_deme in zip(candidate.demes, fresh.demes):
        candidate_deme.refresh_modifiers()
        candidate_deme.refresh_modifiers()
        _assert_conversion_probabilities(
            candidate_deme,
            conversion=conversion,
            rate=new_rate,
        )
        _assert_maps_equal(candidate_deme, fresh_deme)

    candidate.run(2)
    fresh.run(2)
    assert candidate.tick == fresh.tick == 2
    for candidate_deme, fresh_deme in zip(candidate.demes, fresh.demes):
        np.testing.assert_array_equal(
            candidate_deme.state.individual_count,
            fresh_deme.state.individual_count,
        )
        assert np.all(candidate_deme.state.individual_count >= 0.0)


@pytest.mark.parametrize("kind", ["age", "discrete"])
def test_compressed_homing_with_sparse_gtypes_refreshes_exactly(
    kind: PopulationKind,
) -> None:
    """Compressed HomingDrive supports a pruned haplotype-glab product."""
    species = _make_species(f"compressed_sparse_homing_{kind}", extra_glab=True)
    drive = nt.HomingDrive(
        name=f"sparse_homing_{kind}",
        drive_allele="Drive",
        target_allele="WT",
        drive_conversion_rate=0.8,
    )
    pop = _build_population(
        species,
        drive,
        kind=kind,
        compress=True,
        name=f"compressed_sparse_homing_pop_{kind}",
    )
    registry = pop.index_registry
    assert registry.n_gtypes < len(registry.index_to_haplo) * int(pop.config.n_glabs)
    expected_initial = pop.config.zygotes_to_gametes_map.copy()

    pop.refresh_modifiers()
    np.testing.assert_array_equal(
        pop.config.zygotes_to_gametes_map,
        expected_initial,
    )
    pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)
    pop.refresh_modifiers()

    fresh_drive = nt.HomingDrive(
        name=f"sparse_homing_fresh_{kind}",
        drive_allele="Drive",
        target_allele="WT",
        drive_conversion_rate=0.3,
    )
    fresh = _build_population(
        species,
        fresh_drive,
        kind=kind,
        compress=True,
        name=f"compressed_sparse_homing_fresh_pop_{kind}",
    )
    _assert_maps_equal(pop, fresh)
    parent = registry.ztype_index(
        species.get_genotype_from_str("Drive|WT"),
        "default",
    )
    drive_gtype = registry.gtype_index(
        species.get_haploid_genotype_from_str("Drive"),
        "default",
    )
    wt_gtype = registry.gtype_index(
        species.get_haploid_genotype_from_str("WT"),
        "default",
    )
    for sex in (0, 1):
        row = pop.config.zygotes_to_gametes_map[sex, parent]
        assert row[drive_gtype] == pytest.approx(0.65)
        assert row[wt_gtype] == pytest.approx(0.35)
        assert row.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pop.config.zygotes_to_gametes_map.sum(axis=-1), 1.0)


@pytest.mark.parametrize("kind", ["age", "discrete"])
def test_compressed_zygote_conversion_with_sparse_gtypes_refreshes_exactly(
    kind: PopulationKind,
) -> None:
    """Compressed zygote conversion preserves active GType axes on refresh."""
    species = _make_species(f"compressed_sparse_zygote_{kind}", extra_glab=True)
    preset = _make_toxin_preset(
        "zygote",
        0.8,
        name=f"sparse_zygote_{kind}",
    )
    pop = _build_population(
        species,
        preset,
        kind=kind,
        compress=True,
        name=f"compressed_sparse_zygote_pop_{kind}",
    )
    registry = pop.index_registry
    assert registry.n_gtypes < len(registry.index_to_haplo) * int(pop.config.n_glabs)
    expected_initial = pop.config.gametes_to_zygotes_map.copy()

    pop.refresh_modifiers()
    np.testing.assert_array_equal(
        pop.config.gametes_to_zygotes_map,
        expected_initial,
    )
    pop.update().reconfigure_preset(preset, embryo_disruption_rate=0.3)
    pop.refresh_modifiers()

    fresh_preset = _make_toxin_preset(
        "zygote",
        0.3,
        name=f"sparse_zygote_fresh_{kind}",
    )
    fresh = _build_population(
        species,
        fresh_preset,
        kind=kind,
        compress=True,
        name=f"compressed_sparse_zygote_fresh_pop_{kind}",
    )
    _assert_maps_equal(pop, fresh)
    _assert_conversion_probabilities(pop, conversion="zygote", rate=0.3)
    np.testing.assert_allclose(pop.config.gametes_to_zygotes_map.sum(axis=-1), 1.0)


@pytest.mark.parametrize("kind", ["age", "discrete"])
def test_build_time_duplicate_preset_is_identity_deduplicated(
    kind: PopulationKind,
) -> None:
    """One build-time preset instance passed twice is applied exactly once."""
    species = _make_species(f"duplicate_build_preset_{kind}")
    drive = nt.HomingDrive(
        name=f"duplicate_drive_{kind}",
        drive_allele="Drive",
        target_allele="WT",
        drive_conversion_rate=0.8,
    )
    if kind == "age":
        pop = (
            nt.AgeStructuredPopulation.setup(species=species, stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({
                "female": {"Drive|WT": {1: 50}},
                "male": {"Drive|WT": {1: 50}},
            })
            .presets(drive, drive)
            .build()
        )
    else:
        pop = (
            nt.DiscreteGenerationPopulation.setup(species=species, stochastic=False)
            .initial_state({
                "female": {"Drive|WT": 50},
                "male": {"Drive|WT": 50},
            })
            .presets(drive, drive)
            .build()
        )
    parent = pop.index_registry.ztype_index(
        species.get_genotype_from_str("Drive|WT"),
        "default",
    )
    drive_gtype = pop.index_registry.gtype_index(
        species.get_haploid_genotype_from_str("Drive"),
        "default",
    )
    wt_gtype = pop.index_registry.gtype_index(
        species.get_haploid_genotype_from_str("WT"),
        "default",
    )

    assert len(pop.gamete_modifiers) == 1
    for sex in (0, 1):
        row = pop.config.zygotes_to_gametes_map[sex, parent]
        assert row[drive_gtype] == pytest.approx(0.9)
        assert row[wt_gtype] == pytest.approx(0.1)
        assert row.sum() == pytest.approx(1.0)


def test_build_time_preset_failure_can_retry_same_object_exactly() -> None:
    """Failure then retry with the same preset must equal a fresh build."""
    species = _make_species("build_retry_same_preset")
    preset = _ConfigSensitivePreset(
        "build_retry_same_preset_recipe",
        fail_during_rebuild=True,
    )
    configurator = (
        nt.AgeStructuredPopulation.setup(species=species, stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state({
            "female": {"Drive|WT": {1: 50}},
            "male": {"Drive|WT": {1: 50}},
        })
        .competition(
            carrying_capacity=200.0,
            juvenile_growth_mode=nt.NO_COMPETITION,
        )
    )
    original_config = configurator.config

    with pytest.raises(ValueError, match="group modifier failure"):
        configurator.presets(preset)

    assert configurator.config is original_config
    assert preset._bound_species is None  # pyright: ignore[reportPrivateUsage]  # failed registration must restore caller-owned binding.

    preset.fail_during_rebuild = False
    retried = configurator.presets(preset).build()
    fresh = _build_population(
        species,
        _ConfigSensitivePreset("build_retry_fresh_recipe"),
        kind="age",
        compress=False,
        name="build_retry_fresh_pop",
    )

    assert len(retried.presets) == 1
    assert len(retried.gamete_modifiers) == 1
    assert len(retried.zygote_modifiers) == 1
    _assert_maps_equal(retried, fresh)


@pytest.mark.parametrize("kind", ["age", "discrete"])
@pytest.mark.parametrize("failure_stage", ["gamete", "zygote", "fitness"])
@pytest.mark.parametrize("scenario", ["same-call", "append"])
def test_build_time_multi_preset_failure_rolls_back_and_retries_exactly(
    kind: PopulationKind,
    failure_stage: Literal["gamete", "zygote", "fitness"],
    scenario: BuildFailureScenario,
) -> None:
    """Every build-time failure stage is atomic across preset call shapes."""
    axis = f"{kind}_{failure_stage}_{scenario}"
    species = _make_species(f"build_transaction_{axis}")
    successful = _make_fitness_drive(f"build_transaction_successful_{axis}")
    failing = _DeferredFailurePreset(
        f"build_transaction_failing_{axis}",
        fail_during_rebuild=True,
        failure_stage=failure_stage,
    )
    configurator = _build_configurator(
        species,
        kind=kind,
        name=f"build_transaction_pop_{axis}",
    )
    if scenario == "append":
        configurator.presets(successful)

    original_config = configurator.config
    original_registry = configurator._registry  # pyright: ignore[reportPrivateUsage]  # transaction must restore the registry object exactly.
    original_presets = list(configurator._presets)  # pyright: ignore[reportPrivateUsage]  # build-time registration has no public metadata view.
    original_gamete = list(configurator.gamete_modifiers)
    original_zygote = list(configurator.zygote_modifiers)
    original_compression = configurator._compression_applied  # pyright: ignore[reportPrivateUsage]  # rollback covers all Configurator transaction state.
    original_arrays = _copy_config_arrays(configurator.config)
    attempted = (successful, failing) if scenario == "same-call" else (failing,)
    expected_message = (
        "deferred fitness failure"
        if failure_stage == "fitness"
        else "deferred modifier failure"
    )

    with pytest.raises(ValueError, match=expected_message):
        configurator.presets(*attempted)

    assert configurator.config is original_config
    assert configurator._registry is original_registry  # pyright: ignore[reportPrivateUsage]  # exact registry identity is part of atomic rollback.
    assert configurator._presets == original_presets  # pyright: ignore[reportPrivateUsage]  # no failed recipe may remain registered.
    assert configurator.gamete_modifiers == original_gamete
    assert configurator.zygote_modifiers == original_zygote
    assert configurator._compression_applied is original_compression  # pyright: ignore[reportPrivateUsage]  # compression state cannot leak from a failed attempt.
    _assert_config_arrays_equal(configurator.config, original_arrays)
    expected_successful_binding = species if scenario == "append" else None
    assert successful._bound_species is expected_successful_binding  # pyright: ignore[reportPrivateUsage]  # same-call rollback releases earlier inputs; append preserves prior success.
    assert failing._bound_species is None  # pyright: ignore[reportPrivateUsage]  # the failed input remains reusable by its caller.

    failing.fail_during_rebuild = False
    retried = configurator.presets(*attempted).build()
    fresh_successful = _make_fitness_drive(f"build_transaction_fresh_successful_{axis}")
    fresh_failing = _DeferredFailurePreset(
        f"build_transaction_fresh_failing_{axis}",
        failure_stage=failure_stage,
    )
    fresh = _build_configurator(
        species,
        kind=kind,
        name=f"build_transaction_fresh_pop_{axis}",
    ).presets(fresh_successful, fresh_failing).build()

    assert len(retried.presets) == len(fresh.presets) == 2
    assert retried.presets[0] is successful
    assert retried.presets[1] is failing
    assert len(retried.gamete_modifiers) == len(fresh.gamete_modifiers) == 2
    expected_zygote_count = 1 if failure_stage == "zygote" else 0
    assert len(retried.zygote_modifiers) == expected_zygote_count
    assert len(fresh.zygote_modifiers) == expected_zygote_count
    _assert_maps_equal(retried, fresh)
    _assert_config_arrays_equal(retried.config, _copy_config_arrays(fresh.config))
    _assert_fitness_drive_values(retried)
    assert successful._bound_species is species  # pyright: ignore[reportPrivateUsage]  # successful retry commits the original input binding.
    assert failing._bound_species is species  # pyright: ignore[reportPrivateUsage]  # successful retry commits the same formerly failing object.


def test_age_invalid_conversion_rate_is_atomic() -> None:
    """Invalid panmictic reconfiguration leaves every observable unchanged."""
    species = _make_species("invalid_rate_atomic_age")
    drive = nt.HomingDrive(
        name="invalid_rate_age_drive",
        drive_allele="Drive",
        target_allele="WT",
        drive_conversion_rate=0.8,
    )
    pop = _build_population(
        species,
        drive,
        kind="age",
        compress=False,
        name="invalid_rate_atomic_age_pop",
    )
    original_config = pop.config
    original_state = pop.state
    original_counts = pop.state.individual_count.copy()
    original_sperm = pop.state.sperm_storage.copy()
    original_gamete = pop.gamete_modifiers
    original_zygote = pop.zygote_modifiers
    original_presets = pop.presets
    original_z2g = pop.config.zygotes_to_gametes_map.copy()
    original_g2z = pop.config.gametes_to_zygotes_map.copy()
    original_offspring = pop.config.offspring_tensor.copy()

    with pytest.raises(TypeError):
        pop.update().reconfigure_preset(drive, drive_conversion_rate="bad")

    assert drive.drive_conversion_rate == (0.8, 0.8)
    assert pop.config is original_config
    assert pop.state is original_state
    np.testing.assert_array_equal(pop.state.individual_count, original_counts)
    np.testing.assert_array_equal(pop.state.sperm_storage, original_sperm)
    assert pop.gamete_modifiers == original_gamete
    assert pop.zygote_modifiers == original_zygote
    assert pop.presets == original_presets
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, original_z2g)
    np.testing.assert_array_equal(pop.config.gametes_to_zygotes_map, original_g2z)
    np.testing.assert_array_equal(pop.config.offspring_tensor, original_offspring)


def test_spatial_invalid_conversion_rate_is_atomic() -> None:
    """Invalid all-deme reconfiguration leaves every deme unchanged."""
    species = _make_species("invalid_rate_atomic_spatial")
    drive = nt.HomingDrive(
        name="invalid_rate_spatial_drive",
        drive_allele="Drive",
        target_allele="WT",
        drive_conversion_rate=0.8,
    )
    pop = _build_spatial_population(
        species,
        drive,
        kind="age",
        compress=False,
        name="invalid_rate_atomic_spatial_pop",
    )
    original_configs = [deme.config for deme in pop.demes]
    original_states = [deme.state for deme in pop.demes]
    original_counts = [deme.state.individual_count.copy() for deme in pop.demes]
    original_sperm = [deme.state.sperm_storage.copy() for deme in pop.demes]
    original_gamete = [deme.gamete_modifiers for deme in pop.demes]
    original_zygote = [deme.zygote_modifiers for deme in pop.demes]
    original_presets = [deme.presets for deme in pop.demes]
    original_tensors = [deme.config.offspring_tensor.copy() for deme in pop.demes]

    with pytest.raises(TypeError):
        pop.update().reconfigure_preset(drive, drive_conversion_rate="bad")

    assert drive.drive_conversion_rate == (0.8, 0.8)
    for i, deme in enumerate(pop.demes):
        assert deme.config is original_configs[i]
        assert deme.state is original_states[i]
        np.testing.assert_array_equal(deme.state.individual_count, original_counts[i])
        np.testing.assert_array_equal(deme.state.sperm_storage, original_sperm[i])
        assert deme.gamete_modifiers == original_gamete[i]
        assert deme.zygote_modifiers == original_zygote[i]
        assert deme.presets == original_presets[i]
        np.testing.assert_array_equal(
            deme.config.offspring_tensor,
            original_tensors[i],
        )


def test_nonspatial_first_preset_registration_failure_is_atomic() -> None:
    """A failing first registration leaves a panmictic population unchanged."""
    species = _make_species("first_registration_failure_nonspatial")
    baseline = _ConfigSensitivePreset("first_registration_baseline_nonspatial")
    pop = _build_population(
        species, baseline, kind="age", compress=False,
        name="first_registration_failure_nonspatial_pop",
    )
    failing = _DeferredFailurePreset(
        "first_registration_failure_nonspatial_preset",
        fail_during_rebuild=True,
    )
    original_config = pop.config
    original_presets = pop.presets
    original_gamete = pop.gamete_modifiers
    original_zygote = pop.zygote_modifiers
    original_maps = (
        pop.config.zygotes_to_gametes_map.copy(),
        pop.config.gametes_to_zygotes_map.copy(),
        pop.config.offspring_tensor.copy(),
    )

    with pytest.raises(ValueError, match="deferred modifier failure"):
        pop.update().presets(failing)

    assert pop.config is original_config
    assert pop.presets == original_presets
    assert all(registered is not failing for registered in pop.presets)
    assert pop.gamete_modifiers == original_gamete
    assert pop.zygote_modifiers == original_zygote
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, original_maps[0])
    np.testing.assert_array_equal(pop.config.gametes_to_zygotes_map, original_maps[1])
    np.testing.assert_array_equal(pop.config.offspring_tensor, original_maps[2])


def test_spatial_first_preset_registration_failure_is_atomic() -> None:
    """A failing first registration leaves every non-contiguous deme unchanged."""
    layout: tuple[GroupLabel, ...] = ("A", "B", "A")
    species = _make_species("first_registration_failure_spatial")
    baseline = _ConfigSensitivePreset("first_registration_baseline_spatial")
    pop = _build_spatial_population(
        species, baseline, kind="discrete", compress=False,
        name="first_registration_failure_spatial_pop",
        n_demes=len(layout),
    )
    _arrange_noncontiguous_config_groups(pop, layout)
    failing = _DeferredFailurePreset(
        "first_registration_failure_spatial_preset",
        fail_during_rebuild=True,
    )
    original_configs = [deme.config for deme in pop.demes]
    original_presets = [deme.presets for deme in pop.demes]
    original_gamete = [deme.gamete_modifiers for deme in pop.demes]
    original_zygote = [deme.zygote_modifiers for deme in pop.demes]
    original_maps = [
        (
            deme.config.zygotes_to_gametes_map.copy(),
            deme.config.gametes_to_zygotes_map.copy(),
            deme.config.offspring_tensor.copy(),
        )
        for deme in pop.demes
    ]

    with pytest.raises(ValueError, match="deferred modifier failure"):
        pop.update().presets(failing)

    for i, deme in enumerate(pop.demes):
        assert deme.config is original_configs[i]
        assert deme.presets == original_presets[i]
        assert all(registered is not failing for registered in deme.presets)
        assert deme.gamete_modifiers == original_gamete[i]
        assert deme.zygote_modifiers == original_zygote[i]
        np.testing.assert_array_equal(
            deme.config.zygotes_to_gametes_map, original_maps[i][0],
        )
        np.testing.assert_array_equal(
            deme.config.gametes_to_zygotes_map, original_maps[i][1],
        )
        np.testing.assert_array_equal(
            deme.config.offspring_tensor, original_maps[i][2],
        )


@pytest.mark.parametrize("failure_stage", ["zygote", "fitness"])
def test_nonspatial_first_registration_late_stage_failure_is_atomic(
    failure_stage: Literal["zygote", "fitness"],
) -> None:
    """Later preset build stages preserve all arrays and caller ownership."""
    species = _make_species(f"late_stage_failure_nonspatial_{failure_stage}")
    baseline = _ConfigSensitivePreset(
        f"late_stage_failure_baseline_nonspatial_{failure_stage}"
    )
    pop = _build_population(
        species,
        baseline,
        kind="age",
        compress=False,
        name=f"late_stage_failure_nonspatial_pop_{failure_stage}",
    )
    failing = _DeferredFailurePreset(
        f"late_stage_failure_nonspatial_preset_{failure_stage}",
        fail_during_rebuild=True,
        failure_stage=failure_stage,
    )
    original_config = pop.config
    original_state = pop.state
    original_counts = pop.state.individual_count.copy()
    original_presets = pop.presets
    original_gamete = pop.gamete_modifiers
    original_zygote = pop.zygote_modifiers
    original_arrays = tuple(
        getattr(pop.config, field).copy()
        for field in (
            "zygotes_to_gametes_map",
            "gametes_to_zygotes_map",
            "offspring_tensor",
            "viability_fitness",
            "fecundity_fitness",
            "sexual_selection_fitness",
            "zygote_viability_fitness",
        )
    )

    expected_message = (
        "deferred fitness failure"
        if failure_stage == "fitness"
        else "deferred modifier failure"
    )
    with pytest.raises(ValueError, match=expected_message):
        pop.update().presets(failing)

    assert pop.config is original_config
    assert pop.state is original_state
    np.testing.assert_array_equal(pop.state.individual_count, original_counts)
    assert pop.presets == original_presets
    assert pop.gamete_modifiers == original_gamete
    assert pop.zygote_modifiers == original_zygote
    assert failing._bound_species is None  # pyright: ignore[reportPrivateUsage]  # failed registration must release caller-owned preset binding.
    for field, expected in zip(
        (
            "zygotes_to_gametes_map",
            "gametes_to_zygotes_map",
            "offspring_tensor",
            "viability_fitness",
            "fecundity_fitness",
            "sexual_selection_fitness",
            "zygote_viability_fitness",
        ),
        original_arrays,
    ):
        np.testing.assert_array_equal(getattr(pop.config, field), expected)


def test_spatial_single_deme_first_registration_failure_restores_shared_config() -> None:
    """Single-deme failure reverses clone-on-write as part of the transaction."""
    species = _make_species("single_deme_first_registration_failure")
    baseline = _ConfigSensitivePreset("single_deme_first_registration_baseline")
    pop = _build_spatial_population(
        species,
        baseline,
        kind="discrete",
        compress=False,
        name="single_deme_first_registration_failure_pop",
        n_demes=3,
    )
    failing = _DeferredFailurePreset(
        "single_deme_first_registration_failure_preset",
        fail_during_rebuild=True,
        failure_stage="zygote",
    )
    original_configs = [deme.config for deme in pop.demes]
    assert original_configs[0] is original_configs[1] is original_configs[2]
    original_presets = [deme.presets for deme in pop.demes]
    original_gamete = [deme.gamete_modifiers for deme in pop.demes]
    original_zygote = [deme.zygote_modifiers for deme in pop.demes]
    original_tensors = [deme.config.offspring_tensor.copy() for deme in pop.demes]

    with pytest.raises(ValueError, match="deferred modifier failure"):
        pop.update(deme=1).presets(failing)

    for i, deme in enumerate(pop.demes):
        assert deme.config is original_configs[i]
        assert deme.presets == original_presets[i]
        assert deme.gamete_modifiers == original_gamete[i]
        assert deme.zygote_modifiers == original_zygote[i]
        np.testing.assert_array_equal(
            deme.config.offspring_tensor,
            original_tensors[i],
        )
    assert failing._bound_species is None  # pyright: ignore[reportPrivateUsage]  # failed registration must release caller-owned preset binding.


def test_spatial_later_config_group_registration_failure_is_atomic() -> None:
    """A failure in the later B group restores an already-built A group."""
    layout: tuple[GroupLabel, ...] = ("A", "B", "B", "A")
    species = _make_species("later_group_first_registration_failure")
    baseline = _ConfigSensitivePreset("later_group_first_registration_baseline")
    pop = _build_spatial_population(
        species,
        baseline,
        kind="discrete",
        compress=False,
        name="later_group_first_registration_failure_pop",
        n_demes=len(layout),
    )
    _arrange_noncontiguous_config_groups(pop, layout)
    failing = _DeferredFailurePreset(
        "later_group_first_registration_failure_preset",
        fail_during_rebuild=True,
        failure_stage="zygote",
        fail_capacity=700.0,
    )
    original_configs = [deme.config for deme in pop.demes]
    original_presets = [deme.presets for deme in pop.demes]
    original_gamete = [deme.gamete_modifiers for deme in pop.demes]
    original_zygote = [deme.zygote_modifiers for deme in pop.demes]
    original_arrays = [
        tuple(
            getattr(deme.config, field).copy()
            for field in (
                "zygotes_to_gametes_map",
                "gametes_to_zygotes_map",
                "offspring_tensor",
                "viability_fitness",
                "fecundity_fitness",
                "sexual_selection_fitness",
                "zygote_viability_fitness",
            )
        )
        for deme in pop.demes
    ]

    with pytest.raises(ValueError, match="deferred modifier failure"):
        pop.update().presets(failing)

    for i, deme in enumerate(pop.demes):
        assert deme.config is original_configs[i]
        assert deme.presets == original_presets[i]
        assert deme.gamete_modifiers == original_gamete[i]
        assert deme.zygote_modifiers == original_zygote[i]
        for field, expected in zip(
            (
                "zygotes_to_gametes_map",
                "gametes_to_zygotes_map",
                "offspring_tensor",
                "viability_fitness",
                "fecundity_fitness",
                "sexual_selection_fitness",
                "zygote_viability_fitness",
            ),
            original_arrays[i],
        ):
            np.testing.assert_array_equal(getattr(deme.config, field), expected)
    assert failing._bound_species is None  # pyright: ignore[reportPrivateUsage]  # later-group rollback must release caller-owned preset binding.


def test_spatial_first_registration_success_preserves_group_numerics() -> None:
    """Successful first registration keeps sharing and exact group behavior."""
    layout: tuple[GroupLabel, ...] = ("A", "B", "B", "A")
    species = _make_species("first_registration_success_spatial")
    baseline = _DeferredFailurePreset("first_registration_success_baseline")
    pop = _build_spatial_population(
        species,
        baseline,
        kind="discrete",
        compress=False,
        name="first_registration_success_spatial_pop",
        n_demes=len(layout),
    )
    old_a, old_b = _arrange_noncontiguous_config_groups(pop, layout)
    added = _ConfigSensitivePreset("first_registration_success_added")

    pop.update().presets(added)

    new_a = pop.deme(0).config
    new_b = pop.deme(1).config
    assert new_a is not old_a
    assert new_b is not old_b
    assert new_a is not new_b
    assert added._bound_species is species  # pyright: ignore[reportPrivateUsage]  # successful registration must retain the species binding.
    for i, label in enumerate(layout):
        expected_config = new_a if label == "A" else new_b
        expected_rate = 0.2 if label == "A" else 0.8
        deme = pop.deme(i)
        assert deme.config is expected_config
        assert len(deme.presets) == 2
        assert deme.presets[0] is baseline
        assert deme.presets[1] is added
        assert len(deme.gamete_modifiers) == 2
        assert len(deme.zygote_modifiers) == 1
        _assert_group_sensitive_probabilities(deme, expected_rate)

    expected_maps = [
        (
            deme.config.zygotes_to_gametes_map.copy(),
            deme.config.gametes_to_zygotes_map.copy(),
            deme.config.offspring_tensor.copy(),
        )
        for deme in pop.demes
    ]
    for i, deme in enumerate(pop.demes):
        deme.refresh_modifiers()
        np.testing.assert_array_equal(
            deme.config.zygotes_to_gametes_map,
            expected_maps[i][0],
        )
        np.testing.assert_array_equal(
            deme.config.gametes_to_zygotes_map,
            expected_maps[i][1],
        )
        np.testing.assert_array_equal(
            deme.config.offspring_tensor,
            expected_maps[i][2],
        )

    reference_a = _build_population(
        species,
        _DeferredFailurePreset("first_registration_success_reference_a_base"),
        kind="discrete",
        compress=False,
        name="first_registration_success_reference_a",
        carrying_capacity=200.0,
    )
    reference_b = _build_population(
        species,
        _DeferredFailurePreset("first_registration_success_reference_b_base"),
        kind="discrete",
        compress=False,
        name="first_registration_success_reference_b",
        carrying_capacity=700.0,
    )
    reference_a.update().presets(
        _ConfigSensitivePreset("first_registration_success_reference_a_added")
    )
    reference_b.update().presets(
        _ConfigSensitivePreset("first_registration_success_reference_b_added")
    )
    pop.run(2)
    reference_a.run(2)
    reference_b.run(2)
    assert pop.tick == reference_a.tick == reference_b.tick == 2
    for i, label in enumerate(layout):
        expected_state = reference_a.state if label == "A" else reference_b.state
        np.testing.assert_array_equal(
            pop.deme(i).state.individual_count,
            expected_state.individual_count,
        )
        assert np.all(pop.deme(i).state.individual_count >= 0.0)


def _build_dual_modifier_population(name: str) -> tuple[Population, nt.ToxinAntidoteDrive]:
    """Build a population with one preset and both modifier kinds."""
    species = _make_species(f"{name}_species")
    preset = nt.ToxinAntidoteDrive(
        name=f"{name}_preset",
        drive_allele="Drive",
        target_allele="WT",
        disrupted_allele="Disrupted",
        conversion_rate=0.4,
        embryo_disruption_rate=0.4,
    )
    pop = _build_population(
        species,
        preset,
        kind="age",
        compress=False,
        name=f"{name}_pop",
    )
    return pop, preset


def test_presets_property_returns_list_copy() -> None:
    """Clearing and appending to the presets view cannot mutate registration."""
    pop, preset = _build_dual_modifier_population("presets_ownership")
    view = pop.presets

    view.clear()
    view.append(preset)
    view.append(preset)

    assert len(view) == 2
    assert len(pop.presets) == 1
    assert pop.presets[0] is preset
    assert pop.presets is not view


def test_gamete_modifiers_property_returns_list_copy() -> None:
    """Clearing and appending to the gamete view cannot mutate registration."""
    pop, _ = _build_dual_modifier_population("gamete_ownership")
    original = pop.gamete_modifiers
    assert len(original) == 1
    view = pop.gamete_modifiers
    entry = view[0]

    view.clear()
    view.append(entry)
    view.append(entry)

    assert len(view) == 2
    assert pop.gamete_modifiers == original
    assert pop.gamete_modifiers is not view


def test_zygote_modifiers_property_returns_list_copy() -> None:
    """Clearing and appending to the zygote view cannot mutate registration."""
    pop, _ = _build_dual_modifier_population("zygote_ownership")
    original = pop.zygote_modifiers
    assert len(original) == 1
    view = pop.zygote_modifiers
    entry = view[0]

    view.clear()
    view.append(entry)
    view.append(entry)

    assert len(view) == 2
    assert pop.zygote_modifiers == original
    assert pop.zygote_modifiers is not view


def test_deferred_modifier_failure_is_atomic_nonspatial() -> None:
    """A closure failure during trial rebuild cannot mutate the population."""
    species = _make_species("deferred_failure_nonspatial")
    preset = _DeferredFailurePreset("deferred_failure_nonspatial_preset")
    pop = _build_population(
        species,
        preset,
        kind="age",
        compress=False,
        name="deferred_failure_nonspatial_pop",
    )
    original_config = pop.config
    original_state = pop.state
    original_counts = pop.state.individual_count.copy()
    original_sperm = pop.state.sperm_storage.copy()
    original_presets = pop.presets
    original_gamete = pop.gamete_modifiers
    original_zygote = pop.zygote_modifiers
    original_z2g = pop.config.zygotes_to_gametes_map.copy()
    original_g2z = pop.config.gametes_to_zygotes_map.copy()
    original_offspring = pop.config.offspring_tensor.copy()

    with pytest.raises(ValueError, match="deferred modifier failure"):
        pop.update().reconfigure_preset(preset, fail_during_rebuild=True)

    assert preset.fail_during_rebuild is False
    assert pop.config is original_config
    assert pop.state is original_state
    np.testing.assert_array_equal(pop.state.individual_count, original_counts)
    np.testing.assert_array_equal(pop.state.sperm_storage, original_sperm)
    assert pop.presets == original_presets
    assert pop.gamete_modifiers == original_gamete
    assert pop.zygote_modifiers == original_zygote
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, original_z2g)
    np.testing.assert_array_equal(pop.config.gametes_to_zygotes_map, original_g2z)
    np.testing.assert_array_equal(pop.config.offspring_tensor, original_offspring)


def test_deferred_modifier_failure_is_atomic_spatial() -> None:
    """A spatial trial closure failure leaves every target deme unchanged."""
    species = _make_species("deferred_failure_spatial")
    preset = _DeferredFailurePreset("deferred_failure_spatial_preset")
    pop = _build_spatial_population(
        species,
        preset,
        kind="age",
        compress=False,
        name="deferred_failure_spatial_pop",
    )
    original_configs = [deme.config for deme in pop.demes]
    original_states = [deme.state for deme in pop.demes]
    original_counts = [deme.state.individual_count.copy() for deme in pop.demes]
    original_sperm = [deme.state.sperm_storage.copy() for deme in pop.demes]
    original_presets = [deme.presets for deme in pop.demes]
    original_gamete = [deme.gamete_modifiers for deme in pop.demes]
    original_zygote = [deme.zygote_modifiers for deme in pop.demes]
    original_z2g = [deme.config.zygotes_to_gametes_map.copy() for deme in pop.demes]
    original_g2z = [deme.config.gametes_to_zygotes_map.copy() for deme in pop.demes]
    original_offspring = [deme.config.offspring_tensor.copy() for deme in pop.demes]

    with pytest.raises(ValueError, match="deferred modifier failure"):
        pop.update().reconfigure_preset(preset, fail_during_rebuild=True)

    assert preset.fail_during_rebuild is False
    for i, deme in enumerate(pop.demes):
        assert deme.config is original_configs[i]
        assert deme.state is original_states[i]
        np.testing.assert_array_equal(deme.state.individual_count, original_counts[i])
        np.testing.assert_array_equal(deme.state.sperm_storage, original_sperm[i])
        assert deme.presets == original_presets[i]
        assert deme.gamete_modifiers == original_gamete[i]
        assert deme.zygote_modifiers == original_zygote[i]
        np.testing.assert_array_equal(
            deme.config.zygotes_to_gametes_map,
            original_z2g[i],
        )
        np.testing.assert_array_equal(
            deme.config.gametes_to_zygotes_map,
            original_g2z[i],
        )
        np.testing.assert_array_equal(
            deme.config.offspring_tensor,
            original_offspring[i],
        )


def test_invalid_fitness_mode_is_atomic_nonspatial() -> None:
    """Invalid fitness mode cannot mutate preset or fitness tensors."""
    species = _make_species("invalid_fitness_mode_nonspatial")
    drive = nt.HomingDrive(
        name="invalid_fitness_mode_nonspatial_drive",
        drive_allele="Drive",
        target_allele="WT",
        drive_conversion_rate=0.8,
        viability_scaling=0.7,
        viability_mode="multiplicative",
    )
    pop = _build_population(
        species,
        drive,
        kind="age",
        compress=False,
        name="invalid_fitness_mode_nonspatial_pop",
    )
    original_config = pop.config
    original_fitness = (
        pop.config.viability_fitness.copy(),
        pop.config.fecundity_fitness.copy(),
        pop.config.sexual_selection_fitness.copy(),
        pop.config.zygote_viability_fitness.copy(),
    )

    with pytest.raises(ValueError, match="Unknown fitness scaling mode"):
        pop.update().reconfigure_preset(drive, viability_mode="not-a-mode")

    assert drive.viability_mode == "multiplicative"
    assert pop.config is original_config
    np.testing.assert_array_equal(pop.config.viability_fitness, original_fitness[0])
    np.testing.assert_array_equal(pop.config.fecundity_fitness, original_fitness[1])
    np.testing.assert_array_equal(
        pop.config.sexual_selection_fitness,
        original_fitness[2],
    )
    np.testing.assert_array_equal(
        pop.config.zygote_viability_fitness,
        original_fitness[3],
    )


def test_invalid_fitness_mode_is_atomic_spatial() -> None:
    """Invalid all-deme fitness mode leaves every config and tensor unchanged."""
    species = _make_species("invalid_fitness_mode_spatial")
    drive = nt.HomingDrive(
        name="invalid_fitness_mode_spatial_drive",
        drive_allele="Drive",
        target_allele="WT",
        drive_conversion_rate=0.8,
        viability_scaling=0.7,
        viability_mode="multiplicative",
    )
    pop = _build_spatial_population(
        species,
        drive,
        kind="age",
        compress=False,
        name="invalid_fitness_mode_spatial_pop",
    )
    original_configs = [deme.config for deme in pop.demes]
    original_fitness = [
        (
            deme.config.viability_fitness.copy(),
            deme.config.fecundity_fitness.copy(),
            deme.config.sexual_selection_fitness.copy(),
            deme.config.zygote_viability_fitness.copy(),
        )
        for deme in pop.demes
    ]

    with pytest.raises(ValueError, match="Unknown fitness scaling mode"):
        pop.update().reconfigure_preset(drive, viability_mode="not-a-mode")

    assert drive.viability_mode == "multiplicative"
    for i, deme in enumerate(pop.demes):
        assert deme.config is original_configs[i]
        np.testing.assert_array_equal(
            deme.config.viability_fitness,
            original_fitness[i][0],
        )
        np.testing.assert_array_equal(
            deme.config.fecundity_fitness,
            original_fitness[i][1],
        )
        np.testing.assert_array_equal(
            deme.config.sexual_selection_fitness,
            original_fitness[i][2],
        )
        np.testing.assert_array_equal(
            deme.config.zygote_viability_fitness,
            original_fitness[i][3],
        )


@pytest.mark.parametrize(
    "layout",
    [("A", "B", "A"), ("A", "B", "B", "A")],
    ids=["A-B-A", "A-B-B-A"],
)
def test_noncontiguous_config_groups_keep_modifiers_and_run_exactly(
    layout: tuple[GroupLabel, ...],
) -> None:
    """Non-contiguous config groups retain their own modifiers and numerics."""
    axis = "".join(layout)
    species = _make_species(f"noncontiguous_groups_{axis}")
    preset = _ConfigSensitivePreset(f"noncontiguous_groups_preset_{axis}")
    pop = _build_spatial_population(
        species,
        preset,
        kind="discrete",
        compress=False,
        name=f"noncontiguous_groups_pop_{axis}",
        n_demes=len(layout),
    )
    old_a, old_b = _arrange_noncontiguous_config_groups(pop, layout)

    pop.update().presets(preset)

    new_a = pop.deme(layout.index("A")).config
    new_b = pop.deme(layout.index("B")).config
    assert new_a is not old_a
    assert new_b is not old_b
    assert new_a is not new_b
    for i, label in enumerate(layout):
        expected_config = new_a if label == "A" else new_b
        expected_rate = 0.2 if label == "A" else 0.8
        assert pop.deme(i).config is expected_config
        assert len(pop.deme(i).gamete_modifiers) == 1
        assert len(pop.deme(i).zygote_modifiers) == 1
        _assert_group_sensitive_probabilities(pop.deme(i), expected_rate)

    gamete_a = pop.deme(layout.index("A")).gamete_modifiers[0][2]
    gamete_b = pop.deme(layout.index("B")).gamete_modifiers[0][2]
    zygote_a = pop.deme(layout.index("A")).zygote_modifiers[0][2]
    zygote_b = pop.deme(layout.index("B")).zygote_modifiers[0][2]
    assert gamete_a is not gamete_b
    assert zygote_a is not zygote_b
    for i, label in enumerate(layout):
        expected_gamete = gamete_a if label == "A" else gamete_b
        expected_zygote = zygote_a if label == "A" else zygote_b
        assert pop.deme(i).gamete_modifiers[0][2] is expected_gamete
        assert pop.deme(i).zygote_modifiers[0][2] is expected_zygote

    gamete_view = pop.deme(0).gamete_modifiers
    zygote_view = pop.deme(0).zygote_modifiers
    gamete_view.clear()
    zygote_view.clear()
    gamete_view.append(pop.deme(layout.index("B")).gamete_modifiers[0])
    zygote_view.append(pop.deme(layout.index("B")).zygote_modifiers[0])
    assert pop.deme(0).gamete_modifiers[0][2] is gamete_a
    assert pop.deme(0).zygote_modifiers[0][2] is zygote_a

    expected_maps = [
        (
            deme.config.zygotes_to_gametes_map.copy(),
            deme.config.gametes_to_zygotes_map.copy(),
            deme.config.offspring_tensor.copy(),
        )
        for deme in pop.demes
    ]
    for i, deme in enumerate(pop.demes):
        deme.refresh_modifiers()
        np.testing.assert_array_equal(
            deme.config.zygotes_to_gametes_map,
            expected_maps[i][0],
        )
        np.testing.assert_array_equal(
            deme.config.gametes_to_zygotes_map,
            expected_maps[i][1],
        )
        np.testing.assert_array_equal(
            deme.config.offspring_tensor,
            expected_maps[i][2],
        )

    reference_a = _build_population(
        species,
        _ConfigSensitivePreset(f"reference_a_{axis}"),
        kind="discrete",
        compress=False,
        name=f"reference_a_pop_{axis}",
        carrying_capacity=200.0,
    )
    reference_b = _build_population(
        species,
        _ConfigSensitivePreset(f"reference_b_{axis}"),
        kind="discrete",
        compress=False,
        name=f"reference_b_pop_{axis}",
        carrying_capacity=700.0,
    )
    pop.run(2)
    reference_a.run(2)
    reference_b.run(2)
    assert pop.tick == reference_a.tick == reference_b.tick == 2
    for i, label in enumerate(layout):
        expected_state = reference_a.state if label == "A" else reference_b.state
        np.testing.assert_array_equal(
            pop.deme(i).state.individual_count,
            expected_state.individual_count,
        )
        assert np.all(pop.deme(i).state.individual_count >= 0.0)


@pytest.mark.parametrize(
    "layout",
    [("A", "B", "A"), ("A", "B", "B", "A")],
    ids=["A-B-A", "A-B-B-A"],
)
def test_noncontiguous_config_group_failure_is_atomic(
    layout: tuple[GroupLabel, ...],
) -> None:
    """A failed grouped preset rebuild preserves every identity and array."""
    axis = "".join(layout)
    species = _make_species(f"noncontiguous_failure_{axis}")
    preset = _ConfigSensitivePreset(f"noncontiguous_failure_preset_{axis}")
    pop = _build_spatial_population(
        species,
        preset,
        kind="discrete",
        compress=False,
        name=f"noncontiguous_failure_pop_{axis}",
        n_demes=len(layout),
    )
    _arrange_noncontiguous_config_groups(pop, layout)
    pop.update().presets(preset)
    original_configs = [deme.config for deme in pop.demes]
    original_states = [deme.state for deme in pop.demes]
    original_counts = [deme.state.individual_count.copy() for deme in pop.demes]
    original_gamete = [deme.gamete_modifiers for deme in pop.demes]
    original_zygote = [deme.zygote_modifiers for deme in pop.demes]
    original_maps = [
        (
            deme.config.zygotes_to_gametes_map.copy(),
            deme.config.gametes_to_zygotes_map.copy(),
            deme.config.offspring_tensor.copy(),
        )
        for deme in pop.demes
    ]

    with pytest.raises(ValueError, match="group modifier failure"):
        pop.update().reconfigure_preset(preset, fail_during_rebuild=True)

    assert preset.fail_during_rebuild is False
    for i, deme in enumerate(pop.demes):
        assert deme.config is original_configs[i]
        assert deme.state is original_states[i]
        np.testing.assert_array_equal(deme.state.individual_count, original_counts[i])
        assert deme.gamete_modifiers == original_gamete[i]
        assert deme.zygote_modifiers == original_zygote[i]
        np.testing.assert_array_equal(
            deme.config.zygotes_to_gametes_map,
            original_maps[i][0],
        )
        np.testing.assert_array_equal(
            deme.config.gametes_to_zygotes_map,
            original_maps[i][1],
        )
        np.testing.assert_array_equal(
            deme.config.offspring_tensor,
            original_maps[i][2],
        )
