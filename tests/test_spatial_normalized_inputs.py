"""Contract probes for spatial build inputs, cold compilation, and ownership."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

import natal as nt
from natal.frontend.data.definition import ModelDefinition
from natal.frontend.spatial.configurator import SpatialConfigurator, batch_setting
from tests.test_evaluator_compiler_regressions import _OpaqueResourcePreset


def _builder(name: str) -> SpatialConfigurator:
    """Create a two-deme declaration with 100 females and 100 males per deme."""
    species = nt.Species.from_dict(name, {"chr": {"locus": ["WT", "Dr"]}})
    return (
        nt.SpatialPopulation.builder(species, 2, topology=nt.SquareGrid(1, 2), pop_type="discrete_generation")
        .setup(stochastic=False)
        .initial_state(individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
        .survival(female_age0_survival=1, male_age0_survival=1)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1000, low_density_growth_rate=2)
    )


@pytest.mark.parametrize("mode", ["raw", "observation"])
@pytest.mark.parametrize("compress", [False, True])
def test_cold_spatial_definition_preserves_controls_and_isolates_arrays(
    mode: Literal["raw", "observation"], compress: bool,
) -> None:
    """A frozen spatial definition rebuilds the same dynamics after source mutation."""
    capacities = np.array([1000., 2000.])
    adjacency = np.array([[0., 1.], [1., 0.]])
    builder = (
        _builder(f"SpatialInputs_{mode}_{compress}")
        .setup(compress=compress)
        .competition(carrying_capacity=batch_setting(capacities))
        .migration(adjacency=adjacency, strategy="adjacency", migration_rate=0.1)
        .with_observation({"wt": nt.IndividualSelector(ztype="WT|WT")}, collapse_age=True, demes=[1, 0])
        .record_history(mode=mode, max_rows=2)
    )
    pop = builder.build()
    definition = pop.definition
    assert definition is not None
    spatial = definition.spatial
    assert spatial is not None
    assert spatial.observation_demes == (1, 0)
    assert spatial.topology == nt.SquareGrid(1, 2)
    assert spatial.compress is compress
    assert spatial.history_mode == mode
    capacities[:] = 99
    adjacency[:] = 0
    detached_adjacency = spatial.migration["adjacency"]
    assert isinstance(detached_adjacency, np.ndarray)
    detached_adjacency[:] = -1
    cold = SpatialConfigurator._build_from_definition(definition)
    assert cold.demes[0].params.carrying_capacity == 1000
    assert cold.demes[1].params.carrying_capacity == 2000
    assert cold.observe().values.sum() == 400
    for population in (pop, cold):
        population.run(3, record_every=1)
        assert population.history.ticks == (2, 3)
    np.testing.assert_array_equal(cold.observe().values, pop.observe().values)
    assert cold.history.schema == pop.history.schema


def test_spatial_build_consumes_normalized_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual build must use normalized values rather than the mutable builder."""
    builder = _builder("SpatialInputsConsumed").competition(
        carrying_capacity=batch_setting([1000., 2000.]),
    )
    base = builder._definition_for_compile()
    base_spatial = base.spatial
    assert base_spatial is not None
    consumed = base.with_spatial(replace(
        base_spatial, name="compiled-input-name", history_max_rows=1,
        batch_values=(("carrying_capacity", (3000., 4000.)),),
    ))
    monkeypatch.setattr(builder, "_definition_for_compile", lambda: consumed)
    pop = builder.build()
    assert pop.name == "compiled-input-name"
    assert pop.demes[0].params.carrying_capacity == 3000
    assert pop.demes[1].params.carrying_capacity == 4000
    pop.run(3, record_every=1)
    assert pop.history.ticks == (3,)


def test_spatial_batch_callable_expands_once_and_cold_build_uses_frozen_values() -> None:
    """Cold compilation consumes concrete batch values without invoking user code."""
    calls: list[int] = []

    def capacity(index: int) -> float:
        """Record evaluation while providing distinct deme values."""
        calls.append(index)
        return 1000. + index * 100

    pop = _builder("SpatialInputsBatchOnce").competition(carrying_capacity=batch_setting(capacity)).build()
    assert calls == [0, 1]
    definition = pop.definition
    assert definition is not None
    cold = SpatialConfigurator._build_from_definition(definition)
    assert calls == [0, 1]
    assert cold.demes[1].params.carrying_capacity == 1100


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("compress", [False, True])
def test_spatial_normalization_preserves_opaque_resources_and_cached_recipes(batched: bool, compress: bool) -> None:
    """A lock-owning recipe keeps its identity and does not re-run at build."""
    calls: list[str] = []

    class CountingPreset(_OpaqueResourcePreset):
        """Retain an uncopyable resource while counting recipe expansion."""

        def fitness_patch(self) -> None:
            """Produce neutral fitness without taking ownership of the lock."""
            calls.append("fitness")
            return None

    preset = CountingPreset()
    builder = _builder(f"SpatialInputsOpaque_{batched}_{compress}").setup(compress=compress).presets(preset)
    if batched:
        builder = builder.competition(carrying_capacity=batch_setting([1000., 2000.]))
    assert calls == ["fitness"]
    pop = builder.build()
    assert calls == ["fitness"]
    definition = pop.definition
    assert definition is not None
    assert definition.presets == (preset,)
    cold = SpatialConfigurator._build_from_definition(definition)
    assert calls == ["fitness", "fitness"]
    assert cold.demes[0].presets == [preset]


@pytest.mark.parametrize("template_only", [False, True])
def test_spatial_compiler_rejects_incomplete_definition(template_only: bool) -> None:
    """Missing spatial controls cannot fall back to an implicit builder state."""
    builder = _builder(f"SpatialInputsMissing_{template_only}")
    definition = builder._definition_for_compile()
    incomplete = definition.with_spatial(None) if template_only else ModelDefinition(
        definition.species, True,
    )
    with pytest.raises(ValueError, match="normalized spatial inputs"):
        SpatialConfigurator._build_from_definition(incomplete)
