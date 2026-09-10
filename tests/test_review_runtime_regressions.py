"""Independent regressions for the Rust-only plan's runtime contracts.

These tests target the review's concrete counterexamples. Seeded comparisons
verify stream continuity and replay, rather than distributional correctness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

import numpy as np
import pytest

import natal as nt
from natal.frontend.configurator import RuntimeUpdater
from natal.frontend.hooks.tick_context import TickContext
from natal.frontend.population._params_view import ParamsView

ModelKind = Literal["discrete", "age"]
Population = nt.DiscreteGenerationPopulation | nt.AgeStructuredPopulation


class RandomSampler(Protocol):
    """Minimal retained sampler contract used by the lifetime attack."""

    def random(self) -> float:
        """Draw one value from the active event's stream."""
        ...


def _population(
    name: str,
    model: ModelKind = "discrete",
    *,
    stochastic: bool = True,
    callback: Callable[[TickContext], int] | None = None,
    hook_calls: list | None = None,
) -> Population:
    """Build a neutral population with raw history and typed custom values.

    Args:
        name: Species/population name seed.
        model: ``"discrete"`` or ``"age"``.
        stochastic: Declared stochastic flag.
        callback: Optional single callback declared on ``first``.
        hook_calls: Optional ``(items, kwargs)`` pairs declared through
            ``.hooks()`` in the build chain (hook plans compile once at
            ``build()``).
    """
    species = nt.Species.from_dict(
        name=name, structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    if model == "discrete":
        builder = (
            nt.DiscreteGenerationPopulation.setup(species=species, stochastic=stochastic)
            .initial_state(individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        )
    else:
        builder = (
            nt.AgeStructuredPopulation.setup(species=species, stochastic=stochastic)
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(individual_count={"female": {"WT|WT": [0, 100, 0]}, "male": {"WT|WT": [0, 100, 0]}})
            .survival(female_age_based_survival=[1.0, 0.8, 0.0], male_age_based_survival=[1.0, 0.8, 0.0])
        )
    builder = (
        builder.reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .custom(flag=True, cohort=7, temperature=2.5, grid=np.arange(12, dtype=np.float64).reshape(2, 2, 3))
        .record_history(mode="raw")
    )
    if callback is not None:
        builder = builder.hooks(callback, event="first")
    for items, kwargs in hook_calls or []:
        builder = builder.hooks(*items, **kwargs)
    return builder.build()


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_noop_modifier_refresh_preserves_random_stream(model: ModelKind) -> None:
    """A no-op compiler refresh cannot reseed the subsequent trajectory."""
    control = _population(f"ReviewRefreshControl_{model}", model)
    refreshed = _population(f"ReviewRefreshEdited_{model}", model)
    control.run(1)
    refreshed.run(1)
    np.testing.assert_array_equal(control.export_state(), refreshed.export_state())
    refreshed.refresh_modifiers()
    control.run(2)
    refreshed.run(2)
    np.testing.assert_array_equal(refreshed.export_state(), control.export_state())


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_noop_modifier_refresh_preserves_existing_checkpoints(model: ModelKind) -> None:
    """Refreshing genetics cannot erase the session's past restore boundary."""
    pop = _population(f"ReviewRefreshCheckpoint_{model}", model)
    initial = pop.export_state().copy()
    pop.run(2, record_every=1)
    expected = pop.export_state().copy()
    pop.refresh_modifiers()
    pop.run(1, record_every=1)
    pop.restore_checkpoint(0)
    np.testing.assert_array_equal(pop.export_state(), initial)
    pop.run(2, record_every=1)
    np.testing.assert_array_equal(pop.export_state(), expected)


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_config_tensor_is_isolated_even_if_writing_is_reenabled(model: ModelKind) -> None:
    """A returned config cannot change authoritative genetics through aliases."""
    pop = _population(f"ReviewConfigIsolation_{model}", model)
    original = pop.params.viability_fitness.array.copy()
    exposed = pop.config.viability_fitness
    try:
        exposed.flags.writeable = True
        exposed[...] = 0.0
    except ValueError:
        pass  # Immutable storage is also a valid isolation implementation.
    np.testing.assert_array_equal(pop.config.viability_fitness, original)
    np.testing.assert_array_equal(pop.params.viability_fitness.array, original)


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_checkpoint_restores_custom_values_types_and_shapes(model: ModelKind) -> None:
    """Custom slots are checkpointed ecology, including arbitrary-rank arrays."""
    pop = _population(f"ReviewCustomRestore_{model}", model)
    pop.run(1, record_every=1)
    pop.update().custom(flag=False, cohort=19, temperature=9.5, grid=np.full((2, 2, 3), 99.0), added_later=12)
    pop.restore_checkpoint(0)
    actual = pop.config.custom
    assert type(actual["flag"]) is bool and actual["flag"] is True
    assert type(actual["cohort"]) is int and actual["cohort"] == 7
    assert type(actual["temperature"]) is float and actual["temperature"] == 2.5
    assert "added_later" not in actual
    assert isinstance(actual["grid"], np.ndarray)
    np.testing.assert_array_equal(actual["grid"], np.arange(12, dtype=np.float64).reshape(2, 2, 3))


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_failing_callback_discards_candidate_parameters_and_state(model: ModelKind) -> None:
    """State and parameter edits made by a failed callback never commit."""
    def fail(ctx: TickContext) -> int:
        """Edit both candidate faces before the intentional exception."""
        ctx.state.individual_count[...] = 0.0
        ctx.update().competition(carrying_capacity=31.0)
        ctx.update().custom(cohort=99)
        ctx.rng.random()
        raise ValueError("review callback failure")

    pop = _population(f"ReviewCallbackAtomic_{model}", model, callback=fail)
    initial = pop.export_state().copy()
    log = pop.params_log
    with pytest.raises(ValueError, match="review callback failure"):
        pop.run(1)
    np.testing.assert_array_equal(pop.export_state(), initial)
    assert pop.params.carrying_capacity == 100000.0
    assert pop.config.custom["cohort"] == 7
    assert pop.params_log == log


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_failed_session_requires_restore_before_running_again(model: ModelKind) -> None:
    """Disarming a failed hook cannot bypass the session's Failed state."""
    armed = [True]

    def fail_once(ctx: TickContext) -> int:
        """Fail the first attempt without introducing other state edits."""
        if armed[0]:
            raise ValueError("review failed state")
        return 0

    pop = _population(f"ReviewFailedState_{model}", model, callback=fail_once)
    with pytest.raises(ValueError, match="review failed state"):
        pop.run(1)
    armed[0] = False
    with pytest.raises(RuntimeError):
        pop.run(1)
    pop.restore_checkpoint(0)
    pop.run(1)
    assert pop.tick == 1


@pytest.mark.parametrize("model", ["discrete", "age"])
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_invalid_callback_counts_reject_without_committing(model: ModelKind, invalid: float) -> None:
    """Finite nonnegative counts are required before a callback commits."""
    def corrupt(ctx: TickContext) -> int:
        """Inject one invalid candidate coordinate."""
        ctx.state.individual_count.flat[0] = invalid
        return 0

    pop = _population(f"ReviewInvalidCounts_{model}_{invalid}", model, callback=corrupt)
    initial = pop.export_state().copy()
    with pytest.raises(ValueError):
        pop.run(1)
    np.testing.assert_array_equal(pop.export_state(), initial)


def test_context_rng_is_one_sampler_and_replays_after_restore() -> None:
    """Repeated rng access advances one Rust stream; restore rewinds it."""
    samples: list[tuple[float, float]] = []
    identities: list[bool] = []

    def sample(ctx: TickContext) -> int:
        """Record sequential draws without altering biological state."""
        identities.append(ctx.rng is ctx.rng)
        samples.append((float(ctx.rng.random()), float(ctx.rng.random())))
        return 0

    pop = _population("ReviewContextRng", callback=sample)
    pop.run(1)
    expected = pop.export_state().copy()
    pop.restore_checkpoint(0)
    pop.run(1)
    assert all(identities)
    assert samples[0] == samples[1]
    assert samples[0][0] != samples[0][1]
    np.testing.assert_array_equal(pop.export_state(), expected)


@pytest.mark.parametrize("operation", ["params", "update", "rng", "retained_rng", "retained_params", "retained_update"])
def test_retained_callback_context_cannot_mutate_later(operation: str) -> None:
    """Escaped contexts and escaped samplers expire at the event boundary."""
    contexts: list[TickContext] = []
    samplers: list[RandomSampler] = []
    parameter_handles: list[ParamsView] = []
    configurators: list[RuntimeUpdater] = []

    def retain(ctx: TickContext) -> int:
        """Retain both the context and its sampler for the lifetime attack."""
        contexts.append(ctx)
        samplers.append(ctx.rng)
        parameter_handles.append(ctx.params)
        configurators.append(ctx.update())
        return 0

    pop = _population(f"ReviewExpired_{operation}", callback=retain)
    pop.run(1)
    snapshot = pop.export_state().copy()
    with pytest.raises(RuntimeError):
        if operation == "params":
            contexts[0].params.carrying_capacity = 9.0
        elif operation == "update":
            contexts[0].update().competition(carrying_capacity=9.0)
        elif operation == "rng":
            contexts[0].rng.random()
        elif operation == "retained_params":
            parameter_handles[0].carrying_capacity = 9.0
        elif operation == "retained_update":
            configurators[0].competition(carrying_capacity=9.0)
        else:
            samplers[0].random()
    np.testing.assert_array_equal(pop.export_state(), snapshot)
    assert pop.params.carrying_capacity == 100000.0


@pytest.mark.parametrize("model", ["discrete", "age"])
def test_restore_running_boundary_clears_later_finished_state(model: ModelKind) -> None:
    """Restoring an ordinary checkpoint restores Ready as well as counts."""
    pop = _population(f"ReviewRestoreReady_{model}", model)
    pop.run(2, record_every=1, finish=True)
    expected = pop.export_state().copy()
    pop.restore_checkpoint(1)
    pop.run(1)
    np.testing.assert_array_equal(pop.export_state(), expected)


def test_backend_selection_entrypoints_are_absent() -> None:
    """The sole-backend plan forbids user-accessible backend toggles."""
    pop = _population("ReviewRemovedBackend")
    for name in ("enable_rust_backend", "disable_rust_backend", "using_rust_backend"):
        assert not hasattr(pop, name), name
