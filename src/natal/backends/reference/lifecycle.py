"""Unified lifecycle tick orchestration for the reference backend.

This module is the single source of truth for the simulation tick order:

    first hook -> reproduction -> early hook -> survival -> late hook
    -> aging

and for the Wright-Fisher fused tick (first hook only).  All functions are
plain Python; the reference backend is the always-available fallback.
"""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from natal.backends.reference.age_structured_simulator import (
    run_aging,
    run_reproduction,
    run_survival,
)
from natal.backends.reference.discrete_generation_simulator import (
    run_discrete_aging,
    run_discrete_reproduction,
    run_discrete_survival,
)
from natal.backends.reference.simulation.discrete_generation import (
    run_wf_tick as _run_wf_wide,
)
from natal.frontend.data import (
    DiscretePopulationState,
    ModelDraft,
    PopulationState,
)
from natal.frontend.hooks.runtime.csr_kernel import execute_csr_event_program_with_state
from natal.frontend.hooks.types import (
    EVENT_EARLY,
    EVENT_FIRST,
    EVENT_LATE,
    RESULT_CONTINUE,
    RESULT_STOP,
    HookProgram,
)

__all__ = [
    "run",
    "run_discrete_tick",
    "run_structured_tick",
    "run_wf_tick",
]

_StateT = TypeVar("_StateT", PopulationState, DiscretePopulationState)
_LifecycleState = PopulationState | DiscretePopulationState
_LifecycleConfig = ModelDraft
_LifecycleHook = Callable[..., int]


def _run_event(
    event_id: int,
    state: _LifecycleState,
    config: _LifecycleConfig,
    registry: HookProgram,
    event_hook: _LifecycleHook,
    deme_id: int,
    has_sperm_storage: bool,
    sperm_store: Optional[NDArray[np.float64]],
) -> tuple[int, ModelDraft]:
    """Execute one event: declarative CSR ops, then the combined hook.

    ``Op.set_param`` support: the runtime-mutable ecology scalars are
    snapshotted into an ``eco_values`` scratch, the CSR kernel evaluates RPN
    expressions against (and writes results into) that scratch, and changed
    values are flushed back through a ``_replace``-built draft.  Later
    lifecycle stages of the same tick observe the writes.

    Args:
        event_id: Numeric event id (EVENT_FIRST, EVENT_EARLY, EVENT_LATE).
        state: Current population state.  Its arrays are mutated in-place by
            CSR declarative operations and by the combined hook.
        config: Current population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        event_hook: Combined hook with signature
            ``(state, config, deme_id) -> int``.
        deme_id: Deme index.  ``0`` is the panmictic default.
        has_sperm_storage: Whether *sperm_store* contains real data.  When
            False, *sperm_store* must be ``None`` (no dummy array).
        sperm_store: Sperm-storage array or ``None`` for discrete models.

    Returns:
        ``(RESULT_CONTINUE or RESULT_STOP, updated_config)``.
    """
    # ECO_PARAM_NAMES order: carrying_capacity, eggs_per_female, sex_ratio,
    # sperm_displacement_rate, low_density_growth_rate.
    eco_values = np.zeros(5, dtype=np.float64)
    eco_values[0] = config.carrying_capacity
    eco_values[1] = config.eggs_per_female
    eco_values[2] = config.sex_ratio
    eco_values[3] = config.sperm_displacement_rate
    eco_values[4] = config.low_density_growth_rate

    result = execute_csr_event_program_with_state(
        registry,
        event_id,
        state.individual_count,
        sperm_store,
        state.n_tick,
        bool(config.stochastic),
        has_sperm_storage,
        bool(config.continuous_sampling),
        deme_id,
        eco_values,
    )

    # Flush changed ecology scalars through a fresh draft (NamedTuple fields
    # are immutable).
    replacements: dict[str, float] = {}
    if eco_values[0] != config.carrying_capacity:
        replacements["carrying_capacity"] = eco_values[0]
    if eco_values[1] != config.eggs_per_female:
        replacements["eggs_per_female"] = eco_values[1]
    if eco_values[2] != config.sex_ratio:
        replacements["sex_ratio"] = eco_values[2]
    if eco_values[3] != config.sperm_displacement_rate:
        replacements["sperm_displacement_rate"] = eco_values[3]
    if eco_values[4] != config.low_density_growth_rate:
        replacements["low_density_growth_rate"] = eco_values[4]
    if replacements:
        config = config._replace(**replacements)

    if result != RESULT_CONTINUE:
        return RESULT_STOP, config
    result = event_hook(state, config, deme_id)
    return (RESULT_STOP if result != 0 else RESULT_CONTINUE), config


def run_structured_tick(
    state: PopulationState,
    config: ModelDraft,
    registry: HookProgram,
    first_hook: _LifecycleHook,
    early_hook: _LifecycleHook,
    late_hook: _LifecycleHook,
    deme_id: int = 0,
    config_refresh: Optional[Callable[[ModelDraft], ModelDraft]] = None,
) -> tuple[PopulationState, int, ModelDraft]:
    """Execute one age-structured tick with hooks at each lifecycle stage.

    Args:
        state: Current population state with sperm storage.
        config: Population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        first_hook: Combined ``first`` event hook.
        early_hook: Combined ``early`` event hook.
        late_hook: Combined ``late`` event hook.
        deme_id: Deme index.  ``0`` is the panmictic default.
        config_refresh: Optional callback returning the population's
            current config; used after each event so write-channel
            rebinds made by hook execution stay visible for the rest of
            the tick.

    Returns:
        ``(next_state, result_code, updated_config)``.
    """
    tick = state.n_tick
    current = PopulationState(
        n_tick=tick,
        individual_count=state.individual_count.copy(),
        sperm_storage=state.sperm_storage.copy(),
    )

    result, config = _run_event(
        EVENT_FIRST, current, config, registry, first_hook, deme_id, True,
        current.sperm_storage,
    )
    if config_refresh is not None:
        config = config_refresh(config)
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP, config

    current = run_reproduction(current, config)

    result, config = _run_event(
        EVENT_EARLY, current, config, registry, early_hook, deme_id, True,
        current.sperm_storage,
    )
    if config_refresh is not None:
        config = config_refresh(config)
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP, config

    current = run_survival(current, config)

    result, config = _run_event(
        EVENT_LATE, current, config, registry, late_hook, deme_id, True,
        current.sperm_storage,
    )
    if config_refresh is not None:
        config = config_refresh(config)
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP, config

    current = run_aging(current, config)
    return (
        PopulationState(
            n_tick=tick + 1,
            individual_count=current.individual_count,
            sperm_storage=current.sperm_storage,
        ),
        RESULT_CONTINUE,
        config,
    )


def run_discrete_tick(
    state: DiscretePopulationState,
    config: ModelDraft,
    registry: HookProgram,
    first_hook: _LifecycleHook,
    early_hook: _LifecycleHook,
    late_hook: _LifecycleHook,
    deme_id: int = 0,
    config_refresh: Optional[Callable[[ModelDraft], ModelDraft]] = None,
) -> tuple[DiscretePopulationState, int, ModelDraft]:
    """Execute one discrete-generation tick with hooks at each stage.

    Args:
        state: Current discrete population state.
        config: Unified model draft (discrete normalization).
        registry: ``HookProgram`` with CSR declarative operations.
        first_hook: Combined ``first`` event hook.
        early_hook: Combined ``early`` event hook.
        late_hook: Combined ``late`` event hook.
        deme_id: Deme index.  ``0`` is the panmictic default.
        config_refresh: Optional callback returning the population's
            current config; used after each event so write-channel
            rebinds made by hook execution stay visible for the rest of
            the tick.

    Returns:
        ``(next_state, result_code, updated_config)``.
    """
    tick = state.n_tick
    current = DiscretePopulationState(
        n_tick=tick,
        individual_count=state.individual_count.copy(),
    )

    result, config = _run_event(
        EVENT_FIRST, current, config, registry, first_hook, deme_id, False, None,
    )
    if config_refresh is not None:
        config = config_refresh(config)
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP, config

    current = run_discrete_reproduction(current, config)

    result, config = _run_event(
        EVENT_EARLY, current, config, registry, early_hook, deme_id, False, None,
    )
    if config_refresh is not None:
        config = config_refresh(config)
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP, config

    current = run_discrete_survival(current, config)

    result, config = _run_event(
        EVENT_LATE, current, config, registry, late_hook, deme_id, False, None,
    )
    if config_refresh is not None:
        config = config_refresh(config)
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP, config

    current = run_discrete_aging(current, config)
    return DiscretePopulationState(
        n_tick=tick + 1,
        individual_count=current.individual_count,
    ), RESULT_CONTINUE, config


def run_wf_tick(
    state: DiscretePopulationState,
    config: ModelDraft,
    registry: HookProgram,
    first_hook: _LifecycleHook,
    early_hook: _LifecycleHook,
    late_hook: _LifecycleHook,
    deme_id: int = 0,
    config_refresh: Optional[Callable[[ModelDraft], ModelDraft]] = None,
) -> tuple[DiscretePopulationState, int, ModelDraft]:
    """Execute one Wright-Fisher fused tick.

    Only the ``first`` hook runs; ``early_hook`` and ``late_hook`` are
    accepted for signature uniformity but deliberately unused because the
    Wright-Fisher tick has no intermediate lifecycle stages.

    Args:
        state: Current discrete population state.
        config: Unified model draft (discrete normalization).
        registry: ``HookProgram`` with CSR declarative operations.
        first_hook: Combined ``first`` event hook.
        early_hook: Accepted for signature uniformity; unused.
        late_hook: Accepted for signature uniformity; unused.
        deme_id: Deme index.  ``0`` is the panmictic default.
        config_refresh: Optional callback returning the population's
            current config; used after the first event so write-channel
            rebinds made by hook execution stay visible.

    Returns:
        ``(next_state, result_code, updated_config)``.
    """
    tick = state.n_tick
    current = DiscretePopulationState(
        n_tick=tick,
        individual_count=state.individual_count.copy(),
    )

    result, config = _run_event(
        EVENT_FIRST, current, config, registry, first_hook, deme_id, False, None,
    )
    if config_refresh is not None:
        config = config_refresh(config)
    if result != RESULT_CONTINUE:
        return current, RESULT_STOP, config

    # Derived on read: the draft no longer stores the equilibrium
    # metrics (slice 2); compute them from the config's own ecology.
    from natal.frontend.data._engine import (
        derive_equilibrium_metrics_from_draft,
    )

    expected_competition_strength, expected_survival_rate = (
        derive_equilibrium_metrics_from_draft(config)
    )

    new_ind = _run_wf_wide(
        ind_count=current.individual_count,
        offspring_tensor=config.offspring_tensor,
        fecundity_f=config.fecundity_fitness[0],
        fecundity_m=config.fecundity_fitness[1],
        sexual_selection=config.sexual_selection_fitness,
        viability_f=config.viability_fitness[0, 0, :],
        viability_m=config.viability_fitness[1, 0, :],
        eggs_per_female=config.eggs_per_female,
        sex_ratio=config.sex_ratio,
        female_compat=config.female_ztype_compatibility,
        male_compat=config.male_ztype_compatibility,
        female_only=config.female_only_by_sex_chrom,
        male_only=config.male_only_by_sex_chrom,
        has_sex_chromosomes=config.has_sex_chromosomes,
        mode=int(config.extreme_speed_mode),
        stochastic=bool(config.stochastic),
        mating_rate_f=config.age_based_mating_rates[0, 1],
        mating_rate_m=config.age_based_mating_rates[1, 1],
        reproduction_rate=config.age_based_reproduction_rates[1],  # pyright: ignore[reportOptionalSubscript, reportUnknownArgumentType]  # build always resolves the vector (factory fills the mating fallback)
        carrying_capacity=config.carrying_capacity,
        juvenile_growth_mode=config.juvenile_growth_mode,
        low_density_growth_rate=config.low_density_growth_rate,
        expected_competition_strength=expected_competition_strength,
        expected_survival_rate=expected_survival_rate,
    )
    return DiscretePopulationState(
        n_tick=tick + 1,
        individual_count=new_ind,
    ), RESULT_CONTINUE, config


def run(
    tick_fn: Callable[..., tuple[_StateT, int, ModelDraft]],
    state: _StateT,
    config: _LifecycleConfig,
    registry: HookProgram,
    first_hook: _LifecycleHook,
    early_hook: _LifecycleHook,
    late_hook: _LifecycleHook,
    deme_id: int,
    n_steps: int,
    record_every: int,
    record_fn: Callable[[_StateT], None],
    config_refresh: Optional[Callable[[ModelDraft], ModelDraft]] = None,
) -> tuple[_StateT, bool, ModelDraft]:
    """Run *n_steps* lifecycle ticks in pure Python.

    Recording uses the supplied Python callback so the population layer can
    append snapshots through the normal History path.

    Args:
        tick_fn: Single-tick function with signature
            ``(state, config, registry, first_hook, early_hook, late_hook,
            deme_id) -> (state, result, config)``.
        config_refresh: Optional callback returning the population's
            current config; forwarded to *tick_fn* so writes made by hook
            execution stay visible for the rest of each tick.
        state: Initial population state.
        config: Population configuration.
        registry: ``HookProgram`` with CSR declarative operations.
        first_hook: Combined ``first`` event hook.
        early_hook: Combined ``early`` event hook.
        late_hook: Combined ``late`` event hook.
        deme_id: Deme index.  ``0`` is the panmictic default.
        n_steps: Number of ticks to execute.
        record_every: Record after ticks satisfying
            ``tick % record_every == 0``.  ``0`` disables recording.
        record_fn: Callback receiving the completed state after each
            recorded tick.

    Returns:
        ``(final_state, was_stopped, updated_config)``.
    """
    current = state
    for _ in range(n_steps):
        if config_refresh is not None:
            current, result, config = tick_fn(
                current, config, registry, first_hook, early_hook, late_hook,
                deme_id, config_refresh=config_refresh,
            )
        else:
            current, result, config = tick_fn(
                current, config, registry, first_hook, early_hook, late_hook,
                deme_id,
            )
        if result != RESULT_CONTINUE:
            return current, True, config
        if record_every > 0 and (current.n_tick % record_every == 0):
            record_fn(current)
    return current, False, config
