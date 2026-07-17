"""Discrete-generation population model.

Non-overlapping generations with n_ages=2:
- age 0: offspring produced in the current tick
- age 1: reproducing adults

Simulation flow:
first hook → reproduction → early hook → survival → late hook → aging
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.data import (
    DiscretePopulationConfig,
    DiscretePopulationState,
    parse_flattened_discrete_state,
)
from natal.engine.discrete_generation_simulator import (
    run_discrete_aging,
    run_discrete_reproduction,
    run_discrete_survival,
)
from natal.genetics import Genotype, Species
from natal.hooks.types import RESULT_CONTINUE, RESULT_STOP
from natal.population.base import BasePopulation
from natal.registry.index import IndexRegistry
from natal.utils.types import Sex

if TYPE_CHECKING:
    from natal.configurator import (
        DiscreteConfigurator,
    )

__all__ = ["DiscreteGenerationPopulation"]


def _require_discrete_config(config: object) -> DiscretePopulationConfig:
    """Validate that *config* is a well-formed ``DiscretePopulationConfig``.

    The two config models (``PopulationConfig`` for age-structured,
    ``DiscretePopulationConfig`` for discrete-generation) are independent
    NamedTuples with no cross-model conversion.  This helper enforces the
    type boundary at every entry point that stores a config on a
    ``DiscreteGenerationPopulation`` (``__init__``, ``import_config``):
    it rejects any other type — including ``PopulationConfig`` and dict —
    with ``TypeError``, and rejects a ``DiscretePopulationConfig`` whose
    discrete-generation invariants are violated with ``ValueError``.

    The discrete-generation engine hardcodes a 2-age lifecycle
    (age 0 = offspring, age 1 = reproducing adult; adults are replaced
    every tick).  A config with ``n_ages != 2``, ``new_adult_age != 1``,
    or ``adult_ages != [1]`` would run but produce silently wrong dynamics,
    so it is rejected up front rather than silently normalised.

    Args:
        config: The candidate config object.

    Returns:
        *config* itself, narrowed to ``DiscretePopulationConfig``.

    Raises:
        TypeError: If *config* is not a ``DiscretePopulationConfig``.
        ValueError: If *config* is a ``DiscretePopulationConfig`` but
            ``n_ages != 2``, ``new_adult_age != 1``, or
            ``adult_ages != [1]``.
    """
    if not isinstance(config, DiscretePopulationConfig):
        raise TypeError(
            f"DiscreteGenerationPopulation requires a "
            f"DiscretePopulationConfig, got {type(config).__name__}. "
            f"PopulationConfig and other types are not accepted — the two "
            f"config models are independent; build a "
            f"DiscretePopulationConfig via Configurator.for_discrete() "
            f"or build_discrete_engine_config()."
        )
    if config.n_ages != 2 or config.new_adult_age != 1:
        raise ValueError(
            f"DiscretePopulationConfig must satisfy the discrete-generation "
            f"invariants: n_ages == 2 and new_adult_age == 1, got "
            f"n_ages={config.n_ages}, new_adult_age={config.new_adult_age}. "
            f"The discrete engine hardcodes a 2-age lifecycle."
        )
    expected_adult_ages = np.array([1], dtype=np.int64)
    if not np.array_equal(config.adult_ages, expected_adult_ages):
        raise ValueError(
            f"DiscretePopulationConfig.adult_ages must be [1], got "
            f"{config.adult_ages!r}."
        )
    return config


class DiscreteGenerationPopulation(BasePopulation[DiscretePopulationState]):
    """Population with strict non-overlapping generations."""

    def __init__(
        self,
        species: Species,
        population_config: DiscretePopulationConfig,
        name: Optional[str] = None,
        index_registry: Optional[IndexRegistry] = None,
        initial_individual_count: Optional[
            Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]]
        ] = None,
        hooks: Optional[Dict[str, List[Tuple[Any, Optional[str], Optional[int]]]]] = None,
    ):
        """Initialize a discrete-generation population.

        Constructs the population from a species definition and a
        ``DiscretePopulationConfig``, sets up genotype registries and the
        initial age-by-genotype distribution, and registers hooks for
        event-driven intervention.

        Args:
            species: Genetic architecture describing loci, alleles and
                chromosome structure.
            population_config: A fully initialised
                ``DiscretePopulationConfig``.  A ``PopulationConfig`` or
                other type is rejected with ``TypeError`` — the two config
                models are independent; build a ``DiscretePopulationConfig``
                via ``Configurator.for_discrete()`` or
                ``build_discrete_engine_config()``.
            name: Human-readable population name.  Defaults to
                ``"DiscreteGenerationPop"``.
            index_registry: Optional shared registry for index compression.
            initial_individual_count: Optional per-sex, per-genotype
                initial distribution that overrides the config default.
            hooks: Event hook registrations to apply.

        Raises:
            TypeError: If *population_config* is not a
                ``DiscretePopulationConfig``.
            ValueError: If *population_config* violates the discrete
                generation invariants (``n_ages == 2``, ``new_adult_age
                == 1``, ``adult_ages == [1]``).
        """
        if name is None:
            name = "DiscreteGenerationPop"

        super().__init__(species, name, hooks=hooks or {})

        if index_registry is not None:
            self._index_registry = index_registry

        self._config = _require_discrete_config(population_config)

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        n_sexes = self.config.n_sexes
        n_ztypes = self.config.n_ztypes
        n_ages = self.config.n_ages

        # Create an empty state first so we can check whether the config's
        # default initial_individual_count has compatible dimensions --
        # presets from genetic_presets.py often pre-size this to match.
        self._state = DiscretePopulationState.create(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            n_tick=0,
            individual_count=np.zeros((n_sexes, n_ages, n_ztypes), dtype=np.float64),
        )

        cfg_init_ind = self.config.initial_individual_count
        if cfg_init_ind.shape == self.state.individual_count.shape:
            self.state.individual_count[:] = cfg_init_ind

        # An explicit distribution overrides the config default. We zero out
        # the array first because _distribute_initial_population accumulates.
        if initial_individual_count is not None:
            self.state.individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)

        # Keep a pristine copy so reset() can restore the starting state.
        self._initial_population_snapshot = (
            self.state.individual_count.copy(),
            None,
            None,
        )

        self._finalize_hooks()

        # Build self-describing history schema (frozen at construction).
        self._init_history_schema(
            kind="discrete_generation",
            n_demes=1,
            has_sperm_storage=False,
        )

    @classmethod
    def setup(
        cls,
        species: Species,
        name: str = "DiscreteGenerationPop",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        *,
        compress: bool = False,
        declared_zygote_types: Sequence[str] | Sequence[int] | None = None,
        declared_genotypes: Sequence[str] | Sequence[int] | None = None,  # deprecated alias
    ) -> DiscreteConfigurator:
        """Fluent population construction entry point.

        Returns a ``DiscreteConfigurator``.  Chain domain methods and end
        with ``.build()`` to create a Population.
        """
        from natal.configurator import Configurator

        if declared_genotypes is not None:
            if declared_zygote_types is not None:
                raise ValueError(
                    "Cannot specify both declared_zygote_types and "
                    "declared_genotypes (deprecated alias)."
                )
            declared_zygote_types = declared_genotypes
        return Configurator.for_discrete(species).setup(
            name=name,
            stochastic=stochastic,
            continuous_sampling=continuous_sampling,
            fixed_egg_count=fixed_egg_count,
            compress=compress,
            declared_zygote_types=declared_zygote_types,
        )

    def _resolve_age_distribution(
        self,
        age_data: Union[List[int], Dict[int, int], int, float],
    ) -> Tuple[float, float]:
        """Resolve age distribution data into ``(age_0, age_1)`` counts.

        Accepts int/float (all to age 1), list of length 0–2, or dict
        with keys 0 and/or 1.

        Args:
            age_data: Age distribution specification.

        Returns:
            A tuple of ``(age_0_count, age_1_count)``.

        Raises:
            ValueError: If *age_data* format is unsupported.
        """
        if isinstance(age_data, (int, float)):
            return 0.0, float(age_data)
        if isinstance(age_data, list):
            if len(age_data) == 0:
                return 0.0, 0.0
            if len(age_data) == 1:
                return 0.0, float(age_data[0])
            if len(age_data) == 2:
                return float(age_data[0]), float(age_data[1])
            raise ValueError(f"Discrete initial list must have length <= 2, got {len(age_data)}")
        unsupported_keys = [k for k in age_data.keys() if k not in (0, 1)]
        if unsupported_keys:
            raise ValueError(f"Discrete initial dict supports only age keys 0 and 1, got {unsupported_keys}")
        return float(age_data.get(0, 0.0)), float(age_data.get(1, 0.0))

    def _distribute_initial_population(
        self,
        distribution: Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]],
    ) -> None:
        """Distribute individuals across genotypes and ages from a nested dict.

        Args:
            distribution: Dict mapping sex -> {genotype -> age_distribution}.

        Raises:
            ValueError: If sex key is not ``"female"`` or ``"male"``.
        """
        self.state.individual_count.fill(0.0)
        for sex_key, genotype_dist in distribution.items():
            sex_key_norm = sex_key.lower().strip()
            if sex_key_norm == "female":
                sex_idx = int(Sex.FEMALE.value)
            elif sex_key_norm == "male":
                sex_idx = int(Sex.MALE.value)
            else:
                raise ValueError(f"Sex must be 'female' or 'male', got '{sex_key}'")
            for genotype_key, age_data in genotype_dist.items():
                from natal.patterns import (
                    GenotypePatternParser,
                    ZygoteTypePattern,
                )

                if isinstance(genotype_key, str):
                    pattern = ZygoteTypePattern.from_slab_key(genotype_key, self.species)
                else:
                    parser = GenotypePatternParser(self.species)
                    pattern = ZygoteTypePattern(
                        parser.parse(str(genotype_key)), slab=None
                    )

                z_idx = self.registry.resolve_default_ztype_index(pattern)
                age0_count, age1_count = self._resolve_age_distribution(age_data)
                self.state.individual_count[sex_idx, 0, z_idx] = age0_count
                self.state.individual_count[sex_idx, 1, z_idx] = age1_count

    def run(
        self,
        n_steps: int = 1,
        record_every: Optional[int] = None,
        finish: bool = False,
        clear_history_on_start: bool = False,
    ) -> DiscreteGenerationPopulation:
        """Run the population for *n_steps* ticks.

        Uses the pre-compiled discrete lifecycle wrapper when Numba is
        enabled; falls back to a Python dispatch loop otherwise.

        Args:
            n_steps: Number of ticks to simulate.
            record_every: Interval for recording history snapshots.
                Defaults to ``self.record_every``.
            finish: If True, trigger the finish event after the run.
            clear_history_on_start: If True, clear history before running.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the population has already finished.
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. Cannot run() again after finish=True."
            )

        self._running = True
        try:
            # Wright-Fisher extreme-speed path: single multinomial draw per tick.
            if getattr(self.config, "extreme_speed_mode", 0) > 0:
                record_every_resolved = record_every if record_every is not None else self.record_every
                if self.should_use_python_dispatch():
                    return self._run_wright_fisher(
                        n_steps=n_steps,
                        record_every=record_every_resolved,
                        finish=finish,
                        clear_history_on_start=clear_history_on_start,
                    )

                wrappers = self.get_compiled_event_hooks()
                if wrappers.run_wf_fn is None:
                    return self._run_wright_fisher(
                        n_steps=n_steps,
                        record_every=record_every_resolved,
                        finish=finish,
                        clear_history_on_start=clear_history_on_start,
                    )

                obs_mask = self._observation_mask
                n_obs = len(self._observation.labels) if self._observation is not None else 0

                final_state_tuple, history_new, was_stopped = wrappers.run_wf_fn(
                    state=self.state,
                    config=self.config,
                    registry=wrappers.hooks.registry,
                    n_ticks=n_steps,
                    record_interval=record_every_resolved,
                    observation_mask=obs_mask,
                    n_obs_groups=n_obs,
                )

                self._state = DiscretePopulationState(
                    n_tick=int(final_state_tuple[1]),
                    individual_count=final_state_tuple[0],
                )
                self._tick = int(final_state_tuple[1])
                self._process_kernel_history(history_new, clear_history_on_start)

                if was_stopped:
                    self._finished = True
                    self.trigger_event("finish")
                elif finish:
                    self.finish_simulation()

                return self

            if record_every is None:
                record_every = self.record_every
            if self.should_use_python_dispatch():
                return self._run_python_dispatch(
                    n_steps=n_steps,
                    record_every=record_every,
                    finish=finish,
                    clear_history_on_start=clear_history_on_start,
                )

            wrappers = self.get_compiled_event_hooks()
            assert wrappers.hooks.registry is not None, "hooks.registry should always be initialized"

            # No compiled wrapper available -- codegen may have failed (no hooks
            # registered, incompatible hook types, or cache miss). Fall back to
            # the pure Python dispatch loop which is functionally identical.
            if wrappers.run_discrete_fn is None:
                return self._run_python_dispatch(
                    n_steps=n_steps,
                    record_every=record_every,
                    finish=finish,
                    clear_history_on_start=clear_history_on_start,
                )

            obs_mask = self._observation_mask
            n_obs = len(self._observation.labels) if self._observation is not None else 0

            final_state_tuple, history_new, was_stopped = wrappers.run_discrete_fn(
                state=self.state,
                config=self.config,
                registry=wrappers.hooks.registry,
                n_ticks=n_steps,
                record_interval=record_every,
                observation_mask=obs_mask,
                n_obs_groups=n_obs,
            )

            self._state = DiscretePopulationState(
                n_tick=int(final_state_tuple[1]),
                individual_count=final_state_tuple[0],
            )
            self._tick = int(final_state_tuple[1])
            self._process_kernel_history(history_new, clear_history_on_start)

            # A STOP result from any hook means the simulation ended early; mark
            # finished so that downstream code and the caller know not to continue.
            if was_stopped:
                self._finished = True
                self.trigger_event("finish")
            elif finish:
                self.finish_simulation()

            return self
        finally:
            self._running = False

    def _run_wright_fisher(
        self,
        n_steps: int,
        record_every: int,
        finish: bool,
        clear_history_on_start: bool,
    ) -> DiscreteGenerationPopulation:
        """Run using the Wright-Fisher extreme-speed path.

        Only FIRST hooks are supported — they fire before the WF tick.
        EARLY and LATE have no natural insertion point because the WF
        tick fuses reproduction, survival, and aging into one step.
        Both compiled (CSR + njit) and Python-fallback hooks are
        dispatched through ``trigger_event``.
        """
        from natal.engine.simulation.discrete_generation import run_wf_loop

        if clear_history_on_start:
            self.clear_history()

        # Build the HookExecutor lazily — covers CSR, njit custom,
        # selector, and Python wrapper hooks.
        self.ensure_hook_executor()

        state = self._state
        assert state is not None, "Population state must be initialized before running"

        cfg = self.config
        mode = cfg.extreme_speed_mode

        # Record initial state snapshot (mirrors compiled WF wrapper and
        # _run_python_dispatch which both record tick-0 before the loop).
        if record_every > 0 and (state.n_tick % record_every == 0):
            self._tick = state.n_tick
            self._record_current_snapshot(allow_existing=True)

        was_stopped = False
        for _ in range(n_steps):
            # Keep self._tick in sync so that hook ``when`` conditions
            # and CSR condition programs see the correct tick.
            self._tick = state.n_tick

            # ---- FIRST hooks (before reproduction) ----
            if self.trigger_event("first", deme_id=-1) == RESULT_STOP:
                was_stopped = True
                break

            # ---- WF tick (reproduction + survival + aging) ----
            new_ind = run_wf_loop(
                ind_count=state.individual_count,
                n_ticks=1,
                offspring_tensor=cfg.offspring_tensor,
                fecundity_f=cfg.fecundity_f,
                fecundity_m=cfg.fecundity_m,
                sexual_selection=cfg.sexual_selection_fitness,
                viability_f=cfg.viability_f,
                viability_m=cfg.viability_m,
                eggs_per_female=float(cfg.eggs_per_female[()]),
                sex_ratio=float(cfg.sex_ratio[()]),
                female_compat=cfg.female_ztype_compatibility,
                male_compat=cfg.male_ztype_compatibility,
                female_only=cfg.female_only_by_sex_chrom,
                male_only=cfg.male_only_by_sex_chrom,
                has_sex_chromosomes=cfg.has_sex_chromosomes,
                mode=mode,
                stochastic=cfg.stochastic,
                mating_rate_f=cfg.female_adult_mating_rate,
                mating_rate_m=cfg.male_adult_mating_rate,
                reproduction_rate=cfg.reproduction_rate,
                carrying_capacity=float(cfg.carrying_capacity[()]),
                juvenile_growth_mode=int(cfg.juvenile_growth_mode[()]),
                low_density_growth_rate=float(cfg.low_density_growth_rate[()]),
                expected_competition_strength=float(cfg.expected_competition_strength[()]),
                expected_survival_rate=float(cfg.expected_survival_rate[()]),
            )

            next_tick = state.n_tick + 1
            state = DiscretePopulationState(
                n_tick=next_tick, individual_count=new_ind,
            )
            self._state = state  # keep self._state in sync for hooks

            if record_every > 0 and (next_tick % record_every == 0):
                self._tick = next_tick
                self._record_current_snapshot(allow_existing=True)

            # No EARLY / LATE hooks — WF fuses reproduction + survival
            # + aging into one atomic step with no intermediate stages.

        self._state = state
        self._tick = state.n_tick

        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            self.finish_simulation()

        return self

    def run_tick(self) -> DiscreteGenerationPopulation:
        """Run a single simulation tick.

        Returns:
            Self for chaining.
        """
        return self.run(n_steps=1, record_every=self.record_every)

    def _run_python_dispatch(
        self,
        n_steps: int,
        record_every: int,
        finish: bool,
        clear_history_on_start: bool,
    ) -> DiscreteGenerationPopulation:
        """Run simulation ticks using the Python fallback path (no Numba)."""
        from natal.data import DiscretePopulationState

        self.ensure_hook_executor()

        if clear_history_on_start:
            self.clear_history()

        if record_every > 0 and (self._tick % record_every == 0):
            self._record_current_snapshot(allow_existing=True)

        was_stopped = False
        for _ in range(n_steps):
            # Each tick begins with the "first" hook window, giving
            # pre-reproduction intervention a chance to inspect or modify
            # state. A hook returning STOP short-circuits the whole run.
            if self.trigger_event("first", deme_id=-1) != RESULT_CONTINUE:
                was_stopped = True
                break

            # Produce offspring (age 0) from adults (age 1) using fecundity,
            # sex ratio, and density-dependent competition from the config.
            self.state.individual_count[:] = run_discrete_reproduction(
                self.state.individual_count,
                self.config,  # pyright: ignore[reportArgumentType]
            )

            # "Early" hooks fire after reproduction but before survival,
            # allowing interventions such as juvenile mortality modifiers.
            if self.trigger_event("early", deme_id=-1) != RESULT_CONTINUE:
                was_stopped = True
                break

            # Age-independent survival reduces counts based on viability
            # parameters, applied uniformly across all age classes.
            self.state.individual_count[:] = run_discrete_survival(
                self.state.individual_count,
                self.config,  # pyright: ignore[reportArgumentType]
            )

            # "Late" hooks fire after survival, the last opportunity to
            # inspect or alter state before the tick's life cycle ends.
            if self.trigger_event("late", deme_id=-1) != RESULT_CONTINUE:
                was_stopped = True
                break

            # Age all individuals by one year: offspring mature into
            # adults, and the previous adult cohort becomes the new
            # reproducing class.
            self.state.individual_count[:] = run_discrete_aging(
                self.state.individual_count,
            )

            self._tick += 1
            self._state = DiscretePopulationState(
                n_tick=int(self._tick),
                individual_count=self.state.individual_count,
            )

            if record_every > 0 and (self._tick % record_every == 0):
                self._record_current_snapshot(allow_existing=True)

        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            self.finish_simulation()

        return self

    def reset(self) -> None:
        """Reset tick, history, and population state to initial values."""
        self._tick = 0
        if self._history_obj is not None:
            self._history_obj.clear()
        self._finished = False
        # Guard against calls before __init__ finishes (e.g. during
        # BasePopulation.__init__ -> _initialize -> reset chain).
        if hasattr(self, '_initial_population_snapshot'):
            ind_copy, _, _ = self._initial_population_snapshot
            self._state = DiscretePopulationState.create(
                n_sexes=self.config.n_sexes,
                n_ages=self.config.n_ages,
                n_ztypes=self.config.n_ztypes,
                n_tick=0,
                individual_count=ind_copy.copy(),
            )

    def get_total_count(self) -> int:
        """Return the total number of individuals across all categories."""
        return int(round(np.sum(self.state.individual_count)))

    def get_female_count(self) -> int:
        """Return the total number of female individuals."""
        return int(round(np.sum(self.state.individual_count[int(Sex.FEMALE.value)])))

    def get_male_count(self) -> int:
        """Return the total number of male individuals."""
        return int(round(np.sum(self.state.individual_count[int(Sex.MALE.value)])))

    def clear_history(self) -> None:
        """Remove all recorded history snapshots."""
        self.history.clear()

    def export_state(self) -> NDArray[np.float64]:
        """Export the current state as a flat array.

        Returns:
            NDArray: Flattened state array.
        """
        return self.state.flatten_all()

    @property
    def config(self) -> DiscretePopulationConfig:
        """DiscretePopulationConfig: The current configuration."""
        return cast(DiscretePopulationConfig, super().config)

    def export_config(self) -> DiscretePopulationConfig:
        """Return a copy of the current configuration."""
        return self.config

    def import_config(self, config: DiscretePopulationConfig) -> None:
        """Replace the current configuration with *config*.

        Args:
            config: A ``DiscretePopulationConfig`` to install on this
                population.  A ``PopulationConfig`` or other type is
                rejected with ``TypeError``.

        Raises:
            TypeError: If *config* is not a ``DiscretePopulationConfig``.
            ValueError: If *config* violates the discrete-generation
                invariants (``n_ages == 2``, ``new_adult_age == 1``,
                ``adult_ages == [1]``).
        """
        self._config = _require_discrete_config(config)

    def import_state(
        self,
        state: Union[DiscretePopulationState, NDArray[np.float64], Dict[str, np.ndarray]],
    ) -> None:
        """Replace the current state and reset the history timeline.

        All validation happens before any mutation — a failed import leaves the
        population unchanged.

        Args:
            state: New state as a ``DiscretePopulationState``, flat ndarray,
                or dict with ``individual_count`` key.
        """
        # ── Phase 1: parse and validate all inputs ──
        if isinstance(state, np.ndarray):
            state_obj = parse_flattened_discrete_state(
                state,
                n_sexes=self.config.n_sexes,
                n_ages=self.config.n_ages,
                n_ztypes=self.config.n_ztypes,
            )
        elif isinstance(state, DiscretePopulationState):
            state_obj = state
        else:
            state_obj = DiscretePopulationState(
                n_tick=int(state.get("n_tick", self._tick)),
                individual_count=np.asarray(state["individual_count"], dtype=np.float64),
            )

        # ── Phase 2: commit atomically ──
        self._state = DiscretePopulationState(
            n_tick=int(state_obj.n_tick),
            individual_count=state_obj.individual_count.copy(),
        )
        self._tick = int(state_obj.n_tick)
        self.clear_history()

    @property
    def state(self) -> DiscretePopulationState:
        """DiscretePopulationState: The current population state.

        Raises:
            AttributeError: If the state has not been initialized.
        """
        if self._state is None:
            raise AttributeError("Population state has not been initialized.")
        return self._state

    def update(self) -> DiscreteConfigurator:
        """Return a ``DiscreteConfigurator`` for modifying this population's config."""
        return cast('DiscreteConfigurator', self._create_configurator())

    def __repr__(self) -> str:
        """Return a string summary of the discrete-generation population."""
        status = "Finished" if self._finished else "Active"
        return f"<DiscreteGenerationPopulation(name='{self.name}', tick={self.tick}, status={status})>"
