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
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from natal.base_population import BasePopulation
from natal.engine.discrete_generation_simulator import (
    run_discrete_aging,
    run_discrete_reproduction,
    run_discrete_survival,
)
from natal.genetic_entities import Genotype
from natal.genetic_structures import Species
from natal.hooks.types import RESULT_CONTINUE, RESULT_STOP
from natal.index_registry import IndexRegistry
from natal.population_config import (
    DiscretePopulationConfig,
    PopulationConfig,
)
from natal.population_state import (
    DiscretePopulationState,
    parse_flattened_discrete_state,
)
from natal.type_def import Sex

if TYPE_CHECKING:
    from natal.configurator import DiscreteConfigurator
    from natal.population_builder import DiscreteGenerationPopulationBuilder

__all__ = ["DiscreteGenerationPopulation"]


class DiscreteGenerationPopulation(BasePopulation[DiscretePopulationState]):
    """Population with strict non-overlapping generations."""

    @staticmethod
    def _to_discrete_config(config: object) -> DiscretePopulationConfig:
        """Normalize any config to ``DiscretePopulationConfig``.

        For the deprecated builder path, assembles a ``DiscretePopulationConfig``
        directly from a ``PopulationConfig``'s arrays — no public conversion
        function needed.  The modern Configurator path builds
        ``DiscretePopulationConfig`` independently via
        ``build_discrete_engine_config``.
        """
        if isinstance(config, DiscretePopulationConfig):
            return config._replace(
                n_ages=2,
                new_adult_age=1,
                adult_ages=np.array([1], dtype=np.int64),
            )
        if isinstance(config, PopulationConfig):
            cfg = config
            return DiscretePopulationConfig(
                stochastic=cfg.stochastic,
                continuous_sampling=cfg.continuous_sampling,
                n_sexes=int(cfg.n_sexes),
                n_ages=int(cfg.n_ages),
                n_ztypes=cfg.n_ztypes,
                n_haploid_genotypes=cfg.n_haploid_genotypes,
                n_glabs=cfg.n_glabs,
                n_slabs=cfg.n_slabs,
                female_age_based_fertility=cfg.female_age_based_fertility,
                viability_fitness=cfg.viability_fitness,
                fecundity_fitness=cfg.fecundity_fitness,
                zygote_viability_fitness=cfg.zygote_viability_fitness,
                sexual_selection_fitness=cfg.sexual_selection_fitness,
                age_based_relative_competition_strength=cfg.age_based_relative_competition_strength,
                eggs_per_female=cfg.eggs_per_female,
                fixed_egg_count=cfg.fixed_egg_count,
                sex_ratio=cfg.sex_ratio,
                sperm_displacement_rate=cfg.sperm_displacement_rate,
                female_adult_mating_rate=float(cfg.age_based_mating_rates[0, 1]),
                male_adult_mating_rate=float(cfg.age_based_mating_rates[1, 1]),
                reproduction_rate=float(cfg.age_based_reproduction_rates[1]),
                female_age0_survival=float(cfg.age_based_survival_rates[0, 0]),
                male_age0_survival=float(cfg.age_based_survival_rates[1, 0]),
                female_fertility=float(cfg.female_age_based_fertility[0]),
                zygotes_to_gametes_map=cfg.zygotes_to_gametes_map,
                gametes_to_zygotes_map=cfg.gametes_to_zygotes_map,
                offspring_tensor=cfg.offspring_tensor,
                meiosis_f=cfg.zygotes_to_gametes_map[0],
                meiosis_m=cfg.zygotes_to_gametes_map[1],
                fecundity_f=cfg.fecundity_fitness[0],
                fecundity_m=cfg.fecundity_fitness[1],
                viability_f=cfg.viability_fitness[0, 0, :],
                viability_m=cfg.viability_fitness[1, 0, :],
                has_sex_chromosomes=cfg.has_sex_chromosomes,
                female_genotype_compatibility=cfg.female_genotype_compatibility,
                male_genotype_compatibility=cfg.male_genotype_compatibility,
                female_only_by_sex_chrom=cfg.female_only_by_sex_chrom,
                male_only_by_sex_chrom=cfg.male_only_by_sex_chrom,
                juvenile_growth_mode=cfg.juvenile_growth_mode,
                carrying_capacity=cfg.carrying_capacity,
                expected_competition_strength=cfg.expected_competition_strength,
                expected_survival_rate=cfg.expected_survival_rate,
                low_density_growth_rate=cfg.low_density_growth_rate,
                generation_time=cfg.generation_time,
                new_adult_age=1,
                adult_ages=np.array([1], dtype=np.int64),
                initial_individual_count=cfg.initial_individual_count,
                initial_sperm_storage=cfg.initial_sperm_storage,
                hook_slot=int(cfg.hook_slot),
                extreme_speed_mode=0,
                custom=cfg.custom,
            )
        raise TypeError(f"Expected PopulationConfig or DiscretePopulationConfig, got {type(config)}")

    def __init__(
        self,
        species: Species,
        population_config: object,
        name: Optional[str] = None,
        index_registry: Optional[IndexRegistry] = None,
        initial_individual_count: Optional[
            Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]]
        ] = None,
        hooks: Optional[Dict[str, List[Tuple[Any, Optional[str], Optional[int]]]]] = None,
    ):
        """Initialize a discrete-generation population.

        Constructs the population from a species definition and configuration,
        sets up genotype registries and the initial age-by-genotype distribution,
        and registers hooks for event-driven intervention.
        """
        if name is None:
            name = "DiscreteGenerationPop"

        super().__init__(species, name, hooks=hooks or {})

        if index_registry is not None:
            self._index_registry = index_registry

        self._config = self._to_discrete_config(population_config)  # type: ignore[assignment]

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        n_sexes = self.config.n_sexes
        n_genotypes = self.config.n_ztypes
        n_ages = self.config.n_ages

        # Create an empty state first so we can check whether the config's
        # default initial_individual_count has compatible dimensions --
        # presets from genetic_presets.py often pre-size this to match.
        self._state = DiscretePopulationState.create(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_genotypes,
            n_tick=0,
            individual_count=np.zeros((n_sexes, n_ages, n_genotypes), dtype=np.float64),
        )

        cfg_init_ind = self.config.initial_individual_count
        if cfg_init_ind.shape == self.state.individual_count.shape:
            self.state.individual_count[:] = cfg_init_ind

        self._history_shape = (1 + n_sexes * n_ages * n_genotypes,)

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

    def _clone(self, name: str, config: PopulationConfig | DiscretePopulationConfig | None = None) -> Any:
        clone = super()._clone(name, config=config)  # type: ignore[arg-type]
        if config is not None:
            object.__setattr__(clone, "_config", self._to_discrete_config(config))  # type: ignore[assignment]
        return clone

    @classmethod
    @overload
    def setup(
        cls,
        species: Species,
        name: str = "DiscreteGenerationPop",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        *,
        legacy_path: Literal[True],
    ) -> DiscreteGenerationPopulationBuilder:
        """Create a builder for a discrete-generation population (legacy path)."""
        ...

    @classmethod
    @overload
    def setup(
        cls,
        species: Species,
        name: str = "DiscreteGenerationPop",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        *,
        compress: bool = False,
        declared_zygote_types: set[str] | set[int] | None = None,
        declared_genotypes: set[str] | set[int] | None = None,  # deprecated alias
        legacy_path: Literal[False] = False,
    ) -> DiscreteConfigurator:
        """Create a builder for a discrete-generation population."""
        ...

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
        declared_zygote_types: set[str] | set[int] | None = None,
        declared_genotypes: set[str] | set[int] | None = None,  # deprecated alias
        legacy_path: bool = False,
    ) -> DiscreteGenerationPopulationBuilder | DiscreteConfigurator:
        """Fluent population construction entry point.

        By default returns a ``DiscreteConfigurator`` (the new path).  Chain
        domain methods and end with ``.build()`` to create a Population.

        Pass ``legacy_path=True`` to use the classic Builder API.
        """
        if legacy_path:
            import warnings
            warnings.warn(
                "legacy_path=True is deprecated. Use the new Configurator API "
                "(default path) instead. See docs/ for migration guide.",
                FutureWarning, stacklevel=2,
            )
            from natal.population_builder import DiscreteGenerationPopulationBuilder

            return DiscreteGenerationPopulationBuilder(species).setup(
                name=name,
                stochastic=stochastic,
                continuous_sampling=continuous_sampling,
                fixed_egg_count=fixed_egg_count,
            )

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
                from natal.genetic_patterns import (
                    GenotypePatternParser,
                    ZygoteTypePattern,
                )

                if isinstance(genotype_key, str):
                    if self.species.unordered:
                        if "@" in genotype_key:
                            idx = genotype_key.rindex("@")
                            gt_part = genotype_key[:idx]
                            slab_part = genotype_key[idx:]
                        else:
                            gt_part = genotype_key
                            slab_part = ""
                        gt = self.species.get_genotype_from_str(gt_part)
                        gt = self.species.unordered_genotype(gt.maternal, gt.paternal)
                        genotype_key = str(gt) + slab_part
                    pattern = ZygoteTypePattern.parse(genotype_key, self.species)
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
            snapshot_init = state.individual_count.ravel().copy()
            self._history.append((state.n_tick, snapshot_init))
            self._enforce_history_limit()

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
                female_compat=cfg.female_genotype_compatibility,
                male_compat=cfg.male_genotype_compatibility,
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
                snapshot: np.ndarray = state.individual_count.ravel().copy()
                self._history.append((next_tick, snapshot))
                self._enforce_history_limit()

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
        from natal.population_state import DiscretePopulationState

        self.ensure_hook_executor()

        if clear_history_on_start:
            self.clear_history()

        if record_every > 0 and (self._tick % record_every == 0):
            self.create_history_snapshot()

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
                self.create_history_snapshot()

        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            self.finish_simulation()

        return self

    def reset(self) -> None:
        """Reset tick, history, and population state to initial values."""
        self._tick = 0
        self._history = []
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

    def get_history(self) -> np.ndarray:
        """Return recorded history as a 2-D ndarray (rows × flattened state)."""
        if len(self._history) == 0:
            return np.zeros((0, self._history_shape[0]), dtype=np.float64)
        return np.array([rec[1] for rec in self._history], dtype=np.float64)

    def clear_history(self) -> None:
        """Remove all recorded history snapshots."""
        self._history.clear()

    def create_history_snapshot(self) -> None:
        """Record a snapshot of the current state into history."""
        flattened = self.state.flatten_all()
        self._history.append((self._tick, flattened.copy()))
        self._enforce_history_limit()

    def export_state(self) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """Export the current state and history as flat arrays.

        Returns:
            Tuple of ``(state_flat, history)`` where history may be None.
        """
        state_flat = self.state.flatten_all()
        history = self.get_history() if self._history else None
        return state_flat, history

    @property
    def config(self) -> DiscretePopulationConfig:
        """DiscretePopulationConfig: The current configuration."""
        return cast(DiscretePopulationConfig, super().config)

    def export_config(self) -> DiscretePopulationConfig:
        """Return a copy of the current configuration."""
        return self.config

    def import_config(self, config: object) -> None:
        """Replace the current configuration with *config*."""
        self._config = self._to_discrete_config(config)  # type: ignore[assignment]

    def import_state(
        self,
        state: Union[DiscretePopulationState, NDArray[np.float64], Dict[str, np.ndarray]],
        history: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Replace the current state and optionally history.

        Args:
            state: New state as a ``DiscretePopulationState``, flat ndarray,
                or dict with ``individual_count`` key.
            history: Optional 2-D history array to restore.
        """
        # Accept three representations to support checkpoints from
        # export_state(), numpy serialization, and external tooling.
        assert isinstance(state, (np.ndarray, DiscretePopulationState, dict)), \
            "state must be a DiscretePopulationState, flattened ndarray, or dict"
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

        self._state = DiscretePopulationState(
            n_tick=int(state_obj.n_tick),
            individual_count=state_obj.individual_count.copy(),
        )
        self._tick = int(state_obj.n_tick)

        # Rebuild the history buffer row by row from a dense 2-D array as
        # exported by export_state / get_history.
        if history is not None and history.shape[0] > 0:
            self.clear_history()
            for row_idx in range(history.shape[0]):
                flat = history[row_idx, :]
                tick = int(flat[0])
                self._history.append((tick, flat.copy()))

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
        status = "Finished" if self._finished else "Active"
        return f"<DiscreteGenerationPopulation(name='{self.name}', tick={self.tick}, status={status})>"
