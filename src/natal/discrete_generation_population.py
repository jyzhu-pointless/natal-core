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
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from natal.base_population import BasePopulation
from natal.discrete_population_config import (
    DiscretePopulationConfig,
    from_population_config,
)
from natal.genetic_entities import Genotype
from natal.genetic_structures import Species
from natal.hooks.types import RESULT_CONTINUE
from natal.kernels.discrete_kernels import (
    run_discrete_aging,
    run_discrete_reproduction,
    run_discrete_survival,
)
from natal.population_config import PopulationConfig
from natal.population_state import (
    DiscretePopulationState,
    parse_flattened_discrete_state,
)
from natal.type_def import Sex

if TYPE_CHECKING:
    from natal.index_registry import IndexRegistry
    from natal.population_builder import DiscreteGenerationPopulationBuilder


__all__ = ["DiscreteGenerationPopulation"]


class DiscreteGenerationPopulation(BasePopulation[DiscretePopulationState]):
    """Population with strict non-overlapping generations."""

    @staticmethod
    def _to_discrete_config(config: object) -> DiscretePopulationConfig:
        """Normalize and convert any config to ``DiscretePopulationConfig``."""
        if isinstance(config, DiscretePopulationConfig):
            cfg = config._replace(
                n_ages=2,
                new_adult_age=1,
                adult_ages=np.array([1], dtype=np.int64),
            )
            return cfg
        if isinstance(config, PopulationConfig):
            normalized = config._replace(
                n_ages=2,
                new_adult_age=1,
                adult_ages=np.array([1], dtype=np.int64),
            )
            return from_population_config(normalized)
        raise TypeError(f"Expected PopulationConfig or DiscretePopulationConfig, got {type(config)}")

    def __init__(
        self,
        species: Species,
        population_config: object,
        name: Optional[str] = None,
        initial_individual_count: Optional[
            Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]]
        ] = None,
        hooks: Optional[Dict[str, List[Tuple[Any, Optional[str], Optional[int]]]]] = None,
    ):
        if name is None:
            name = "DiscreteGenerationPop"

        super().__init__(species, name, hooks=hooks or {})

        self._config = self._to_discrete_config(population_config)  # type: ignore[assignment]

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        n_sexes = self._config_nn.n_sexes
        n_genotypes = self._config_nn.n_genotypes
        n_ages = self._config_nn.n_ages

        self._state = DiscretePopulationState.create(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_genotypes=n_genotypes,
            n_tick=0,
            individual_count=np.zeros((n_sexes, n_ages, n_genotypes), dtype=np.float64),
        )

        cfg_init_ind = self._config_nn.get_scaled_initial_individual_count()
        if cfg_init_ind.shape == self._state_nn.individual_count.shape:
            self._state_nn.individual_count[:] = cfg_init_ind

        self._history_shape = (1 + n_sexes * n_ages * n_genotypes,)

        if initial_individual_count is not None:
            self._state_nn.individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)

        self._initial_population_snapshot = (
            self._state_nn.individual_count.copy(),
            None,
            None,
        )

        self._finalize_hooks()

    def _refresh_modifier_maps(self) -> None:
        """Rebuild genotype/gamete/zygote maps and offspring_tensor from registered modifiers."""
        if self._config is None or self._registry is None:
            return

        haploid_genotypes = self._registry.index_to_haplo
        diploid_genotypes = self._registry.index_to_genotype
        if not haploid_genotypes or not diploid_genotypes:
            return

        from natal.modifiers import build_modifier_wrappers
        from natal.population_config import (
            initialize_gamete_map,
            initialize_zygote_map,
        )

        n_glabs = self._config.n_glabs
        gamete_funcs, zygote_funcs = build_modifier_wrappers(
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            population=self,
            index_registry=self._index_registry,
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
        )

        z2g = initialize_gamete_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            gamete_modifiers=gamete_funcs,
        )
        g2z = initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            zygote_modifiers=zygote_funcs,
        )

        import natal.kernels.algorithms as _alg

        self._config = self._config._replace(  # type: ignore[assignment]
            genotype_to_gametes_map=z2g,
            gametes_to_zygote_map=g2z,
            offspring_tensor=_alg.compute_offspring_probability_tensor(
                meiosis_f=z2g[0], meiosis_m=z2g[1],
                haplo_to_genotype_map=g2z,
                n_genotypes=self._config.n_genotypes,
                n_haplogenotypes=self._config.n_haploid_genotypes,
                n_glabs=n_glabs,
            ),
        )

    def _clone(self, name: str, config: PopulationConfig | DiscretePopulationConfig | None = None) -> Any:
        clone = super()._clone(name, config=config)  # type: ignore[arg-type]
        if config is not None:
            object.__setattr__(clone, "_config", self._to_discrete_config(config))  # type: ignore[assignment]
        return clone

    @classmethod
    def setup(
        cls,
        species: Species,
        name: str = "DiscreteGenerationPop",
        stochastic: bool = True,
        use_continuous_sampling: bool = False,
        use_fixed_egg_count: bool = False,
    ) -> DiscreteGenerationPopulationBuilder:
        from natal.population_builder import DiscreteGenerationPopulationBuilder

        builder = DiscreteGenerationPopulationBuilder(species)
        builder.setup(
            name=name,
            stochastic=stochastic,
            use_continuous_sampling=use_continuous_sampling,
            use_fixed_egg_count=use_fixed_egg_count,
        )
        return builder

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
        self._state_nn.individual_count.fill(0.0)
        for sex_key, genotype_dist in distribution.items():
            sex_key_norm = sex_key.lower().strip()
            if sex_key_norm == "female":
                sex_idx = int(Sex.FEMALE.value)
            elif sex_key_norm == "male":
                sex_idx = int(Sex.MALE.value)
            else:
                raise ValueError(f"Sex must be 'female' or 'male', got '{sex_key}'")
            for genotype_key, age_data in genotype_dist.items():
                genotype = self._resolve_genotype_key(genotype_key)
                genotype_idx = self._registry_nn.genotype_to_index[genotype]
                age0_count, age1_count = self._resolve_age_distribution(age_data)
                self._state_nn.individual_count[sex_idx, 0, genotype_idx] = age0_count
                self._state_nn.individual_count[sex_idx, 1, genotype_idx] = age1_count

    def run(
        self,
        n_steps: int = 1,
        record_every: Optional[int] = None,
        finish: bool = False,
        clear_history_on_start: bool = False,
    ) -> DiscreteGenerationPopulation:
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. Cannot run() again after finish=True."
            )
        if record_every is None:
            record_every = self.record_every
        if self.should_use_python_dispatch():
            return self._run_python_dispatch(
                n_steps=n_steps,
                record_every=record_every,
                finish=finish,
                clear_history_on_start=clear_history_on_start,
            )

        hooks = self.get_compiled_event_hooks()
        assert hooks.registry is not None, "hooks.registry should always be initialized"

        if hooks.run_discrete_fn is None:
            return self._run_python_dispatch(
                n_steps=n_steps,
                record_every=record_every,
                finish=finish,
                clear_history_on_start=clear_history_on_start,
            )

        obs_mask = self._observation_mask
        n_obs = len(self._observation.labels) if self._observation is not None else 0

        final_state_tuple, history_new, was_stopped = hooks.run_discrete_fn(
            state=self._state_nn,
            config=self._config_nn,
            registry=hooks.registry,
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

        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            self.finish_simulation()

        return self

    def run_tick(self) -> DiscreteGenerationPopulation:
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
            if self.trigger_event("first", deme_id=-1) != RESULT_CONTINUE:
                was_stopped = True
                break

            self._state_nn.individual_count[:] = run_discrete_reproduction(
                self._state_nn.individual_count,
                self._config_nn,  # pyright: ignore[reportArgumentType]
            )

            if self.trigger_event("early", deme_id=-1) != RESULT_CONTINUE:
                was_stopped = True
                break

            self._state_nn.individual_count[:] = run_discrete_survival(
                self._state_nn.individual_count,
                self._config_nn,  # pyright: ignore[reportArgumentType]
            )

            if self.trigger_event("late", deme_id=-1) != RESULT_CONTINUE:
                was_stopped = True
                break

            self._state_nn.individual_count[:] = run_discrete_aging(
                self._state_nn.individual_count,
            )

            self._tick += 1
            self._state = DiscretePopulationState(
                n_tick=int(self._tick),
                individual_count=self._state_nn.individual_count,
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
        self._tick = 0
        self._history = []
        self._finished = False
        if hasattr(self, '_initial_population_snapshot'):
            ind_copy, _, _ = self._initial_population_snapshot
            self._state = DiscretePopulationState.create(
                n_sexes=self._config_nn.n_sexes,
                n_ages=self._config_nn.n_ages,
                n_genotypes=self._config_nn.n_genotypes,
                n_tick=0,
                individual_count=ind_copy.copy(),
            )

    def get_total_count(self) -> int:
        return int(round(np.sum(self._state_nn.individual_count)))

    def get_female_count(self) -> int:
        return int(round(np.sum(self._state_nn.individual_count[int(Sex.FEMALE.value)])))

    def get_male_count(self) -> int:
        return int(round(np.sum(self._state_nn.individual_count[int(Sex.MALE.value)])))

    def get_history(self) -> np.ndarray:
        if len(self._history) == 0:
            return np.zeros((0, self._history_shape[0]), dtype=np.float64)
        return np.array([rec[1] for rec in self._history], dtype=np.float64)

    def clear_history(self) -> None:
        self._history.clear()

    def create_history_snapshot(self) -> None:
        flattened = self._state_nn.flatten_all()
        self._history.append((self._tick, flattened.copy()))
        self._enforce_history_limit()

    def export_state(self) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        state_flat = self._state_nn.flatten_all()
        history = self.get_history() if self._history else None
        return state_flat, history

    def export_config(self) -> DiscretePopulationConfig:
        return self._config_nn

    def import_config(self, config: object) -> None:
        self._config = self._to_discrete_config(config)  # type: ignore[assignment]

    def import_state(
        self,
        state: Union[DiscretePopulationState, NDArray[np.float64], Dict[str, np.ndarray]],
        history: Optional[NDArray[np.float64]] = None,
    ) -> None:
        assert isinstance(state, (np.ndarray, DiscretePopulationState, dict)), \
            "state must be a DiscretePopulationState, flattened ndarray, or dict"
        if isinstance(state, np.ndarray):
            state_obj = parse_flattened_discrete_state(
                state,
                n_sexes=self._config_nn.n_sexes,
                n_ages=self._config_nn.n_ages,
                n_genotypes=self._config_nn.n_genotypes,
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

        if history is not None and history.shape[0] > 0:
            self.clear_history()
            for row_idx in range(history.shape[0]):
                flat = history[row_idx, :]
                tick = int(flat[0])
                self._history.append((tick, flat.copy()))

    @property
    def _state_nn(self) -> DiscretePopulationState:
        return self._require_state()

    @property
    def _config_nn(self) -> DiscretePopulationConfig:
        config = self._require_config()
        assert isinstance(config, DiscretePopulationConfig)
        return config

    @property
    def _registry_nn(self) -> IndexRegistry:
        return self._require_registry()

    def __repr__(self) -> str:
        status = "Finished" if self._finished else "Active"
        return f"<DiscreteGenerationPopulation(name='{self.name}', tick={self.tick}, status={status})>"
