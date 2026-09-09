"""Age-structured population models.

This module implements age-structured (overlapping generation) population
models and utilities for survival, reproduction, juvenile recruitment, and
fitness management.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.frontend.data import ModelDraft, PopulationState
from natal.frontend.genetics import Genotype, Species
from natal.frontend.population.base import BasePopulation
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.utils.types import Sex

if TYPE_CHECKING:
    from natal.backends.rust.rust_backend import RustLifecycleBackend
    from natal.frontend.configurator import Configurator

__all__ = ["AgeStructuredPopulation"]

# Type alias for hooks
HookCallback = Callable[..., object]
# =============================================================================
# Age-structured population model (based on BasePopulation)
# =============================================================================


class AgeStructuredPopulation(BasePopulation[PopulationState]):
    """Age-structured population model (overlapping generations).

    An age-structured population built on ``BasePopulation`` and
    ``PopulationState``. Supports age-dependent survival and fecundity,
    juvenile recruitment modes, optional sperm-storage mechanics, and a
    hook/modifier system for user extensions.

    Attributes:
        snapshots (Dict[str, object]): Storage for custom state snapshots.
    """

    def __init__(
        self,
        species: Species,
        population_config: ModelDraft,
        name: Optional[str] = None,
        index_registry: Optional[IndexRegistry] = None,
        initial_individual_count: Optional[
            Mapping[
                str, Mapping[Union[Genotype, str], Union[List[int], Dict[int, int]]]
            ]
        ] = None,
        initial_sperm_storage: Optional[
            Mapping[
                Union[Genotype, str],
                Mapping[
                    Union[Genotype, str], Union[Dict[int, float], List[float], float]
                ],
            ]
        ] = None,
        hook_items: Optional[List[object]] = None,
    ):
        """Initialize an age-structured population instance using a ModelDraft.

        Args:
            species: Species object describing genetic architecture.
            population_config: Fully initialized ModelDraft instance.
            name: Human-readable population name. If None, uses "AgeStructuredPop".
            initial_individual_count: Initial population distribution.
                Format: {sex: {genotype: counts_by_age}}
            initial_sperm_storage: Initial sperm storage state (if supported).
            hook_items: Hook registrations (``Op`` objects, ``@hook``-
                decorated functions, or single-parameter callables).

        Examples:
            >>> pop_config = build_population_config(species, ...)
            >>> pop = AgeStructuredPopulation(
            ...     species,
            ...     pop_config,
            ...     name="MyPop",
            ...     initial_individual_count={...}
            ... )
        """
        if name is None:
            name = "AgeStructuredPop"

        super().__init__(species, name, hook_items=hook_items)

        if index_registry is not None:
            self._index_registry = index_registry

        self._config = population_config

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        self._state = PopulationState.create(
            n_ztypes=population_config.n_ztypes,
            n_sexes=population_config.n_sexes,
            n_ages=population_config.n_ages,
        )

        # Initialize from builder-injected config arrays if available.
        cfg_init_ind = population_config.initial_individual_count
        if cfg_init_ind.shape == self._live_state().individual_count.shape:
            self._live_state().individual_count[:] = cfg_init_ind
        cfg_init_sperm = population_config.initial_sperm_storage
        if cfg_init_sperm.shape == self._live_state().sperm_storage.shape:
            self._live_state().sperm_storage[:] = cfg_init_sperm

        self.snapshots = {}
        self._rust_run_active = False
        self._rust_lifecycle_backend: RustLifecycleBackend | None = None
        self._rust_backend_seed: int | None = None
        # Structural changes (hook registration, modifier maps, blueprint
        # flags) rebuild the session before the next run; value changes go
        # straight to the session through the writers and the run-boundary
        # ecology flush.
        self._rust_needs_rebuild: bool = False

        if initial_individual_count is not None:
            self._live_state().individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)

        if initial_sperm_storage is not None:
            # TODO: add population_config.use_sperm_storage
            self._distribute_initial_sperm_storage(species, initial_sperm_storage)

        self._initial_population_snapshot = (
            self._live_state().individual_count.copy(),
            self._live_state().sperm_storage.copy(),
            None,
        )

        self._initialize_registry()
        self._finalize_hooks()

        # Build self-describing history schema (frozen at construction).
        self._init_history_schema(
            kind="age_structured",
            n_demes=1,
            has_sperm_storage=True,
        )

    @classmethod
    def setup(
        cls,
        species: Species,
        name: str = "AgeStructuredPop",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        *,
        compress: bool = False,
        declared_zygote_types: Sequence[str] | Sequence[int] | None = None,
        declared_genotypes: Sequence[str]
        | Sequence[int]
        | None = None,  # deprecated alias
    ) -> Configurator:
        """Start building an age-structured population with overlapping generations.

        This is the fluent entry point for constructing an
        ``AgeStructuredPopulation``.  It returns the unified
        ``Configurator`` that you configure by chaining domain methods
        (``initial_state()``, ``reproduction()``, ``competition()``, etc.)
        and finalize with ``build()``.

        Args:
            species: Species object describing the population's genetic
                architecture (chromosomes, loci, alleles).
            name: Human-readable name for the population.
                Defaults to ``"AgeStructuredPop"``.
            stochastic: If ``False``, use deterministic (median) outcomes for
                reproduction and survival. Defaults to ``True``.
            continuous_sampling: If ``True``, sample from continuous
                distributions instead of discrete counts.
                Defaults to ``False``.
            fixed_egg_count: If ``True``, disable Poisson noise on egg counts
                so each female produces exactly the specified number of eggs.
                Defaults to ``False``.
            compress: If ``True``, enable full index compression at build
                time, pruning unreachable GTypes and ZTypes to shrink
                internal arrays. Defaults to ``False``.
            declared_zygote_types: Optional sequence of genotype strings
                (``"WT|WT"``) or integer indices that are treated as
                reachable even if absent from the initial state.  Use this
                to prevent compression from pruning genotypes that may
                appear later via hooks or runtime presets.
            declared_genotypes: Deprecated alias for
                *declared_zygote_types*.

        Returns:
            A ``Configurator`` ready for domain-method chaining.
            Call ``.build()`` to produce an ``AgeStructuredPopulation``.

        Raises:
            ValueError: If both ``declared_zygote_types`` and
                ``declared_genotypes`` (deprecated alias) are specified
                simultaneously.
        """
        from natal.frontend.configurator import Configurator

        if declared_genotypes is not None:
            if declared_zygote_types is not None:
                raise ValueError(
                    "Cannot specify both declared_zygote_types and "
                    "declared_genotypes (deprecated alias)."
                )
            declared_zygote_types = declared_genotypes
        return Configurator.from_species(species).setup(
            name=name,
            stochastic=stochastic,
            continuous_sampling=continuous_sampling,
            fixed_egg_count=fixed_egg_count,
            compress=compress,
            declared_zygote_types=declared_zygote_types,
        )

    def _distribute_initial_population(
        self, distribution: Mapping[str, Mapping[Union[Genotype, str], object]]
    ) -> None:
        """Distribute initial population from a specification dictionary.

        Args:
            distribution: Format {sex: {genotype: age_counts}}
                where age_counts can be a list or dict of age -> count.

        Raises:
            ValueError: If sex key is invalid.
            TypeError: If age data is not a list or dict.
        """
        self._live_state().individual_count.fill(0.0)
        for sex_key, genotype_dist in distribution.items():
            sex_key_norm = sex_key.lower().strip()
            if sex_key_norm == "female":
                sex_idx = int(Sex.FEMALE.value)
            elif sex_key_norm == "male":
                sex_idx = int(Sex.MALE.value)
            else:
                raise ValueError(f"Sex must be 'female' or 'male', got '{sex_key}'")

            for genotype_key, age_data in genotype_dist.items():
                from natal.frontend.patterns import (
                    GenotypePatternParser,
                    ZygoteTypePattern,
                )

                if isinstance(genotype_key, str):
                    pattern = ZygoteTypePattern.from_slab_key(
                        genotype_key, self.species
                    )
                else:
                    parser = GenotypePatternParser(self.species)
                    pattern = ZygoteTypePattern(
                        parser.parse(str(genotype_key)), slab=None
                    )

                z_idx = self.registry.resolve_default_ztype_index(pattern)

                if isinstance(age_data, list):
                    for age, raw_count in enumerate(cast(List[object], age_data)):
                        if not isinstance(raw_count, (int, float)) or isinstance(
                            raw_count, bool
                        ):
                            raise TypeError(
                                f"Age count must be numeric, got {type(raw_count)}"
                            )
                        count = float(raw_count)
                        if age < self.config.n_ages and count > 0:
                            self._live_state().individual_count[sex_idx, age, z_idx] = (
                                count
                            )
                elif isinstance(age_data, dict):
                    for age_raw, raw_count in cast(
                        Dict[object, object], age_data
                    ).items():
                        if not isinstance(age_raw, int):
                            raise TypeError(f"Age key must be int, got {type(age_raw)}")
                        if not isinstance(raw_count, (int, float)) or isinstance(
                            raw_count, bool
                        ):
                            raise TypeError(
                                f"Age count must be numeric, got {type(raw_count)}"
                            )
                        age = age_raw
                        count = float(raw_count)
                        if age < self.config.n_ages and count > 0:
                            self._live_state().individual_count[sex_idx, age, z_idx] = (
                                count
                            )
                else:
                    raise TypeError(
                        f"age_data must be a list or dict, got {type(age_data)}"
                    )

    def _distribute_initial_sperm_storage(
        self,
        species: Species,
        sperm_storage_dist: Mapping[
            Union[Genotype, str],
            Mapping[
                Union[Genotype, str],
                Union[Dict[int, float], List[float], Tuple[float, ...], float, int],
            ],
        ],
    ) -> None:
        """Populate the internal sperm storage from user-provided initial distribution.

        Note:
            Supported formats for age_data (innermost value):
            - Dict[int, float]: Sparse mapping {age: count, ...}
            - List[float]: Dense list [count_age0, count_age1, ...]
            - float/int: Scalar value applied to all adult ages (>= new_adult_age)

        Args:
            species: Species object for genotype parsing.
            sperm_storage_dist: Mapping of {female_genotype: {male_genotype: age_data}}.

        Raises:
            TypeError: If genotype keys or age data have incorrect types.
            ValueError: If sperm counts or ages are out of range.
        """
        self._live_state().sperm_storage.fill(0.0)
        from natal.frontend.patterns import GenotypePatternParser, ZygoteTypePattern

        for female_key, male_dict in sperm_storage_dist.items():
            assert isinstance(female_key, (str, Genotype)), (
                f"Female genotype key must be Genotype or str, got {type(female_key)}"
            )

            if isinstance(female_key, str):
                female_pattern = ZygoteTypePattern.from_slab_key(female_key, species)
            else:
                parser = GenotypePatternParser(species)
                female_pattern = ZygoteTypePattern(
                    parser.parse(str(female_key)), slab=None
                )

            f_z = self.registry.resolve_default_ztype_index(female_pattern)

            for male_key, age_data in male_dict.items():
                assert isinstance(male_key, (str, Genotype)), (
                    f"Male genotype key must be Genotype or str, got {type(male_key)}"
                )

                if isinstance(male_key, str):
                    male_pattern = ZygoteTypePattern.from_slab_key(male_key, species)
                else:
                    parser = GenotypePatternParser(species)
                    male_pattern = ZygoteTypePattern(
                        parser.parse(str(male_key)), slab=None
                    )

                m_z = self.registry.resolve_default_ztype_index(male_pattern)

                assert isinstance(age_data, (dict, list, tuple, int, float)), (
                    f"Age data must be Dict, List, or numeric scalar, got {type(age_data)}"
                )

                # Parse age_data: supports multiple formats
                if isinstance(age_data, dict):
                    # Dict format: {age: count, ...}
                    for age_raw, raw_count in cast(
                        Dict[object, object], age_data
                    ).items():
                        if not isinstance(age_raw, int):
                            raise TypeError(f"Age must be int, got {type(age_raw)}")
                        if not isinstance(raw_count, (int, float)) or isinstance(
                            raw_count, bool
                        ):
                            raise TypeError(
                                f"Sperm count must be numeric, got {type(raw_count)}"
                            )
                        age = age_raw
                        count = float(raw_count)
                        if age < 0 or age >= self.n_ages:
                            raise ValueError(
                                f"Age {age} out of range [0, {self.n_ages})"
                            )
                        if count < 0:
                            raise ValueError(
                                f"Sperm count must be non-negative, got {count}"
                            )
                        if count > 0:
                            self._live_state().sperm_storage[age, f_z, m_z] = count

                elif isinstance(age_data, list):
                    # List format: [count_age0, count_age1, ...]
                    for age, raw_count in enumerate(cast(List[object], age_data)):
                        if not isinstance(raw_count, (int, float)) or isinstance(
                            raw_count, bool
                        ):
                            raise TypeError(
                                f"Sperm count must be numeric, got {type(raw_count)}"
                            )
                        count = float(raw_count)
                        if age >= self.n_ages:
                            break
                        if count < 0:
                            raise ValueError(
                                f"Sperm count must be non-negative, got {count}"
                            )
                        if count > 0:
                            self._live_state().sperm_storage[age, f_z, m_z] = count

                elif isinstance(age_data, tuple):
                    # Tuple format: (count_age0, count_age1, ...)
                    for age, raw_count in enumerate(cast(Tuple[object, ...], age_data)):
                        if not isinstance(raw_count, (int, float)) or isinstance(
                            raw_count, bool
                        ):
                            raise TypeError(
                                f"Sperm count must be numeric, got {type(raw_count)}"
                            )
                        count = float(raw_count)
                        if age >= self.n_ages:
                            break
                        if count < 0:
                            raise ValueError(
                                f"Sperm count must be non-negative, got {count}"
                            )
                        if count > 0:
                            self._live_state().sperm_storage[age, f_z, m_z] = count

                else:
                    # Scalar format: apply to all adult ages
                    if age_data < 0:
                        raise ValueError(
                            f"Sperm count must be non-negative, got {age_data}"
                        )
                    if age_data > 0:
                        for age in range(self.new_adult_age, self.n_ages):
                            self._live_state().sperm_storage[age, f_z, m_z] = float(
                                age_data
                            )

    def _refresh_state_cache_from_session(self) -> None:
        """Pull the session-owned state into the local cache.

        The Rust session owns the counts, sperm storage, and tick; the
        cache container is rebuilt from a fresh snapshot and the mirror
        tick follows.
        """
        backend = self._rust_lifecycle_backend
        if backend is None:
            return
        tick, ind_flat, sperm_flat = backend.state_snapshot()
        n_ages = int(self.config.n_ages)
        n_ztypes = int(self.config.n_ztypes)
        self._state = PopulationState(
            n_tick=int(tick),
            individual_count=ind_flat.reshape(2, n_ages, n_ztypes).copy(),
            sperm_storage=sperm_flat.reshape(n_ages, n_ztypes, n_ztypes).copy(),
        )
        self._tick = int(tick)
        self._state_cache_stale = False

    def _snapshot_state(self) -> PopulationState:
        """Copy the live state container for the public :attr:`state` snapshot.

        Returns:
            A fresh ``PopulationState`` with copied count and sperm
            arrays; writes through it never reach the engine.
        """
        src = self._live_state()  # lazily pulls the session snapshot under Rust
        assert src is not None  # the base property guards initialization
        return PopulationState(
            n_tick=int(src.n_tick),
            individual_count=src.individual_count.copy(),
            sperm_storage=src.sperm_storage.copy(),
        )

    def reset(self) -> None:
        """Reset the population to its initial state.

        Restores individual counts and sperm storage to original values.
        """
        self._require_standalone_owner("reset")
        self._tick = 0
        if self._history_obj is not None:
            self._history_obj.clear()
        self._finished = False
        self._failed = False
        if hasattr(self, "_initial_population_snapshot"):
            ind_copy, sperm_copy, _ = self._initial_population_snapshot
            assert sperm_copy is not None

            self._state = PopulationState.create(
                n_ztypes=self.config.n_ztypes,
                n_sexes=self.config.n_sexes,
                n_ages=self.config.n_ages,
                n_tick=0,
                individual_count=ind_copy.copy(),
                sperm_storage=sperm_copy.copy(),
            )
        backend = self._rust_lifecycle_backend
        if backend is not None and self._state is not None:
            # The session owns the runtime state: the reset container
            # becomes the new session state, and RNG restarts from the
            # most recently configured explicit seed.
            backend.set_state(self._state)
            backend.reseed(int(self._rust_backend_seed or 0))
            backend.clear_checkpoints()
            self._state_cache_stale = False

    @property
    def n_ages(self) -> int:
        """int: Number of age classes in this population."""
        return self.config.n_ages

    @property
    def new_adult_age(self) -> int:
        """int: Minimum age at which individuals are considered adults."""
        return self.config.new_adult_age

    def get_total_count(self) -> float:
        """Return the total number of individuals in the population.

        Returns:
            float: Grand total across all sexes, ages, and genotypes.
        """
        return self._live_state().individual_count.sum()

    def get_female_count(self) -> float:
        """Return the total number of female individuals.

        Returns:
            float: Sum of all female individual counts.
        """
        return self._live_state().individual_count[Sex.FEMALE.value, :, :].sum()

    def get_male_count(self) -> float:
        """Return the total number of male individuals.

        Returns:
            float: Sum of all male individual counts.
        """
        return self._live_state().individual_count[Sex.MALE.value, :, :].sum()

    def get_adult_count(self, sex: str = "both") -> int:
        """Return the number of adult individuals for the given sex.

        Args:
            sex: One of ``'female'``, ``'male'``, or ``'both'`` (aliases accepted).

        Returns:
            float: Total number of adults for the requested sex(es).

        Raises:
            ValueError: If the sex identifier is not recognized.
        """
        if sex not in ("female", "male", "both", "F", "M"):
            raise ValueError(f"sex must be 'female', 'male', or 'both', got '{sex}'")

        total = 0

        if sex in ("female", "F", "both"):
            total += (
                self._live_state()
                .individual_count[Sex.FEMALE.value, self.new_adult_age : self.n_ages, :]
                .sum()
            )

        if sex in ("male", "M", "both"):
            total += (
                self._live_state()
                .individual_count[Sex.MALE.value, self.new_adult_age : self.n_ages, :]
                .sum()
            )

        return int(total)

    @property
    def config(self) -> ModelDraft:
        """ModelDraft: The current configuration."""
        return super().config

    # ========================================================================
    # State export/import (simulator interface)
    # ========================================================================

    def export_config(self) -> ModelDraft:
        """Export a detached model configuration.

        Returns:
            ModelDraft: A copy of the current population configuration.
        """
        return self.config

    def import_config(self, config: ModelDraft) -> None:
        """Import configuration into the population.

        Args:
            config: Model configuration to install.
        """
        self._require_standalone_owner("import_config")
        # Configuration is usually read-only (used by run_tick),
        # kept here for completeness.
        self._install_config(config)

    def clear_history(self) -> None:
        """Clear history rows and the paired session checkpoints."""
        self._require_standalone_owner("clear_history")
        super().clear_history()

    def export_state(self) -> NDArray[np.float64]:
        """Export population state as a flattened array.

        Returns:
            NDArray: Flattened state array ``[n_tick, ind_count.ravel(), sperm_storage.ravel()]``.
        """
        return self._live_state().flatten_all()

    def import_state(
        self,
        state: Union[
            PopulationState,
            NDArray[np.float64],
            Dict[str, np.ndarray],
            Tuple[np.ndarray, np.ndarray],
        ],
    ) -> None:
        """Import state and reset the history timeline.

        All validation happens before any mutation — a failed import leaves the
        population unchanged.

        Args:
            state: Flattened array, PopulationState object, or data dictionary.
        """
        self._require_standalone_owner("import_state")
        from natal.frontend.data import PopulationState, parse_flattened_state

        n_sexes, n_ages, n_ztypes = self._live_state().individual_count.shape

        # ── Phase 1: parse and validate all inputs ──
        if isinstance(state, np.ndarray):
            state_obj = parse_flattened_state(state, n_sexes, n_ages, n_ztypes)
        elif isinstance(state, PopulationState):
            state_obj = state
        elif isinstance(state, dict):
            state_obj = PopulationState(
                n_tick=int(state.get("n_tick", self._tick)),
                individual_count=np.asarray(
                    state["individual_count"], dtype=np.float64
                ),
                sperm_storage=np.asarray(state["sperm_storage"], dtype=np.float64),
            )
        else:
            if len(state) != 2:
                raise ValueError(f"Tuple state must have length 2, got {len(state)}")
            state_obj = PopulationState(
                n_tick=self._tick,
                individual_count=np.asarray(state[0], dtype=np.float64),
                sperm_storage=np.asarray(state[1], dtype=np.float64),
            )

        expected_individual_shape = self._live_state().individual_count.shape
        if state_obj.individual_count.shape != expected_individual_shape:
            raise ValueError(
                "individual_count shape mismatch: expected "
                f"{expected_individual_shape}, got {state_obj.individual_count.shape}"
            )
        expected_sperm_shape = self._live_state().sperm_storage.shape
        if state_obj.sperm_storage.shape != expected_sperm_shape:
            raise ValueError(
                "sperm_storage shape mismatch: expected "
                f"{expected_sperm_shape}, got {state_obj.sperm_storage.shape}"
            )

        self._validate_import_values(state_obj)
        candidate = PopulationState(
            n_tick=int(state_obj.n_tick),
            individual_count=state_obj.individual_count.copy(),
            sperm_storage=state_obj.sperm_storage.copy(),
        )
        backend = self._rust_lifecycle_backend
        if backend is not None:
            # Native validation and commit must succeed before changing even
            # the Python cache, tick, or history associated with this owner.
            backend.set_state(candidate)
        self._state = candidate
        self._tick = candidate.n_tick
        self._state_cache_stale = False
        self.clear_history()

    # ========================================================================
    # History restoration helpers
    # ========================================================================

    def restore_checkpoint(self, tick: int) -> None:
        """Restore the population to a specific raw-history tick.

        Args:
            tick: The target tick number.

        Raises:
            ValueError: If no record is found for the specified tick.
        """
        self._require_standalone_owner("restore_checkpoint")
        super().restore_checkpoint(tick)

    # ========================================================================
    # Hooks system
    # ========================================================================

    # [Allowed hook events]
    #
    #     Before simulation:  [initialization]
    #                                |
    #                                v
    #     For tick in T:    |-------------------------------------------------------------------------|
    #                       |     [first] -->  reproduction  --> [early] -->  survival  --> [late]    |
    #                       |        ^                                                         |      |
    #                       |        |<--------------------------------------------------------|      |
    #                       |-------------------------------------------------------------------------|
    #                                |
    #                                v
    #     After simulation:      [finish]
    #

    # ========================================================================
    # Evolution logic
    # ========================================================================

    def _get_kernel_config(self) -> Tuple[Any, ...]:
        """Build configuration tuple for simulation engine.

        Returns:
            tuple: An engine-compatible configuration tuple.
        """
        return self.export_config()

    def _initialize_session(self, seed: int = 0) -> AgeStructuredPopulation:
        """Enable the Rust lifecycle backend for subsequent runs.

        CSR declarative hooks travel to Rust inside the ``HookProgram``;
        single-parameter Python callbacks are bridged through the
        session's ``python_callbacks`` channel (fired at event boundaries
        interleaved with the CSR slots by priority, state copies written back per call).

        The current config is materialized into the contract pair once; the
        session owns its copies.  Value changes flow straight to the
        session through the writers (no rebuild, no RNG reset); in-run
        writes defer to the draft and the run boundary flushes them.
        Structural changes (hooks, blueprint flags) rebuild the session
        before the next run.

        Args:
            seed: Seed for the Rust RNG used in stochastic simulations.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the Rust extension is unavailable.
        """
        self._require_standalone_owner("_initialize_session")
        from natal.backends.rust.rust_backend import (
            RustLifecycleBackend,
            rust_backend_available,
        )

        if not rust_backend_available():
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "before enabling the Rust backend."
            )
        hook_program = self._build_hook_program()
        self._run_program = self._run_program._replace(hooks=hook_program)
        backend = RustLifecycleBackend(
            # Materialization copies the first owned draft itself; only an
            # existing session needs a current native parameter snapshot.
            self._config if self._rust_lifecycle_backend is None and self._config is not None else self.config,
            hook_program,
            seed=seed,
        )
        self._register_rust_callbacks(backend)
        # Capture the state BEFORE the field switch: under a structural
        # rebuild the lazy pull must read the OLD session, not the freshly
        # constructed one whose state is the blueprint initial population
        # again.
        state_to_install = self._live_state()
        self._rust_lifecycle_backend = backend
        self._rust_backend_seed = seed
        self._rust_needs_rebuild = False
        # The session owns the state from here on: install the
        # live Python state so the freshly seeded RNG continues from the
        # population's current counts and tick.
        backend.set_state(state_to_install)
        self._state_cache_stale = False
        return self

    def _run_startup_sync(self) -> None:
        """Install changed execution flags and hooks without replacing state or RNG."""
        if not self._rust_needs_rebuild:
            return
        backend = self._rust_lifecycle_backend
        if backend is None:
            raise RuntimeError("The population session has not been initialized.")
        config = self.config
        program = self._build_hook_program()
        backend.configure_program(program, config)
        self._register_rust_callbacks(backend)
        self._run_program = self._run_program._replace(hooks=program)
        self._rust_needs_rebuild = False

    def _run_rust_lifecycle(
        self,
        n_steps: int,
        record_every: int,
        finish: bool,
        clear_history_on_start: bool,
    ) -> AgeStructuredPopulation:
        """Run the Rust batch kernel and commit its history rows."""
        self._run_startup_sync()
        backend = self._rust_lifecycle_backend
        if backend is None:
            # Instances that skipped ``build()`` (clones, direct
            # construction) lazily create their engine session at the
            # first run boundary; ``_initialize_session`` installs the
            # current live state into the fresh session.
            self._initialize_session(seed=int(self._rust_backend_seed or 0))
            backend = self._rust_lifecycle_backend
            assert backend is not None  # enable either returns or raises

        # Raw-mode runs keep a full record-aligned checkpoint per recorded
        # tick inside the session (state + RNG + ecology), so the public
        # restore_checkpoint rolls back everything, not just counts.
        history_obj = getattr(self, "_history_obj", None)
        checkpoint_every = (
            record_every
            if record_every > 0
            and history_obj is not None
            and history_obj.schema.mode == "raw"
            else 0
        )

        if history_obj is not None:
            if clear_history_on_start:
                self.clear_history()
            if history_obj.schema.mode == "observation":
                history_obj._configure_observation(self.observation)
            backend.bind_history(history_obj._store, self._params_log)
            history_obj._bind_checkpoint_pruner(backend.retain_checkpoints_from)  # pyright: ignore[reportPrivateUsage]  # capacity changes synchronously release native checkpoints.

        self._rust_run_active = True
        try:
            final_tick, _history_new, was_stopped = backend.run(
                n_steps=n_steps,
                record_every=record_every,
                observation_mask=self._observation_mask,
                checkpoint_every=checkpoint_every,
            )
        finally:
            self._rust_run_active = False

        # The session owns the state: only the mirror tick updates eagerly;
        # the cached container refreshes lazily on the next read.
        self._tick = int(final_tick)
        self._mark_state_cache_stale()
        # Bound native HistoryStore receives records during the session run.

        if was_stopped:
            self._finished = True
            self.trigger_event("finish", deme_id=self._deme_id)
        elif finish:
            self.finish_simulation()

        return self

    def run(
        self,
        n_steps: int,
        record_every: Optional[int] = None,
        finish: bool = False,
        clear_history_on_start: bool = False,
    ) -> AgeStructuredPopulation:
        """Run multi-step evolution through the Rust engine session.

        Args:
            n_steps: Number of steps to evolve.
            record_every: Interval for recording snapshots.
                If None, uses self.record_every. If 0, no snapshots are recorded.
            finish: Whether to mark the population as finished after the run.
            clear_history_on_start: Whether to clear existing history before starting.

        Returns:
            AgeStructuredPopulation: Self for chaining.

        Raises:
            RuntimeError: If the population is already finished and cannot
                continue, or if the native engine extension is unavailable
                (the session is created by ``build()`` or lazily at the
                first run).
        """
        self._require_standalone_owner("run")
        if getattr(self, "_running", False):
            raise RuntimeError("Nested run is forbidden")
        if getattr(self, "_failed", False):
            raise RuntimeError("Population has failed; restore a checkpoint or reset before run")
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. "
                "Cannot run() again after finish=True."
            )

        self._running = True
        try:
            if record_every is None:
                record_every = self.record_every

            return self._run_rust_lifecycle(
                n_steps=n_steps,
                record_every=record_every,
                finish=finish,
                clear_history_on_start=clear_history_on_start,
            )
        except BaseException:
            self._failed = True
            self._mark_state_cache_stale()
            raise
        finally:
            self._running = False

    def run_tick(self) -> AgeStructuredPopulation:
        """
        Execute a single tick of evolution.

        Returns:
            AgeStructuredPopulation: Self for chaining.

        Raises:
            RuntimeError: If the population is already finished and cannot continue.
        """
        return self.run(
            n_steps=1, record_every=self.record_every, clear_history_on_start=False
        )

    def get_age_distribution(self, sex: str = "both") -> np.ndarray:
        """Return the age distribution for the requested sex.

        Args:
            sex: One of ``'female'``, ``'male'``, or ``'both'``.

        Returns:
            NDArray[np.float64]: Age distribution array with shape (n_ages,).

        Raises:
            ValueError: If sex identifier is invalid.
        """
        if sex not in ("female", "male", "both", "F", "M"):
            raise ValueError(f"sex must be 'female', 'male', or 'both', got '{sex}'")

        # Access directly from PopulationState
        if sex in ("female", "F"):
            return (
                self._live_state().individual_count[Sex.FEMALE.value, :, :].sum(axis=1)
            )
        elif sex in ("male", "M"):
            return self._live_state().individual_count[Sex.MALE.value, :, :].sum(axis=1)
        else:
            return self._live_state().individual_count.sum(axis=(0, 2))

    def get_genotype_count(self, genotype: Genotype) -> Tuple[float, float]:
        """Return total counts for a genotype as (female_count, male_count).

        .. deprecated::
            Use ``self.registry.ztype_index()`` + manual array sum instead.
        """
        import warnings

        warnings.warn(
            "get_genotype_count is deprecated; use registry + manual sum",
            DeprecationWarning,
            stacklevel=2,
        )
        genotype_idx = self.registry.ztype_index(genotype, self.registry.slab_labels[0])
        female_count = (
            self._live_state().individual_count[Sex.FEMALE.value, :, genotype_idx].sum()
        )
        male_count = (
            self._live_state().individual_count[Sex.MALE.value, :, genotype_idx].sum()
        )
        return (female_count, male_count)

    @property
    def genotypes_present(self) -> Set[Genotype]:
        """Set[Genotype]: Returns the set of genotypes with count > 0.

        .. deprecated::
            Use ``self.registry.index_to_genotype`` + manual count check
            instead.
        """
        import warnings

        warnings.warn(
            "genotypes_present is deprecated; use registry + manual count check",
            DeprecationWarning,
            stacklevel=2,
        )
        present: Set[Genotype] = set()
        for z_idx, (genotype, _slab) in enumerate(self.registry.index_to_ztype):
            total_count = self._live_state().individual_count[:, :, z_idx].sum()
            if total_count > 0:
                present.add(genotype)
        return present

    def update(self) -> Configurator:
        """Return a ``Configurator`` for modifying this population's config."""
        return self._create_configurator()

    def __repr__(self) -> str:
        """Return a compact string representation of the population."""
        return (
            f"AgeStructuredPopulation(name='{self.name}', n_ages={self.n_ages}, "
            f"total_count={self.get_total_count()}, "
            f"adult_females={self.get_adult_count('female')}, "
            f"adult_males={self.get_adult_count('male')})"
        )
