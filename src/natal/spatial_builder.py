"""Spatial population builder with fluent API and batch-setting support.

Provides ``SpatialBuilder`` for constructing ``SpatialPopulation`` instances
via a chainable API. Supports both homogeneous (all demes identical) and
heterogeneous (per-deme varying parameters via ``batch_setting``) construction.

Examples:
    >>> species = Species.from_dict(...)
    >>> pop = (SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10))
    ...     .setup(name="demo", stochastic=False)
    ...     .initial_state(female={"WT|WT": 5000}, male={"WT|WT": 5000})
    ...     .reproduction(eggs_per_female=50)
    ...     .competition(carrying_capacity=10000)
    ...     .presets(drive)
    ...     .migration(kernel=my_kernel, migration_rate=0.1)
    ...     .build())
"""

from __future__ import annotations

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.genetic_structures import Species

# Lazy imports to avoid circular dependency at module level.
# SpatialPopulation and builder classes are imported at function-call time.

__all__ = [
    "BatchSetting",
    "batch_setting",
    "SpatialBuilder",
]



# ---------------------------------------------------------------------------
# BatchSetting
# ---------------------------------------------------------------------------

class BatchSetting:
    """Deferred per-deme parameter specification.

    Wraps one of three value kinds used by ``SpatialBuilder`` to express
    parameters that vary across demes:

    - **scalar**: A Python sequence (list/tuple), one element per deme.
      Each element can be a scalar or an array (e.g. per-deme
      equilibrium distributions).
    - **array**: A 1D numpy array, one element per deme.
    - **spatial**: A callable ``(topology, deme_idx) -> float``, expanded at
      build time.

    ``SpatialBuilder`` detects ``BatchSetting`` values in builder method
    calls, stores them, and expands them during ``build()``.
    """

    _KIND_SCALAR = "scalar"
    _KIND_ARRAY = "array"
    _KIND_SPATIAL = "spatial"

    def __init__(
        self,
        values: Union[Sequence[Any], NDArray[np.floating[Any]], Callable[..., float]],
    ):
        self._fn: Optional[Callable[..., float]] = None
        self._values: Optional[List[Any]] = None
        self._values_array: Optional[NDArray[np.floating[Any]]] = None
        self._n_demes: Optional[int] = None

        if callable(values):
            self._kind: str = self._KIND_SPATIAL
            self._fn = values
        elif isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise ValueError(
                    f"BatchSetting array must be 1D, got shape {values.shape}"
                )
            self._kind = self._KIND_ARRAY
            self._values_array = np.asarray(values)
            self._n_demes = int(self._values_array.shape[0])
        else:
            self._kind = self._KIND_SCALAR
            self._values = list(values)
            self._n_demes = len(self._values)

    @property
    def kind(self) -> str:
        """str: The kind of value source (``"scalar"``, ``"array"``, or ``"spatial"``)."""
        return self._kind

    def __repr__(self) -> str:
        if self._kind == self._KIND_SPATIAL:
            return f"BatchSetting(kind={self._kind!r})"
        return f"BatchSetting(kind={self._kind!r}, n={self._n_demes})"

    def expand(
        self,
        n_demes: int,
        topology: Any = None,  # GridTopology, imported lazily
    ) -> List[Any]:
        """Expand to a concrete list of per-deme values.

        Args:
            n_demes: Number of demes to expand to.
            topology: Optional ``GridTopology`` required for spatial kind.

        Returns:
            List of per-deme values (scalars or arrays).

        Raises:
            ValueError: If length mismatch or spatial kind without topology.
        """
        if self._kind == self._KIND_SCALAR:
            if self._values is None:
                raise ValueError("BatchSetting scalar values are None")
            if len(self._values) != n_demes:
                raise ValueError(
                    f"BatchSetting has {len(self._values)} values "
                    f"but {n_demes} demes are required"
                )
            return list(self._values)

        elif self._kind == self._KIND_ARRAY:
            if self._values_array is None:
                raise ValueError("BatchSetting array values are None")
            if self._n_demes != n_demes:
                raise ValueError(
                    f"BatchSetting array has {self._n_demes} values "
                    f"but {n_demes} demes are required"
                )
            return self._values_array.tolist()

        elif self._kind == self._KIND_SPATIAL:
            if topology is None:
                raise ValueError(
                    "Spatial BatchSetting requires topology for expansion."
                )
            fn = cast(Callable[..., float], self._fn)
            return [float(fn(topology, i)) for i in range(n_demes)]

        raise ValueError(f"Unknown kind: {self._kind}")

    def first_value(self) -> Any:
        """Return a single concrete element for template-builder delegation.

        ``SpatialBuilder`` holds a single-deme template builder internally.
        When a parameter is wrapped in ``batch_setting`` (a per-deme list),
        the template builder still needs one scalar value to proceed through
        ``setup() → … → build()``. This method provides that value —
        typically the first element of the list or array.

        For spatial kind (lazy callable), returns ``None`` because the value
        cannot be resolved without topology expansion at build time.

        Returns:
            The first element for scalar/array kinds, or ``None`` for spatial kind.
        """
        if self._kind == self._KIND_SCALAR:
            return self._values[0] if self._values else None
        elif self._kind == self._KIND_ARRAY:
            if self._values_array is not None and len(self._values_array) > 0:
                return self._values_array[0].item() if hasattr(self._values_array[0], 'item') else self._values_array[0]
            return None
        return None  # spatial kind: deferred until expand() has topology


def batch_setting(
    values: Union[Sequence[Any], NDArray[np.floating[Any]], Callable[..., float], BatchSetting],
) -> BatchSetting:
    """Create a ``BatchSetting`` for per-deme parameter specification.

    Args:
        values: One of:
            - A list/tuple of scalars of length ``n_demes``.
            - A 1D numpy array of length ``n_demes``.
            - A callable ``(topology: GridTopology, deme_idx: int) -> float``.
            - An existing ``BatchSetting`` (returned as-is).

    Returns:
        A ``BatchSetting`` instance that ``SpatialBuilder`` detects and
        expands at build time.
    """
    if isinstance(values, BatchSetting):
        return values
    return BatchSetting(values)


# ---------------------------------------------------------------------------
# _make_hashable
# ---------------------------------------------------------------------------


def _make_hashable(value: Any) -> Any:
    """Convert a value to a hashable form for config-equivalence grouping.

    Scalars pass through; lists/tuples convert to nested tuples;
    dicts convert to sorted (key, hashable_value) tuples;
    numpy arrays convert to bytes.
    """
    if isinstance(value, np.ndarray):
        return ("__ndarray__", value.tobytes())
    if isinstance(value, dict):
        d = cast(Dict[Any, Any], value)
        items = sorted(d.items(), key=lambda x: str(x[0]))
        return ("__dict__", tuple((k, _make_hashable(v)) for k, v in items))
    if isinstance(value, list):
        lst = cast(List[Any], value)
        return tuple(_make_hashable(v) for v in lst)
    if isinstance(value, tuple):
        tup = cast(tuple[Any, ...], value)
        return tuple(_make_hashable(v) for v in tup)
    return value


# ---------------------------------------------------------------------------
# _clone_deme
# ---------------------------------------------------------------------------


def _clone_deme(
    template: Any,
    config: Any,
    name: str,
) -> Any:
    """Create a lightweight functional copy of a template deme.

    Delegates to the population instance's ``_clone`` method, which shares
    compiled hooks, config arrays, registry, and modifiers with the template.
    Only state arrays are independent.

    Args:
        template: A fully-built population instance (``AgeStructuredPopulation``
            or ``DiscreteGenerationPopulation``).
        config: The ``PopulationConfig`` for the clone (shared by reference).
        name: Unique name for the clone.

    Returns:
        A new population instance of the same type as *template*.
    """
    return template._clone(name=name, config=config)


# ---------------------------------------------------------------------------
# _replace optimization: builder-kwarg → config-field mappings
# ---------------------------------------------------------------------------
#
# ``_build_heterogeneous`` uses ``PopulationConfig._replace`` to share heavy
# arrays across groups.  Most builder kwargs map directly to a same-named
# config field; only the exceptions below need explicit mappings.
#
# The dispatch in ``_build_variant_config`` works like this:
#
#   1. *array kwarg* (individual_count, sperm_storage)
#      → convert dict → array via PopulationConfigBuilder, then _replace.
#   2. *multi-field kwarg* (carrying_capacity variants)
#      → _replace into both base_carrying_capacity and the scaled
#        carrying_capacity.
#   3. *rename kwarg* (eggs_per_female → expected_eggs_per_female, etc.)
#      → _replace under the renamed config field.
#   4. *any other kwarg*
#      → try ``hasattr(base_config, kwarg)``; if the config field exists,
#        _replace directly.  If not, fall back to full builder replay.
#
# This means adding a new batch-able scalar parameter typically requires
# zero changes here — as long as the builder kwarg and config field share
# the same name.
# ---------------------------------------------------------------------------

# Builder kwarg names that require dict → numpy-array conversion.
# The output array replaces the named config field.
_ARRAY_KWARGS: frozenset[str] = frozenset({"individual_count", "sperm_storage"})

# Builder kwargs that map to *multiple* config fields (handled specially).
# Each value is (base_field, scaled_field) — both are written with
# population_scale applied to scaled_field.
_KWARG_MULTI_FIELD: dict[str, tuple[str, str]] = {
    "carrying_capacity": ("base_carrying_capacity", "carrying_capacity"),
    "age_1_carrying_capacity": ("base_carrying_capacity", "carrying_capacity"),
    "old_juvenile_carrying_capacity": ("base_carrying_capacity", "carrying_capacity"),
}

# Builder kwarg → config field renames.
# Kwargs not listed here are tried directly with ``hasattr(base_config, name)``.
_KWARG_RENAMES: dict[str, str] = {
    "eggs_per_female": "expected_eggs_per_female",
    "expected_num_adult_females": "base_expected_num_adult_females",
}

# Builder kwargs that affect equilibrium metrics.
# When any of these change, expected_competition_strength and
# expected_survival_rate are recomputed after _replace.
_EQUILIBRIUM_SENSITIVE_KWARGS: frozenset[str] = frozenset({
    "carrying_capacity", "age_1_carrying_capacity", "old_juvenile_carrying_capacity",
    "eggs_per_female", "sex_ratio",
})


# ---------------------------------------------------------------------------
# SpatialBuilder
# ---------------------------------------------------------------------------

class SpatialBuilder:
    """Fluent builder for ``SpatialPopulation``.

    Wraps a single-deme population builder (``AgeStructuredPopulationBuilder``
    or ``DiscreteGenerationPopulationBuilder``) as a template. All chainable
    configuration methods delegate to the template builder and return ``self``.

    Spatial-specific parameters (topology, migration, adjacency) are stored
    directly and forwarded to ``SpatialPopulation`` at build time.
    """

    def __init__(
        self,
        species: Species,
        n_demes: int,
        topology: Any = None,  # GridTopology, imported lazily
        *,
        pop_type: Literal["age_structured", "discrete_generation"] = "age_structured",
    ):
        if n_demes < 1:
            raise ValueError(f"n_demes must be >= 1, got {n_demes}")

        self._species = species
        self._n_demes = n_demes
        self._topology = topology
        self._pop_type: Literal["age_structured", "discrete_generation"] = pop_type

        # Create the template single-deme builder.
        if pop_type == "age_structured":
            from natal.population_builder import AgeStructuredPopulationBuilder
            self._template: Any = AgeStructuredPopulationBuilder(species)
        else:
            from natal.population_builder import DiscreteGenerationPopulationBuilder
            self._template: Any = DiscreteGenerationPopulationBuilder(species)

        # Accumulated batch settings: param_name -> BatchSetting.
        self._batch_settings: Dict[str, BatchSetting] = {}

        # Replay log: list of (method_name, kwargs_with_batch_settings).
        self._replay_log: List[tuple[str, Dict[str, Any]]] = []

        # Spatial migration parameters.
        self._migration_kernel: Optional[NDArray[np.float64]] = None
        self._migration_rate: float = 0.0
        self._migration_strategy: str = "auto"
        self._migration_adjacency: Optional[object] = None
        self._kernel_bank: Optional[Sequence[NDArray[np.float64]]] = None
        self._deme_kernel_ids: Optional[NDArray[np.int64]] = None
        self._kernel_include_center: bool = False

        # Container-level name.
        self._spatial_name: str = "SpatialPopulation"

    # ------------------------------------------------------------------
    # Internal: batch detection and delegation
    # ------------------------------------------------------------------

    def _detect_and_delegate(
        self,
        method_name: str,
        kwargs: Dict[str, Any],
    ) -> SpatialBuilder:
        """Detect BatchSetting values in kwargs, store them, and delegate
        concrete (non-batch) values to the template builder's method.

        Args:
            method_name: Name of the method on the template builder.
            kwargs: Keyword arguments passed by the user.

        Returns:
            Self for chaining.
        """
        concrete: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchSetting):
                self._batch_settings[key] = value
                # Template builder only understands scalar values —
                # feed it the first element so it can proceed through
                # its own build() pipeline. The full per-deme list is
                # stored in _batch_settings for later expansion.
                first = value.first_value()
                if first is not None:
                    concrete[key] = first
            else:
                concrete[key] = value

        # Record the original call (with BatchSetting values preserved).
        self._replay_log.append((method_name, dict(kwargs)))

        # Delegate to template builder (filter out None values).
        template_method = getattr(self._template, method_name)
        filtered = {k: v for k, v in concrete.items() if v is not None}
        template_method(**filtered)
        return self

    def _delegate_positional(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> SpatialBuilder:
        """Like ``_detect_and_delegate`` but accepts positional args.

        Positional args are assumed to never be BatchSetting; only kwargs
        are checked.
        """
        concrete_kwargs: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchSetting):
                self._batch_settings[key] = value
                # Template builder only understands scalar values —
                # feed it the first element so it can proceed through
                # its own build() pipeline. The full per-deme list is
                # stored in _batch_settings for later expansion.
                first = value.first_value()
                if first is not None:
                    concrete_kwargs[key] = first
            else:
                concrete_kwargs[key] = value

        self._replay_log.append((method_name, dict(kwargs)))

        template_method = getattr(self._template, method_name)
        filtered = {k: v for k, v in concrete_kwargs.items() if v is not None}
        template_method(*args, **filtered)
        return self

    # ------------------------------------------------------------------
    # Chainable configuration methods
    # ------------------------------------------------------------------

    def setup(
        self,
        name: str = "SpatialPopulation",
        stochastic: bool = True,
        use_continuous_sampling: bool = False,
        use_fixed_egg_count: bool = False,
    ) -> SpatialBuilder:
        """Configure basic population settings.

        Args:
            name: Human-readable population name.
            stochastic: Whether to use stochastic sampling.
            use_continuous_sampling: If True, use Dirichlet sampling.
            use_fixed_egg_count: If True, egg count is fixed.

        Returns:
            Self for chaining.
        """
        self._spatial_name = name
        self._replay_log.append(("setup", {
            "name": name,
            "stochastic": stochastic,
            "use_continuous_sampling": use_continuous_sampling,
            "use_fixed_egg_count": use_fixed_egg_count,
        }))
        self._template.setup(
            name=name,
            stochastic=stochastic,
            use_continuous_sampling=use_continuous_sampling,
            use_fixed_egg_count=use_fixed_egg_count,
        )
        return self

    def age_structure(
        self,
        n_ages: int = 8,
        new_adult_age: int = 2,
        generation_time: Optional[int] = None,
        equilibrium_distribution: Optional[Union[List[float], NDArray[np.float64]]] = None,
    ) -> SpatialBuilder:
        """Configure age structure (age-structured models only).

        Args:
            n_ages: Number of age classes.
            new_adult_age: Age at which individuals become adults.
            generation_time: Optional pre-computed generation time.
            equilibrium_distribution: Optional equilibrium distribution.

        Returns:
            Self for chaining.
        """
        if self._pop_type != "age_structured":
            raise TypeError("age_structure() is only valid for age_structured pop_type")
        return self._detect_and_delegate(
            "age_structure",
            {
                "n_ages": n_ages,
                "new_adult_age": new_adult_age,
                "generation_time": generation_time,
                "equilibrium_distribution": equilibrium_distribution,
            },
        )

    def initial_state(
        self,
        individual_count: Any,
        sperm_storage: Optional[Any] = None,
    ) -> SpatialBuilder:
        """Configure the initial population state.

        Args:
            individual_count: Initial abundance mapping.
            sperm_storage: Optional initial sperm storage (age-structured only).

        Returns:
            Self for chaining.
        """
        kwargs: Dict[str, Any] = {"individual_count": individual_count}
        if sperm_storage is not None:
            kwargs["sperm_storage"] = sperm_storage
        return self._detect_and_delegate("initial_state", kwargs)

    def survival(
        self,
        # Age-structured params
        female_age_based_survival_rates: Optional[Any] = None,
        male_age_based_survival_rates: Optional[Any] = None,
        generation_time: Optional[int] = None,
        equilibrium_distribution: Optional[Any] = None,
        # Discrete-generation params
        female_age0_survival: Optional[float] = None,
        male_age0_survival: Optional[float] = None,
        adult_survival: Optional[float] = None,
    ) -> SpatialBuilder:
        """Configure survival rates.

        Args:
            female_age_based_survival_rates: Per-age female survival (age-structured).
            male_age_based_survival_rates: Per-age male survival (age-structured).
            generation_time: Optional generation time override.
            equilibrium_distribution: Optional equilibrium distribution.
            female_age0_survival: Female age-0 survival (discrete-generation).
            male_age0_survival: Male age-0 survival (discrete-generation).
            adult_survival: Adult survival (discrete-generation).

        Returns:
            Self for chaining.
        """
        if self._pop_type == "age_structured":
            return self._detect_and_delegate(
                "survival",
                {
                    "female_age_based_survival_rates": female_age_based_survival_rates,
                    "male_age_based_survival_rates": male_age_based_survival_rates,
                    "generation_time": generation_time,
                    "equilibrium_distribution": equilibrium_distribution,
                },
            )
        else:
            return self._detect_and_delegate(
                "survival",
                {
                    "female_age0_survival": female_age0_survival,
                    "male_age0_survival": male_age0_survival,
                    "adult_survival": adult_survival,
                },
            )

    def reproduction(
        self,
        # Shared params
        eggs_per_female: Any = 50.0,
        sex_ratio: Any = 0.5,
        use_fixed_egg_count: bool = False,
        # Age-structured params
        female_age_based_mating_rates: Optional[Any] = None,
        male_age_based_mating_rates: Optional[Any] = None,
        female_age_based_reproduction_rates: Optional[Any] = None,
        female_age_based_relative_fertility: Optional[Any] = None,
        use_sperm_storage: bool = True,
        sperm_displacement_rate: float = 0.05,
        # Discrete-generation params
        female_adult_mating_rate: float = 1.0,
        male_adult_mating_rate: float = 1.0,
    ) -> SpatialBuilder:
        """Configure reproduction and mating parameters.

        Args:
            eggs_per_female: Expected offspring per adult female. Accepts ``BatchSetting``.
            sex_ratio: Proportion of female offspring.
            use_fixed_egg_count: If True, egg count is deterministic.
            female_age_based_mating_rates: Female mating rates (age-structured).
            male_age_based_mating_rates: Male mating rates (age-structured).
            female_age_based_reproduction_rates: Reproduction participation rates.
            female_age_based_relative_fertility: Fertility weights.
            use_sperm_storage: Whether to model sperm storage (age-structured).
            sperm_displacement_rate: Rate of sperm displacement (age-structured).
            female_adult_mating_rate: Adult female mating rate (discrete-generation).
            male_adult_mating_rate: Adult male mating rate (discrete-generation).

        Returns:
            Self for chaining.
        """
        if self._pop_type == "age_structured":
            return self._detect_and_delegate(
                "reproduction",
                {
                    "female_age_based_mating_rates": female_age_based_mating_rates,
                    "male_age_based_mating_rates": male_age_based_mating_rates,
                    "female_age_based_reproduction_rates": female_age_based_reproduction_rates,
                    "female_age_based_relative_fertility": female_age_based_relative_fertility,
                    "eggs_per_female": eggs_per_female,
                    "use_fixed_egg_count": use_fixed_egg_count,
                    "sex_ratio": sex_ratio,
                    "use_sperm_storage": use_sperm_storage,
                    "sperm_displacement_rate": sperm_displacement_rate,
                },
            )
        else:
            return self._detect_and_delegate(
                "reproduction",
                {
                    "eggs_per_female": eggs_per_female,
                    "sex_ratio": sex_ratio,
                    "female_adult_mating_rate": female_adult_mating_rate,
                    "male_adult_mating_rate": male_adult_mating_rate,
                },
            )

    def competition(
        self,
        # Age-structured params
        competition_strength: float = 5.0,
        juvenile_growth_mode: Any = "logistic",
        low_density_growth_rate: Any = 6.0,
        age_1_carrying_capacity: Any = None,
        old_juvenile_carrying_capacity: Any = None,
        expected_num_adult_females: Any = None,
        equilibrium_distribution: Any = None,
        # Discrete-generation params
        carrying_capacity: Any = None,
    ) -> SpatialBuilder:
        """Configure competition and density-dependence.

        Args:
            competition_strength: Relative competition factor for age-1 juveniles
                (age-structured only).
            juvenile_growth_mode: Growth model identifier. Accepts ``BatchSetting``.
            low_density_growth_rate: Growth rate at low density. Accepts ``BatchSetting``.
            age_1_carrying_capacity: Carrying capacity at age=1 (age-structured).
                Accepts ``BatchSetting``.
            old_juvenile_carrying_capacity: Alias for ``age_1_carrying_capacity``.
            expected_num_adult_females: Equilibrium adult females. Accepts ``BatchSetting``.
            equilibrium_distribution: Optional equilibrium distribution.
            carrying_capacity: Carrying capacity (discrete-generation). Accepts ``BatchSetting``.

        Returns:
            Self for chaining.
        """
        if self._pop_type == "age_structured":
            # Alias carrying_capacity / old_juvenile_carrying_capacity → age_1_carrying_capacity
            resolved_cc = age_1_carrying_capacity
            if resolved_cc is None:
                resolved_cc = old_juvenile_carrying_capacity
            if resolved_cc is None:
                resolved_cc = carrying_capacity

            return self._detect_and_delegate(
                "competition",
                {
                    "competition_strength": competition_strength,
                    "juvenile_growth_mode": juvenile_growth_mode,
                    "low_density_growth_rate": low_density_growth_rate,
                    "age_1_carrying_capacity": resolved_cc,
                    "expected_num_adult_females": expected_num_adult_females,
                    "equilibrium_distribution": equilibrium_distribution,
                },
            )
        else:
            return self._detect_and_delegate(
                "competition",
                {
                    "juvenile_growth_mode": juvenile_growth_mode,
                    "low_density_growth_rate": low_density_growth_rate,
                    "carrying_capacity": carrying_capacity,
                },
            )

    def presets(self, *preset_list: Any) -> SpatialBuilder:
        """Add gene-drive presets (applied during build).

        Each positional argument may be a ``BatchSetting`` of preset objects,
        allowing different demes to receive different presets.

        Args:
            *preset_list: One or more preset objects, or ``BatchSetting``
                instances wrapping per-deme preset values.

        Returns:
            Self for chaining.
        """
        # Detect BatchSetting in positional args.
        concrete_args: list[Any] = []
        for i, item in enumerate(preset_list):
            if isinstance(item, BatchSetting):
                self._batch_settings[f"_preset_{i}"] = item
                first = item.first_value()
                if first is not None:
                    concrete_args.append(first)
            else:
                concrete_args.append(item)

        self._replay_log.append(("presets", {"preset_list": preset_list}))
        self._template.presets(*concrete_args)
        return self

    def fitness(
        self,
        viability: Optional[Any] = None,
        fecundity: Optional[Any] = None,
        sexual_selection: Optional[Any] = None,
        zygote_viability: Optional[Any] = None,
        mode: str = "replace",
    ) -> SpatialBuilder:
        """Configure fitness values (applied after presets).

        Args:
            viability: Genotype selectors to viability fitness values.
            fecundity: Genotype selectors to fecundity fitness values.
            sexual_selection: Mating preference mapping.
            zygote_viability: Genotype selectors to zygote viability values.
            mode: ``"replace"`` (default) or ``"multiply"``.

        Returns:
            Self for chaining.
        """
        return self._detect_and_delegate(
            "fitness",
            {
                "viability": viability,
                "fecundity": fecundity,
                "sexual_selection": sexual_selection,
                "zygote_viability": zygote_viability,
                "mode": mode,
            },
        )

    def hooks(self, *hook_items: Any) -> SpatialBuilder:
        """Register lifecycle hooks.

        Args:
            *hook_items: Functions decorated with ``@hook`` or hook mappings.

        Returns:
            Self for chaining.
        """
        self._replay_log.append(("hooks", {"hook_items": hook_items}))
        self._template.hooks(*hook_items)
        return self

    def modifiers(
        self,
        gamete_modifiers: Optional[Any] = None,
        zygote_modifiers: Optional[Any] = None,
    ) -> SpatialBuilder:
        """Configure custom modifier functions.

        Args:
            gamete_modifiers: Modifiers for gamete production.
            zygote_modifiers: Modifiers for zygote formation.

        Returns:
            Self for chaining.
        """
        return self._detect_and_delegate(
            "modifiers",
            {
                "gamete_modifiers": gamete_modifiers,
                "zygote_modifiers": zygote_modifiers,
            },
        )

    # ------------------------------------------------------------------
    # Spatial-specific methods
    # ------------------------------------------------------------------

    def migration(
        self,
        kernel: Optional[NDArray[np.float64]] = None,
        migration_rate: float = 0.0,
        strategy: Literal["auto", "adjacency", "kernel", "hybrid"] = "auto",
        adjacency: Optional[object] = None,
        kernel_bank: Optional[Sequence[NDArray[np.float64]]] = None,
        deme_kernel_ids: Optional[NDArray[np.int64]] = None,
        kernel_include_center: bool = False,
    ) -> SpatialBuilder:
        """Configure spatial migration parameters.

        Args:
            kernel: Odd-shaped 2D migration kernel.
            migration_rate: Fraction of each deme that migrates.
            strategy: Migration strategy (``"auto"``, ``"adjacency"``,
                ``"kernel"``, ``"hybrid"``).
            adjacency: Explicit adjacency matrix.
            kernel_bank: Optional heterogeneous kernel bank.
            deme_kernel_ids: Per-deme kernel ids into ``kernel_bank``.
            kernel_include_center: Whether kernel includes center cell.

        Returns:
            Self for chaining.
        """
        if kernel is not None:
            self._migration_kernel = np.asarray(kernel, dtype=np.float64)
        self._migration_rate = float(migration_rate)
        self._migration_strategy = strategy
        if adjacency is not None:
            self._migration_adjacency = adjacency
        if kernel_bank is not None:
            self._kernel_bank = kernel_bank
        if deme_kernel_ids is not None:
            self._deme_kernel_ids = np.asarray(deme_kernel_ids, dtype=np.int64)
        self._kernel_include_center = bool(kernel_include_center)
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Any:
        """Build and return the configured ``SpatialPopulation``.

        Returns:
            A ``SpatialPopulation`` with all demes initialised.
        """
        if not self._batch_settings:
            return self._build_homogeneous()
        return self._build_heterogeneous()

    def _build_homogeneous(self) -> Any:
        """Phase 1a: Build one template deme, clone N-1 times."""
        from natal.spatial_population import SpatialPopulation

        template = self._template.build()
        tpl_config = template.export_config()

        demes: List[Any] = [template]
        for i in range(1, self._n_demes):
            clone = _clone_deme(
                template,
                config=tpl_config,
                name=f"{self._spatial_name}_deme_{i}",
            )
            demes.append(clone)

        return SpatialPopulation(
            demes=demes,
            topology=self._topology,
            adjacency=self._migration_adjacency,
            migration_kernel=self._migration_kernel,
            migration_strategy=cast(Any, self._migration_strategy),
            kernel_bank=self._kernel_bank,
            deme_kernel_ids=self._deme_kernel_ids,
            kernel_include_center=self._kernel_include_center,
            migration_rate=self._migration_rate,
            name=self._spatial_name,
        )

    def _build_heterogeneous(self) -> Any:
        """Phase 1b: Group demes by config equivalence, build one template
        per group.

        The first group is built via the full builder pipeline. Subsequent
        groups use ``_replace`` on the base config when only scalar fields
        differ — this shares all heavy arrays (genotype maps, fitness arrays,
        etc.) and avoids re-running the builder pipeline. When batch settings
        include non-scalar values (initial_state dicts, rate arrays) a full
        rebuild is used as fallback.
        """
        from natal.spatial_population import SpatialPopulation

        # 1. Expand all batch settings to per-deme value lists.
        expanded: Dict[str, List[Any]] = {}
        for param_name, batch in self._batch_settings.items():
            expanded[param_name] = batch.expand(self._n_demes, self._topology)

        # 2. Compute per-deme config signatures from batch param values.
        batch_param_names = sorted(expanded.keys())
        signatures: List[tuple[tuple[str, Any], ...]] = []
        for i in range(self._n_demes):
            sig: tuple[tuple[str, Any], ...] = tuple(
                (name, _make_hashable(expanded[name][i]))
                for name in batch_param_names
            )
            signatures.append(sig)

        # 3. Group deme indices by signature.
        groups: defaultdict[tuple[tuple[str, Any], ...], List[int]] = defaultdict(list)
        for idx, sig in enumerate(signatures):
            groups[sig].append(idx)

        # 4. Build one template per group. The first group is built via the
        #    full builder pipeline; subsequent groups use _replace when possible
        #    to share heavy arrays (genotype maps, fitness arrays, etc.).
        demes: List[Any] = [None] * self._n_demes
        base_config: Any = None
        base_template: Any = None

        for _sig, indices in groups.items():
            first_idx = indices[0]
            sig_map: Dict[str, Any] = {
                name: expanded[name][first_idx]
                for name in batch_param_names
            }

            if base_config is None:
                # First group: full builder pipeline.
                group_template = self._build_template_for_group(sig_map)
                base_config = group_template.export_config()
                base_template = group_template
            elif self._can_use_replace(sig_map, base_config):
                # Only known-replaceable fields differ: share big arrays
                # via _replace, rebuilding only the arrays that changed.
                variant_config = self._build_variant_config(
                    sig_map, base_config,
                    species=self._species,
                    pop_type=self._pop_type,
                )
                group_template = _clone_deme(
                    base_template,
                    config=variant_config,
                    name=f"{self._spatial_name}_deme_{first_idx}",
                )
                # _clone copies state and snapshot from base_template;
                # overwrite with per-group config values.
                state = group_template._require_state()
                if "individual_count" in sig_map:
                    state.individual_count[:] = variant_config.initial_individual_count
                if "sperm_storage" in sig_map:
                    ss = getattr(state, 'sperm_storage', None)
                    if ss is not None:
                        ss[:] = variant_config.initial_sperm_storage
                # Update snapshot so reset() restores per-group state.
                ss_snap = getattr(state, 'sperm_storage', None)
                object.__setattr__(group_template, '_initial_population_snapshot', (
                    state.individual_count.copy(),
                    ss_snap.copy() if ss_snap is not None else None,
                    None,
                ))
            else:
                # Non-scalar differences (initial_state dicts, etc.): full rebuild.
                group_template = self._build_template_for_group(sig_map)

            demes[first_idx] = group_template

            tpl_config = group_template.export_config()
            for idx in indices[1:]:
                demes[idx] = _clone_deme(
                    group_template,
                    config=tpl_config,
                    name=f"{self._spatial_name}_deme_{idx}",
                )

        return SpatialPopulation(
            demes=demes,
            topology=self._topology,
            adjacency=self._migration_adjacency,
            migration_kernel=self._migration_kernel,
            migration_strategy=cast(Any, self._migration_strategy),
            kernel_bank=self._kernel_bank,
            deme_kernel_ids=self._deme_kernel_ids,
            kernel_include_center=self._kernel_include_center,
            migration_rate=self._migration_rate,
            name=self._spatial_name,
        )

    @staticmethod
    def _can_use_replace(sig_map: Dict[str, Any], base_config: Any) -> bool:
        """Return True if every kwarg in *sig_map* maps to a known or
        discoverable config field."""
        for name in sig_map:
            if name in _ARRAY_KWARGS:
                continue
            if name in _KWARG_MULTI_FIELD:
                continue
            if name in _KWARG_RENAMES:
                continue
            # Dynamic: try direct field name match on the config object
            if hasattr(base_config, name):
                continue
            return False
        return True

    @staticmethod
    def _build_variant_config(
        sig_map: Dict[str, Any],
        base_config: Any,
        *,
        species: Any = None,
        pop_type: str = "age_structured",
    ) -> Any:
        """Create a variant config via ``_replace``, sharing all heavy arrays.

        Dispatch order:
        1. Array kwargs (individual_count, sperm_storage) — dict→array conversion.
        2. Multi-field kwargs (carrying_capacity) — replace base + scaled fields.
        3. Rename kwargs (eggs_per_female→expected_eggs_per_female).
        4. Any other kwarg — direct ``_replace`` if the field exists on the
           config (validated by ``_can_use_replace`` beforehand).

        Equilibrium metrics are recomputed when capacity / eggs / sex-ratio change.

        Args:
            sig_map: Mapping from batch kwarg name to group's concrete value.
            base_config: The base ``PopulationConfig`` to derive from.
            species: ``Species`` instance, needed for genotype resolution.
            pop_type: ``"age_structured"`` or ``"discrete_generation"``.

        Returns:
            A new ``PopulationConfig`` sharing all unchanged array references
            with *base_config*.
        """
        from natal.algorithms import compute_equilibrium_metrics
        from natal.population_builder import PopulationConfigBuilder

        replace_kwargs: Dict[str, Any] = {}
        needs_equilibrium = False

        for kwarg, val in sig_map.items():
            # --- 1. array-valued: dict → array conversion ---
            if kwarg == "individual_count":
                if pop_type == "age_structured":
                    array = PopulationConfigBuilder.resolve_age_structured_initial_individual_count(
                        species=species,
                        distribution=val,
                        n_ages=int(base_config.n_ages),
                        new_adult_age=int(base_config.new_adult_age),
                    )
                else:
                    array = PopulationConfigBuilder.resolve_discrete_initial_individual_count(
                        species=species,
                        distribution=val,
                    )
                array *= float(base_config.population_scale)
                replace_kwargs["initial_individual_count"] = array
                continue

            if kwarg == "sperm_storage":
                if pop_type == "age_structured":
                    array = PopulationConfigBuilder.resolve_age_structured_initial_sperm_storage(
                        species=species,
                        sperm_storage=val,
                        n_ages=int(base_config.n_ages),
                        new_adult_age=int(base_config.new_adult_age),
                    )
                    array *= float(base_config.population_scale)
                    replace_kwargs["initial_sperm_storage"] = array
                continue

            # --- 2. multi-field: carrying_capacity variants ---
            multi = _KWARG_MULTI_FIELD.get(kwarg)
            if multi is not None:
                base_field, scaled_field = multi
                replace_kwargs[base_field] = float(val)
                replace_kwargs[scaled_field] = float(val) * float(base_config.population_scale)
                if kwarg in _EQUILIBRIUM_SENSITIVE_KWARGS:
                    needs_equilibrium = True
                continue

            # --- 3. rename ---
            config_field = _KWARG_RENAMES.get(kwarg, kwarg)
            replace_kwargs[config_field] = val

            if kwarg in _EQUILIBRIUM_SENSITIVE_KWARGS:
                needs_equilibrium = True

        variant = base_config._replace(**replace_kwargs)

        if needs_equilibrium:
            new_comp, new_surv = compute_equilibrium_metrics(
                carrying_capacity=float(variant.carrying_capacity),
                expected_eggs_per_female=float(variant.expected_eggs_per_female),
                age_based_survival_rates=variant.age_based_survival_rates,
                age_based_mating_rates=variant.age_based_mating_rates,
                female_age_based_relative_fertility=variant.female_age_based_relative_fertility,
                relative_competition_strength=variant.age_based_relative_competition_strength,
                sex_ratio=float(variant.sex_ratio),
                new_adult_age=int(variant.new_adult_age),
                n_ages=int(variant.n_ages),
                age_based_reproduction_rates=variant.age_based_reproduction_rates,
            )
            variant = variant._replace(
                expected_competition_strength=float(new_comp),
                expected_survival_rate=float(new_surv),
            )

        return variant

    def _build_template_for_group(self, sig_map: Dict[str, Any]) -> Any:
        """Build a single template deme for one config-signature group.

        Replays all recorded method calls on a fresh builder, substituting
        batch-setting parameters with the group-specific scalar values from
        *sig_map*.

        Args:
            sig_map: Mapping from batch parameter name to the group's
                concrete value.

        Returns:
            A fully-built population instance for this group.
        """
        if self._pop_type == "age_structured":
            from natal.population_builder import AgeStructuredPopulationBuilder
            builder: Any = AgeStructuredPopulationBuilder(self._species)
        else:
            from natal.population_builder import DiscreteGenerationPopulationBuilder
            builder = DiscreteGenerationPopulationBuilder(self._species)

        for method_name, kwargs in self._replay_log:
            method = getattr(builder, method_name, None)
            if method is None:
                continue

            # Substitute batch-setting values for this group.
            resolved: Dict[str, Any] = {}
            for key, value in kwargs.items():
                if key in sig_map:
                    resolved[key] = sig_map[key]
                elif isinstance(value, BatchSetting):
                    # This shouldn't happen if sig_map covers all batch names,
                    # but fall back to first value just in case.
                    first = value.first_value()
                    if first is not None:
                        resolved[key] = first
                else:
                    resolved[key] = value

            # Handle positional args (presets, hooks).
            if method_name == "presets":
                # Cast to Any: dict.pop with default=() types as tuple[()], which
                # makes enumerate yield Never elements.
                raw_preset_list: Any = cast(Any, resolved.pop("preset_list", ()))
                expanded_presets: list[Any] = []
                for i, item in enumerate(raw_preset_list):
                    key = f"_preset_{i}"
                    val: Any = sig_map.get(key)
                    if val is not None:
                        expanded_presets.append(val)
                    elif isinstance(item, BatchSetting):
                        first = item.first_value()
                        if first is not None:
                            expanded_presets.append(first)
                    else:
                        expanded_presets.append(item)
                filtered = {k: v for k, v in resolved.items() if v is not None}
                method(*expanded_presets, **filtered)
            elif method_name == "hooks":
                hook_items = resolved.pop("hook_items", ())
                filtered = {k: v for k, v in resolved.items() if v is not None}
                method(*hook_items, **filtered)
            else:
                filtered = {k: v for k, v in resolved.items() if v is not None}
                method(**filtered)

        return builder.build()
