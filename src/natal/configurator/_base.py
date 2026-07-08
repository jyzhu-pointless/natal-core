"""Mutable wrapper for PopulationConfig, with a chainable API.

Functionality:
  - Read/write config fields through a chainable API.
  - Register custom named parameters (stored as a structured numpy array).
  - Freeze changes back to an immutable NamedTuple via ``_replace``
    (cheap — all ndarray fields are shared by reference).
  - ``_pop_ref`` write-back: when wired to a live Population via
    ``for_population()``, config mutations propagate back to the
    Population automatically through ``set_config()``.

Why this module exists:
  ``PopulationConfig`` / ``DiscretePopulationConfig`` are immutable
  NamedTuples — fields cannot be modified once created.  However,
  during simulation setup (and inside hooks at runtime), parameters
  need real-time adjustment.  The ``Configurator`` provides a mutable
  layer on top: all modifications write into the config arrays
  in-place, and the final immutable config is materialised via
  ``build()``.

  The adapter class ``ConfigContext`` lets genetic presets and
  modifiers operate on config arrays without needing a live Population
  object.  The standalone :func:`set_param` function is also usable
  from within Numba-compiled hooks via ``objmode``.

Key classes:
  - ``Configurator`` — base class with chainable domain methods.
  - ``DiscreteConfigurator`` — subclass for non-overlapping generations.
  - ``AgeStructuredConfigurator`` — subclass for overlapping generations.

See also:
  :func:`set_param` — low-level scalar writer.
  :func:`hook_set_param` — Numba-safe wrapper for use in hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Self, Sequence, cast

import numpy as np
from numba import (  # pyright: ignore[reportMissingTypeStubs]
    objmode,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)
from numpy.typing import NDArray

from natal.configurator._params import (
    resolve_param,
)
from natal.configurator._registry_builder import (
    ConfigContext,
    build_registry,
    rebuild_config_maps,
)
from natal.data import (
    DiscretePopulationConfig,
    PopulationConfig,
)
from natal.fitness._writer import write_fitness_field
from natal.genetics import Species
from natal.numba.utils import njit_switch
from natal.presets import CytoplasmicPreset
from natal.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.configurator.age_structured import AgeStructuredConfigurator
    from natal.configurator.discrete import DiscreteConfigurator
    from natal.modifiers.module import GameteModifier, ZygoteModifier
    from natal.population.age_structured import AgeStructuredPopulation
    from natal.population.base import BasePopulation
    from natal.population.discrete_generation import DiscreteGenerationPopulation
    from natal.presets import GeneticPreset

__all__ = [
    "Configurator",
    "hook_set_param",
    "set_param",
]

# ── Type aliases for hook registrations ──────────────────────────────────────

# A single hook registration: (func, name?, priority?)
# ``Any`` return is required for compatibility with ``HookRegistration`` in
# base_population.py (tuple invariance — ``Callable[..., int]`` would fail).
_HookReg = tuple[Callable[..., Any], str | None, int | None]
# Hook registration map keyed by event name.
HookMap = dict[str, list[_HookReg]]
# A hook item can be a raw dict or a callable with @hook metadata.
_HookItem = Callable[..., Any] | HookMap


# These parameters affect the steady-state equilibrium.  When any of them
# change, ``expected_competition_strength`` and ``expected_survival_rate``
# must be recomputed via ``sync_equilibrium_metrics``.
_EQUILIBRIUM_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "competition.carrying_capacity",
    "reproduction.eggs_per_female",
    "reproduction.sex_ratio",
})


# ── Core runtime setter ────────────────────────────────────────────────────────


def set_param(
    config: PopulationConfig | DiscretePopulationConfig,
    name: str,
    value: float | int | bool,
    *,
    _sync_equilibrium: bool = True,
) -> None:
    """Set a simulation parameter by its user-facing name.

    Looks up *name* in ``ALL_PARAMETERS`` (the declarative registry in
    :mod:`natal.parameters`) to find the correct config field + index path,
    then writes *value* in-place into the 0-d ndarray at that position.

    The registry maps each parameter name to a :class:`~natal.parameters.
    ParamDescriptor` with these fields used here:

    - ``config_field`` — name of the attribute on the config object
    - ``config_path`` — index tuple into the ndarray (empty for 0-d)
    - ``is_tensor`` / ``is_array`` — guards to reject non-scalar writes
    - ``domain`` — domain string grouping related parameters
      (e.g. ``"competition"``, ``"reproduction"``)

    After writing, ``sync_equilibrium_metrics`` is called automatically
    for equilibrium-sensitive keys (carrying capacity, eggs per female,
    sex ratio) unless ``_sync_equilibrium=False`` is passed.

    Usable from pure Python, ``with objmode():`` inside njit hooks, and
    Configurator chain methods.

    If *name* matches a custom field on ``config.custom`` (registered via
    :meth:`Configurator.custom`), it is written directly — no registry
    lookup needed. This means ``hook_set_param`` inside Numba hooks can
    also address custom fields.

    Args:
        config: The PopulationConfig or DiscretePopulationConfig to modify.
        name: Parameter name — full key ``"competition.carrying_capacity"``,
              short name ``"carrying_capacity"``, or alias.
        value: New value (scalar). For tensor parameters, use
               direct array access instead.

    Raises:
        KeyError: If *name* is not a registered parameter or custom field.
        ValueError: If *name* refers to a tensor (non-scalar) parameter.

    Examples::

        set_param(config, "competition.carrying_capacity", 5000.0)
        set_param(config, "carrying_capacity", 5000.0)        # short name
        set_param(config, "reproduction.eggs_per_female", 100.0)
    """
    desc = resolve_param(name)  # noqa: F821
    if desc is None:
        # Fallback: check custom fields (not in the parameter registry).
        if hasattr(config, 'custom') and config.custom.dtype.names and name in config.custom.dtype.names:
            config.custom[name][()] = value
            return
        raise KeyError(f"Unknown parameter: {name!r}")
    if desc.config_field is None:
        raise ValueError(
            f"{name!r} is a spatial-only parameter and cannot be set "
            f"on a non-spatial config. Use pop.update(...) on a "
            f"SpatialPopulation instead."
        )
    if desc.is_tensor or desc.is_array:
        raise ValueError(
            f"set_param does not support tensor or array parameters "
            f"like {name!r}. Use direct array access instead."
        )

    field = getattr(config, desc.config_field)

    # Write through the index path:
    #   ()      → 0-d ndarray → field[()] = value
    #   (1,)    → 1-D index   → field[1] = value
    #   (0, 0)  → 2-D index   → field[0, 0] = value
    if desc.config_path:
        field[desc.config_path] = value
    elif isinstance(field, np.ndarray) and field.ndim == 0:
        field[()] = value
    elif isinstance(field, np.ndarray):
        raise ValueError(
            f"Cannot set {name!r} via set_param: field is a {field.ndim}d array "
            f"but config_path is empty. Use the corresponding Configurator method "
            f"or write to the array directly."
        )
    else:
        raise TypeError(
            f"Cannot set {name!r} via set_param: field is a Python "
            f"{type(field).__name__} on an immutable config. "
            f"Use the corresponding Configurator method instead."
        )

    # Auto-sync equilibrium when sensitive params change.
    key = f"{desc.domain}.{desc.name}"
    if _sync_equilibrium and key in _EQUILIBRIUM_SENSITIVE_KEYS:
        from natal.engine.simulation.age_structured import sync_equilibrium_metrics

        if not isinstance(config, DiscretePopulationConfig):
            sync_equilibrium_metrics(config)


# ── Helpers: hook merging and fitness field writing ────────────────────────────


def merge_hooks(hook_items: list[_HookItem]) -> HookMap:
    """Merge @hook-decorated items into a hook registration map.

    Each item can be a raw dict or a function with @hook metadata.
    """
    result: HookMap = {}
    for item in hook_items:
        if isinstance(item, dict):
            hook_dict = cast(HookMap, item)
            for event, registrations in hook_dict.items():
                result.setdefault(event, []).extend(registrations)
        elif callable(item):
            # @hook-decorated functions store metadata in a .meta dict,
            # but some older decorators set attributes directly.  Check
            # both for backward compatibility.
            meta = getattr(item, "meta", {})
            event = meta.get("event") or getattr(item, "event", None)
            priority = meta.get("priority", getattr(item, "priority", 0))
            name = getattr(item, "__name__", None)
            if event:
                result.setdefault(event, []).append((item, name, priority))
            else:
                import warnings
                warnings.warn(
                    f"Hook {name or '<anonymous>'!r} has no event metadata. "
                    f"Decorate it with @nt.hook(event='...') to register it.",
                    UserWarning, stacklevel=2,
                )
        else:
            import warnings
            warnings.warn(
                f"Ignoring unsupported hook item of type {type(item).__name__!r}. "
                f"Expected a dict or a callable decorated with @nt.hook.",
                UserWarning, stacklevel=2,
            )
    return result




# ── Configurator ───────────────────────────────────────────────────────────────


class Configurator:
    """Parameter configurator — unified API for build-time and runtime use.

    Wraps a PopulationConfig and provides chainable domain methods
    (``.competition()``, ``.reproduction()``, etc.) that immediately write
    parameters via :func:`set_param`.  Presets, modifiers, and fitness
    are applied immediately — no deferred execution.

    Usage::

        # Build-time (from a blank config)
        cfg = Configurator(blank_config)
        cfg.competition(carrying_capacity=10000).reproduction(eggs_per_female=50)
        cfg.presets(drive).apply()

        # Runtime (modify an existing config)
        Configurator(pop.config).competition(carrying_capacity=5000)
    """

    def __init__(
        self,
        config: PopulationConfig | DiscretePopulationConfig,
        species: Species | None = None,
    ) -> None:
        """Wrap a config for chainable modification.

        Args:
            config: An existing PopulationConfig or DiscretePopulationConfig.
            species: Required for methods that need genotype resolution
                (initial_state, presets, modifiers, fitness).  Can be
                omitted when the Configurator is only used for scalar
                parameter updates via set_param.
        """
        self._config = config
        self._species = species  # needed for initial_state / preset resolution

        # _registry is lazily built on first _make_ctx() call, avoiding the
        # cost of genotype enumeration for simple scalar-param updates.
        self._registry: IndexRegistry | None = None

        # Modifier lists — accumulated across presets() / modifiers() calls,
        # then applied when maps are rebuilt.
        self.gamete_modifiers: list[tuple[int, str | None, GameteModifier]] = []
        self.zygote_modifiers: list[tuple[int, str | None, ZygoteModifier]] = []

        # Accumulated kwargs for custom structured-array fields.  Each
        # .custom() call adds to this dict; build_custom_array() is called
        # only at the end.
        self._custom_kwargs: dict[str, object] = {}

        # optional backref for writing config updates back to a Population when created via for_population()
        self._pop_ref: BasePopulation[Any] | None = None

        # Discrete-specific scalar overrides (stored here so build() can
        # extract them into DiscretePopulationConfig at the last moment).
        self._female_adult_mating_rate: float | None = None
        self._male_adult_mating_rate: float | None = None
        self._female_age0_survival: float | None = None
        self._male_age0_survival: float | None = None

        # Index compression flag — enabled via setup(compress=True).
        # Applied during rebuild_config_maps (build-time) or
        # refresh_modifier_maps() (runtime).
        # GType and ZType compression always run together — one BFS produces
        # both masks and they must be applied in tandem.
        self._compress: bool = False
        self._compression_applied: bool = False
        self._declared_zygote_types: set[str] | set[int] | None = None

    @property
    def config(self) -> PopulationConfig | DiscretePopulationConfig:
        """The wrapped PopulationConfig (read-only accessor)."""
        return self._config

    # -- adapter factory ------------------------------------------------------

    def _make_ctx(self) -> ConfigContext:
        """Build a :class:`ConfigContext` seeded from the current state.

        Creates an adapter that mimics ``BasePopulation``'s attribute
        surface so that ``apply_preset_to_population`` and modifier
        functions can operate without a live Population.

        The context receives a shallow copy of the modifier lists so that
        preset / modifier calls can append without mutating the originals
        until :meth:`_sync_from_ctx` explicitly commits them back.

        Lazily builds ``self._registry`` from the species on first call.

        Returns:
            A new ``ConfigContext`` pre-populated with species, config,
            registry, and modifier lists.

        Raises:
            RuntimeError: If ``_species`` is ``None`` (Configurator was
                created via the raw constructor without a Species).
        """
        # _species and _registry are set either:
        #   - by from_species() (build path) — _species directly, registry lazy
        #   - by for_population() (update path) — both from the Population
        # so the existing guards below work for both paths without changes.
        if self._species is None:
            raise RuntimeError(
                "presets() / modifiers() / fitness() require a Species. "
                "Use Configurator.from_species() to create this instance."
            )
        if self._registry is None:
            self._registry = build_registry(self._species)

        ctx = ConfigContext(
            self._species, self._config, self._registry,
            compress=False,  # compression is only enabled in build()
        )
        ctx.declared_zygote_types = self._declared_zygote_types
        ctx.gamete_modifiers = list(self.gamete_modifiers)
        ctx.zygote_modifiers = list(self.zygote_modifiers)
        return ctx

    def _sync_from_ctx(self, ctx: ConfigContext) -> None:
        """Commit adapter-side mutations back into the Configurator.

        Called after ``apply_preset_to_population`` or ``rebuild_config_maps``
        has finished writing into *ctx.config* and the modifier lists.
        Copies the mutated config and modifier lists back, and records
        whether compression was applied.

        Args:
            ctx: The ``ConfigContext`` whose state to consume.
        """
        self._config = ctx.config
        self.gamete_modifiers = ctx.gamete_modifiers
        self.zygote_modifiers = ctx.zygote_modifiers
        if ctx.compression_applied:
            self._compression_applied = True

    # -- factory ---------------------------------------------------------------

    @classmethod
    def from_species(
        cls,
        species: Species,
        *,
        discrete: bool = False,
    ) -> DiscreteConfigurator | AgeStructuredConfigurator:  # type: ignore[name-defined]  # noqa: F821  # lazy-imported subclass forward ref
        """Create a Configurator from a Species with a minimal config.

        This is the primary factory.  Pass ``discrete=True`` for
        non-overlapping generations; otherwise an age-structured config
        with overlapping generations is returned.

        Args:
            species: The genetic architecture for the population.
            discrete: If ``True``, return a ``DiscreteConfigurator``
                (Wright-Fisher, non-overlapping generations).  Default
                ``False`` → ``AgeStructuredConfigurator``.

        Returns:
            A ``DiscreteConfigurator`` or ``AgeStructuredConfigurator``
            ready for further chaining.
        """
        from natal.configurator.age_structured import AgeStructuredConfigurator
        from natal.configurator.discrete import DiscreteConfigurator

        bp = species.get_config_blueprint()
        n_g = bp["n_genotypes"]
        n_hg = bp["n_gtypes"]
        n_gl = bp["n_glabs"]
        n_sl = bp.get("n_slabs", 1)
        z2g = bp["zygotes_to_gametes_map"]
        g2z = bp["gametes_to_zygotes_map"]
        has_sc = getattr(species, "has_sex_chromosomes", False)

        if discrete:
            from natal.data import build_discrete_engine_config

            config = build_discrete_engine_config(
                n_genotypes=n_g, n_gtypes=n_hg, n_glabs=n_gl,
                n_slabs=n_sl,
                gamete_labels=species.gamete_labels or ["default"],
                somatic_labels=species.somatic_labels or ["default"],
                zygotes_to_gametes_map=z2g, gametes_to_zygotes_map=g2z,
                has_sex_chromosomes=has_sc,
            )
            result = DiscreteConfigurator(config, species=species)
            object.__setattr__(result, "_name", "DiscreteGenerationPop")
        else:
            from natal.data import build_population_config

            config = build_population_config(
                n_genotypes=n_g, n_gtypes=n_hg, n_glabs=n_gl,
                n_slabs=n_sl,
                gamete_labels=species.gamete_labels or ["default"],
                somatic_labels=species.somatic_labels or ["default"],
                zygotes_to_gametes_map=z2g, gametes_to_zygotes_map=g2z,
                n_ages=2, new_adult_age=1, carrying_capacity=1000.0,
                has_sex_chromosomes=has_sc,
            )
            result = AgeStructuredConfigurator(config, species=species)
            object.__setattr__(result, "_name", "AgeStructuredPop")
        return result

    @classmethod
    def for_discrete(cls, species: Species) -> DiscreteConfigurator:  # type: ignore[name-defined]  # noqa: F821  # lazy-imported subclass forward ref
        """Shorthand for ``from_species(species, discrete=True)``."""
        from natal.configurator.discrete import DiscreteConfigurator as _DC
        return cast(_DC, cls.from_species(species, discrete=True))

    @classmethod
    def for_age_structured(cls, species: Species) -> AgeStructuredConfigurator:  # type: ignore[name-defined]  # noqa: F821  # lazy-imported subclass forward ref
        """Shorthand for ``from_species(species)``."""
        from natal.configurator.age_structured import AgeStructuredConfigurator as _ASC
        return cast(_ASC, cls.from_species(species))

    @staticmethod
    def for_config(
        config: PopulationConfig | DiscretePopulationConfig,
    ) -> DiscreteConfigurator | AgeStructuredConfigurator:  # type: ignore[name-defined]  # noqa: F821  # lazy-imported subclass forward ref
        """Return the right Configurator subclass for the given config type.

        Args:
            config: The config to wrap.

        Returns:
            ``DiscreteConfigurator`` if *config* is a
            ``DiscretePopulationConfig``, otherwise
            ``AgeStructuredConfigurator``.
        """
        from natal.configurator.age_structured import AgeStructuredConfigurator
        from natal.configurator.discrete import DiscreteConfigurator

        if isinstance(config, DiscretePopulationConfig):
            return DiscreteConfigurator(config)
        return AgeStructuredConfigurator(config)

    @staticmethod
    def for_population(pop: BasePopulation[Any]) -> DiscreteConfigurator | AgeStructuredConfigurator:  # type: ignore[name-defined]  # noqa: F821  # lazy-imported forward ref; Any: Generic population reference, species type irrelevant
        """Create a Configurator wired to *pop* for runtime updates.

        Binds ``_pop_ref``, ``_species``, and ``_registry``
        and ``_registry`` from the Population so that all chain methods work without
        further setup. This is the single entry point for ``pop.update()`` paths.

        Args:
            pop: The population to wire to.

        Returns:
            A ``DiscreteConfigurator`` or ``AgeStructuredConfigurator``
            ready for further chaining.
        """
        cfg = Configurator.for_config(pop.config)

        # Record the Population reference for write-back
        cfg._pop_ref = pop

        # Bind species and registry from the Population so
        # _make_ctx() and fitness() work correctly.
        cfg._species = pop.species
        cfg._registry = pop.index_registry

        return cfg

    # -- setup flags -----------------------------------------------------------

    def setup(
        self,
        *,
        name: str | None = None,
        stochastic: bool | None = None,
        continuous_sampling: bool | None = None,
        fixed_egg_count: bool | None = None,
        compress: bool = False,
        declared_zygote_types: set[str] | set[int] | None = None,
        declared_genotypes: set[str] | set[int] | None = None,  # deprecated alias
    ) -> Self:
        """Configure simulation flags and optional population name.

        *name* is stored and used by ``build()`` when no explicit name is given.

        *compress* enables index compression at build time.  It enables both
        GType (gamete-axis) and ZType (genotype-axis) compression in one flag.
        The older ``compress_gametes()`` / ``compress_genotypes()`` chain
        methods have been removed — use this parameter instead.

        *declared_zygote_types* is a set of genotype strings (``"WT|WT"``) or
        integer indices that are treated as reachable by the BFS even if they
        have zero individuals in the initial state.  Use this to prevent
        compression from pruning genotypes that may appear later via hooks or
        runtime presets.

        .. deprecated:: 0.1
            The parameter name ``declared_genotypes`` is a deprecated alias
            for ``declared_zygote_types`` and still works.

        Args:
            name: Population name (falls back to ``"Population"`` at build time).
            stochastic: If ``False``, use deterministic (median) outcomes.
            continuous_sampling: If ``True``, sample from continuous
                distributions instead of discrete counts.
            fixed_egg_count: If ``True``, disable Poisson noise on egg counts.
            compress: If ``True``, enable full index compression at build time.
            declared_zygote_types: Optional set of genotype selectors to protect
                from compression pruning.

        Returns:
            Self for chaining.
        """
        if name is not None:
            self._name = name
        if compress:
            self._compress = True
        if declared_genotypes is not None:
            import warnings
            warnings.warn(
                "declared_genotypes is deprecated. Use declared_zygote_types instead.",
                FutureWarning, stacklevel=2,
            )
            if declared_zygote_types is not None:
                raise ValueError(
                    "Cannot specify both declared_zygote_types and "
                    "declared_genotypes (deprecated alias)."
                )
            declared_zygote_types = declared_genotypes
        if declared_zygote_types is not None:
            self._declared_zygote_types = declared_zygote_types
        overrides: dict[str, bool] = {}
        if stochastic is not None:
            overrides["stochastic"] = stochastic
        if continuous_sampling is not None:
            overrides["continuous_sampling"] = continuous_sampling
        if fixed_egg_count is not None:
            overrides["fixed_egg_count"] = fixed_egg_count
        if overrides:
            self._config = self._config._replace(**overrides)
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
        return self

    # -- domain methods --------------------------------------------------------

    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[str, float | Sequence[int | float] | Mapping[int, int | float]]],
        sperm_storage: Mapping[str, Mapping[str, float | Sequence[int | float] | Mapping[int, int | float]]] | None = None,
    ) -> Self:
        """Set the initial population distribution (deferred — applied at build time).

        *individual_count* is a dict like
        ``{"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}}``.
        Genotype selectors accept both strings and ``Genotype`` objects.

        The distribution is NOT written to config immediately.  Instead it is
        stored in a deferred buffer and applied during :meth:`build` — after
        index compression, so that genotype selectors resolve to compressed
        indices.

        Args:
            individual_count: Per-sex, per-genotype initial counts.
                Nested as ``{sex: {genotype_selector: count}}``.
            sperm_storage: Per-sex, per-genotype initial stored sperm,
                same nesting structure as *individual_count*.

        Returns:
            Self for chaining.
        """
        if self._species is None:
            raise RuntimeError(
                "initial_state() requires a Species reference. "
                "Use Configurator.from_species() to create the instance."
            )
        from natal.configurator._factory import PopulationConfigBuilder

        n_ages = self._config.n_ages
        new_adult_age = self._config.new_adult_age
        array = PopulationConfigBuilder.resolve_age_structured_initial_individual_count(
            species=self._species,
            distribution=individual_count,
            n_ages=n_ages,
            new_adult_age=new_adult_age,
        )
        overrides: dict[str, object] = {"initial_individual_count": array}
        if sperm_storage is not None:
            overrides["initial_sperm_storage"] = \
                PopulationConfigBuilder.resolve_age_structured_initial_sperm_storage(
                    species=self._species,
                    sperm_storage=sperm_storage,
                    n_ages=n_ages,
                    new_adult_age=new_adult_age,
                )
        self._config = self._config._replace(**overrides)
        return self

    # -- custom fields ---------------------------------------------------------

    def custom(self, **kwargs: bool | int | float | NDArray[np.float64]) -> Self:
        """Register custom named fields on ``config.custom``.

        Multiple calls accumulate — ``.custom(a=1).custom(b=2)`` stores both.

        Unlike most domain methods which modify 0-d ndarrays in-place
        (sharing the same ``PopulationConfig`` reference with the Population),
        ``custom()`` must call ``_replace`` to create a new config object
        with a rebuilt ``custom`` structured array.  When called via
        ``pop.update().custom(...)``, the ``_pop_ref`` back-reference
        (set by ``BasePopulation._create_configurator()``) is used to write
        the new config back into the Population.

        Args:
            **kwargs: Name-value pairs for custom fields.  Values must be
                ``bool``, ``int``, ``float``, or ``NDArray[np.float64]``.

        Returns:
            Self for chaining.
        """
        from natal.data import build_custom_array

        self._custom_kwargs.update(kwargs)
        names = self._config.custom.dtype.names or ()

        if any(k not in names for k in kwargs):
            self._config = self._config._replace(
                custom=build_custom_array(self._custom_kwargs)
            )
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
        else:
            # All fields already exist — write incrementally in-place.
            # self._config shares array references with pop.config, so
            # no write-back is needed for existing fields.
            for key, value in kwargs.items():
                self._config.custom[key][()] = value
        return self

    # -- presets / modifiers / fitness (immediate — applied directly to config) --

    def presets(self, *presets: GeneticPreset) -> Self:
        """Apply genetic presets to config arrays.

        Each preset encapsulates modifier callables, fitness patches,
        and optionally a cytoplasmic tag.  Presets are applied in order.
        Modifier lists are accumulated — calling ``presets()`` again
        appends additional modifiers rather than replacing existing ones.

        When wired to a Population (via ``for_population()``), presets are
        applied directly to the Population — no adapter, no write-back needed.
        Otherwise the ``ConfigContext`` adapter path is used for build-time.

        Args:
            *presets: One or more ``GeneticPreset`` instances
                (e.g. ``HomingDrive``, ``ToxinAntidoteDrive``).

        Returns:
            Self for chaining.
        """
        if self._pop_ref is not None:
            # Collect presets, then apply in priority order.
            for preset in presets:
                self._pop_ref.add_preset(preset)
            self._pop_ref.refresh_modifiers()
            self._pop_ref.reapply_preset_fitness()
            self._config = self._pop_ref.config
            return self

        from natal.presets import apply_preset_to_population

        ctx = self._make_ctx()
        ctx.presets = list(presets)
        for preset in presets:
            apply_preset_to_population(ctx, preset)  # pyright: ignore[reportArgumentType]
        # Trigger map rebuild for cytoplasmic presets (which have no
        # gamete/zygote modifiers and thus do not auto-trigger rebuilds).
        has_cytoplasmic = any(isinstance(p, CytoplasmicPreset) for p in presets)
        if has_cytoplasmic:
            rebuild_config_maps(ctx)
        self._sync_from_ctx(ctx)
        return self

    def modifiers(
        self,
        gamete_modifiers: list[GameteModifier] | None = None,
        zygote_modifiers: list[ZygoteModifier] | None = None,
    ) -> Self:
        """Register gamete / zygote modifiers and rebuild maps immediately.

        Args:
            gamete_modifiers: List of :class:`~natal.modifiers.GameteModifier`
                instances affecting meiosis (genotype → gamete mapping).
            zygote_modifiers: List of :class:`~natal.modifiers.ZygoteModifier`
                instances affecting fertilisation (gamete → zygote mapping).

        Returns:
            Self for chaining.
        """
        if self._pop_ref is not None:
            if gamete_modifiers:
                for mod in gamete_modifiers:
                    self._pop_ref.add_gamete_modifier(mod)
            if zygote_modifiers:
                for mod in zygote_modifiers:
                    self._pop_ref.add_zygote_modifier(mod)
            if gamete_modifiers or zygote_modifiers:
                self._pop_ref.refresh_modifier_maps()
            self._config = self._pop_ref.config
            return self

        ctx = self._make_ctx()
        next_gid = ConfigContext.next_modifier_id(ctx.gamete_modifiers)
        if gamete_modifiers:
            for mod in gamete_modifiers:
                ctx.gamete_modifiers.append((next_gid, None, mod))
                next_gid += 1
        next_zid = ConfigContext.next_modifier_id(ctx.zygote_modifiers)
        if zygote_modifiers:
            for mod in zygote_modifiers:
                ctx.zygote_modifiers.append((next_zid, None, mod))
                next_zid += 1
        if gamete_modifiers or zygote_modifiers:
            rebuild_config_maps(ctx)
        self._sync_from_ctx(ctx)
        return self

    def fitness(
        self,
        viability: Mapping[str, float | Mapping[str, float]] | None = None,
        fecundity: Mapping[str, float | Mapping[str, float]] | None = None,
        sexual_selection: Mapping[str, float | Mapping[str, float]] | None = None,
        zygote_viability: Mapping[str, float | Mapping[str, float]] | None = None,
        mode: str = "replace",
    ) -> Self:
        """Write fitness values directly into config arrays.

        Each dict maps genotype-pattern strings (e.g. ``"WT|WT"``) to
        fitness multipliers.  *mode* can be ``"replace"`` (overwrite the
        fitness tensor) or ``"multiply"`` (scale existing values).

        Sex-specific fitness can be specified with nested dicts:
        ``{"female": {"WT|WT": 0.9}, "male": {"WT|WT": 1.0}}``.

        The ``@slab`` suffix is supported for slab-aware writing::

            cfg.fitness(viability={"A|a@infected": 0.5})
            # Writes 0.5 to ZType index = genotype_index × n_slabs + slab_index

        Without ``@slab``, the value is written to ALL slab columns
        (backward compatible).

        Args:
            viability: Per-genotype viability (juvenile survival) fitness.
            fecundity: Per-genotype fecundity (egg production) fitness.
            sexual_selection: Per-genotype mating success fitness (pair format).
            zygote_viability: Per-genotype zygote-stage survival fitness.
            mode: ``"replace"`` (overwrite) or ``"multiply"`` (scale existing).

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If ``_species`` is ``None`` — fitness resolution
                requires genotype information from the Species.
        """
        # Species guard — works for both from_species and for_population paths.
        if self._species is None:
            raise RuntimeError(
                "fitness() requires a Species. "
                "Use Configurator.from_species() to create this instance."
            )
        if self._registry is None:
            self._registry = build_registry(self._species)

        registry = self._registry
        all_genotypes = registry.index_to_genotype
        for patch_name, patch_dict in [
            ("viability", viability),
            ("fecundity", fecundity),
            ("sexual_selection", sexual_selection),
            ("zygote_viability", zygote_viability),
        ]:
            if patch_dict is not None:
                write_fitness_field(
                    self._config, patch_name, patch_dict, mode,
                    species=self._species, registry=registry,
                    all_genotypes=all_genotypes,
                )
        return self

    # -- deprecated compression methods (use setup(compress=True)) ----------------

    def compress_gametes(self, enabled: bool = True) -> Self:
        """Enable GType compression (deprecated — use setup(compress=True)).

        .. deprecated::
            Use ``setup(compress=True)`` instead.  This method will be
            removed in a future version.
        """
        import warnings
        warnings.warn(
            "compress_gametes() is deprecated. Use setup(compress=True) instead.",
            FutureWarning, stacklevel=2,
        )
        self._compress = enabled
        return self

    def compress_genotypes(self, enabled: bool = True) -> Self:
        """Enable ZType compression (deprecated — use setup(compress=True)).

        .. deprecated::
            Use ``setup(compress=True)`` instead.  This method will be
            removed in a future version.
        """
        import warnings
        warnings.warn(
            "compress_genotypes() is deprecated. Use setup(compress=True) instead.",
            FutureWarning, stacklevel=2,
        )
        self._compress = enabled
        return self

    # -- hooks ------------------------------------------------------------------

    def hooks(self, *hook_items: _HookItem) -> Self:
        """Register event hooks.

        Hooks are passed through to the Population constructor at
        ``build()`` time — they are *not* config writes.

        Args:
            *hook_items: Hook registrations. Each item can be a raw
                ``{event: [(func, name, priority), ...]}`` dict or a
                callable decorated with ``@hook(event='...')``.

        Returns:
            Self for chaining.
        """
        # Lazy allocation: most chain methods never touch hooks, so
        # we avoid creating the list until the first .hooks() call.
        if not hasattr(self, "_hook_items"):
            self._hook_items: list[_HookItem] = []
        self._hook_items.extend(hook_items)
        return self

    # -- observations ------------------------------------------------------------

    def with_observation(
        self,
        groups: object,
        *,
        collapse_age: bool = False,
    ) -> Self:
        """Register observation groups, applied at ``build()`` time.

        Args:
            groups: Observation groups (dict of name→spec, list of specs,
                or None for one-group-per-genotype).
            collapse_age: Whether to collapse the age axis in exports.

        Returns:
            Self for chaining.
        """
        self._observation_groups = groups
        self._observation_collapse_age = collapse_age
        return self

    # -- preset reconfiguration -------------------------------------------------

    def reconfigure_preset(self, preset: GeneticPreset, **changes: object) -> Self:
        """Modify a registered preset parameter and re-apply.

        Because ``presets()`` appends modifiers cumulatively, calling it
        again after changing a preset attribute would double-apply.  This
        method clears the modifier lists first, then re-applies the preset
        so it writes onto a clean slate.

        Args:
            preset: A preset previously registered via :meth:`presets`.
            **changes: Attribute name / value pairs to update on *preset*.

        Returns:
            Self for chaining.
        """
        for attr, value in changes.items():
            if not hasattr(preset, attr):
                raise AttributeError(
                    f"{type(preset).__name__} {preset.name!r} has no "
                    f"attribute {attr!r}. Cannot reconfigure a non-existent "
                    f"parameter — this would silently create a stray attribute "
                    f"on the preset object."
                )
            setattr(preset, attr, value)

        if self._pop_ref is not None:
            pop = self._pop_ref
            pop.refresh_modifiers()
            pop.reapply_preset_fitness()
            self._config = pop.config
        else:
            raise RuntimeError(
                "reconfigure_preset() requires a live Population. "
                "Use pop.update().reconfigure_preset(...) or "
                "Configurator.for_population(pop).reconfigure_preset(...)."
            )

        from natal.engine.simulation.age_structured import sync_equilibrium_metrics
        sync_equilibrium_metrics(self._config)
        return self

    # -- apply / build ---------------------------------------------------------

    def apply(self) -> Self:
        """Sync derived values (equilibrium metrics).

        All parameters are now applied immediately, so this is only
        needed when you modify config arrays directly (outside Configurator).

        Returns:
            Self for chaining.
        """
        self._sync_equilibrium()
        return self

    def _sync_equilibrium(self) -> None:
        """Recompute ``expected_competition_strength`` and ``expected_survival_rate``.

        Called by :meth:`apply` and after equilibrium-sensitive parameter
        changes (carrying capacity, eggs per female, sex ratio).  Results
        are written directly into the config's 0-d ndarray fields.

        If a custom ``equilibrium_distribution`` was stored (via
        ``competition(equilibrium_distribution=...)``), it is used as the
        target age structure for the Champer model.  If the user explicitly
        set ``expected_num_new_adult_females``, external egg counts are computed
        from that value instead of the distribution.

        For ``DiscretePopulationConfig``, survival/mating/reproduction arrays
        are constructed manually from scalar fields (age0_survival,
        adult_mating_rate, etc.) before calling
        ``compute_equilibrium_metrics``.
        """
        from natal.configurator._params import compute_expected_eggs_from_females
        from natal.engine.simulation.age_structured import (
            compute_equilibrium_metrics,
        )

        eq_dist: NDArray[np.float64] | None = getattr(
            self, "_equilibrium_distribution", None
        )
        # Reshape flat equilibrium_distribution to (n_sexes, n_ages) if needed.
        config = self._config
        if isinstance(eq_dist, np.ndarray) and eq_dist.ndim == 1:
            n_ages = int(config.n_ages)
            eq_dist = eq_dist.reshape(2, n_ages)

        # Compute external_expected_eggs from expected_num_new_adult_females
        # Only when the user explicitly set it (avoid default config value)
        external_eggs: float | None = None
        if getattr(self, "_has_user_expected_new_adult_females", False):
            if isinstance(config, DiscretePopulationConfig):
                ext_surv: NDArray[np.float64] = np.array([
                    [config.female_age0_survival, 0.0],
                    [config.male_age0_survival, 0.0],
                ])
                ext_repro: NDArray[np.float64] = np.array([
                    0.0, config.reproduction_rate,
                ])
            else:
                ext_surv = config.age_based_survival_rates
                ext_repro = config.age_based_reproduction_rates

            external_eggs = compute_expected_eggs_from_females(
                expected_num_new_adult_females=getattr(self, "_user_expected_new_adult_females", 500.0),
                eggs_per_female=float(config.eggs_per_female),
                age_based_survival_rates=ext_surv,
                age_based_reproduction_rates=ext_repro,
                female_age_based_fertility=config.female_age_based_fertility,
                sex_ratio=float(config.sex_ratio),
                new_adult_age=int(config.new_adult_age),
                n_ages=int(config.n_ages),
            )

        if isinstance(config, DiscretePopulationConfig):
            surv: NDArray[np.float64] = np.array([
                [config.female_age0_survival, 0.0],
                [config.male_age0_survival, 0.0],
            ])
            mate: NDArray[np.float64] = np.array([
                [0.0, config.female_adult_mating_rate],
                [0.0, config.male_adult_mating_rate],
            ])
            repro: NDArray[np.float64] = np.array([
                0.0, config.reproduction_rate,
            ])
        else:
            surv = config.age_based_survival_rates
            mate = config.age_based_mating_rates
            repro = config.age_based_reproduction_rates

        expected_comp, expected_surv = compute_equilibrium_metrics(
            carrying_capacity=float(config.carrying_capacity),
            eggs_per_female=float(config.eggs_per_female),
            age_based_survival_rates=surv,
            age_based_mating_rates=mate,
            age_based_reproduction_rates=repro,
            female_age_based_fertility=config.female_age_based_fertility,
            relative_competition_strength=config.age_based_relative_competition_strength,
            sex_ratio=float(config.sex_ratio),
            new_adult_age=int(config.new_adult_age),
            n_ages=int(config.n_ages),
            equilibrium_individual_count=eq_dist,
            external_expected_eggs=external_eggs,
        )
        config.expected_competition_strength[()] = expected_comp
        config.expected_survival_rate[()] = expected_surv

    def build(
        self,
        name: str | None = None,
        hooks: HookMap | None = None,
    ) -> DiscreteGenerationPopulation | AgeStructuredPopulation:
        """Finalise the config and create a Population.

        This is the terminal method of the build chain::

            Configurator.from_species()
                .age_structure(5, 2)
                .competition(K=5000)
                .reproduction(eggs=100)
                .build(name="pop")

        Internally it: (1) syncs equilibrium metrics via :meth:`apply`,
        (2) runs compression if enabled, (3) flushes deferred-write buffers,
        (4) merges hooks, and (5) passes ``self._config`` to the Population
        constructor.

        .. note::

           ``fitness()`` and ``initial_state()`` store their patches in
           deferred buffers rather than writing config immediately.  The
           buffers are flushed here — AFTER compression — so all genotype
           selectors resolve to compressed indices.  Genotypes pruned by
           compression raise ``ValueError`` at flush time instead of being
           silently dropped.

        Args:
            name: Population name (falls back to ``.setup(name=...)``
                or ``"Population"``).
            hooks: Additional hook registrations merged with any stored
                via :meth:`hooks`.

        Returns:
            ``AgeStructuredPopulation`` or ``DiscreteGenerationPopulation``,
            depending on whether *self._config* is a ``PopulationConfig``
            or ``DiscretePopulationConfig``.
        """
        # Sync equilibrium metrics and apply index compression (if enabled).
        self.apply()

        # Compression runs on a COPY — self._config stays in G_orig space.
        # Population receives the compressed config.  All user writes
        # (fitness, initial_state) already happened on the full-size
        # config; compression subslices the arrays naturally.
        final_config = self._config
        if self._compress and not self._compression_applied:
            ctx = self._make_ctx()
            ctx.compress = True  # compression only happens at build time
            rebuild_config_maps(ctx)
            self._sync_from_ctx(ctx)
            final_config = ctx.config  # compressed copy

        # Custom kwargs (accumulated by .custom()) applied to final config.
        if self._custom_kwargs:
            from natal.data import build_custom_array
            final_config = final_config._replace(
                custom=build_custom_array(self._custom_kwargs)
            )

        # Resolve name: explicit argument > setup(name=...) > default
        if name is None:
            name = getattr(self, "_name", "Population")

        # Merge stored hooks (from .hooks()) with passed hooks.
        stored_hooks: list[_HookItem] | None = getattr(self, "_hook_items", None)
        if stored_hooks:
            hook_map = merge_hooks(stored_hooks)
            if hooks:
                # Merge external hooks into stored ones
                for event, items in (hooks or {}).items():
                    hook_map.setdefault(event, []).extend(items)
            hooks = hook_map

        # Determine population class from config type.
        if self._species is None:
            raise RuntimeError(
                "Cannot build Population: no Species set. "
                "Use Configurator.from_species() to create this instance."
            )

        if isinstance(final_config, DiscretePopulationConfig):
            # Sync pre-extracted discrete fields from the latest source maps.
            # These may be stale if gamete/fitness maps were rebuilt by presets
            # or if equilibrium metrics were recomputed after construction.
            replace_kwargs: dict[str, object] = {
                "meiosis_f": final_config.zygotes_to_gametes_map[0],
                "meiosis_m": final_config.zygotes_to_gametes_map[1],
                "fecundity_f": final_config.fecundity_fitness[0],
                "fecundity_m": final_config.fecundity_fitness[1],
            }
            # Viability source array is only present on full PopulationConfig.
            if hasattr(final_config, "viability_fitness"):
                replace_kwargs["viability_f"] = final_config.viability_fitness[0, 0, :]  # type: ignore[reportAttributeAccessIssue]
                replace_kwargs["viability_m"] = final_config.viability_fitness[1, 0, :]  # type: ignore[reportAttributeAccessIssue]
            final_config = final_config._replace(**replace_kwargs)
            from natal.population.discrete_generation import (
                DiscreteGenerationPopulation,
            )

            pop: DiscreteGenerationPopulation | AgeStructuredPopulation = \
                DiscreteGenerationPopulation(
                    species=self._species,
                    population_config=final_config,
                    index_registry=self._registry,
                    name=name,
                    hooks=hooks,
                )
        else:
            from natal.population.age_structured import (
                AgeStructuredPopulation,
            )

            pop = AgeStructuredPopulation(
                species=self._species,
                population_config=final_config,
                index_registry=self._registry,
                name=name,
                hooks=hooks,
            )

        # Apply observation groups if registered.
        obs_groups = getattr(self, "_observation_groups", None)
        if obs_groups is not None:
            collapse = getattr(self, "_observation_collapse_age", False)
            pop.set_observations(obs_groups, collapse_age=collapse)
        return pop


# ── Numba hook wrapper ─────────────────────────────────────────────────────────


@njit_switch(cache=True)
def hook_set_param(config: object, name: str, value: float) -> None:
    """Set a simulation parameter from inside a Numba hook.

    Wraps :func:`set_param` in an objmode context so the call is valid
    from nopython-compiled hook functions.  Use when you need the
    flexibility of string-name lookup at the cost of an objmode boundary
    (~microseconds per call).

    For the fastest path, write ``config.field[()] = v`` directly in
    nopython — no objmode overhead.

    Args:
        config: The PopulationConfig or DiscretePopulationConfig.
        name: Parameter name — ``"competition.carrying_capacity"``,
              ``"carrying_capacity"``, or any registered alias.
        value: New scalar value.
    """
    with objmode():
        set_param(config, name, value)  # pyright: ignore[reportArgumentType] — objmode converts njit types to Python




