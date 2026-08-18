"""Mutable wrapper for PopulationConfig, with a chainable API.

Provides read/write access to config fields through a chainable API,
registration of custom named parameters stored as structured numpy arrays,
and freeze-back to an immutable NamedTuple via ``_replace`` (cheap — all
ndarray fields are shared by reference).  When wired to a live Population
through ``for_population()``, config mutations propagate back to the
Population automatically through ``set_config()``.

``PopulationConfig`` and ``DiscretePopulationConfig`` are immutable
NamedTuples whose fields cannot be modified once created.  During
simulation setup and inside hooks at runtime, however, parameters need
real-time adjustment.  The ``Configurator`` provides a mutable layer on
top: all modifications write into the config arrays in-place, and the
final immutable config is materialised via ``build()``.

The adapter class ``ConfigContext`` lets genetic presets and modifiers
operate on config arrays without needing a live Population object.  The
standalone :func:`set_param` function is also usable from within
Numba-compiled hooks via ``objmode``.

Key classes are ``Configurator`` (base with chainable domain methods),
``DiscreteConfigurator`` (non-overlapping generations), and
``AgeStructuredConfigurator`` (overlapping generations).

See also :func:`set_param` (low-level scalar writer) and
:func:`hook_set_param` (Numba-safe wrapper for use in hooks).
"""

from __future__ import annotations

from copy import copy, deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    Optional,
    Self,
    Sequence,
    cast,
)

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
    from natal.patterns import IndividualSelector
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


def normalize_observation_groups(groups: object) -> dict[str, IndividualSelector]:
    """Validate the runtime boundary for observation group definitions.

    ``object`` is intentional here: Python callers can pass values that do not
    satisfy the public type annotation, and this boundary must reject them with
    the documented exception instead of failing later during compilation.

    Args:
        groups: Runtime value supplied to ``with_observation()``.

    Returns:
        A new insertion-ordered mapping of validated labels and selectors.

    Raises:
        TypeError: If the value is not a mapping of selectors.
        ValueError: If the mapping or a label is empty.
    """
    from collections.abc import Mapping as MappingABC

    from natal.patterns import IndividualSelector

    if not isinstance(groups, MappingABC):
        raise TypeError("groups must be a non-empty mapping of IndividualSelector values")
    if not groups:
        raise ValueError("groups must be non-empty")
    typed_groups = cast(Mapping[object, object], groups)
    normalized: dict[str, IndividualSelector] = {}
    for label, selector in typed_groups.items():
        if not isinstance(label, str):
            raise TypeError("Observation group labels must be strings")
        if not label:
            raise ValueError("Observation group labels must be non-empty")
        if not isinstance(selector, IndividualSelector):
            raise TypeError(
                f"Observation group {label!r} must use IndividualSelector"
            )
        normalized[label] = selector
    return normalized


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


def _collect_genotype_strings(genotype_ref: str | Sequence[str]) -> set[str]:
    """Extract non-wildcard genotype strings from a HookOp.genotypes value."""
    result: set[str] = set()
    if isinstance(genotype_ref, str):
        if genotype_ref != "*":
            result.add(genotype_ref)
    else:
        for s in genotype_ref:
            if s != "*":
                result.add(s)
    return result


def collect_hook_genotype_refs(hook_items: list[_HookItem]) -> set[str]:
    """Extract genotype string references from hook items for compression seeds.

    - Selector hooks: reads ``func.selectors`` metadata directly.
    - Declarative hooks: calls function once, extracts ``op.genotypes``.
    - Custom hooks: skipped.
    """
    refs: set[str] = set()
    for item in hook_items:
        if isinstance(item, dict):
            hook_dict: HookMap = cast(HookMap, item)
            for registrations in hook_dict.values():
                for hook_reg in registrations:
                    refs.update(_extract_refs_from_callable(hook_reg[0]))
        elif callable(item):
            refs.update(_extract_refs_from_callable(item))
    return refs


def _extract_refs_from_callable(func: Callable[..., Any]) -> set[str]:
    """Extract genotype strings from a single hook callable."""
    selectors = getattr(func, "selectors", None)
    if selectors:
        result: set[str] = set()
        for val in selectors.values():
            if isinstance(val, str):
                result.update(_collect_genotype_strings(val))
            elif isinstance(val, (list, tuple)):
                for v in val:  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(v, str):
                        result.update(_collect_genotype_strings(v))
        return result

    meta = getattr(func, "meta", None)
    is_custom = getattr(func, "custom", False) or (meta and meta.get("custom"))
    if not is_custom and meta:
        try:
            ops = func()
            if isinstance(ops, list):
                result: set[str] = set()
                for op in ops:  # pyright: ignore[reportUnknownVariableType]
                    genotypes = getattr(op, "genotypes", None)  # pyright: ignore[reportUnknownArgumentType]
                    if isinstance(genotypes, str):
                        result.update(_collect_genotype_strings(genotypes))
                    elif isinstance(genotypes, (list, tuple)):
                        for g in genotypes:  # pyright: ignore[reportUnknownVariableType]
                            if isinstance(g, str):
                                result.update(_collect_genotype_strings(g))
                return result
        except Exception:
            pass

    return set()


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
        # Preset identity must survive build() so runtime refresh and
        # reconfiguration can reconstruct modifiers from the original recipes.
        self._presets: list[GeneticPreset] = []

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

        # Observation and History are independent build-time policies.
        self._observation_groups: Mapping[str, IndividualSelector] | None = None
        self._observation_collapse_age = False
        self._record_history_mode: Literal["raw", "observation"] = "raw"
        self._record_history_max_rows: int | None = None

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
        backend: Literal["auto", "rust", "numba", "python"] | None = None,
        declared_zygote_types: Sequence[str] | Sequence[int] | None = None,
        declared_genotypes: Sequence[str] | Sequence[int] | None = None,  # deprecated alias
    ) -> Self:
        """Configure simulation flags and optional population name.

        *name* is stored and used by ``build()`` when no explicit name is given.

        *compress* enables index compression at build time.  It enables both
        GType (gamete-axis) and ZType (genotype-axis) compression in one flag.
        The older ``compress_gametes()`` / ``compress_genotypes()`` chain
        methods have been removed — use this parameter instead.

        *declared_zygote_types* is a sequence of genotype strings (``"WT|WT"``) or
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
            backend: Lifecycle backend selector used by ``build()``.
                ``None`` preserves any earlier setting; ``"numba"`` preserves
                the legacy JIT path; ``"python"`` forces the pure-Python
                fallback; ``"auto"`` selects Rust when available and CSR-only;
                ``"rust"`` forces the Rust backend.
            declared_zygote_types: Optional sequence of genotype selectors to protect
                from compression pruning.

        Returns:
            Self for chaining.
        """
        if name is not None:
            self._name = name
        if compress:
            self._compress = True
        if backend is not None:
            self._backend: Literal["auto", "rust", "numba", "python"] = backend
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
            self._declared_zygote_types = cast("set[str] | set[int]", set(declared_zygote_types))
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

        .. note::

            ``initial_state()`` is a build-time operation and is **not**
            available at runtime via ``pop.update()``.  The initial
            distribution is baked into the config at construction time;
            changing it after the population has been built has no effect
            on the ongoing simulation state.

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
            pop = self._pop_ref
            original_config = pop.config
            original_presets = pop.presets
            original_gamete = pop.gamete_modifiers
            original_zygote = pop.zygote_modifiers
            preset_bindings: list[tuple[GeneticPreset, Species | None]] = [
                (preset, preset._bound_species)  # pyright: ignore[reportPrivateUsage]  # rollback must preserve binding after a failed first registration.
                for preset in presets
            ]
            # Fitness writers mutate arrays in place.  Rebuild against an
            # isolated config so an exception can restore the exact original
            # config identity and every observable array.
            pop.set_config(deepcopy(original_config))
            try:
                for preset in presets:
                    pop.add_preset(preset)
                pop.refresh_modifiers()
                pop.reapply_preset_fitness()
            except Exception:
                pop._presets = original_presets  # pyright: ignore[reportPrivateUsage]  # transactional rollback restores internal registration metadata.
                pop._gamete_modifiers = original_gamete  # pyright: ignore[reportPrivateUsage]  # transactional rollback restores derived modifier metadata.
                pop._zygote_modifiers = original_zygote  # pyright: ignore[reportPrivateUsage]  # transactional rollback restores derived modifier metadata.
                pop.set_config(original_config)
                for preset, bound_species in preset_bindings:
                    preset._bound_species = bound_species  # pyright: ignore[reportPrivateUsage]  # restore the caller-owned preset exactly.
                self._config = original_config
                raise
            self._config = pop.config
            return self

        from natal.presets import apply_preset_to_population

        new_presets: list[GeneticPreset] = []
        for preset in presets:
            if any(registered is preset for registered in self._presets) or any(
                registered is preset for registered in new_presets
            ):
                continue
            new_presets.append(preset)
        if not new_presets:
            return self

        original_config = self._config
        original_registry = self._registry
        original_gamete = list(self.gamete_modifiers)
        original_zygote = list(self.zygote_modifiers)
        original_presets = list(self._presets)
        original_compression_applied = self._compression_applied
        preset_bindings: list[tuple[GeneticPreset, Species | None]] = [
            (preset, preset._bound_species)  # pyright: ignore[reportPrivateUsage]  # rollback must preserve binding after a failed build-time registration.
            for preset in new_presets
        ]
        # Preset fitness and modifier rebuilding may mutate arrays in place.
        # Work against an isolated config and publish it only after every new
        # preset has completed successfully.
        if isinstance(original_config, DiscretePopulationConfig):
            isolated_config: PopulationConfig | DiscretePopulationConfig = deepcopy(
                original_config
            )
        else:
            isolated_config = deepcopy(original_config)
        self._config = isolated_config
        self._presets.extend(new_presets)
        try:
            ctx = self._make_ctx()
            ctx.presets = list(self._presets)
            for preset in new_presets:
                apply_preset_to_population(ctx, preset)  # pyright: ignore[reportArgumentType]
            # Cytoplasmic presets have no gamete/zygote modifier that would
            # otherwise trigger a map rebuild.
            if any(isinstance(p, CytoplasmicPreset) for p in new_presets):
                rebuild_config_maps(ctx)
            self._sync_from_ctx(ctx)
        except Exception:
            self._config = original_config
            self._registry = original_registry
            self.gamete_modifiers = original_gamete
            self.zygote_modifiers = original_zygote
            self._presets = original_presets
            self._compression_applied = original_compression_applied
            for preset, bound_species in preset_bindings:
                preset._bound_species = bound_species  # pyright: ignore[reportPrivateUsage]  # restore the caller-owned preset exactly.
            raise
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

        Raises:
            RuntimeError: When called on a runtime Configurator
                (i.e. via ``pop.update().hooks()``).  Use
                ``pop.set_hook()`` for runtime hook registration.
        """
        if self._pop_ref is not None:
            raise RuntimeError(
                "hooks can only be registered at build time. "
                "Use pop.set_hook() for runtime hook registration."
            )
        if not hasattr(self, "_hook_items"):
            self._hook_items: list[_HookItem] = []
        self._hook_items.extend(hook_items)
        return self

    # -- observations ------------------------------------------------------------

    def with_observation(
        self,
        groups: Mapping[str, IndividualSelector],
        *,
        collapse_age: bool = False,
    ) -> Self:
        """Register observation groups, applied at ``build()`` time.

        Only valid during the build phase (``_pop_ref is None``).
        Calling this on a runtime Configurator raises ``RuntimeError``.

        Args:
            groups: Non-empty ordered mapping from labels to selectors.
            collapse_age: Whether to collapse the age axis in exports.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: When called on a runtime Configurator.
            TypeError: If groups is not a mapping of selectors.
            ValueError: If groups or a group label is empty.
        """
        if self._pop_ref is not None:
            raise RuntimeError(
                "with_observation() is only valid during the build phase. "
                "Observation rules cannot change after the Population has been built."
            )
        self._observation_groups = normalize_observation_groups(groups)
        self._observation_collapse_age = collapse_age
        return self

    def record_history(
        self,
        *,
        mode: Literal["raw", "observation"] = "raw",
        max_rows: Optional[int] = None,
    ) -> Self:
        """Set the recording mode and capacity for this population's history.

        Must be called during the build phase.  Calling this on a runtime
        Configurator raises ``RuntimeError``.

        When ``mode="observation"`` and no ``.with_observation()`` has been
        called, an identity observation (one group per ZType) is
        automatically generated.

        Args:
            mode: ``"raw"`` for full-state recording or ``"observation"``
                for compressed observation-aggregate recording.
            max_rows: Maximum number of records to keep (FIFO eviction).
                ``None`` means unlimited.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: When called on a runtime Configurator.
            ValueError: When mode is invalid or ``max_rows`` is less than one.
        """
        if self._pop_ref is not None:
            raise RuntimeError(
                "record_history() is only valid during the build phase. "
                "Recording settings cannot change after the Population has "
                "been built. Use pop.clear_history() to reset."
            )
        if mode not in ("raw", "observation"):
            raise ValueError(
                f"mode must be 'raw' or 'observation', got {mode!r}"
            )
        if max_rows is not None and max_rows < 1:
            raise ValueError(
                f"max_rows must be >= 1 or None, got {max_rows}"
            )
        self._record_history_mode = mode
        self._record_history_max_rows: Optional[int] = max_rows
        return self

    # -- preset reconfiguration -------------------------------------------------

    def reconfigure_preset(self, preset: GeneticPreset, **changes: object) -> Self:
        """Modify a registered preset parameter and re-apply.

        Because ``presets()`` appends modifiers cumulatively, calling it
        again after changing a preset attribute would double-apply.  This
        method clears the modifier lists first, then re-applies the preset
        so it writes onto a clean slate.

        Validation happens entirely before any mutation: if the preset is
        not registered or an attribute name is invalid, the exception is
        raised and the preset object is left unchanged (error-path state
        invariant).

        Args:
            preset: A preset previously registered via :meth:`presets`.
            **changes: Attribute name / value pairs to update on *preset*.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If *preset* is not registered on this population.
            AttributeError: If any key in *changes* is not an attribute of
                *preset*.
            TypeError: If a changed value is incompatible with the preset.
            RuntimeError: If called without a live Population backref.
        """
        # ── Validate phase (zero side effects) ──
        if self._pop_ref is None:
            raise RuntimeError(
                "reconfigure_preset() requires a live Population. "
                "Use pop.update().reconfigure_preset(...) or "
                "Configurator.for_population(pop).reconfigure_preset(...)."
            )
        pop = self._pop_ref
        if preset not in pop._presets:  # pyright: ignore[reportPrivateUsage]
            raise ValueError(
                f"Preset {preset.name!r} is not registered on this "
                f"population. Use presets() to register it first."
            )
        for attr in changes:
            if not hasattr(preset, attr):
                raise AttributeError(
                    f"{type(preset).__name__} {preset.name!r} has no "
                    f"attribute {attr!r}. Cannot reconfigure a non-existent "
                    f"parameter — this would silently create a stray attribute "
                    f"on the preset object."
                )

        # Exercise the complete rebuild on an isolated population before
        # mutating the registered preset.  Calling only the recipe factories
        # is insufficient because a custom modifier may fail later, when its
        # returned callable is invoked by refresh_modifier_maps().
        candidate = copy(preset)
        for attr, value in changes.items():
            setattr(candidate, attr, value)
        trial = pop._clone(  # pyright: ignore[reportPrivateUsage]
            f"{pop.name}__preset_validation__",
            config=deepcopy(pop.config),
        )
        trial._presets = [  # pyright: ignore[reportPrivateUsage]
            candidate if registered is preset else registered
            for registered in trial._presets  # pyright: ignore[reportPrivateUsage]
        ]
        trial.refresh_modifiers()
        trial.reapply_preset_fitness()

        # ── Commit phase ──
        for attr, value in changes.items():
            setattr(preset, attr, value)

        pop.refresh_modifiers()
        pop.reapply_preset_fitness()
        self._config = pop.config

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
            # Auto-collect genotype refs from hooks so genotypes introduced
            # only via hooks survive BFS pruning.
            stored_hooks: list[_HookItem] | None = getattr(
                self, "_hook_items", None
            )
            if stored_hooks:
                hook_refs = collect_hook_genotype_refs(stored_hooks)
                if hook_refs:
                    existing = self._declared_zygote_types
                    self._declared_zygote_types = cast(
                        "set[str] | set[int]",
                        (existing | hook_refs)
                        if existing is not None
                        else hook_refs,
                    )

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

        # Configurator applies modifiers before Population construction.  Carry
        # both the recipe objects and their current derived callables across the
        # boundary so refresh_modifiers() and reconfigure_preset() behave the
        # same for build-time and runtime preset registration.
        pop._presets = list(self._presets)  # pyright: ignore[reportPrivateUsage]
        pop._gamete_modifiers = list(self.gamete_modifiers)  # pyright: ignore[reportPrivateUsage]
        pop._zygote_modifiers = list(self.zygote_modifiers)  # pyright: ignore[reportPrivateUsage]

        # Apply the requested lifecycle backend.  ``auto`` silently falls
        # back to Numba when the Rust extension is unavailable or custom
        # hooks are present; ``rust`` propagates the failure.
        backend = getattr(self, "_backend", "numba")
        if backend == "python":
            pop._python_backend = True  # pyright: ignore[reportPrivateUsage]
        elif backend in ("auto", "rust"):
            from natal.engine.backends.rust_backend import rust_backend_available

            if backend == "rust" or rust_backend_available():
                try:
                    pop.enable_rust_backend()  # type: ignore[reportAttributeAccessIssue]  # both concrete Population classes expose this method
                except RuntimeError:
                    if backend == "rust":
                        raise

        # Compile and freeze the recording plan.
        self._compile_recording_plan(pop)
        return pop

    def _compile_recording_plan(self, pop: AgeStructuredPopulation | DiscreteGenerationPopulation) -> None:
        """Compile and freeze the :class:`RecordingPlan` on the population.

        Called at the end of :meth:`build` after all observations and
        initial state have been applied.  The plan is immutable for the
        remainder of the population's lifetime.
        """
        from natal.data import DiscretePopulationConfig
        from natal.output._recording import compile_recording_plan
        from natal.output.history import History
        from natal.output.observation import build_identity_observation

        config = pop.config
        if isinstance(config, DiscretePopulationConfig):
            kind = "discrete_generation"
            has_sperm = False
        else:
            kind = "age_structured"
            has_sperm = True

        from natal.output.observation import ObservationFilter

        obs_groups = self._observation_groups
        if obs_groups is None:
            observation = build_identity_observation(
                pop.index_registry,
                n_ztypes=pop.index_registry.n_ztypes,
                n_sexes=pop.config.n_sexes,
                n_ages=pop.config.n_ages,
            )
        else:
            observation = ObservationFilter(pop.index_registry).build_from_selectors(
                groups=dict(obs_groups),
                collapse_age=self._observation_collapse_age,
                n_sexes=pop.config.n_sexes,
                n_ages=pop.config.n_ages,
                n_ztypes=pop.index_registry.n_ztypes,
            )
        pop._observation = observation  # type: ignore[reportPrivateUsage]  # build-time installation of the immutable canonical rule
        record_mode = self._record_history_mode
        max_rows = self._record_history_max_rows

        plan = compile_recording_plan(
            pop,
            mode=record_mode,
            kind=kind,
            n_demes=1,
            has_sperm_storage=has_sperm,
            observation=observation,
        )
        from dataclasses import replace

        observation = replace(
            observation,
            population_fingerprint=plan.schema.population.fingerprint,
        )
        pop._observation = observation  # type: ignore[reportPrivateUsage]  # bind canonical rule to the frozen PopulationLayout
        # ``_observation_mask`` is an engine recording input, not the
        # canonical query rule. Keeping it ``None`` in raw mode prevents the
        # lifecycle wrapper from silently switching the row layout merely
        # because every Population now owns an Observation.
        pop._observation_mask = plan.observation_mask  # type: ignore[reportPrivateUsage]  # frozen engine input derived from RecordingPlan
        pop._recording_plan = plan  # type: ignore[reportPrivateUsage]  # configurator sets private attr on population
        pop._history_obj = History(plan.schema, max_rows=max_rows)  # type: ignore[reportPrivateUsage]  # configurator sets private attr


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
