"""Mutable wrapper for ModelDraft, with a chainable API.

Provides read/write access to draft fields through a chainable API,
registration of custom named parameters stored as structured numpy arrays,
and shallow-copy via ``_replace`` (cheap — all
ndarray fields are shared by reference).  When wired to a live Population
through ``for_population()``, config mutations propagate back to the
Population automatically through ``set_config()`` and the live Rust
session (when present) receives the same values immediately.

``ModelDraft`` is an immutable NamedTuple whose fields cannot be
replaced once created.  During
simulation setup and inside hooks at runtime, however, parameters need
real-time adjustment.  The ``Configurator`` provides a mutable layer on
top: all modifications route through the declarative route table
(:mod:`natal.frontend.configurator._routes`) via batch writers
(:mod:`natal.frontend.configurator._writers`), and the
final draft is materialized via ``build()``.

The adapter class ``ConfigContext`` lets genetic presets and modifiers
operate on config arrays without needing a live Population object.

Since slice 3 there is exactly one ``Configurator`` class: the former
``AgeStructuredConfigurator`` / ``DiscreteConfigurator`` split was a
code duplication of parameter shapes, now expressed as data in the
route table.  Discrete-specific vocabulary (``female_age0_survival``,
``female_adult_mating_rate``, ...) keeps working — those names route to
single cells of the unified vectors.

See also :func:`set_param` (low-level scalar writer).
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
from numpy.typing import NDArray

from natal.contracts.materialize import (
    gtype_names_from_registry,
    ztype_names_from_registry,
)
from natal.frontend.configurator._params import (
    compute_expected_eggs_from_females,
)
from natal.frontend.configurator._registry_builder import (
    ConfigContext,
    build_registry,
    rebuild_config_maps,
)
from natal.frontend.configurator._routes import (
    dispatch,
    lookup_or_none,
    sync_equilibrium_for_draft,
)
from natal.frontend.configurator._writers import (
    ConfigWriter,
    CoreConfigWriter,
    DraftWriter,
)
from natal.frontend.data import (
    ModelDraft,
)
from natal.frontend.genetics import Species
from natal.frontend.hooks.types import DemeSelector
from natal.frontend.presets import CytoplasmicPreset
from natal.frontend.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
    from natal.frontend.patterns import IndividualSelector
    from natal.frontend.population.age_structured import AgeStructuredPopulation
    from natal.frontend.population.base import BasePopulation
    from natal.frontend.population.discrete_generation import (
        DiscreteGenerationPopulation,
    )
    from natal.frontend.presets import GeneticPreset

__all__ = [
    "Configurator",
    "set_param",
]

# ── Type aliases for hook registrations ──────────────────────────────────────

# One stored .hooks() call: (items, kwargs) pairs, replayed onto the
# population at build time or forwarded immediately at runtime.
HookCall = tuple[tuple[object, ...], dict[str, object]]
# A hook item: an Op, an op list, or a callable (decorated or a plain
# single-parameter callback).
_HookItem = object


# Genetics tensors rebuilt wholesale by preset/modifier application.  The
# Rust bridge refreshes these as whole tensors instead of tracking cells.
_RUST_GENETICS_TENSORS: frozenset[str] = frozenset({
    "viability_fitness",
    "fecundity_fitness",
    "sexual_selection_fitness",
    "zygote_viability_fitness",
    "offspring_tensor",
    "meiosis_map",
    "female_ztype_compatibility",
    "male_ztype_compatibility",
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

    from natal.frontend.patterns import IndividualSelector

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
    config: ModelDraft,
    name: str,
    value: float | int | bool,
    *,
    _sync_equilibrium: bool = True,
    _dirty: set[str] | None = None,
) -> ModelDraft:
    """Set a simulation parameter by its user-facing name.

    The write is routed through the declarative route table
    (:mod:`natal.frontend.configurator._routes`): the name (full key,
    short name, or alias) resolves to a route entry, the value is
    parsed and validated according to the entry's ``kind``, and the
    write is committed into the draft.  Entries flagged ``sensitive``
    in ``parameters.jsonc`` (carrying capacity, eggs per female, sex
    ratio, the Champer overrides) automatically refresh the equilibrium
    metric caches unless ``_sync_equilibrium=False`` is passed.

    Scalar NamedTuple slots (including the ecology scalars) are written
    through ``_replace``, so the returned draft must be rebound to the
    caller's variable::

        config = set_param(config, "competition.carrying_capacity", 5000.0)

    Array-backed fields (custom slots and vector/tensor contents) are
    mutated in place and the same draft is returned.

    Usable from pure Python and Configurator chain methods.

    If *name* matches a custom field on ``config.custom`` (registered via
    :meth:`Configurator.custom`), it is written directly — no route
    lookup needed.

    Args:
        config: The ModelDraft to modify.
        name: Parameter name — full key ``"competition.carrying_capacity"``,
              short name ``"carrying_capacity"``, or alias.
        value: New value (scalar). For tensor, vector, and row
               parameters, use the Configurator methods or
               ``pop.params.tensor_write`` instead.
        _sync_equilibrium: Refresh equilibrium caches for sensitive keys
               (internal; callers that sync explicitly pass ``False``).
        _dirty: Optional sink set receiving the corresponding contract
               (Params) field name after a successful write — the Rust
               dirty bridge. ``None`` skips marking entirely.

    Returns:
        The (possibly replaced) draft carrying the committed write.

    Raises:
        KeyError: If *name* is not a registered parameter or custom field.
        TypeError: If *name* refers to an immutable structural field.
        ValueError: If *name* refers to a tensor/vector/row parameter,
               or the value fails bounds validation.
    """
    entry = lookup_or_none(name)
    if entry is None:
        # Fallback: check custom slots (not in the route table).  The
        # slot dict is shared mutable content on the draft, so the write
        # is visible to the population without a _replace.
        from natal.frontend.data import build_custom_slots

        if name in getattr(config, "custom", ()):
            config.custom[name] = build_custom_slots({name: value})[name]
            return config
        raise KeyError(f"Unknown parameter: {name!r}")
    if entry.kind in ("geno_tensor", "age_vec", "sex_row"):
        raise ValueError(
            f"set_param does not support tensor or array parameters "
            f"like {name!r}. Use the corresponding Configurator method "
            f"or pop.params.tensor_write instead."
        )
    return dispatch(config, name, value, dirty_sink=_dirty, sync_sensitive=_sync_equilibrium)


# ── Helpers: fitness field writing ─────────────────────────────────────────────


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


def collect_hook_genotype_refs(hook_calls: list[HookCall]) -> set[str]:
    """Extract genotype string references from hook calls for compression seeds.

    Ensures genotypes introduced only via hooks survive BFS pruning:
    declarative op lists contribute their ``op.genotypes`` strings, and
    selector hooks contribute their resolved ``selectors`` specs.
    """
    from natal.frontend.hooks.types import HookOp

    refs: set[str] = set()

    def _collect_item(item: object) -> None:
        if isinstance(item, HookOp):
            refs.update(_collect_genotype_strings(item.genotypes))
        elif isinstance(item, (list, tuple)):
            for inner in cast("Sequence[object]", item):
                _collect_item(inner)
        elif callable(item):
            refs.update(_extract_refs_from_callable(item))

    for items, _kwargs in hook_calls:
        for item in items:
            _collect_item(item)
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
    if meta:
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

    Wraps a ModelDraft and provides chainable domain methods
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
        config: ModelDraft,
        species: Species | None = None,
    ) -> None:
        """Wrap a config for chainable modification.

        Args:
            An existing ModelDraft.
            species: Required for methods that need genotype resolution
                (initial_state, presets, modifiers, fitness).  Can be
                omitted when the Configurator is only used for scalar
                parameter updates via set_param.
        """
        self._config: ModelDraft = config
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

        # Accumulated kwargs for user custom slots.  Each .custom() call
        # adds to this dict; build_custom_slots() normalizes it whenever
        # the values are (re)applied to the draft.
        self._custom_kwargs: dict[str, object] = {}

        # optional backref for writing config updates back to a Population when created via for_population()
        self._pop_ref: BasePopulation[Any] | None = None

        # Discrete-specific scalar overrides (stored here so build() can
        # (discrete scalars now normalize into the unified draft vectors
        # at write time — no end-of-build extraction exists anymore).

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

        # Stored .hooks() calls (build path); replayed onto the population
        # after construction and before backend enable.
        self._hook_calls: list[HookCall] = []

    @property
    def config(self) -> ModelDraft:
        """The wrapped ModelDraft (read-only accessor)."""
        return self._config

    def _rust_dirty_sink(self) -> set[str] | None:
        """Return the bound population's live rust-dirty set, if any.

        Write paths add contract field names to this set; the population
        drains it into the Rust session before the next run.  ``None``
        means there is no live population (build path) and nothing needs
        marking.

        Returns:
            The dirty set of ``_pop_ref`` when wired, else ``None``.
        """
        pop = self._pop_ref
        if pop is None:
            return None
        # getattr guard: spatial _clone builds instances via __new__.
        return getattr(pop, "_rust_dirty", None)

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
        # rebuild_config_maps rewrote the inheritance maps; mark the whole
        # genetics section so the Rust session refreshes every tensor.
        sink = self._rust_dirty_sink()
        if sink is not None:
            sink.update(_RUST_GENETICS_TENSORS)

    # -- factory ---------------------------------------------------------------

    @classmethod
    def from_species(
        cls,
        species: Species,
        *,
        discrete: bool = False,
    ) -> Configurator:
        """Create a Configurator from a Species with a minimal config.

        This is the primary factory.  Pass ``discrete=True`` for
        non-overlapping generations; otherwise an age-structured config
        with overlapping generations is returned.  Since slice 3 both
        granularities share this single Configurator class — the choice
        only selects the normalized draft shape.

        Args:
            species: The genetic architecture for the population.
            discrete: If ``True``, build the discrete-generation
                (Wright-Fisher) normalized draft.  Default ``False`` →
                age-structured draft.

        Returns:
            A ``Configurator`` ready for further chaining.
        """
        bp = species.get_config_blueprint()
        n_g = bp["n_genotypes"]
        n_hg = bp["n_gtypes"]
        n_gl = bp["n_glabs"]
        n_sl = bp.get("n_slabs", 1)
        z2g = bp["zygotes_to_gametes_map"]
        g2z = bp["gametes_to_zygotes_map"]
        has_sc = getattr(species, "has_sex_chromosomes", False)

        if discrete:
            from natal.frontend.data import build_discrete_engine_config

            config = build_discrete_engine_config(
                n_genotypes=n_g, n_gtypes=n_hg, n_glabs=n_gl,
                n_slabs=n_sl,
                gamete_labels=species.gamete_labels or ["default"],
                somatic_labels=species.somatic_labels or ["default"],
                zygotes_to_gametes_map=z2g, gametes_to_zygotes_map=g2z,
                has_sex_chromosomes=has_sc,
            )
            result = Configurator(config, species=species)
            object.__setattr__(result, "_name", "DiscreteGenerationPop")
        else:
            from natal.frontend.data import build_population_config

            config = build_population_config(
                n_genotypes=n_g, n_gtypes=n_hg, n_glabs=n_gl,
                n_slabs=n_sl,
                gamete_labels=species.gamete_labels or ["default"],
                somatic_labels=species.somatic_labels or ["default"],
                zygotes_to_gametes_map=z2g, gametes_to_zygotes_map=g2z,
                n_ages=2, new_adult_age=1, carrying_capacity=1000.0,
                has_sex_chromosomes=has_sc,
            )
            result = Configurator(config, species=species)
            object.__setattr__(result, "_name", "AgeStructuredPop")
        return result

    @classmethod
    def for_discrete(cls, species: Species) -> Configurator:
        """Shorthand for ``from_species(species, discrete=True)``.

        Args:
            species: The genetic architecture for the population.

        Returns:
            A ``Configurator`` wrapping a discrete-normalized draft.
        """
        return cls.from_species(species, discrete=True)

    @classmethod
    def for_age_structured(cls, species: Species) -> Configurator:
        """Shorthand for ``from_species(species)``.

        Args:
            species: The genetic architecture for the population.

        Returns:
            A ``Configurator`` wrapping an age-structured draft.
        """
        return cls.from_species(species)

    @staticmethod
    def for_config(
        config: ModelDraft,
    ) -> Configurator:
        """Wrap *config* with the right granularity of the unified class.

        The returned instance is identical either way — the draft's
        ``discrete_generation`` flag carries the granularity as data —
        but the constructor runs through the canonical entry point.

        Args:
            config: The draft to wrap.

        Returns:
            A ``Configurator`` around *config*.
        """
        return Configurator(config)

    @staticmethod
    def for_population(pop: BasePopulation[Any]) -> Configurator:
        """Create a Configurator wired to *pop* for runtime updates.

        Binds ``_pop_ref``, ``_species``, and ``_registry`` from the
        Population so that all chain methods work without further
        setup. This is the single entry point for ``pop.update()`` paths.

        Args:
            pop: The population to wire to.

        Returns:
            A ``Configurator`` ready for further chaining.
        """
        cfg = Configurator.for_config(pop.config)

        # Record the Population reference for write-back
        cfg._pop_ref = pop

        # Bind species and registry from the Population so
        # _make_ctx() and fitness() work correctly.
        cfg._species = pop.species
        cfg._registry = pop.index_registry

        return cfg

    # -- batch writer ----------------------------------------------------------

    def _make_writer(self) -> ConfigWriter:
        """Create a batch writer bound to the current draft state.

        Build path (no live population): a :class:`DraftWriter` writing
        the draft through the route table.  Runtime path: a
        :class:`CoreConfigWriter` which also pushes the committed
        values straight into the live Rust session when one exists,
        while still marking the dirty bridge exactly as before.

        Returns:
            A fresh :class:`ConfigWriter` bound to this instance.
        """

        def _publish(draft: ModelDraft) -> None:
            # ``_replace`` writes swap the draft identity: keep the
            # Configurator's view and the live Population in sync.
            self._config = draft
            if self._pop_ref is not None:
                self._pop_ref.set_config(draft)

        sink = self._rust_dirty_sink()
        if self._pop_ref is not None:
            # getattr guard: the backend only exists on Rust-enabled
            # populations; reference-path populations pass None.  During an
            # active Rust run the session cannot be written (PyO3 borrow);
            # writes defer to the next run through the dirty bridge.
            backend: object = None
            if not getattr(self._pop_ref, "_rust_run_active", False):
                backend = getattr(self._pop_ref, "_rust_lifecycle_backend", None)
            return CoreConfigWriter(
                self._config, sink, backend,
                on_replace=_publish,
                species=self._species, registry=self._registry,
                param_log=self._pop_ref.log_param_change,
            )
        return DraftWriter(
            self._config, sink,
            on_replace=_publish,
            species=self._species, registry=self._registry,
        )

    # -- setup flags -----------------------------------------------------------

    def setup(
        self,
        *,
        name: str | None = None,
        stochastic: bool | None = None,
        continuous_sampling: bool | None = None,
        fixed_egg_count: bool | None = None,
        compress: bool = False,
        backend: Literal["auto", "rust", "python"] | None = None,
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
                ``None`` preserves any earlier setting; ``"python"`` forces
                the pure-Python reference; ``"auto"`` (the default) selects
                Rust when the extension is available and falls back to the
                reference otherwise; ``"rust"`` forces the Rust backend.
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
            self._backend: Literal["auto", "rust", "python"] = backend
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
                # Execution flags live on the frozen Blueprint contract, so
                # a live session must be rebuilt, not value-refreshed.
                sink = self._rust_dirty_sink()
                if sink is not None:
                    sink.add("__blueprint__")
        return self

    # -- domain methods --------------------------------------------------------
    #
    # Every domain method is a thin shell: parse kwargs -> build a writes
    # dict -> one ``writer.apply(writes)``.  All differences between
    # parameters live in the route table (parameters.jsonc), not in the
    # method bodies.


    def age_structure(
        self, n_ages: int, new_adult_age: int,
        generation_time: float | None = None,
    ) -> Self:
        """Lock population dimensions.

        Args:
            n_ages: Total number of age classes.
            new_adult_age: First adult age.
            generation_time: Optional marker for model interpretation.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: When called on a discrete-generation draft
                (fixed at 2 ages by normalization) or after domain
                methods have already been called.
            ValueError: When *n_ages*/*new_adult_age* are inconsistent.

        Note:
            Must be called before any domain method (competition,
            reproduction, survival, etc.).  Calling it after domain
            methods will raise ``RuntimeError``.
        """
        if self._config.discrete_generation:
            raise RuntimeError(
                "age_structure() is not applicable to discrete-generation "
                "populations: their draft is normalized to 2 age classes."
            )
        if getattr(self, "_has_domain_params", False):
            raise RuntimeError(
                "age_structure() must be called before any domain method "
                "(competition(), reproduction(), survival(), etc.). "
                "Domain methods have already been called on this configurator."
            )
        if n_ages <= 1:
            raise ValueError(f"n_ages must be at least 2, got {n_ages}")
        if new_adult_age < 0 or new_adult_age >= n_ages:
            raise ValueError(
                f"new_adult_age must be in [0, {n_ages}), got {new_adult_age}"
            )
        from natal.frontend.data import build_population_config

        old = self._config
        # Use species blueprint maps (unexpanded) so that
        # build_population_config applies slab expansion exactly once.
        if self._species is not None:
            bp = self._species.get_config_blueprint()
            n_g_orig = bp["n_genotypes"]
            n_hg_orig = bp["n_gtypes"]
            z2g_bp = bp["zygotes_to_gametes_map"]
            g2z_bp = bp["gametes_to_zygotes_map"]
        else:
            n_g_orig = old.n_ztypes
            n_hg_orig = old.n_gtypes
            z2g_bp = old.zygotes_to_gametes_map
            g2z_bp = old.gametes_to_zygotes_map

        self._config = build_population_config(
            n_genotypes=n_g_orig,
            n_gtypes=n_hg_orig,
            n_ages=n_ages,
            n_glabs=old.n_glabs,
            n_slabs=old.n_slabs,
            gamete_labels=self._species.gamete_labels if self._species else None,
            somatic_labels=self._species.somatic_labels if self._species else None,
            zygotes_to_gametes_map=z2g_bp,
            gametes_to_zygotes_map=g2z_bp,
            new_adult_age=new_adult_age,
            generation_time=generation_time,
            stochastic=bool(old.stochastic),
            continuous_sampling=bool(old.continuous_sampling),
            fixed_egg_count=bool(old.fixed_egg_count),
            has_sex_chromosomes=old.has_sex_chromosomes,
        )
        # Rebuild registry for the new n_ages (affects genotype lookup dims).
        if self._species is not None:
            from natal.frontend.configurator._base import build_registry
            self._registry = build_registry(self._species)
        return self

    def competition(
        self,
        *,
        carrying_capacity: float | None = None,
        low_density_growth_rate: float | None = None,
        juvenile_growth_mode: int | str | None = None,
        growth_mode: int | str | None = None,
        competition_strength: float | None = None,
        expected_num_new_adult_females: float | None = None,
        equilibrium_distribution: NDArray[np.float64] | None = None,
        age_1_carrying_capacity: float | None = None,
        old_juvenile_carrying_capacity: float | None = None,
    ) -> Self:
        """Configure density-dependent competition.

        Args:
            carrying_capacity: Equilibrium population at age 1 (K).
            low_density_growth_rate: Per-capita growth at low density (r).
            juvenile_growth_mode: Regulation function (string or int) —
                historical spelling of *growth_mode*.
            growth_mode: Regulation function (string or int):
                ``no_competition``/``fixed``/``linear`` (``logistic``
                alias)/``beverton_holt``/``ricker`` or the integer.
            competition_strength: Larval competition weight.
            expected_num_new_adult_females: Target adult females
                (Champer model); derived egg override is computed and
                declared on the draft.
            equilibrium_distribution: Custom (2, n_ages) array for the
                Champer equilibrium computation.
            age_1_carrying_capacity: Legacy alias for *carrying_capacity*.
            old_juvenile_carrying_capacity: Legacy alias.

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        mode_value = (
            juvenile_growth_mode if juvenile_growth_mode is not None else growth_mode
        )
        # ---- carrying capacity (K) fallback chain ----
        k_value = carrying_capacity
        if k_value is None and age_1_carrying_capacity is not None:
            k_value = age_1_carrying_capacity
        if k_value is None and old_juvenile_carrying_capacity is not None:
            k_value = old_juvenile_carrying_capacity
        # Only auto-detect K during initial build (no live Population).
        if k_value is None and self._pop_ref is None:
            init_ind = self._config.initial_individual_count
            if init_ind.size > 0 and init_ind.ndim >= 2 and init_ind.shape[1] >= 2:
                age_1_count = float(init_ind[:, 1, :].sum())
                if age_1_count >= 0.5:
                    k_value = age_1_count
                else:
                    total = float(init_ind.sum())
                    if total >= 0.5:
                        k_value = total
        writes: dict[str, object] = {}
        if k_value is not None:
            writes["carrying_capacity"] = k_value
        if low_density_growth_rate is not None:
            writes["low_density_growth_rate"] = low_density_growth_rate
        if mode_value is not None:
            writes["growth_mode"] = mode_value
        if competition_strength is not None:
            writes["competition_strength"] = competition_strength
        if equilibrium_distribution is not None:
            writes["equilibrium_distribution"] = equilibrium_distribution
        if writes:
            writer = self._make_writer()
            writer.apply(writes)
            self._config = writer.draft
        if expected_num_new_adult_females is not None:
            self._declare_expected_females(float(expected_num_new_adult_females))
        return self

    def _declare_expected_females(self, target_females: float) -> None:
        """Compute and declare the Champer egg override on the draft.

        Args:
            target_females: Target number of new adult females; the
                equivalent total egg production is derived from the
                current demographics and persisted via the
                ``external_expected_eggs`` route (a sensitive write, so
                the equilibrium caches refresh).
        """
        cfg = self._config
        eggs = compute_expected_eggs_from_females(
            expected_num_new_adult_females=target_females,
            eggs_per_female=float(cfg.eggs_per_female),
            age_based_survival_rates=cfg.age_based_survival_rates,
            age_based_reproduction_rates=cfg.age_based_reproduction_rates,
            female_age_based_fertility=cfg.female_age_based_fertility,
            sex_ratio=float(cfg.sex_ratio),
            new_adult_age=int(cfg.new_adult_age),
            n_ages=int(cfg.n_ages),
        )
        writer = self._make_writer()
        writer.apply({"external_expected_eggs": eggs})
        self._config = writer.draft

    def reproduction(
        self,
        *,
        eggs_per_female: float | None = None,
        sex_ratio: float | None = None,
        sperm_displacement_rate: float | None = None,
        female_age_based_mating_rate: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        male_age_based_mating_rate: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        age_based_reproduction_rate: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        female_age_based_fertility: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        female_adult_mating_rate: float | None = None,
        male_adult_mating_rate: float | None = None,
        fixed_egg_count: bool | None = None,
    ) -> Self:
        """Configure reproduction.

        Every keyword is a route-table name: scalars go through the
        ``scalar`` parser, per-age vectors through the flexible
        ``sex_row``/``age_vec`` resolution, discrete adult mating
        probabilities through ``slot`` cells, and the egg-count flag
        through the ``bool`` channel.

        Args:
            eggs_per_female: Base eggs per reproducing female.
            sex_ratio: Female fraction of offspring (0–1).
            sperm_displacement_rate: Fraction of stored sperm displaced.
            female_age_based_mating_rate: Per-age female mating probability.
            male_age_based_mating_rate: Per-age male mating probability.
            age_based_reproduction_rate: Per-age reproduction participation.
            female_age_based_fertility: Per-age fertility weight.
            female_adult_mating_rate: Adult female mating probability
                (discrete vocabulary; writes the adult cell).
            male_adult_mating_rate: Adult male mating probability
                (discrete vocabulary; writes the adult cell).
            fixed_egg_count: Disable Poisson noise.

        Returns:
            Self for chaining.

        Raises:
            TypeError: When a per-age parameter is passed on a
                discrete-generation draft (use the discrete vocabulary
                instead).
        """
        self._has_domain_params = True
        # Discrete drafts are normalized to 2 ages where age-0 does not
        # mate: per-age flexible specs are meaningless there and keep the
        # historical rejection of the former DiscreteConfigurator.
        if self._config.discrete_generation and (
            female_age_based_mating_rate is not None
            or male_age_based_mating_rate is not None
            or age_based_reproduction_rate is not None
            or female_age_based_fertility is not None
        ):
            raise TypeError(
                "reproduction() rejects per-age parameters on "
                "discrete-generation populations; use the discrete "
                "vocabulary (eggs_per_female, sex_ratio, "
                "female_adult_mating_rate, male_adult_mating_rate)"
            )
        writes: dict[str, object] = {}
        if eggs_per_female is not None:
            writes["eggs_per_female"] = eggs_per_female
        if sex_ratio is not None:
            writes["sex_ratio"] = sex_ratio
        if sperm_displacement_rate is not None:
            writes["sperm_displacement_rate"] = sperm_displacement_rate
        if female_age_based_mating_rate is not None:
            writes["female_age_based_mating_rate"] = female_age_based_mating_rate
        if male_age_based_mating_rate is not None:
            writes["male_age_based_mating_rate"] = male_age_based_mating_rate
        if age_based_reproduction_rate is not None:
            writes["age_based_reproduction_rate"] = age_based_reproduction_rate
        if female_age_based_fertility is not None:
            writes["female_age_based_fertility"] = female_age_based_fertility
        if female_adult_mating_rate is not None:
            writes["female_adult_mating_rate"] = female_adult_mating_rate
        if male_adult_mating_rate is not None:
            writes["male_adult_mating_rate"] = male_adult_mating_rate
        if fixed_egg_count is not None:
            writes["fixed_egg_count"] = fixed_egg_count
        if writes:
            writer = self._make_writer()
            writer.apply(writes)
            self._config = writer.draft
        return self

    def survival(
        self,
        *,
        female_age_based_survival: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        male_age_based_survival: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        female_age0_survival: float | None = None,
        male_age0_survival: float | None = None,
    ) -> Self:
        """Configure survival rates.

        Per-age params accept flexible forms (scalar, list, dict, or
        callable).  The discrete ``female_age0_survival`` /
        ``male_age0_survival`` names keep working on both
        granularities — they route to single cells of the unified
        ``(2, n_ages)`` survival vector.

        Args:
            female_age_based_survival: Female survival rates (flexible form).
            male_age_based_survival: Male survival rates (same forms).
            female_age0_survival: Female juvenile (age-0) survival.
            male_age0_survival: Male juvenile (age-0) survival.

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        writes: dict[str, object] = {}
        if female_age_based_survival is not None:
            writes["female_age_based_survival"] = female_age_based_survival
        if male_age_based_survival is not None:
            writes["male_age_based_survival"] = male_age_based_survival
        if female_age0_survival is not None:
            writes["female_age0_survival"] = female_age0_survival
        if male_age0_survival is not None:
            writes["male_age0_survival"] = male_age0_survival
        if writes:
            writer = self._make_writer()
            writer.apply(writes)
            self._config = writer.draft
        return self

    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[str, float | Sequence[int | float] | Mapping[int, int | float]]],
        sperm_storage: Mapping[str, Mapping[str, float | Sequence[int | float] | Mapping[int, int | float]]] | None = None,
    ) -> Self:
        """Set the initial population distribution (deferred — applied at build time).

        *individual_count* is a dict like
        ``{"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}}``.
        Genotype selectors accept both strings and ``Genotype`` objects.

        The resolution granularity follows the draft's
        ``discrete_generation`` flag: discrete drafts use the flat
        discrete resolution and ignore sperm storage; age-structured
        drafts resolve per-age distributions.

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
                same nesting structure as *individual_count*.  Ignored
                (with a warning) for discrete-generation models.

        Returns:
            Self for chaining.
        """
        if self._species is None:
            raise RuntimeError(
                "initial_state() requires a Species reference. "
                "Use Configurator.from_species() to create the instance."
            )
        from natal.frontend.configurator._factory import PopulationConfigBuilder

        if self._config.discrete_generation:
            array = PopulationConfigBuilder.resolve_discrete_initial_individual_count(
                species=self._species,
                distribution=individual_count,
            )
            overrides: dict[str, object] = {"initial_individual_count": array}
            if sperm_storage is not None:
                import warnings
                warnings.warn(
                    "sperm_storage is ignored for discrete-generation populations.",
                    UserWarning, stacklevel=2,
                )
            self._config = self._config._replace(**overrides)
            return self

        n_ages = self._config.n_ages
        new_adult_age = self._config.new_adult_age
        array = PopulationConfigBuilder.resolve_age_structured_initial_individual_count(
            species=self._species,
            distribution=individual_count,
            n_ages=n_ages,
            new_adult_age=new_adult_age,
        )
        overrides = {"initial_individual_count": array}
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
        """Register custom named slots on ``config.custom``.

        Multiple calls accumulate — ``.custom(a=1).custom(b=2)`` stores both.

        ``config.custom`` is a plain ``{name: value}`` dict shared between
        the Configurator and the Population's draft, so values are written
        in place and are immediately visible on both sides — no
        ``_replace`` or write-back round-trip is needed.  When called via
        ``pop.update().custom(...)``, the ``_rust_dirty`` sink is marked
        so the Rust session refreshes ``custom_slots`` before the next run.

        Args:
            **kwargs: Name-value pairs for custom slots.  Values must be
                ``bool``, ``int``, ``float``, or ``NDArray[np.float64]``.

        Returns:
            Self for chaining.
        """
        from natal.frontend.data import build_custom_slots

        self._custom_kwargs.update(kwargs)
        normalized = build_custom_slots(self._custom_kwargs)
        # Keep the same dict object (shared with the population's draft):
        # clear + update in place instead of rebinding the slot.
        self._config.custom.clear()
        self._config.custom.update(normalized)
        sink = self._rust_dirty_sink()
        if sink is not None:
            sink.add("custom_slots")
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

        from natal.frontend.presets import apply_preset_to_population

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
        isolated_config: ModelDraft = deepcopy(original_config)
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
            gamete_modifiers: List of :class:`~natal.frontend.modifiers.GameteModifier`
                instances affecting meiosis (genotype → gamete mapping).
            zygote_modifiers: List of :class:`~natal.frontend.modifiers.ZygoteModifier`
                instances affecting fertilization (gamete → zygote mapping).

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

        writes: dict[str, object] = {}
        for patch_name, patch_dict in [
            ("viability", viability),
            ("fecundity", fecundity),
            ("sexual_selection", sexual_selection),
            ("zygote_viability", zygote_viability),
        ]:
            if patch_dict is not None:
                writes[patch_name] = patch_dict
        if writes:
            # geno_tensor kind: pattern dicts delegate to
            # write_fitness_field inside the writer; the writer also
            # pushes the whole tensors to the live Rust session.
            writer = self._make_writer()
            writer.apply(writes, mode=mode)
            self._config = writer.draft
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

    def hooks(
        self,
        *hook_items: _HookItem,
        event: str | None = None,
        priority: int = 0,
        deme: DemeSelector = "*",
        name: str | None = None,
    ) -> Self:
        """Register event hooks — the single entry, build and runtime.

        Accepted items: ``Op.*`` objects (or lists of them), functions
        decorated with ``@hook(event='...')``, and plain single-parameter
        callables (``def hook(pop) -> int``).

        Args:
            *hook_items: Hook registrations.
            event: Default event for items that do not carry one
                (``"first"``, ``"early"``, ``"late"``, ``"finish"``).
            priority: Execution priority — lower values run first.
            deme: Deme selector for spatial populations.
            name: Optional name for grouped op registrations.

        Returns:
            Self for chaining.

        Raises:
            TypeError: If an item has an unsupported shape (including the
                removed ``(state, config, deme_id)`` signature).
            ValueError: If an event name is unknown or cannot be resolved.
        """
        if self._pop_ref is not None:
            # Runtime: register immediately on the live population.
            self._pop_ref.register_hooks(
                *hook_items, event=event, priority=priority, deme=deme, name=name
            )
            return self
        self._hook_calls.append((hook_items, {
            "event": event, "priority": priority, "deme": deme, "name": name,
        }))
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

        # Execute the recipe exactly once, on the live population, with the
        # candidate preset swapped into the registry (plan 5.1: no
        # clone-to-validate followed by a second execution on the real
        # object).  A failure rolls the population back from snapshots
        # before re-raising, so neither the session nor the committed
        # declaration is polluted.  Outside this guarantee: the user
        # recipe's own side effects on the world (files, globals) and any
        # NATAL state it mutates beyond the modifier/fitness surfaces
        # below (e.g. calling pop.add_gamete_modifier itself).
        candidate = copy(preset)
        for attr, value in changes.items():
            setattr(candidate, attr, value)

        saved_presets = list(pop._presets)  # pyright: ignore[reportPrivateUsage]
        saved_config = pop._config  # pyright: ignore[reportPrivateUsage]
        assert saved_config is not None  # a live population always has a config
        saved_gamete_modifiers = list(pop._gamete_modifiers)  # pyright: ignore[reportPrivateUsage]
        saved_zygote_modifiers = list(pop._zygote_modifiers)  # pyright: ignore[reportPrivateUsage]
        # refresh_modifier_maps marks the Rust bridge mid-rebuild; leaving
        # those markers behind would force a full session rebuild (and RNG
        # reset) on the next run.  Pending markers the user created before
        # the reconfigure must survive the restore, so snapshot-and-restore
        # rather than clear.
        saved_dirty = set(pop._rust_dirty)  # pyright: ignore[reportPrivateUsage]
        # reapply_preset_fitness clears the shared fitness arrays in place,
        # so a failed run must restore their contents, not just the config
        # reference.
        saved_fitness = tuple(
            tensor.copy()
            for tensor in (
                saved_config.viability_fitness,
                saved_config.fecundity_fitness,
                saved_config.sexual_selection_fitness,
                saved_config.zygote_viability_fitness,
            )
        )
        try:
            pop._presets = [  # pyright: ignore[reportPrivateUsage]
                candidate if registered is preset else registered
                for registered in saved_presets
            ]
            pop.refresh_modifiers()
            pop.reapply_preset_fitness()
        except BaseException:
            pop._presets = saved_presets  # pyright: ignore[reportPrivateUsage]
            pop._config = saved_config  # pyright: ignore[reportPrivateUsage]
            pop._gamete_modifiers[:] = saved_gamete_modifiers  # pyright: ignore[reportPrivateUsage]
            pop._zygote_modifiers[:] = saved_zygote_modifiers  # pyright: ignore[reportPrivateUsage]
            pop._rust_dirty.clear()  # pyright: ignore[reportPrivateUsage]
            pop._rust_dirty.update(saved_dirty)  # pyright: ignore[reportPrivateUsage]
            for tensor, saved in zip(
                (
                    saved_config.viability_fitness,
                    saved_config.fecundity_fitness,
                    saved_config.sexual_selection_fitness,
                    saved_config.zygote_viability_fitness,
                ),
                saved_fitness,
            ):
                tensor[...] = saved
            raise

        # ── Commit phase: the candidate products are already live on the
        # population; adopt the attribute changes onto the registered
        # object and restore its identity in the registry so add_preset's
        # idempotency-by-identity keeps working.
        for attr, value in changes.items():
            setattr(preset, attr, value)
        pop._presets = [  # pyright: ignore[reportPrivateUsage]
            preset if registered is candidate else registered
            for registered in pop._presets  # pyright: ignore[reportPrivateUsage]
        ]
        self._config = pop.config

        self._config = sync_equilibrium_for_draft(self._config)
        return self

    # -- apply / build ---------------------------------------------------------

    def apply(self) -> Self:
        """Sync derived values (equilibrium metrics).

        All routed writes already refresh the equilibrium caches on
        their own (driven by the jsonc ``sensitive`` column), so this is
        only needed when you modify config arrays directly (outside
        Configurator) or want to force a re-derivation before build.

        Returns:
            Self for chaining.
        """
        self._config = sync_equilibrium_for_draft(self._config)
        return self

    def build(
        self,
        name: str | None = None,
        hook_items: Sequence[object] | None = None,
    ) -> DiscreteGenerationPopulation | AgeStructuredPopulation:
        """Finalize the config and create a Population.

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
            hook_items: Additional hook registrations (same item shapes
                as :meth:`hooks`), registered together with any stored
                via :meth:`hooks`.

        Returns:
            ``AgeStructuredPopulation`` or ``DiscreteGenerationPopulation``,
            depending on whether *self._config* carries the
            discrete-generation flag.
        """
        # Sync equilibrium metrics and apply index compression (if enabled).
        self.apply()

        # Inject the symbolic name directory from the registry (building the
        # registry lazily when presets/fitness never forced it).  Happens
        # before compression so compress_config subslices real names.
        if self._species is None:
            raise RuntimeError(
                "Cannot build Population: no Species set. "
                "Use Configurator.from_species() to create this instance."
            )
        if self._registry is None:
            self._registry = build_registry(self._species)
        self._config = self._config._replace(
            ztype_names=ztype_names_from_registry(self._registry.index_to_ztype),
            gtype_names=gtype_names_from_registry(self._registry.index_to_gtype),
        )

        # Compression runs on a COPY — self._config stays in G_orig space.
        # Population receives the compressed config.  All user writes
        # (fitness, initial_state) already happened on the full-size
        # config; compression subslices the arrays naturally.
        final_config = self._config
        if self._compress and not self._compression_applied:
            # Auto-collect genotype refs from hooks so genotypes introduced
            # only via hooks survive BFS pruning.
            if self._hook_calls:
                hook_refs = collect_hook_genotype_refs(self._hook_calls)
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
            from natal.frontend.data import build_custom_slots
            final_config = final_config._replace(
                custom=build_custom_slots(self._custom_kwargs)
            )

        # Resolve name: explicit argument > setup(name=...) > default
        if name is None:
            name = getattr(self, "_name", "Population")

        if final_config.discrete_generation:
            from natal.frontend.population.discrete_generation import (
                DiscreteGenerationPopulation,
            )

            pop: DiscreteGenerationPopulation | AgeStructuredPopulation = \
                DiscreteGenerationPopulation(
                    species=self._species,
                    population_config=final_config,
                    index_registry=self._registry,
                    name=name,
                )
        else:
            from natal.frontend.population.age_structured import (
                AgeStructuredPopulation,
            )

            pop = AgeStructuredPopulation(
                species=self._species,
                population_config=final_config,
                index_registry=self._registry,
                name=name,
            )

        # Configurator applies modifiers before Population construction.  Carry
        # both the recipe objects and their current derived callables across the
        # boundary so refresh_modifiers() and reconfigure_preset() behave the
        # same for build-time and runtime preset registration.
        pop._presets = list(self._presets)  # pyright: ignore[reportPrivateUsage]
        pop._gamete_modifiers = list(self.gamete_modifiers)  # pyright: ignore[reportPrivateUsage]
        pop._zygote_modifiers = list(self.zygote_modifiers)  # pyright: ignore[reportPrivateUsage]

        # Replay stored .hooks() calls (plus any passed inline) BEFORE the
        # backend enable below: enable_rust_backend snapshots both the CSR
        # program and the Python-callback bridges.
        hook_calls: list[HookCall] = list(self._hook_calls)
        if hook_items:
            hook_calls.append((tuple(hook_items), {}))
        for items, kwargs in hook_calls:
            pop.register_hooks(  # pyright: ignore[reportPrivateUsage]
                *items,
                event=cast("str | None", kwargs["event"]),
                priority=cast("int", kwargs["priority"]),
                deme=cast("DemeSelector", kwargs["deme"]),
                name=cast("str | None", kwargs["name"]),
            )

        # Apply the requested lifecycle backend.  ``auto`` (the default)
        # selects Rust when the extension is importable and silently falls
        # back to the pure-Python reference otherwise; ``rust`` propagates
        # the failure.  The legacy compiled-backend selector was retired:
        # request ``rust`` or ``python`` explicitly.
        backend = getattr(self, "_backend", "auto")
        if backend == "python":
            pop._python_backend = True  # pyright: ignore[reportPrivateUsage]
        elif backend in ("auto", "rust"):
            from natal.backends.rust.rust_backend import rust_backend_available

            if backend == "rust" or rust_backend_available():
                try:
                    pop.enable_rust_backend()  # type: ignore[reportAttributeAccessIssue]  # both concrete Population classes expose this method
                except RuntimeError:
                    if backend == "rust":
                        raise
        elif backend == "numba":
            raise ValueError(
                "backend='numba' was removed. Use backend='rust' (native "
                "extension) or backend='python' (pure-Python reference)."
            )

        # Compile and freeze the recording plan.
        self._compile_recording_plan(pop)
        return pop

    def _compile_recording_plan(self, pop: AgeStructuredPopulation | DiscreteGenerationPopulation) -> None:
        """Compile and freeze the :class:`RecordingPlan` on the population.

        Called at the end of :meth:`build` after all observations and
        initial state have been applied.  The plan is immutable for the
        remainder of the population's lifetime.
        """
        from natal.frontend.output._recording import compile_recording_plan
        from natal.frontend.output.history import History
        from natal.frontend.output.observation import build_identity_observation

        config = pop.config
        if config.discrete_generation:
            kind = "discrete_generation"
            has_sperm = False
        else:
            kind = "age_structured"
            has_sperm = True

        from natal.frontend.output.observation import ObservationFilter

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

