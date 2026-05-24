"""Mutable wrapper for PopulationConfig and DiscretePopulationConfig.

Provides a chainable API for reading/writing config fields, registering
custom named parameters (stored as a structured numpy array), and
freezing changes back to an immutable NamedTuple via ``_replace``
(cheap — all ndarray fields are shared by reference).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Self, cast

import numpy as np
from numba import objmode  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from natal.discrete_population_config import DiscretePopulationConfig
from natal.genetic_structures import Species
from natal.index_registry import IndexRegistry
from natal.numba_utils import njit_switch
from natal.parameters import ALL_PARAMETERS, ParamDescriptor
from natal.population_config import PopulationConfig

if TYPE_CHECKING:
    from natal.age_structured_population import AgeStructuredPopulation
    from natal.discrete_generation_population import DiscreteGenerationPopulation
    from natal.genetic_presets import GeneticPreset
    from natal.modifiers import GameteModifier, ZygoteModifier

__all__ = ["Configurator", "DiscreteConfigurator", "AgeStructuredConfigurator"]

# ── Type aliases for hook registrations ──────────────────────────────────────

# A single hook registration: (func, name?, priority?)
_HookReg = tuple[Callable[..., Any], str | None, int | None]
# Hook registration map keyed by event name.
_HookMap = dict[str, list[_HookReg]]
# A hook item can be a raw dict or a callable with @hook metadata.
_HookItem = Callable[..., Any] | _HookMap


# ── Registry builder (shared by Configurator and adapter) ──────────────────────


def _build_registry(species: Species) -> IndexRegistry:
    """Build an IndexRegistry pre-populated with all genotypes/haplotypes from a Species."""
    registry = IndexRegistry()
    for genotype in species.get_all_genotypes():
        registry.register_genotype(genotype)
    haploid_genotypes = species.get_all_haploid_genotypes()
    if haploid_genotypes:
        for hg in haploid_genotypes:
            registry.register_haplogenotype(hg)
    raw_glabs = getattr(species, "gamete_labels", None)
    glabs = raw_glabs or ["default"]
    for glab in glabs:
        registry.register_gamete_label(glab)
    return registry


# ── Adapter: lets presets/modifiers/fitness operate without a Population ──


class _ConfigContext:
    """Lightweight adapter exposing Species + IndexRegistry + PopConfig.

    Provides the minimal interface that :func:`apply_preset_to_population`,
    ``build_modifier_wrappers``, and ``_apply_preset_fitness_patch`` expect.
    This decouples preset/modifier/fitness application from the Population
    object lifecycle — they write directly into config arrays.
    """

    def __init__(
        self,
        species: Species,
        config: PopulationConfig | DiscretePopulationConfig,
        registry: IndexRegistry,
    ) -> None:
        self.species = species
        self.config = config
        self.registry = registry
        self.index_registry = registry
        self.gamete_modifiers: list[tuple[int, str | None, Any]] = []
        self.zygote_modifiers: list[tuple[int, str | None, Any]] = []

    # -- modifier registration (mimics BasePopulation) ----------------------

    def add_gamete_modifier(
        self,
        modifier: GameteModifier,
        *,
        name: str | None = None,
        modifier_id: int | None = None,
        refresh: bool = True,
    ) -> None:
        resolved_id = _ConfigContext.next_modifier_id(self.gamete_modifiers) if modifier_id is None else modifier_id
        self.gamete_modifiers.append((resolved_id, name, modifier))
        self.gamete_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self.refresh_modifier_maps()

    def add_zygote_modifier(
        self,
        modifier: ZygoteModifier,
        *,
        name: str | None = None,
        modifier_id: int | None = None,
        refresh: bool = True,
    ) -> None:
        resolved_id = _ConfigContext.next_modifier_id(self.zygote_modifiers) if modifier_id is None else modifier_id
        self.zygote_modifiers.append((resolved_id, name, modifier))
        self.zygote_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self.refresh_modifier_maps()

    def refresh_modifier_maps(self) -> None:
        _rebuild_config_maps(self)

    @staticmethod
    def next_modifier_id(modifiers: list[tuple[int, str | None, Any]]) -> int:
        ids = [mid for mid, _, _ in modifiers]
        return (max(ids) + 1) if ids else 0


# ── Core: rebuild genotype/gamete/zygote maps from modifier lists ─────────────


def _rebuild_config_maps(ctx: _ConfigContext) -> None:
    """Apply gamete/zygote modifiers and rebuild ``offspring_tensor``.

    Starts from the species-level Mendelian baseline (cached via
    :meth:`Species.get_config_blueprint`) and applies modifier callables
    in-place, avoiding redundant O(n²) baseline recomputation.
    """
    from natal.engine.simulation.age_structured import (
        compute_offspring_probability_tensor,
    )
    from natal.modifiers import build_modifier_wrappers

    haploid_genotypes = ctx.registry.index_to_haplo
    diploid_genotypes = ctx.registry.index_to_genotype
    if not haploid_genotypes or not diploid_genotypes:
        return

    n_glabs = int(ctx.config.n_glabs)
    gamete_funcs, zygote_funcs = build_modifier_wrappers(
        gamete_modifiers=ctx.gamete_modifiers,
        zygote_modifiers=ctx.zygote_modifiers,
        population=None,
        index_registry=ctx.registry,
        haploid_genotypes=haploid_genotypes,
        diploid_genotypes=diploid_genotypes,
        n_glabs=n_glabs,
    )

    # Copy the Mendelian baseline from the species cache — no need to
    # recompute it from scratch.
    bp = ctx.species.get_config_blueprint()
    genotype_to_gametes_map = bp["genotype_to_gametes_map"].copy()
    gametes_to_zygote_map = bp["gametes_to_zygote_map"].copy()

    # Apply modifier callables on top of the baseline.
    for fn in gamete_funcs:
        genotype_to_gametes_map = fn(genotype_to_gametes_map)
    for fn in zygote_funcs:
        gametes_to_zygote_map = fn(gametes_to_zygote_map)

    offspring_tensor = compute_offspring_probability_tensor(
        meiosis_f=genotype_to_gametes_map[0],
        meiosis_m=genotype_to_gametes_map[1],
        haplo_to_genotype_map=gametes_to_zygote_map,
        n_genotypes=int(ctx.config.n_genotypes),
        n_haplogenotypes=int(ctx.config.n_haploid_genotypes),
        n_glabs=n_glabs,
    )

    # DiscretePopulationConfig has pre-extracted slices (meiosis_f, meiosis_m,
    # viability_f, etc.) that must stay in sync with the source maps.
    from natal.discrete_population_config import DiscretePopulationConfig

    overrides: dict[str, Any] = {
        "genotype_to_gametes_map": genotype_to_gametes_map,
        "gametes_to_zygote_map": gametes_to_zygote_map,
        "offspring_tensor": offspring_tensor,
    }
    if isinstance(ctx.config, DiscretePopulationConfig):
        overrides["meiosis_f"] = genotype_to_gametes_map[0]
        overrides["meiosis_m"] = genotype_to_gametes_map[1]
        overrides["fecundity_f"] = ctx.config.fecundity_fitness[0]
        overrides["fecundity_m"] = ctx.config.fecundity_fitness[1]
        overrides["viability_f"] = ctx.config.viability_fitness[0, 0, :]
        overrides["viability_m"] = ctx.config.viability_fitness[1, 0, :]
    ctx.config = ctx.config._replace(**overrides)


# Full parameter-name keys that need equilibrium sync.
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
) -> None:
    """Set a simulation parameter by its user-facing name.

    Resolves *name* through ``ALL_PARAMETERS`` (the parameters.py registry)
    to the correct config field and index path, writes the value in-place,
    and automatically syncs equilibrium metrics when needed.

    Usable from pure Python, ``with objmode():`` inside njit hooks, and
    Configurator chain methods.

    Args:
        config: The PopulationConfig or DiscretePopulationConfig to modify.
        name: Parameter name — full key ``"competition.carrying_capacity"``,
              short name ``"carrying_capacity"``, or alias.
        value: New value (scalar). For tensor parameters, use
               direct array access instead.

    Raises:
        KeyError: If *name* is not a registered parameter.
        ValueError: If *name* refers to a tensor (non-scalar) parameter.

    Examples::

        set_param(config, "competition.carrying_capacity", 5000.0)
        set_param(config, "carrying_capacity", 5000.0)        # short name
        set_param(config, "reproduction.eggs_per_female", 100.0)
    """
    desc = _resolve_param(name)
    if desc is None:
        raise KeyError(f"Unknown parameter: {name!r}")
    if desc.config_field is None:
        return  # spatial-only parameter, nothing to write
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
    else:
        field[()] = value  # fallback for 0-d fields

    # Auto-sync equilibrium when sensitive params change.
    key = f"{desc.domain.value}.{desc.name}"
    if key in _EQUILIBRIUM_SENSITIVE_KEYS:
        from natal.engine.simulation.age_structured import sync_equilibrium_metrics

        sync_equilibrium_metrics(config)


# ── Helpers: hook merging and fitness field writing ────────────────────────────


def _merge_hooks(hook_items: list[_HookItem]) -> _HookMap:
    """Merge @hook-decorated items into a hook registration map.

    Each item can be a raw dict or a function with @hook metadata.
    """
    result: _HookMap = {}
    for item in hook_items:
        if isinstance(item, dict):
            hook_dict = cast(_HookMap, item)
            for event, registrations in hook_dict.items():
                result.setdefault(event, []).extend(registrations)
        elif callable(item):
            meta = getattr(item, "meta", {})
            event = meta.get("event") or getattr(item, "event", None)
            priority = meta.get("priority", getattr(item, "priority", 0))
            name = getattr(item, "__name__", None)
            if event:
                result.setdefault(event, []).append((item, name, priority))
    return result


def _write_fitness_field(
    config: PopulationConfig | DiscretePopulationConfig,
    field_name: str,
    patch: Mapping[str, float | Mapping[str, float]],
    mode: str,
    *,
    species: Species,
    registry: IndexRegistry,
    all_genotypes: list[Any],
) -> None:
    """Resolve genotype-pattern strings and write into a fitness tensor.

    *field_name* is one of ``"viability"``, ``"fecundity"``,
    ``"sexual_selection"``, or ``"zygote_viability"``.  *patch* is a
    dict mapping genotype selectors to fitness values, with optional
    sex-keyed nesting.
    """
    # Resolve sex-keyed dict: {"female": {genotype: val}, "male": {...}}
    if patch and all(k in ("female", "male") for k in patch):
        sex_patch = cast(Mapping[str, Mapping[str, float]], patch)
        for sex_key, geno_dict in sex_patch.items():
            sex_idx = 0 if sex_key == "female" else 1
            _write_fitness_field_flat(
                config, field_name, geno_dict, mode,
                sex_idx=sex_idx,
                species=species, registry=registry,
                all_genotypes=all_genotypes,
            )
        return

    # Flat dict: apply to both sexes
    flat_patch = cast(Mapping[str, float], patch)
    for sex_idx in (0, 1):
        _write_fitness_field_flat(
            config, field_name, flat_patch, mode,
            sex_idx=sex_idx,
            species=species, registry=registry,
            all_genotypes=all_genotypes,
        )


def _write_fitness_field_flat(
    config: PopulationConfig | DiscretePopulationConfig,
    field_name: str,
    patch: Mapping[str, float],
    mode: str,
    *,
    sex_idx: int,
    species: Species,
    registry: IndexRegistry,
    all_genotypes: list[Any],
) -> None:
    """Write a flat genotype→value dict into one sex's fitness slice."""
    for selector, value in patch.items():
        matched = species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context=f"fitness.{field_name}",
        )
        for genotype in matched:
            gidx = registry.genotype_to_index[genotype]

            if field_name == "viability":
                arr = config.viability_fitness
                if mode == "replace":
                    arr[sex_idx, :, gidx] = float(value)
                else:
                    arr[sex_idx, :, gidx] *= float(value)
            elif field_name == "fecundity":
                arr = config.fecundity_fitness
                if mode == "replace":
                    arr[sex_idx, gidx] = float(value)
                else:
                    arr[sex_idx, gidx] *= float(value)
            elif field_name == "sexual_selection":
                arr = config.sexual_selection_fitness
                # sexual_selection is indexed differently: [female_idx, male_idx]
                if mode == "replace":
                    arr[sex_idx, gidx] = float(value)
                else:
                    arr[sex_idx, gidx] *= float(value)
            elif field_name == "zygote_viability":
                arr = config.zygote_viability_fitness
                if mode == "replace":
                    arr[sex_idx, gidx] = float(value)
                else:
                    arr[sex_idx, gidx] *= float(value)


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
        self._config = config
        self._species = species  # needed for initial_state / preset resolution
        self._registry: IndexRegistry | None = None
        # Modifier lists — accumulated across presets() / modifiers() calls,
        # then applied when maps are rebuilt.
        self.gamete_modifiers: list[tuple[int, str | None, Any]] = []
        self.zygote_modifiers: list[tuple[int, str | None, Any]] = []

    @property
    def config(self) -> PopulationConfig | DiscretePopulationConfig:
        """The wrapped PopulationConfig (read-only accessor)."""
        return self._config

    # -- adapter factory ------------------------------------------------------

    def _make_ctx(self) -> _ConfigContext:
        """Build a :class:`_ConfigContext` for the current config + species + registry."""
        if self._species is None:
            raise RuntimeError(
                "presets() / modifiers() / fitness() require a Species. "
                "Use Configurator.from_species() to create this instance."
            )
        if self._registry is None:
            self._registry = _build_registry(self._species)
        ctx = _ConfigContext(self._species, self._config, self._registry)
        ctx.gamete_modifiers = list(self.gamete_modifiers)
        ctx.zygote_modifiers = list(self.zygote_modifiers)
        return ctx

    def _sync_from_ctx(self, ctx: _ConfigContext) -> None:
        """Sync config and modifier lists back from an adapter after mutations."""
        self._config = ctx.config
        self.gamete_modifiers = ctx.gamete_modifiers
        self.zygote_modifiers = ctx.zygote_modifiers

    # -- factory ---------------------------------------------------------------

    @classmethod
    def from_species(cls, species: Species) -> AgeStructuredConfigurator:
        """Create an ``AgeStructuredConfigurator`` with a minimal config.

        The returned config contains species-derived arrays (genotype maps,
        offspring tensor, compatibility) and a default ``n_ages=2``,
        ``new_adult_age=1``.  All other fields are unset placeholders.

        Call ``.age_structure()`` before methods that depend on per-age arrays.
        """
        from natal.population_config import build_population_config

        bp = species.get_config_blueprint()
        config = build_population_config(
            n_genotypes=bp["n_genotypes"],
            n_haploid_genotypes=bp["n_haploid_genotypes"],
            n_ages=2,
            n_glabs=bp["n_glabs"],
            genotype_to_gametes_map=bp["genotype_to_gametes_map"],
            gametes_to_zygote_map=bp["gametes_to_zygote_map"],
            new_adult_age=1,
            carrying_capacity=1000.0,
            has_sex_chromosomes=getattr(species, "has_sex_chromosomes", False),
        )
        return AgeStructuredConfigurator(config, species=species)

    @classmethod
    def for_discrete(cls, species: Species) -> DiscreteConfigurator:
        """Create a ``DiscreteConfigurator`` for building a discrete-generation population.

        The underlying config is a ``DiscretePopulationConfig``, so ``.build()``
        returns ``DiscreteGenerationPopulation``.
        """
        from natal.discrete_population_config import from_population_config
        from natal.population_config import build_population_config

        bp = species.get_config_blueprint()
        config = build_population_config(
            n_genotypes=bp["n_genotypes"],
            n_haploid_genotypes=bp["n_haploid_genotypes"],
            n_ages=2,
            n_glabs=bp["n_glabs"],
            genotype_to_gametes_map=bp["genotype_to_gametes_map"],
            gametes_to_zygote_map=bp["gametes_to_zygote_map"],
            new_adult_age=1,
            carrying_capacity=1000.0,
            has_sex_chromosomes=getattr(species, "has_sex_chromosomes", False),
        )
        # Age-structured default sets juvenile (age0) survival to 0.0, but the
        # discrete model needs age0 = 1.0 (juveniles become adults next tick).
        config.age_based_survival_rates[:, 0] = 1.0
        return DiscreteConfigurator(from_population_config(config), species=species)

    @classmethod
    def for_age_structured(cls, species: Species) -> AgeStructuredConfigurator:
        """Create an ``AgeStructuredConfigurator`` for building an age-structured population.

        The underlying config is a ``PopulationConfig``, so ``.build()``
        returns ``AgeStructuredPopulation``.
        """
        cfg = cls.from_species(species)
        return AgeStructuredConfigurator(cfg._config, species=cfg._species)

    @staticmethod
    def for_config(
        config: PopulationConfig | DiscretePopulationConfig,
    ) -> DiscreteConfigurator | AgeStructuredConfigurator:
        """Return the right Configurator subclass for the given config type."""
        if isinstance(config, DiscretePopulationConfig):
            return DiscreteConfigurator(config)
        return AgeStructuredConfigurator(config)

    # -- dimension lock --------------------------------------------------------

    def age_structure(
        self,
        n_ages: int,
        new_adult_age: int,
        generation_time: float | None = None,
    ) -> Self:
        """Lock the population dimensions.  Must be called first.

        Rebuilds the config with the given *n_ages* and *new_adult_age*,
        preserving species-derived arrays (genotype maps, offspring tensor)
        from the old config so they are not recomputed.

        Apply dimension-dependent parameters (survival, reproduction,
        initial_state) AFTER this call.
        """
        from natal.population_config import build_population_config

        old = self._config
        self._config = build_population_config(
            n_genotypes=old.n_genotypes,
            n_haploid_genotypes=old.n_haploid_genotypes,
            n_ages=n_ages,
            n_glabs=old.n_glabs,
            genotype_to_gametes_map=old.genotype_to_gametes_map,
            gametes_to_zygote_map=old.gametes_to_zygote_map,
            new_adult_age=new_adult_age,
            generation_time=generation_time,
            is_stochastic=bool(old.is_stochastic),
            use_continuous_sampling=bool(old.use_continuous_sampling),
            use_fixed_egg_count=bool(old.use_fixed_egg_count),
            has_sex_chromosomes=old.has_sex_chromosomes,
        )
        # Rebuild registry for the new n_ages (affects genotype lookup dims).
        if self._species is not None:
            self._registry = _build_registry(self._species)
        return self

    # -- setup flags -----------------------------------------------------------

    def setup(
        self,
        *,
        name: str | None = None,
        stochastic: bool | None = None,
        use_continuous_sampling: bool | None = None,
        use_fixed_egg_count: bool | None = None,
    ) -> Self:
        """Configure simulation flags and optional population name.

        *name* is stored and used by ``build()`` when no explicit name is given.
        """
        if name is not None:
            self._name = name
        overrides: dict[str, bool] = {}
        if stochastic is not None:
            overrides["is_stochastic"] = stochastic
        if use_continuous_sampling is not None:
            overrides["use_continuous_sampling"] = use_continuous_sampling
        if use_fixed_egg_count is not None:
            overrides["use_fixed_egg_count"] = use_fixed_egg_count
        if overrides:
            self._config = self._config._replace(**overrides)
        return self

    # -- domain methods --------------------------------------------------------


    def initial_state(
        self,
        individual_count: dict[str, dict[str, float | list[int] | dict[int, int]]],
        sperm_storage: dict[str, dict[str, float | list[int] | dict[int, int]]] | None = None,
    ) -> Self:
        """Set the initial population distribution.

        *individual_count* is a dict like
        ``{"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}}``.
        Resolved to a 3-D array using the same logic as the Builder.
        """
        from natal.population_builder import PopulationConfigBuilder

        if self._species is None:
            raise RuntimeError(
                "initial_state() requires a Species reference. "
                "Use Configurator.from_species() to create the instance."
            )
        n_ages = self._config.n_ages
        new_adult_age = self._config.new_adult_age
        array = PopulationConfigBuilder.resolve_age_structured_initial_individual_count(
            species=self._species,
            distribution=individual_count,
            n_ages=n_ages,
            new_adult_age=new_adult_age,
        )
        self._config = self._config._replace(
            initial_individual_count=array * float(self._config.population_scale)
        )
        if sperm_storage is not None:
            ss_arr = PopulationConfigBuilder.resolve_age_structured_initial_sperm_storage(
                species=self._species,
                sperm_storage=sperm_storage,
                n_ages=n_ages,
                new_adult_age=new_adult_age,
            )
            self._config = self._config._replace(
                initial_sperm_storage=ss_arr * float(self._config.population_scale)
            )
        return self

    # -- custom fields ---------------------------------------------------------

    def custom(self, **kwargs: bool | int | float | NDArray[np.float64]) -> Self:
        """Register custom named fields on ``config.custom``."""
        from natal.population_builder import build_custom_array

        self._config = self._config._replace(
            custom=build_custom_array(kwargs)
        )
        # If wrapping a Population's config, sync the new config reference
        # back so the Population sees the updated custom fields.
        source = getattr(self, "_pop_ref", None)
        if source is not None:
            object.__setattr__(source, '_config', self._config)
        return self

    # -- presets / modifiers / fitness (immediate — applied directly to config) --

    def presets(self, *presets: GeneticPreset) -> Self:
        """Apply genetic presets directly to config arrays.

        Each preset's gamete/zygote modifiers and fitness patch are
        resolved against the current Species + IndexRegistry and
        written into the config immediately.  No deferred execution.
        """
        from natal.genetic_presets import apply_preset_to_population

        ctx = self._make_ctx()
        for preset in presets:
            # _ConfigContext provides the same interface as BasePopulation
            # (species, config, index_registry, add_*_modifier, refresh_modifier_maps)
            apply_preset_to_population(ctx, preset)  # pyright: ignore[reportArgumentType] — adapter implements protocol
        self._sync_from_ctx(ctx)
        self._save_baselines()
        return self

    def modifiers(
        self,
        gamete_modifiers: list[GameteModifier] | None = None,
        zygote_modifiers: list[ZygoteModifier] | None = None,
    ) -> Self:
        """Register gamete / zygote modifiers and rebuild maps immediately."""
        ctx = self._make_ctx()
        next_gid = _ConfigContext.next_modifier_id(ctx.gamete_modifiers)
        if gamete_modifiers:
            for mod in gamete_modifiers:
                ctx.gamete_modifiers.append((next_gid, None, mod))
                next_gid += 1
        next_zid = _ConfigContext.next_modifier_id(ctx.zygote_modifiers)
        if zygote_modifiers:
            for mod in zygote_modifiers:
                ctx.zygote_modifiers.append((next_zid, None, mod))
                next_zid += 1
        if gamete_modifiers or zygote_modifiers:
            _rebuild_config_maps(ctx)
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
        """
        if self._species is None:
            raise RuntimeError(
                "fitness() requires a Species. "
                "Use Configurator.from_species() to create this instance."
            )
        if self._registry is None:
            self._registry = _build_registry(self._species)

        registry = self._registry
        all_genotypes = list(registry.genotype_to_index.keys())

        for patch_name, patch_dict in [
            ("viability", viability),
            ("fecundity", fecundity),
            ("sexual_selection", sexual_selection),
            ("zygote_viability", zygote_viability),
        ]:
            if patch_dict is None:
                continue
            _write_fitness_field(
                self._config, patch_name, patch_dict, mode,
                species=self._species, registry=registry,
                all_genotypes=all_genotypes,
            )
        return self

    # -- hooks ------------------------------------------------------------------

    def hooks(self, *hook_items: _HookItem) -> Self:
        """Register event hooks.

        Hooks are passed through to the Population constructor at
        ``build()`` time — they are *not* config writes.
        """
        if not hasattr(self, "_hook_items"):
            self._hook_items: list[_HookItem] = []
        self._hook_items.extend(hook_items)
        return self

    # -- preset reconfiguration -------------------------------------------------

    def reconfigure_preset(self, preset: GeneticPreset, **changes: object) -> Self:
        """Modify a registered preset parameter and re-apply from baselines.

        Restores baseline fitness / gamete arrays, applies the updated
        preset parameters, and syncs equilibrium.
        """
        baseline = getattr(self, "_baseline", None)
        if baseline is None:
            raise RuntimeError(
                "No baselines saved.  Call .presets() first to register the preset."
            )

        for attr, value in changes.items():
            setattr(preset, attr, value)

        # Restore baselines
        self._config.viability_fitness[...] = baseline["viability"]
        self._config.fecundity_fitness[...] = baseline["fecundity"]
        self._config.sexual_selection_fitness[...] = baseline["sexual_selection"]
        self._config.zygote_viability_fitness[...] = baseline["zygote_viability"]
        self._config.genotype_to_gametes_map[...] = baseline["gamete_map"]
        self._config.gametes_to_zygote_map[...] = baseline["zygote_map"]

        # Clear modifier lists and re-apply
        self.gamete_modifiers.clear()
        self.zygote_modifiers.clear()
        self.presets(preset)
        from natal.engine.simulation.age_structured import sync_equilibrium_metrics

        sync_equilibrium_metrics(self._config)
        return self

    # -- apply / build ---------------------------------------------------------

    def apply(self) -> Self:
        """Sync derived values (equilibrium metrics).

        All parameters are now applied immediately, so this is only
        needed when you modify config arrays directly (outside Configurator).
        """
        self._sync_equilibrium()
        return self

    def _sync_equilibrium(self) -> None:
        """Recompute equilibrium metrics, respecting stored Champer distribution."""
        from natal.engine.simulation.age_structured import (
            compute_equilibrium_metrics,
        )

        eq_dist: NDArray[np.float64] | None = getattr(
            self, "_equilibrium_distribution", None
        )
        # Reshape flat equilibrium_distribution to (n_sexes, n_ages) if needed.
        config = self._config
        if eq_dist is not None and eq_dist.ndim == 1:
            n_ages = int(config.n_ages)
            eq_dist = eq_dist.reshape(2, n_ages)

        # Compute external_expected_eggs from expected_num_adult_females
        # Only when the user explicitly set it (avoid default config value)
        external_eggs: float | None = None
        if getattr(self, "_has_user_expected_females", False):
            from natal.population_builder import PopulationConfigBuilder

            external_eggs = PopulationConfigBuilder.compute_expected_eggs_from_females(
                expected_num_adult_females=float(config.base_expected_num_adult_females),
                expected_eggs_per_female=float(config.expected_eggs_per_female),
                age_based_survival_rates=config.age_based_survival_rates,
                age_based_reproduction_rates=config.age_based_reproduction_rates,
                female_age_based_relative_fertility=config.female_age_based_relative_fertility,
                sex_ratio=float(config.sex_ratio),
                new_adult_age=int(config.new_adult_age),
                n_ages=int(config.n_ages),
            )

        expected_comp, expected_surv = compute_equilibrium_metrics(
            carrying_capacity=float(config.carrying_capacity),
            expected_eggs_per_female=float(config.expected_eggs_per_female),
            age_based_survival_rates=config.age_based_survival_rates,
            age_based_mating_rates=config.age_based_mating_rates,
            age_based_reproduction_rates=config.age_based_reproduction_rates,
            female_age_based_relative_fertility=config.female_age_based_relative_fertility,
            relative_competition_strength=config.age_based_relative_competition_strength,
            sex_ratio=float(config.sex_ratio),
            new_adult_age=int(config.new_adult_age),
            n_ages=int(config.n_ages),
            equilibrium_individual_count=eq_dist,
            external_expected_eggs=external_eggs,
        )
        config.expected_competition_strength[()] = expected_comp
        config.expected_survival_rate[()] = expected_surv

    def _save_baselines(self) -> None:
        """Save pre-preset config arrays for later preset reconfiguration."""
        self._baseline = {
            "viability": self._config.viability_fitness.copy(),
            "fecundity": self._config.fecundity_fitness.copy(),
            "sexual_selection": self._config.sexual_selection_fitness.copy(),
            "zygote_viability": self._config.zygote_viability_fitness.copy(),
            "gamete_map": self._config.genotype_to_gametes_map.copy(),
            "zygote_map": self._config.gametes_to_zygote_map.copy(),
        }

    # -- Internal implementations (called by subclass chain methods) ----------

    def _competition_impl(
        self,
        *,
        carrying_capacity: float | None = None,
        low_density_growth_rate: float | None = None,
        juvenile_growth_mode: int | str | None = None,
        competition_strength: float | None = None,
        expected_num_adult_females: float | None = None,
        equilibrium_distribution: NDArray[np.float64] | None = None,
        age_1_carrying_capacity: float | None = None,
        old_juvenile_carrying_capacity: float | None = None,
    ) -> None:
        mode_value: int | None = None
        if isinstance(juvenile_growth_mode, str):
            from natal.population_config import (
                BEVERTON_HOLT,
                CONCAVE,
                FIXED,
                LINEAR,
                LOGISTIC,
                NO_COMPETITION,
            )
            _MODE_MAP: dict[str, int] = {
                "concave": CONCAVE, "linear": LINEAR, "logistic": LOGISTIC,
                "beverton_holt": BEVERTON_HOLT, "fixed": FIXED,
                "no_competition": NO_COMPETITION,
            }
            mode_value = _MODE_MAP[juvenile_growth_mode.lower()]
        elif juvenile_growth_mode is not None:
            mode_value = juvenile_growth_mode
        k_value = carrying_capacity
        if k_value is None and age_1_carrying_capacity is not None:
            k_value = age_1_carrying_capacity
        if k_value is None and old_juvenile_carrying_capacity is not None:
            k_value = old_juvenile_carrying_capacity
        if k_value is None:
            init_ind = self._config.get_scaled_initial_individual_count()
            if init_ind.size > 0 and init_ind.ndim >= 2 and init_ind.shape[1] >= 2:
                age_1_count = float(init_ind[:, 1, :].sum())
                if age_1_count >= 0.5:
                    k_value = age_1_count
                else:
                    total = float(init_ind.sum())
                    if total >= 0.5:
                        k_value = total
        for name, value in [
            ("carrying_capacity", k_value),
            ("low_density_growth_rate", low_density_growth_rate),
            ("juvenile_growth_mode", mode_value),
            ("competition_strength", competition_strength),
        ]:
            if value is not None:
                set_param(self._config, f"competition.{name}", value)
        if expected_num_adult_females is not None:
            self._config = self._config._replace(
                base_expected_num_adult_females=float(expected_num_adult_females)
            )
            self._has_user_expected_females = True
        if equilibrium_distribution is not None:
            self._equilibrium_distribution = equilibrium_distribution

    def _reproduction_impl(
        self,
        *,
        eggs_per_female: float | None = None,
        sex_ratio: float | None = None,
        sperm_displacement_rate: float | None = None,
        female_age_based_mating_rates: float | list[float] | dict[int, float] | Callable[..., float] | None = None,
        male_age_based_mating_rates: float | list[float] | dict[int, float] | Callable[..., float] | None = None,
        female_age_based_reproduction_rates: float | list[float] | dict[int, float] | Callable[..., float] | None = None,
        female_age_based_relative_fertility: float | list[float] | dict[int, float] | Callable[..., float] | None = None,
        female_adult_mating_rate: float | None = None,
        male_adult_mating_rate: float | None = None,
        use_fixed_egg_count: bool | None = None,
        use_sperm_storage: bool | None = None,
    ) -> None:
        from natal.population_builder import PopulationConfigBuilder

        _ = use_sperm_storage
        n_ages = self._config.n_ages
        resolve = PopulationConfigBuilder.resolve_age_param
        for name, value in [
            ("eggs_per_female", eggs_per_female),
            ("sex_ratio", sex_ratio),
            ("sperm_displacement_rate", sperm_displacement_rate),
        ]:
            if value is not None:
                set_param(self._config, f"reproduction.{name}", value)
        if female_age_based_mating_rates is not None:
            self._config.age_based_mating_rates[0, :] = resolve(
                female_age_based_mating_rates, n_ages, np.zeros(n_ages)
            )
        if male_age_based_mating_rates is not None:
            self._config.age_based_mating_rates[1, :] = resolve(
                male_age_based_mating_rates, n_ages, np.zeros(n_ages)
            )
        if female_age_based_reproduction_rates is not None:
            self._config.age_based_reproduction_rates[:] = resolve(
                female_age_based_reproduction_rates, n_ages, np.ones(n_ages)
            )
        if female_age_based_relative_fertility is not None:
            self._config.female_age_based_relative_fertility[:] = resolve(
                female_age_based_relative_fertility, n_ages, np.ones(n_ages)
            )
        if female_adult_mating_rate is not None:
            self._config.age_based_mating_rates[0, 1] = float(female_adult_mating_rate)
        if male_adult_mating_rate is not None:
            self._config.age_based_mating_rates[1, 1] = float(male_adult_mating_rate)
        if use_fixed_egg_count is not None:
            self._config = self._config._replace(use_fixed_egg_count=use_fixed_egg_count)

    def _survival_impl(
        self,
        *,
        female: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        male: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        female_age_based_survival_rates: list[float] | None = None,
        male_age_based_survival_rates: list[float] | None = None,
        female_age0_survival: float | None = None,
        male_age0_survival: float | None = None,
        adult_survival: float | None = None,
    ) -> None:
        from natal.population_builder import PopulationConfigBuilder

        n_ages = self._config.n_ages
        if female is not None:
            arr = PopulationConfigBuilder.resolve_age_param(
                female, n_ages, np.ones(n_ages)
            )
            self._config.age_based_survival_rates[0, :] = arr
        if male is not None:
            arr = PopulationConfigBuilder.resolve_age_param(
                male, n_ages, np.ones(n_ages)
            )
            self._config.age_based_survival_rates[1, :] = arr
        if female_age_based_survival_rates is not None:
            arr = PopulationConfigBuilder.resolve_age_param(
                female_age_based_survival_rates, n_ages, np.ones(n_ages)
            )
            self._config.age_based_survival_rates[0, :] = arr
        if male_age_based_survival_rates is not None:
            arr = PopulationConfigBuilder.resolve_age_param(
                male_age_based_survival_rates, n_ages, np.ones(n_ages)
            )
            self._config.age_based_survival_rates[1, :] = arr
        for name, value in [
            ("female_age0_survival", female_age0_survival),
            ("male_age0_survival", male_age0_survival),
        ]:
            if value is not None:
                set_param(self._config, f"survival.{name}", value)
        if adult_survival is not None:
            new_adult_age = self._config.new_adult_age
            self._config.age_based_survival_rates[:, new_adult_age:] = float(adult_survival)

    def build(
        self,
        name: str | None = None,
        hooks: _HookMap | None = None,
    ) -> DiscreteGenerationPopulation | AgeStructuredPopulation:
        """Create a Population from the current config.

        Args:
            name: Human-readable population name (defaults to ``self._name``
                  if set via ``.setup(name=...)``, otherwise ``"Population"``).
            hooks: Optional hook registrations.

        Returns:
            An ``AgeStructuredPopulation`` or ``DiscreteGenerationPopulation``
            depending on the config type.
        """
        # Sync equilibrium metrics.
        self.apply()

        # Resolve name: explicit argument > setup(name=...) > default
        if name is None:
            name = getattr(self, "_name", "Population")

        # Merge stored hooks (from .hooks()) with passed hooks.
        stored_hooks: list[_HookItem] | None = getattr(self, "_hook_items", None)
        if stored_hooks:
            hook_map = _merge_hooks(stored_hooks)
            if hooks:
                # Merge external hooks into stored ones
                for event, items in (hooks or {}).items():
                    hook_map.setdefault(event, []).extend(items)
            hooks = hook_map

        # Determine population class from config type.
        from natal.discrete_population_config import DiscretePopulationConfig

        if self._species is None:
            raise RuntimeError(
                "Cannot build Population: no Species set. "
                "Use Configurator.from_species() to create this instance."
            )

        if isinstance(self._config, DiscretePopulationConfig):
            # Sync pre-extracted discrete fields from the latest source maps.
            # These may be stale if gamete/fitness maps were rebuilt by presets
            # or if equilibrium metrics were recomputed after construction.
            self._config = self._config._replace(
                meiosis_f=self._config.genotype_to_gametes_map[0],
                meiosis_m=self._config.genotype_to_gametes_map[1],
                fecundity_f=self._config.fecundity_fitness[0],
                fecundity_m=self._config.fecundity_fitness[1],
                viability_f=self._config.viability_fitness[0, 0, :],
                viability_m=self._config.viability_fitness[1, 0, :],
            )
            from natal.discrete_generation_population import (
                DiscreteGenerationPopulation,
            )

            return DiscreteGenerationPopulation(
                species=self._species,
                population_config=self._config,
                name=name,
                hooks=hooks,
            )
        else:
            from natal.age_structured_population import (
                AgeStructuredPopulation,
            )

            return AgeStructuredPopulation(
                species=self._species,
                population_config=self._config,
                name=name,
                hooks=hooks,
            )


# ── Typed Configurator subclasses ────────────────────────────────────────────
# Each subclass overrides build() with the specific Population return type,
# so the IDE / type checker knows exactly what you get after the chain.



# ── Model-specific Configurators ────────────────────────────────────────────


class DiscreteConfigurator(Configurator):
    """Configurator for ``DiscreteGenerationPopulation`` (Wright-Fisher).

    Two age classes (age-0 juveniles, age-1 adults).  Non-overlapping
    generations — adults are replaced each tick.

    Create via ``Configurator.for_discrete(species)`` or
    ``DiscreteGenerationPopulation.setup(species)``.
    """

    def age_structure(
        self, n_ages: int = 2, new_adult_age: int = 1,
        generation_time: float | None = None,
    ) -> DiscreteConfigurator:
        """Lock dimensions.  Discrete uses 2 ages, adult at age 1."""
        super().age_structure(n_ages=n_ages, new_adult_age=new_adult_age,
                              generation_time=generation_time)
        return self

    def competition(
        self,
        *,
        carrying_capacity: float | None = None,
        low_density_growth_rate: float | None = None,
        juvenile_growth_mode: int | str | None = None,
        age_1_carrying_capacity: float | None = None,
    ) -> DiscreteConfigurator:
        """Configure density-dependent competition.

        Args:
            carrying_capacity (K): Equilibrium total adults at age 1.
            low_density_growth_rate (r): Per-capita growth at low density.
            juvenile_growth_mode: ``"concave"``, ``"logistic"``, … or int.
            age_1_carrying_capacity: Legacy alias for *carrying_capacity*.
        """
        self._competition_impl(
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=low_density_growth_rate,
            juvenile_growth_mode=juvenile_growth_mode,
            age_1_carrying_capacity=age_1_carrying_capacity,
        )
        return self

    def reproduction(
        self,
        *,
        eggs_per_female: float | None = None,
        sex_ratio: float | None = None,
        female_adult_mating_rate: float | None = None,
        male_adult_mating_rate: float | None = None,
        use_fixed_egg_count: bool | None = None,
    ) -> DiscreteConfigurator:
        """Configure reproduction for the discrete-generation model.

        Args:
            eggs_per_female: Eggs per reproducing female per tick.
            sex_ratio: Female fraction of offspring (0–1).
            female_adult_mating_rate: Adult female mating probability.
            male_adult_mating_rate: Adult male mating probability.
            use_fixed_egg_count: Disable Poisson noise.
        """
        self._reproduction_impl(
            eggs_per_female=eggs_per_female,
            sex_ratio=sex_ratio,
            female_adult_mating_rate=female_adult_mating_rate,
            male_adult_mating_rate=male_adult_mating_rate,
            use_fixed_egg_count=use_fixed_egg_count,
        )
        return self

    def survival(
        self,
        *,
        female_age0_survival: float | None = None,
        male_age0_survival: float | None = None,
        adult_survival: float | None = None,
    ) -> DiscreteConfigurator:
        """Configure survival.  Only age-0 (juvenile→adult) matters.

        Both default to 1.0.

        Args:
            female_age0_survival: Female juvenile survival probability.
            male_age0_survival: Male juvenile survival probability.
            adult_survival: Accepted for compat, no-op in discrete model.
        """
        self._survival_impl(
            female_age0_survival=female_age0_survival,
            male_age0_survival=male_age0_survival,
            adult_survival=adult_survival,
        )
        return self

    def build(
        self, name: str | None = None, hooks: _HookMap | None = None,
    ) -> DiscreteGenerationPopulation:
        """Build and return a ``DiscreteGenerationPopulation``."""
        from natal.discrete_generation_population import (
            DiscreteGenerationPopulation as DGP,
        )
        result = super().build(name=name, hooks=hooks)
        return cast(DGP, result)


class AgeStructuredConfigurator(Configurator):
    """Configurator for ``AgeStructuredPopulation`` (overlapping generations).

    Supports arbitrary age classes with per-age survival, mating, and
    fertility.  Adults survive across ticks — generations overlap.

    Per-age parameters accept flexible input: scalar, list, dict, callable.

    Create via ``Configurator.for_age_structured(species)`` or
    ``AgeStructuredPopulation.setup(species, legacy_path=False)``.
    """

    def age_structure(
        self, n_ages: int, new_adult_age: int,
        generation_time: float | None = None,
    ) -> AgeStructuredConfigurator:
        """Lock population dimensions.

        Args:
            n_ages: Total number of age classes.
            new_adult_age: First adult age.
            generation_time: Optional marker for model interpretation.
        """
        super().age_structure(
            n_ages=n_ages, new_adult_age=new_adult_age,
            generation_time=generation_time,
        )
        return self

    def competition(
        self,
        *,
        carrying_capacity: float | None = None,
        low_density_growth_rate: float | None = None,
        juvenile_growth_mode: int | str | None = None,
        competition_strength: float | None = None,
        expected_num_adult_females: float | None = None,
        equilibrium_distribution: NDArray[np.float64] | None = None,
        age_1_carrying_capacity: float | None = None,
        old_juvenile_carrying_capacity: float | None = None,
    ) -> AgeStructuredConfigurator:
        """Configure density-dependent competition.

        Args:
            carrying_capacity (K): Equilibrium population at age 1.
            low_density_growth_rate (r): Per-capita growth at low density.
            juvenile_growth_mode: Regulation function (string or int).
            competition_strength: Larval competition weight.
            expected_num_adult_females: Target adult females (Champer model).
            equilibrium_distribution: Custom (n_sexes, n_ages) array for
                Champer equilibrium computation.
            age_1_carrying_capacity: Legacy alias for *carrying_capacity*.
            old_juvenile_carrying_capacity: Legacy alias.
        """
        self._competition_impl(
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=low_density_growth_rate,
            juvenile_growth_mode=juvenile_growth_mode,
            competition_strength=competition_strength,
            expected_num_adult_females=expected_num_adult_females,
            equilibrium_distribution=equilibrium_distribution,
            age_1_carrying_capacity=age_1_carrying_capacity,
            old_juvenile_carrying_capacity=old_juvenile_carrying_capacity,
        )
        return self

    def reproduction(
        self,
        *,
        eggs_per_female: float | None = None,
        sex_ratio: float | None = None,
        sperm_displacement_rate: float | None = None,
        female_age_based_mating_rates: float | list[float] | dict[int, float] | Callable[..., float] | None = None,
        male_age_based_mating_rates: float | list[float] | dict[int, float] | Callable[..., float] | None = None,
        female_age_based_reproduction_rates: float | list[float] | dict[int, float] | Callable[..., float] | None = None,
        female_age_based_relative_fertility: float | list[float] | dict[int, float] | Callable[..., float] | None = None,
        use_fixed_egg_count: bool | None = None,
        use_sperm_storage: bool | None = None,
    ) -> AgeStructuredConfigurator:
        """Configure reproduction for the age-structured model.

        Args:
            eggs_per_female: Base eggs per reproducing female.
            sex_ratio: Female fraction of offspring (0–1).
            sperm_displacement_rate: Fraction of stored sperm displaced.
            female_age_based_mating_rates: Per-age female mating probability.
            male_age_based_mating_rates: Per-age male mating probability.
            female_age_based_reproduction_rates: Per-age reproduction participation.
            female_age_based_relative_fertility: Per-age fertility weight.
            use_fixed_egg_count: Disable Poisson noise.
            use_sperm_storage: Enable sperm storage.
        """
        self._reproduction_impl(
            eggs_per_female=eggs_per_female,
            sex_ratio=sex_ratio,
            sperm_displacement_rate=sperm_displacement_rate,
            female_age_based_mating_rates=female_age_based_mating_rates,
            male_age_based_mating_rates=male_age_based_mating_rates,
            female_age_based_reproduction_rates=female_age_based_reproduction_rates,
            female_age_based_relative_fertility=female_age_based_relative_fertility,
            use_fixed_egg_count=use_fixed_egg_count,
            use_sperm_storage=use_sperm_storage,
        )
        return self

    def survival(
        self,
        *,
        female: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        male: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        female_age_based_survival_rates: list[float] | None = None,
        male_age_based_survival_rates: list[float] | None = None,
        adult_survival: float | None = None,
        female_age0_survival: float | None = None,
        male_age0_survival: float | None = None,
    ) -> AgeStructuredConfigurator:
        """Configure survival rates.  Per-age params accept flexible forms.

        Args:
            female: Female survival rates (scalar, list, dict, or callable).
            male: Male survival rates (same forms).
            female_age_based_survival_rates: Legacy alias for *female*.
            male_age_based_survival_rates: Legacy alias for *male*.
            adult_survival: Shorthand — sets all adult ages uniformly.
            female_age0_survival: Discrete-model shortcut for age-0 female.
            male_age0_survival: Discrete-model shortcut for age-0 male.
        """
        self._survival_impl(
            female=female, male=male,
            female_age_based_survival_rates=female_age_based_survival_rates,
            male_age_based_survival_rates=male_age_based_survival_rates,
            adult_survival=adult_survival,
            female_age0_survival=female_age0_survival,
            male_age0_survival=male_age0_survival,
        )
        return self

    def build(
        self, name: str | None = None, hooks: _HookMap | None = None,
    ) -> AgeStructuredPopulation:
        """Build and return an ``AgeStructuredPopulation``."""
        from natal.age_structured_population import (
            AgeStructuredPopulation as ASP,
        )
        result = super().build(name=name, hooks=hooks)
        return cast(ASP, result)


# -- parameter name resolution -----------------------------------------------


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


def _resolve_param(name: str) -> ParamDescriptor | None:
    """Look up a parameter name in ALL_PARAMETERS, checking aliases.

    Returns the ParamDescriptor or None.
    """
    if name in ALL_PARAMETERS:
        return ALL_PARAMETERS[name]

    for key, desc in ALL_PARAMETERS.items():
        if key.endswith(f".{name}"):
            return desc
        if name in desc.aliases:
            return desc

    return None
