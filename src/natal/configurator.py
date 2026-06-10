"""Mutable wrapper for PopulationConfig, with a chainable API.

Functionality:
  - Read/write config fields through a chainable API.
  - Register custom named parameters (stored as a structured numpy array).
  - Freeze changes back to an immutable NamedTuple via ``_replace``
    (cheap — all ndarray fields are shared by reference).

Why this module exists:
  ``PopulationConfig`` / ``DiscretePopulationConfig`` are immutable
  NamedTuples — fields cannot be modified once created.  However,
  during simulation setup (and inside hooks at runtime), parameters
  need real-time adjustment.  The ``Configurator`` provides a mutable
  layer on top: all modifications write into the config arrays
  in-place, and the final immutable config is materialised via
  ``build()``.

  The adapter class ``_ConfigContext`` lets genetic presets and
  modifiers operate on config arrays without needing a live Population
  object.  The standalone :func:`set_param` function is also usable
  from within Numba-compiled hooks via ``objmode``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Self, cast

import numpy as np
from numba import objmode  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from natal.genetic_structures import Species
from natal.index_registry import IndexRegistry
from natal.numba_utils import njit_switch
from natal.parameters import ALL_PARAMETERS, ParamDescriptor
from natal.population_config import (
    DiscretePopulationConfig,
    PopulationConfig,
)

if TYPE_CHECKING:
    from natal.age_structured_population import AgeStructuredPopulation
    from natal.base_population import BasePopulation
    from natal.discrete_generation_population import DiscreteGenerationPopulation
    from natal.genetic_entities import Genotype
    from natal.genetic_presets import GeneticPreset
    from natal.modifiers import GameteModifier, ZygoteModifier

__all__ = ["Configurator", "DiscreteConfigurator", "AgeStructuredConfigurator"]

# ── Type aliases for hook registrations ──────────────────────────────────────

# A single hook registration: (func, name?, priority?)
# ``Any`` return is required for compatibility with ``HookRegistration`` in
# base_population.py (tuple invariance — ``Callable[..., int]`` would fail).
_HookReg = tuple[Callable[..., Any], str | None, int | None]
# Hook registration map keyed by event name.
_HookMap = dict[str, list[_HookReg]]
# A hook item can be a raw dict or a callable with @hook metadata.
_HookItem = Callable[..., Any] | _HookMap


# ── Registry builder (shared by Configurator and adapter) ──────────────────────


def _build_registry(species: Species) -> IndexRegistry:
    """Build an IndexRegistry pre-populated with all genotypes/haplotypes from a Species.

    Note:
        When *species* does not define ``gamete_labels`` (or they are empty),
        a single ``"default"`` label is registered so that modifier code
        always has at least one gamete-label slot to index into.
    """
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
    """Adapter that lets presets/modifiers/fitness write into config arrays
    without needing a live Population object.

    ``apply_preset_to_population``, ``build_modifier_wrappers``, and
    ``_apply_preset_fitness_patch`` are all designed to work against
    ``BasePopulation``.  But during ``Configurator`` builds there *is* no
    Population yet.  This class exposes the four things those functions
    actually need — *species*, *config*, *index_registry*, and the two
    modifier lists — with the same attribute names and mutation patterns
    as ``BasePopulation``.

    After the preset/modifier/fitness call returns, :meth:`Configurator.
    _sync_from_ctx` pulls the mutated *config* and modifier lists back
    into the Configurator.
    """

    def __init__(
        self,
        species: Species,
        config: PopulationConfig | DiscretePopulationConfig,
        registry: IndexRegistry,
    ) -> None:
        """Initialise the adapter with species, config, and registry.

        Args:
            species: The genetic architecture for the population.
            config: The PopulationConfig or DiscretePopulationConfig to wrap.
            registry: An IndexRegistry pre-populated with genotypes/haplotypes.
        """
        self.species = species
        self.config = config
        self.registry = registry
        self.index_registry = registry
        # Mirror names matching BasePopulation for drop-in compatibility.
        self.gamete_modifiers: list[tuple[int, str | None, GameteModifier]] = []
        self.zygote_modifiers: list[tuple[int, str | None, ZygoteModifier]] = []

    # -- modifier registration (mimics BasePopulation) ----------------------

    def add_gamete_modifier(
        self,
        modifier: GameteModifier,
        *,
        name: str | None = None,
        modifier_id: int | None = None,
        refresh: bool = True,
    ) -> None:
        """Register a gamete modifier on the adapter.

        Args:
            modifier: The GameteModifier callable.
            name: Optional name string for identification.
            modifier_id: Explicit modifier ID (auto-assigned if ``None``).
            refresh: If ``True`` (default), rebuild modifier maps immediately.
        """
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
        """Register a zygote modifier on the adapter.

        Args:
            modifier: The ZygoteModifier callable.
            name: Optional name string for identification.
            modifier_id: Explicit modifier ID (auto-assigned if ``None``).
            refresh: If ``True`` (default), rebuild modifier maps immediately.
        """
        resolved_id = _ConfigContext.next_modifier_id(self.zygote_modifiers) if modifier_id is None else modifier_id
        self.zygote_modifiers.append((resolved_id, name, modifier))
        self.zygote_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self.refresh_modifier_maps()

    def refresh_modifier_maps(self) -> None:
        """Rebuild config maps from the current modifier lists.

        Delegates to :func:`_rebuild_config_maps`, which recomputes the
        offspring probability tensor from the accumulated gamete and
        zygote modifier callables.
        """
        _rebuild_config_maps(self)

    # ``Any`` for the 3rd tuple element is justified: the function only
    # reads ``id`` and ``name`` (1st/2nd elements); the modifier object
    # itself is never accessed, so its type is irrelevant.
    @staticmethod
    def next_modifier_id(modifiers: list[tuple[int, str | None, Any]]) -> int:
        """Return the next available modifier ID.

        Scans the existing modifier IDs and returns ``max + 1``
        (or ``0`` if the list is empty).

        Args:
            modifiers: List of ``(id, name, modifier)`` tuples.

        Returns:
            The next available integer ID.
        """
        ids = [mid for mid, _, _ in modifiers]
        return (max(ids) + 1) if ids else 0


# ── Core: rebuild genotype/gamete/zygote maps from modifier lists ─────────────


def _rebuild_config_maps(ctx: _ConfigContext) -> None:
    """Apply gamete/zygote modifiers and rebuild ``offspring_tensor``.

    Starts from the species-level Mendelian baseline (cached via
    :meth:`Species.get_config_blueprint`) and applies modifier callables
    in-place, avoiding redundant O(n²) baseline recomputation.

    Returns:
        ``None`` — the mutated config is written back through
        ``ctx.config``, which the caller is expected to sync via
        :meth:`Configurator._sync_from_ctx`.
    """
    from natal.engine.simulation.age_structured import (
        compute_offspring_probability_tensor,
    )
    from natal.modifiers import build_modifier_wrappers

    # ---- resolve genotype/haplotype lists from the registry ----
    haploid_genotypes = ctx.registry.index_to_haplo
    diploid_genotypes = ctx.registry.index_to_genotype
    if not haploid_genotypes or not diploid_genotypes:
        return  # species has no haploid genotypes (no sex chromosomes)

    n_glabs = int(ctx.config.n_glabs)
    # ---- compile modifier callables from the accumulated modifier lists ----
    # build_modifier_wrappers converts user-supplied GameteModifier /
    # ZygoteModifier objects into tensor → tensor callables.
    gamete_funcs, zygote_funcs = build_modifier_wrappers(
        gamete_modifiers=ctx.gamete_modifiers,
        zygote_modifiers=ctx.zygote_modifiers,
        population=None,
        index_registry=ctx.registry,
        haploid_genotypes=haploid_genotypes,
        diploid_genotypes=diploid_genotypes,
        n_glabs=n_glabs,
    )

    # ---- fetch the Mendelian baseline from the species cache ----
    # Because get_config_blueprint() is cached per-species, repeatedly
    # calling it is cheap.  We copy the arrays so that modifier callables
    # can mutate them without corrupting the cached originals.
    bp = ctx.species.get_config_blueprint()
    genotype_to_gametes_map = bp["genotype_to_gametes_map"].copy()
    gametes_to_zygote_map = bp["gametes_to_zygote_map"].copy()

    # ---- chain modifier callables on top of the baseline ----
    # Each callable accepts and returns a tensor of the same shape,
    # allowing modifiers to be composed in registration order.
    for fn in gamete_funcs:
        genotype_to_gametes_map = fn(genotype_to_gametes_map)
    for fn in zygote_funcs:
        gametes_to_zygote_map = fn(gametes_to_zygote_map)

    # ---- recompute offspring probability tensor from the updated maps ----
    # This tensor pre-computes P(offspring_genotype | mother, father) for
    # all genotype pairs.  It depends on meiosis maps and zygote maps,
    # both of which may have been altered by modifiers above.
    offspring_tensor = compute_offspring_probability_tensor(
        meiosis_f=genotype_to_gametes_map[0],
        meiosis_m=genotype_to_gametes_map[1],
        haplo_to_genotype_map=gametes_to_zygote_map,
        n_genotypes=int(ctx.config.n_genotypes),
        n_haplogenotypes=int(ctx.config.n_haploid_genotypes),
        n_glabs=n_glabs,
    )

    # ---- write everything back into the config via _replace ----
    # The three maps replace their config counterparts.  For
    # DiscretePopulationConfig we must additionally update the
    # pre-extracted per-sex slices (meiosis_f, viability_f, etc.),
    # because those are what the discrete engine reads at runtime.
    overrides: dict[str, Any] = {
        "genotype_to_gametes_map": genotype_to_gametes_map,
        "gametes_to_zygote_map": gametes_to_zygote_map,
        "offspring_tensor": offspring_tensor,
    }
    if isinstance(ctx.config, DiscretePopulationConfig):
        # Keep the pre-extracted slices in sync with the source maps.
        overrides["meiosis_f"] = genotype_to_gametes_map[0]
        overrides["meiosis_m"] = genotype_to_gametes_map[1]
        overrides["fecundity_f"] = ctx.config.fecundity_fitness[0]
        overrides["fecundity_m"] = ctx.config.fecundity_fitness[1]
        overrides["viability_f"] = ctx.config.viability_fitness[0, 0, :]
        overrides["viability_m"] = ctx.config.viability_fitness[1, 0, :]
        # Pre-extracted scalars must also stay in sync.
        cfg = ctx.config
        overrides["female_mating_rate"] = cfg.age_based_mating_rates[0, 1]
        overrides["male_mating_rate"] = cfg.age_based_mating_rates[1, 1]
        overrides["reproduction_rate"] = cfg.age_based_reproduction_rates[1]
        overrides["base_survival_f"] = cfg.age_based_survival_rates[0, 0]
        overrides["base_survival_m"] = cfg.age_based_survival_rates[1, 0]
    ctx.config = ctx.config._replace(**overrides)


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


def _write_fitness_field(
    config: PopulationConfig | DiscretePopulationConfig,
    field_name: str,
    patch: Mapping[str, float | Mapping[str, float]],
    mode: str,
    *,
    species: Species,
    registry: IndexRegistry,
    all_genotypes: list[Genotype],
) -> None:
    """Resolve genotype-pattern strings and write into a fitness tensor.

    *field_name* is one of ``"viability"``, ``"fecundity"``,
    ``"sexual_selection"``, or ``"zygote_viability"``.  *patch* is a
    dict mapping genotype selectors to fitness values, with optional
    sex-keyed or genotype-keyed nesting.

    The function detects the format of *patch* and dispatches to one of
    four branches.  See inline comments for the detection rules.

    Supported formats::

        {genotype: val}                                        # scalar → both sexes, all ages
        {genotype: {"female": val, "male": val}}               # per-selector sex-keyed
        {genotype: {0: val, 1: val}}                           # per-selector age-keyed
        {genotype: {"female": {0: val}}}                       # per-selector sex+age keyed
        {"female": {genotype: val}, "male": {...}}             # top-level sex-keyed
        {female_g: {male_g: val}}                              # sexual_selection pair format
    """
    # ══════════════════════════════════════════════════════════════════════
    # BRANCH 1: top-level sex-keyed
    #   {"female": {genotype: val}, "male": {genotype: val}}
    #
    # Detection: ALL top-level keys are "female" or "male".
    # Action: iterate sex→genotype_dict, delegate each to _write_fitness_field_flat.
    # ══════════════════════════════════════════════════════════════════════
    if patch and all(k in ("female", "male") for k in patch):
        # ---- guard: every value must itself be a dict {genotype: val} ----
        if not all(isinstance(v, Mapping) for v in patch.values()):
            raise TypeError(
                "sex-keyed fitness dict values must be genotype→value mappings"
            )
        sex_patch = cast(Mapping[str, Mapping[str, float]], patch)
        # ---- write female slice, then male slice ----
        for sex_key, geno_dict in sex_patch.items():
            sex_idx = 0 if sex_key == "female" else 1
            _write_fitness_field_flat(
                config, field_name, geno_dict, mode,
                sex_idx=sex_idx,
                species=species, registry=registry,
                all_genotypes=all_genotypes,
            )
        return

    # ══════════════════════════════════════════════════════════════════════
    # BRANCH 2: sexual_selection — nested female→male pair format
    #   {female_g: {male_g: val}}
    #
    # Detection: field is "sexual_selection" AND any value is a Mapping.
    # Action: resolve female & male selectors independently,
    #         then write into each [f_idx, m_idx] cell.
    # ══════════════════════════════════════════════════════════════════════
    if field_name == "sexual_selection":
        arr = config.sexual_selection_fitness          # shape: (g, g) — [female_idx, male_idx]
        has_nested = any(isinstance(v, Mapping) for v in patch.values())
        if has_nested:
            for female_selector, male_map in patch.items():           # outer: female genotype key
                if not isinstance(male_map, Mapping):                 # guard: must be {male_g: val}
                    raise TypeError(
                        "Mixed scalar/nested format in sexual_selection. "
                        "When using nested female→male pairs, all values "
                        "must be dicts mapping male selectors to values."
                    )
                for male_selector, value in male_map.items():         # inner: male genotype key → float
                    # ---- resolve both selectors to genotype indices ----
                    matched_f = species.resolve_genotype_selectors(
                        selector=female_selector,
                        all_genotypes=all_genotypes,
                        context="sexual_selection (female)",
                    )
                    matched_m = species.resolve_genotype_selectors(
                        selector=male_selector,
                        all_genotypes=all_genotypes,
                        context="sexual_selection (male)",
                    )
                    # ---- write every female×male combination ----
                    for f_geno in matched_f:
                        f_idx = registry.genotype_to_index[f_geno]
                        for m_geno in matched_m:
                            m_idx = registry.genotype_to_index[m_geno]
                            val = float(value)
                            if mode == "replace":
                                arr[f_idx, m_idx] = val
                            else:
                                arr[f_idx, m_idx] *= val
            return

        # ═══════════════════════════════════════════════════════════════
        # BRANCH 3: sexual_selection — flat male-keyed format
        #   {male_g: val}
        #
        # Detection: field is "sexual_selection" AND no nested values.
        # Action: resolve male selector, write value to ALL female rows
        #         of the matched male column (arr[:, m_idx]).
        # ═══════════════════════════════════════════════════════════════
        for male_selector, value in patch.items():
            if isinstance(value, Mapping):                            # guard: mixed scalar/nested
                raise TypeError(
                    "Mixed scalar/nested format in sexual_selection. "
                    "When using nested female→male pairs, all values "
                    "must be dicts."
                )
            # ---- resolve the male genotype to an index ----
            matched_m = species.resolve_genotype_selectors(
                selector=male_selector,
                all_genotypes=all_genotypes,
                context="sexual_selection (male)",
            )
            for m_geno in matched_m:
                m_idx = registry.genotype_to_index[m_geno]
                val = float(value)
                if mode == "replace":
                    arr[:, m_idx] = val        # broadcast: all females × this male
                else:
                    arr[:, m_idx] *= val
        return

    # ══════════════════════════════════════════════════════════════════════
    # BRANCH 4: per-selector resolution (viability / fecundity / zygote)
    #   {genotype: val}
    #   {genotype: {"female": val, "male": val}}           — sex-keyed
    #   {genotype: {0: val, 1: val}}                       — age-keyed
    #   {genotype: {"female": {0: val}}}                   — sex+age keyed
    #
    # Detection: everything not caught by branches 1-3.
    # Each selector value may be:
    #   - scalar → apply to both sexes, all ages
    #   - Mapping → inspect the first key to decide the format
    # ══════════════════════════════════════════════════════════════════════
    for selector, value in patch.items():
        if isinstance(value, Mapping):
            # Inspect the first key to determine the nesting structure.
            first_key = next(iter(value.keys()))
            if isinstance(first_key, int) and not isinstance(first_key, bool):  # type: ignore[unnecessary-isinstance] — bool ⊂ int in Python
                # ---- age-keyed: {genotype: {0: val, 1: val}} ----
                for age_key, age_val in value.items():          # type: ignore[var-unknown]  # Mapping values are Any without explicit type params
                    if age_val is None:                         # type: ignore[unnecessary-comparison]  # user may pass {age: None} to skip
                        continue
                    age = int(age_key)
                    for sex_idx in (0, 1):
                        _write_fitness_field_flat(
                            config, field_name,
                            {selector: float(age_val)}, mode,
                            sex_idx=sex_idx, age_idx=age,
                            species=species, registry=registry,
                            all_genotypes=all_genotypes,
                        )
            elif first_key in ("female", "male"):
                # ---- sex-keyed: {genotype: {"female": val, "male": val}} ----
                for sex_key, sex_val in value.items():
                    sex_idx = 0 if sex_key == "female" else 1
                    if isinstance(sex_val, Mapping):
                        # ---- sex+age keyed: {genotype: {"female": {0: val}}} ----
                        for age_key, age_val in sex_val.items():          # type: ignore[var-unknown]  # Mapping values are Any without explicit type params
                            if age_val is None:
                                continue
                            _write_fitness_field_flat(
                                config, field_name,
                                {selector: float(age_val)}, mode,  # type: ignore[arg-type]  # age_val is Unknown from unparameterized Mapping
                                sex_idx=sex_idx, age_idx=int(age_key),  # type: ignore[arg-type]  # age_key is Unknown from unparameterized Mapping
                                species=species, registry=registry,
                                all_genotypes=all_genotypes,
                            )
                    else:
                        # ---- simple sex-keyed (existing behavior) ----
                        _write_fitness_field_flat(
                            config, field_name,
                            {selector: float(sex_val)}, mode,
                            sex_idx=sex_idx,
                            species=species, registry=registry,
                            all_genotypes=all_genotypes,
                        )
            else:
                raise TypeError(
                    f"Unrecognised key in fitness value dict: {first_key!r}. "
                    f"Expected 'female'/'male' (sex-keyed) or int (age-keyed)."
                )
        else:
            # ---- scalar format: {genotype: val} → apply to both sexes, all ages ----
            for sex_idx in (0, 1):
                _write_fitness_field_flat(
                    config, field_name,
                    {selector: float(value)}, mode,
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
    all_genotypes: list[Genotype],
    age_idx: int | None = None,
) -> None:
    """Write a flat (per-genotype) fitness patch into the correct config array.

    The target array shape depends on *field_name*:

    - ``"viability"`` → ``(n_sexes, n_ages, n_genotypes)`` — writes ``[sex_idx, default_age, gidx]``
    - ``"fecundity"`` → ``(n_sexes, n_genotypes)`` — no age axis
    - ``"sexual_selection"`` → ``(n_genotypes, n_genotypes)`` — no age axis
    - ``"zygote_viability"`` → ``(n_sexes, n_genotypes)`` — no age axis

    When *age_idx* is ``None`` (the default), the write targets the
    last juvenile age (``new_adult_age - 1``) — viability fitness
    normally represents larval / juvenile survival, not adult fitness.
    ``fecundity`` and ``zygote_viability`` have no age axis so
    *age_idx* is ignored for them.

    Args:
        config: The PopulationConfig or DiscretePopulationConfig to modify.
        field_name: One of ``"viability"``, ``"fecundity"``,
            ``"sexual_selection"``, or ``"zygote_viability"``.
        patch: A flat ``{genotype_selector: value}`` mapping.
        mode: ``"replace"`` (overwrite) or ``"multiply"`` (scale existing).
        sex_idx: Index for the sex axis (0 = female, 1 = male).
        species: The Species for genotype-selector resolution.
        registry: The IndexRegistry mapping genotypes to indices.
        all_genotypes: List of all genotype objects for selector matching.
        age_idx: Age index for the write (defaults to ``new_adult_age - 1``).
    """
    # Default to last juvenile age: viability typically affects
    # larvae/juveniles, not adults.  DiscretePopulationConfig has
    # no ``new_adult_age`` field (always 2 ages, adult at age 1).
    resolved_age: int = age_idx if age_idx is not None else (
        getattr(config, "new_adult_age", 1) - 1
    )

    for selector, value in patch.items():
        matched = species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context=f"fitness.{field_name}",
        )
        for genotype in matched:
            gidx = registry.genotype_to_index[genotype]
            age_slice = slice(resolved_age, resolved_age + 1)

            if field_name == "viability":
                arr = config.viability_fitness
                if mode == "replace":
                    arr[sex_idx, age_slice, gidx] = float(value)
                else:
                    arr[sex_idx, age_slice, gidx] *= float(value)
            elif field_name == "fecundity":
                arr = config.fecundity_fitness          # no age axis — age_idx ignored
                if mode == "replace":
                    arr[sex_idx, gidx] = float(value)
                else:
                    arr[sex_idx, gidx] *= float(value)
            elif field_name == "sexual_selection":
                arr = config.sexual_selection_fitness
                # sexual_selection is indexed [female_idx, male_idx].
                # sex_idx=0 → genotype is female → write row (arr[gidx, :])
                # sex_idx=1 → genotype is male   → write col (arr[:, gidx])
                if mode == "replace":
                    if sex_idx == 0:
                        arr[gidx, :] = float(value)
                    else:
                        arr[:, gidx] = float(value)
                else:
                    if sex_idx == 0:
                        arr[gidx, :] *= float(value)
                    else:
                        arr[:, gidx] *= float(value)
            elif field_name == "zygote_viability":
                arr = config.zygote_viability_fitness  # no age axis — age_idx ignored
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
        self._female_mating_rate: float | None = None
        self._male_mating_rate: float | None = None
        self._base_survival_f: float | None = None
        self._base_survival_m: float | None = None

    @property
    def config(self) -> PopulationConfig | DiscretePopulationConfig:
        """The wrapped PopulationConfig (read-only accessor)."""
        return self._config

    # -- adapter factory ------------------------------------------------------

    def _make_ctx(self) -> _ConfigContext:
        """Build a :class:`_ConfigContext` seeded from the current state.

        The context receives a shallow copy of the modifier lists so that
        preset / modifier calls can append without mutating the originals
        until :meth:`_sync_from_ctx` explicitly commits them back.

        Raises:
            RuntimeError: If ``_species`` is ``None`` — the Configurator
                must be created via :meth:`from_species` when using
                presets, modifiers, or fitness.
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
            self._registry = _build_registry(self._species)
        ctx = _ConfigContext(self._species, self._config, self._registry)
        ctx.gamete_modifiers = list(self.gamete_modifiers)
        ctx.zygote_modifiers = list(self.zygote_modifiers)
        return ctx

    def _sync_from_ctx(self, ctx: _ConfigContext) -> None:
        """Commit adapter-side mutations back into the Configurator.

        Called after ``apply_preset_to_population`` or ``_rebuild_config_maps``
        has finished writing into *ctx.config* and the modifier lists.
        """
        self._config = ctx.config
        self.gamete_modifiers = ctx.gamete_modifiers
        self.zygote_modifiers = ctx.zygote_modifiers

    # -- factory ---------------------------------------------------------------

    @classmethod
    def from_species(
        cls,
        species: Species,
        *,
        discrete: bool = False,
    ) -> DiscreteConfigurator | AgeStructuredConfigurator:
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
        bp = species.get_config_blueprint()
        n_g = bp["n_genotypes"]
        n_hg = bp["n_haploid_genotypes"]
        n_gl = bp["n_glabs"]
        g2g = bp["genotype_to_gametes_map"]
        g2z = bp["gametes_to_zygote_map"]
        has_sc = getattr(species, "has_sex_chromosomes", False)

        if discrete:
            from natal.population_config import build_discrete_population_config

            config = build_discrete_population_config(
                n_genotypes=n_g, n_haploid_genotypes=n_hg, n_glabs=n_gl,
                genotype_to_gametes_map=g2g, gametes_to_zygote_map=g2z,
                has_sex_chromosomes=has_sc,
            )
            result = DiscreteConfigurator(config, species=species)
            object.__setattr__(result, "_name", "DiscreteGenerationPop")
        else:
            from natal.population_config import build_population_config

            config = build_population_config(
                n_genotypes=n_g, n_haploid_genotypes=n_hg, n_glabs=n_gl,
                genotype_to_gametes_map=g2g, gametes_to_zygote_map=g2z,
                n_ages=2, new_adult_age=1, carrying_capacity=1000.0,
                has_sex_chromosomes=has_sc,
            )
            result = AgeStructuredConfigurator(config, species=species)
            object.__setattr__(result, "_name", "AgeStructuredPop")
        return result

    @classmethod
    def for_discrete(cls, species: Species) -> DiscreteConfigurator:
        """Shorthand for ``from_species(species, discrete=True)``."""
        return cast(DiscreteConfigurator, cls.from_species(species, discrete=True))

    @classmethod
    def for_age_structured(cls, species: Species) -> AgeStructuredConfigurator:
        """Shorthand for ``from_species(species)``."""
        return cast(AgeStructuredConfigurator, cls.from_species(species))

    @staticmethod
    def for_config(
        config: PopulationConfig | DiscretePopulationConfig,
    ) -> DiscreteConfigurator | AgeStructuredConfigurator:
        """Return the right Configurator subclass for the given config type.

        Args:
            config: The config to wrap.

        Returns:
            ``DiscreteConfigurator`` if *config* is a
            ``DiscretePopulationConfig``, otherwise
            ``AgeStructuredConfigurator``.
        """
        if isinstance(config, DiscretePopulationConfig):
            return DiscreteConfigurator(config)
        return AgeStructuredConfigurator(config)

    @staticmethod
    def for_population(pop: BasePopulation[Any]) -> DiscreteConfigurator | AgeStructuredConfigurator:
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
        use_continuous_sampling: bool | None = None,
        use_fixed_egg_count: bool | None = None,
    ) -> Self:
        """Configure simulation flags and optional population name.

        *name* is stored and used by ``build()`` when no explicit name is given.

        Args:
            name: Population name (falls back to ``"Population"`` at build time).
            stochastic: If ``False``, use deterministic (median) outcomes.
            use_continuous_sampling: If ``True``, sample from continuous
                distributions instead of discrete counts.
            use_fixed_egg_count: If ``True``, disable Poisson noise on egg counts.

        Returns:
            Self for chaining.
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
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
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
        Genotype selectors accept both strings and ``Genotype`` objects.
        Resolved to a 3-D array using the same logic as the Builder.

        Args:
            individual_count: Per-sex, per-genotype initial counts.
                Nested as ``{sex: {genotype_selector: count}}``.
            sperm_storage: Per-sex, per-genotype initial stored sperm,
                same nesting structure as *individual_count*.

        Returns:
            Self for chaining.
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
        from natal.population_config import build_custom_array

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
        """Apply genetic presets directly to config arrays.

        When wired to a Population (via ``for_population()``), presets are
        applied directly to the Population — no adapter, no write-back needed.
        Otherwise the ``_ConfigContext`` adapter path is used for build-time.
        """
        if self._pop_ref is not None:
            # Collect presets, then apply in priority order.
            for preset in presets:
                self._pop_ref.add_preset(preset)
            self._pop_ref.rebuild_from_presets()
            self._config = self._pop_ref.config
            return self

        from natal.genetic_presets import apply_preset_to_population

        ctx = self._make_ctx()
        for preset in presets:
            apply_preset_to_population(ctx, preset)  # pyright: ignore[reportArgumentType]
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
            pop.rebuild_from_presets()
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
        changes.  If a custom ``equilibrium_distribution`` was stored (via
        ``competition(equilibrium_distribution=...)``), it is used as the
        target age structure for the Champer model.  If the user explicitly
        set ``expected_num_adult_females``, external egg counts are computed
        from that value instead of the distribution.
        """
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

        # Compute external_expected_eggs from expected_num_adult_females
        # Only when the user explicitly set it (avoid default config value)
        external_eggs: float | None = None
        if getattr(self, "_has_user_expected_females", False):
            from natal.population_builder import PopulationConfigBuilder

            external_eggs = PopulationConfigBuilder.compute_expected_eggs_from_females(
                expected_num_adult_females=getattr(self, "_user_expected_adult_females", 500.0),
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

    def build(
        self,
        name: str | None = None,
        hooks: _HookMap | None = None,
    ) -> DiscreteGenerationPopulation | AgeStructuredPopulation:
        """Finalise the config and create a Population.

        This is the terminal method of the build chain::

            Configurator.from_species()
                .age_structure(5, 2)
                .competition(K=5000)
                .reproduction(eggs=100)
                .build(name="pop")

        Internally it: (1) syncs equilibrium metrics via :meth:`apply`,
        (2) merges hooks registered via :meth:`hooks` with the *hooks*
        argument, and (3) passes ``self._config`` to the Population
        constructor.  After this point the Configurator no longer owns
        the config — the Population does.

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

            pop: DiscreteGenerationPopulation | AgeStructuredPopulation = \
                DiscreteGenerationPopulation(
                    species=self._species,
                    population_config=self._config,
                    name=name,
                    hooks=hooks,
                )
        else:
            from natal.age_structured_population import (
                AgeStructuredPopulation,
            )

            pop = AgeStructuredPopulation(
                species=self._species,
                population_config=self._config,
                name=name,
                hooks=hooks,
            )

        # Apply observation groups if registered.
        obs_groups = getattr(self, "_observation_groups", None)
        if obs_groups is not None:
            collapse = getattr(self, "_observation_collapse_age", False)
            pop.set_observations(obs_groups, collapse_age=collapse)
        return pop


# ── Typed Configurator subclasses ────────────────────────────────────────────
# Each subclass overrides build() with the specific Population return type,
# so the IDE / type checker knows exactly what you get after the chain.



# ── Model-specific Configurators ────────────────────────────────────────────
# Each subclass overrides every chain method for two reasons:
#   1. Narrow the return type (so .competition() → DiscreteConfigurator,
#      not Configurator).
#   2. Filter the parameter list to only those relevant to the model
#      (e.g. discrete doesn't expose competition_strength).
# The actual logic lives in the three Configurator._*_impl methods;
# subclass methods are thin wrappers: filter args → call _impl → return self.


class DiscreteConfigurator(Configurator):
    """Configurator for ``DiscreteGenerationPopulation`` (discrete-generation model).

    Two age classes (age-0 juveniles, age-1 adults).  Non-overlapping
    generations — adults are replaced each tick.

    Create via ``Configurator.for_discrete(species)`` or
    ``DiscreteGenerationPopulation.setup(species)``.
    """

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
            carrying_capacity: Equilibrium total adults at age 1 (K).
            low_density_growth_rate: Per-capita growth at low density (r).
            juvenile_growth_mode: ``"concave"``, ``"logistic"``, … or int.
            age_1_carrying_capacity: Legacy alias for *carrying_capacity*.

        Returns:
            Self for chaining.

        Note:
            When *carrying_capacity* (K) is set and the user has not
            explicitly called ``expected_num_adult_females=``,
            ``expected_num_adult_females`` is auto-computed as
            ``K * sex_ratio`` for the discrete model.
        """
        self._has_domain_params = True
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
            mode_value = _MODE_MAP.get(juvenile_growth_mode.lower())
            if mode_value is None:
                raise ValueError(
                    f"Unknown growth mode string: {juvenile_growth_mode!r}. "
                    f"Expected one of: {', '.join(sorted(_MODE_MAP))}."
                )
        elif juvenile_growth_mode is not None:
            mode_value = juvenile_growth_mode
        k_value = carrying_capacity
        if k_value is None and age_1_carrying_capacity is not None:
            k_value = age_1_carrying_capacity
        # K auto-detection: only during initial build.
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
        for name, value in [
            ("carrying_capacity", k_value),
            ("low_density_growth_rate", low_density_growth_rate),
            ("juvenile_growth_mode", mode_value),
        ]:
            if value is not None:
                set_param(self._config, f"competition.{name}", value)
        # Auto-compute expected_num_adult_females = K × sex_ratio
        if not getattr(self, "_has_user_expected_females", False):
            k = float(self._config.carrying_capacity)
            sr = float(self._config.sex_ratio)
            self._user_expected_adult_females = k * sr
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

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        # 0-d ndarray fields — write in-place, no staleness risk.
        if eggs_per_female is not None:
            set_param(self._config, "reproduction.eggs_per_female", eggs_per_female,
                      _sync_equilibrium=False)
        if sex_ratio is not None:
            set_param(self._config, "reproduction.sex_ratio", sex_ratio,
                      _sync_equilibrium=False)
        if eggs_per_female is not None or sex_ratio is not None:
            from natal.engine.simulation.age_structured import sync_equilibrium_metrics
            sync_equilibrium_metrics(self._config)
        # Scalars — write to config immediately (runtime) and store for build().
        scalar_overrides: dict[str, float] = {}
        if female_adult_mating_rate is not None:
            val = float(female_adult_mating_rate)
            self._female_mating_rate = val
            scalar_overrides["female_mating_rate"] = val
        if male_adult_mating_rate is not None:
            val = float(male_adult_mating_rate)
            self._male_mating_rate = val
            scalar_overrides["male_mating_rate"] = val
        if scalar_overrides:
            self._config = self._config._replace(**scalar_overrides)
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
        # Boolean flag — must use _replace (not a 0-d ndarray).
        if use_fixed_egg_count is not None:
            self._config = self._config._replace(use_fixed_egg_count=use_fixed_egg_count)
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
        return self

    def initial_state(
        self,
        individual_count: dict[str, dict[str, float | list[int] | dict[int, int]]],
        sperm_storage: dict[str, dict[str, float | list[int] | dict[int, int]]] | None = None,
    ) -> DiscreteConfigurator:
        """Set the initial population for a discrete-generation model.

        Uses the discrete resolution (flat JSON-style dict) rather than the
        age-structured resolution.

        Args:
            individual_count: Per-sex, per-genotype initial counts.
                Nested as ``{sex: {genotype_selector: count}}``.
            sperm_storage: Ignored for discrete-generation models
                (sperm storage is not applicable).

        Returns:
            Self for chaining.
        """
        if not self._species:
            raise RuntimeError(
                "initial_state() requires a Species. "
                "Use Configurator.for_discrete() or setup(species, ...)."
            )
        from natal.population_builder import PopulationConfigBuilder

        array = PopulationConfigBuilder.resolve_discrete_initial_individual_count(
            species=self._species,
            distribution=individual_count,
        )
        self._config = self._config._replace(
            initial_individual_count=array,
        )
        # Discrete models don't use sperm storage.
        if sperm_storage is not None:
            import warnings
            warnings.warn(
                "sperm_storage is ignored for discrete-generation populations.",
                UserWarning, stacklevel=2,
            )
        return self

    def survival(
        self,
        *,
        female_age0_survival: float | None = None,
        male_age0_survival: float | None = None,
    ) -> DiscreteConfigurator:
        """Configure survival.  Only age-0 (juvenile→adult) matters.

        Both default to 1.0.  ``adult_survival`` is NOT accepted — in
        discrete-generation models, adults are fully replaced each tick,
        so adult survival is always 0.0 and cannot be overridden.

        Args:
            female_age0_survival: Female juvenile survival probability.
            male_age0_survival: Male juvenile survival probability.

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        overrides: dict[str, float] = {}
        if female_age0_survival is not None:
            val = float(female_age0_survival)
            self._base_survival_f = val
            overrides["base_survival_f"] = val
        if male_age0_survival is not None:
            val = float(male_age0_survival)
            self._base_survival_m = val
            overrides["base_survival_m"] = val
        if overrides:
            self._config = self._config._replace(**overrides)
            if self._pop_ref is not None:
                self._pop_ref.set_config(self._config)
        return self

    def build(
        self, name: str | None = None, hooks: _HookMap | None = None,
    ) -> DiscreteGenerationPopulation:
        """Build and return a ``DiscreteGenerationPopulation``.

        Extracts discrete-specific scalars from stored override values
        before handing off to the base ``build()`` for finalisation.
        """
        # Extract scalars that replace the default values burned into
        # DiscretePopulationConfig at construction time.  This is the
        # only correct place — survival() and reproduction() store
        # overrides here, and the engine reads these scalars directly.
        overrides: dict[str, float] = {}
        if self._female_mating_rate is not None:
            overrides["female_mating_rate"] = self._female_mating_rate
        if self._male_mating_rate is not None:
            overrides["male_mating_rate"] = self._male_mating_rate
        if self._base_survival_f is not None:
            overrides["base_survival_f"] = self._base_survival_f
        if self._base_survival_m is not None:
            overrides["base_survival_m"] = self._base_survival_m
        if overrides:
            self._config = self._config._replace(**overrides)

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

    Create via ``Configurator.from_species()`` or
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

        Returns:
            Self for chaining.

        Note:
            Must be called before any domain method (competition,
            reproduction, survival, etc.).  Calling it after domain
            methods will raise ``RuntimeError``.
        """
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
            carrying_capacity: Equilibrium population at age 1 (K).
            low_density_growth_rate: Per-capita growth at low density (r).
            juvenile_growth_mode: Regulation function (string or int).
            competition_strength: Larval competition weight.
            expected_num_adult_females: Target adult females (Champer model).
            equilibrium_distribution: Custom (n_sexes, n_ages) array for
                Champer equilibrium computation.
            age_1_carrying_capacity: Legacy alias for *carrying_capacity*.
            old_juvenile_carrying_capacity: Legacy alias.

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
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
            mode_value = _MODE_MAP.get(juvenile_growth_mode.lower())
            if mode_value is None:
                raise ValueError(
                    f"Unknown growth mode string: {juvenile_growth_mode!r}. "
                    f"Expected one of: {', '.join(sorted(_MODE_MAP))}."
                )
        elif juvenile_growth_mode is not None:
            mode_value = juvenile_growth_mode
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
        for name, value in [
            ("carrying_capacity", k_value),
            ("low_density_growth_rate", low_density_growth_rate),
            ("juvenile_growth_mode", mode_value),
            ("competition_strength", competition_strength),
        ]:
            if value is not None:
                set_param(self._config, f"competition.{name}", value)
        if expected_num_adult_females is not None:
            self._user_expected_adult_females = float(expected_num_adult_females)
            self._has_user_expected_females = True
        if equilibrium_distribution is not None:
            self._equilibrium_distribution = equilibrium_distribution
        return self

    def reproduction(
        self,
        *,
        eggs_per_female: float | None = None,
        sex_ratio: float | None = None,
        sperm_displacement_rate: float | None = None,
        female_age_based_mating_rates: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        male_age_based_mating_rates: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        female_age_based_reproduction_rates: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
        female_age_based_relative_fertility: float | list[float] | dict[int, float] | Callable[[int], float] | None = None,
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

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        from natal.population_builder import PopulationConfigBuilder

        if use_sperm_storage is not None:
            import warnings
            warnings.warn(
                "use_sperm_storage parameter has never been functional — "
                "sperm storage is always enabled regardless of this setting. "
                "The parameter is accepted for compatibility but has no effect.",
                FutureWarning, stacklevel=2,
            )
        n_ages = self._config.n_ages
        resolve = PopulationConfigBuilder.resolve_age_param
        for name, value in [
            ("eggs_per_female", eggs_per_female),
            ("sex_ratio", sex_ratio),
            ("sperm_displacement_rate", sperm_displacement_rate),
        ]:
            if value is not None:
                set_param(self._config, f"reproduction.{name}", value,
                          _sync_equilibrium=False)
        if eggs_per_female is not None or sex_ratio is not None:
            from natal.engine.simulation.age_structured import sync_equilibrium_metrics
            sync_equilibrium_metrics(self._config)
        if female_age_based_mating_rates is not None:
            self._config.age_based_mating_rates[0, :] = resolve(
                female_age_based_mating_rates, n_ages, np.zeros(n_ages))
        if male_age_based_mating_rates is not None:
            self._config.age_based_mating_rates[1, :] = resolve(
                male_age_based_mating_rates, n_ages, np.zeros(n_ages))
        if female_age_based_reproduction_rates is not None:
            self._config.age_based_reproduction_rates[:] = resolve(
                female_age_based_reproduction_rates, n_ages, np.ones(n_ages))
        if female_age_based_relative_fertility is not None:
            self._config.female_age_based_relative_fertility[:] = resolve(
                female_age_based_relative_fertility, n_ages, np.ones(n_ages))
        if use_fixed_egg_count is not None:
            self._config = self._config._replace(use_fixed_egg_count=use_fixed_egg_count)
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

        Returns:
            Self for chaining.
        """
        self._has_domain_params = True
        from natal.population_builder import PopulationConfigBuilder

        n_ages = self._config.n_ages
        if female is not None:
            self._config.age_based_survival_rates[0, :] = (
                PopulationConfigBuilder.resolve_age_param(
                    female, n_ages, np.ones(n_ages)))
        if male is not None:
            self._config.age_based_survival_rates[1, :] = (
                PopulationConfigBuilder.resolve_age_param(
                    male, n_ages, np.ones(n_ages)))
        if female_age_based_survival_rates is not None:
            self._config.age_based_survival_rates[0, :] = (
                PopulationConfigBuilder.resolve_age_param(
                    female_age_based_survival_rates, n_ages, np.ones(n_ages)))
        if male_age_based_survival_rates is not None:
            self._config.age_based_survival_rates[1, :] = (
                PopulationConfigBuilder.resolve_age_param(
                    male_age_based_survival_rates, n_ages, np.ones(n_ages)))
        for name, value in [
            ("female_age0_survival", female_age0_survival),
            ("male_age0_survival", male_age0_survival),
        ]:
            if value is not None:
                set_param(self._config, f"survival.{name}", value)
        if adult_survival is not None:
            new_adult_age = self._config.new_adult_age
            self._config.age_based_survival_rates[:, new_adult_age:] = float(adult_survival)
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
    """Look up a parameter name in ``ALL_PARAMETERS`` with three fallback tiers.

    Tier 1: exact match — ``"competition.carrying_capacity"``.
    Tier 2: short-name match — ``"carrying_capacity"`` matches via
            ``key.endswith(".carrying_capacity")``.
    Tier 3: alias match — user-friendly names defined in each
            ``ParamDescriptor.aliases``.

    Returns the ``ParamDescriptor`` or ``None``.
    """
    # Tier 1: O(1) exact key lookup.
    if name in ALL_PARAMETERS:
        return ALL_PARAMETERS[name]

    # Tier 2-3: linear scan for short-name / alias match.
    for key, desc in ALL_PARAMETERS.items():
        if key.endswith(f".{name}"):
            return desc
        if name in desc.aliases:
            return desc

    return None
