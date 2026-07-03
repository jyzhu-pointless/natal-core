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

from typing import TYPE_CHECKING, Any, Callable, Mapping, Self, Sequence, cast

import numpy as np
from numba import objmode  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from natal.data import (
    DiscretePopulationConfig,
    PopulationConfig,
    compress_config,
)
from natal.genetics import Species, build_compression_mask
from natal.numba.utils import njit_switch
from natal.presets import CytoplasmicPreset
from natal.registry.index import IndexRegistry
from natal.utils.parameters import ALL_PARAMETERS, ParamDescriptor

if TYPE_CHECKING:
    from natal.configurator.age_structured import AgeStructuredConfigurator
    from natal.configurator.discrete import DiscreteConfigurator
    from natal.genetics import Genotype
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


# ── Registry builder (shared by Configurator and adapter) ──────────────────────


def build_registry(species: Species) -> IndexRegistry:
    """Build an IndexRegistry pre-populated with all genotypes/haplotypes from a Species.

    Note:
        When *species* does not define ``gamete_labels`` (or they are empty),
        a single ``"default"`` label is registered so that modifier code
        always has at least one gamete-label slot to index into.

        Labels MUST be registered before genotypes/haplotypes so that
        the auto-cross-product creates ZType/GType entries for ALL slabs/glabs.
    """
    registry = IndexRegistry()

    # 1. Register labels FIRST — so auto-cross-product covers all of them.
    raw_glabs = getattr(species, "gamete_labels", None)
    glabs = raw_glabs or ["default"]
    for glab in glabs:
        registry.register_gamete_label(glab)

    raw_slabs = getattr(species, "somatic_labels", None)
    slabs = raw_slabs or ["default"]
    for slab in slabs:
        registry.register_somatic_label(slab)

    # 2. Register genotypes — auto-cross-products with ALL slab_labels above.
    for genotype in species.get_all_genotypes(unordered=species.unordered):
        registry.register_genotype(genotype)

    # 3. Register haplotypes — auto-cross-products with ALL glab_labels above.
    haploid_genotypes = species.get_all_haploid_genotypes()
    if haploid_genotypes:
        for hg in haploid_genotypes:
            registry.register_haplogenotype(hg)

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
        compress: bool = False,
    ) -> None:
        """Initialise the adapter with species, config, and registry.

        Args:
            species: The genetic architecture for the population.
            config: The PopulationConfig or DiscretePopulationConfig to wrap.
            registry: An IndexRegistry pre-populated with genotypes/haplotypes.
            compress: Enable both GType and ZType index compression at once.
        """
        self.species = species
        self.config = config
        self.registry = registry
        self.index_registry = registry
        self.compress = compress
        self.compression_applied: bool = False
        self.declared_zygote_types: set[str] | set[int] | None = None
        self.gamete_modifiers: list[tuple[int, str | None, GameteModifier]] = []
        self.zygote_modifiers: list[tuple[int, str | None, ZygoteModifier]] = []
        self.presets: list[GeneticPreset] = []  # for cytoplasmic preset post-processing

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
            _rebuild_config_maps(self)

    def refresh_modifier_maps(self) -> None:
        """Rebuild config maps from the current modifier lists.

        Mirrors :meth:`BasePopulation.refresh_modifier_maps` for the
        adapter — required by :func:`apply_preset_to_population` which
        accepts both Population and _ConfigContext objects.
        """
        _rebuild_config_maps(self)

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
    from natal.modifiers.module import build_modifier_wrappers

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
        expand_to_ztypes=int(ctx.config.n_slabs) > 1,
    )

    # ---- fetch the Mendelian baseline from the species cache ----
    # Because get_config_blueprint() is cached per-species, repeatedly
    # calling it is cheap.  We copy the arrays so that modifier callables
    # can mutate them without corrupting the cached originals.
    bp = ctx.species.get_config_blueprint()
    zygotes_to_gametes_map = bp["zygotes_to_gametes_map"].copy()
    gametes_to_zygotes_map = bp["gametes_to_zygotes_map"].copy()

    # ---- chain modifier callables on top of the baseline ----
    # Each callable accepts and returns a tensor of the same shape,
    # allowing modifiers to be composed in registration order.
    # Blueprint maps are pre-expanded (G×S), so modifier wrappers
    # expand genotype indices to ZType indices internally.
    for fn in gamete_funcs:
        zygotes_to_gametes_map = fn(zygotes_to_gametes_map)
    for fn in zygote_funcs:
        gametes_to_zygotes_map = fn(gametes_to_zygotes_map)

    # ---- index compression (optional) ----
    n_g_compressed = int(ctx.config.n_ztypes)
    n_hg_effective = int(ctx.config.n_gtypes) // n_glabs
    n_glabs_effective = n_glabs
    gtype_mask = np.array([], dtype=np.int32)
    ztype_mask = np.array([], dtype=np.int32)

    if ctx.compress:
        ctx.compression_applied = True

        # Resolve declared_zygote_types to integer indices for the BFS.
        # Each declared genotype is expanded to all slab variants because
        # the BFS operates in the slab-expanded space (G = G_orig × n_slabs).
        declared_ints: set[int] | None = None
        if ctx.declared_zygote_types is not None:
            declared_ints = set()
            n_slabs = int(ctx.config.n_slabs)
            for dg in ctx.declared_zygote_types:
                if isinstance(dg, str):
                    try:
                        gt = ctx.species.get_genotype_from_str(dg)
                        if gt in diploid_genotypes:
                            for s in range(n_slabs):
                                declared_ints.add(
                                    ctx.registry.ztype_index(gt, ctx.registry.slab_labels[s])
                                )
                    except Exception:
                        pass
                else:
                    g_orig = int(dg)
                    for s in range(n_slabs):
                        declared_ints.add(
                            ctx.registry.ztype_index(
                                diploid_genotypes[g_orig],
                                ctx.registry.slab_labels[s],
                            )
                        )

        _gt_mask, _, _zt_mask, _ = (
            build_compression_mask(
                zygotes_to_gametes_map,
                gametes_to_zygotes_map,
                ctx.config.initial_individual_count,
                n_glabs=n_glabs,
                n_slabs=1,  # maps are pre-expanded; mask lives in G_orig space
                declared_genotypes=declared_ints,
            )
        )
        gtype_mask = _gt_mask
        ztype_mask = _zt_mask

        # Guard: if no genotypes or gametes are reachable, skip compression
        # entirely (initial state is empty and no declared_genotypes given).
        # Without this guard the compression code produces zero-length arrays
        # that crash downstream code.
        has_reachable = (gtype_mask >= 0).any() or (ztype_mask >= 0).any()
        if not has_reachable:
            return

    # GType (gamete-axis) compression.
    n_hg_effective = int(ctx.config.n_gtypes) // n_glabs
    n_glabs_effective = n_glabs
    gtype_compressed = False
    if gtype_mask.size > 0:
        _hl_active = gtype_mask >= 0
        n_hl_compressed = int(_hl_active.sum())
        if n_hl_compressed < zygotes_to_gametes_map.shape[2]:
            zygotes_to_gametes_map = zygotes_to_gametes_map[:, :, _hl_active]
            gametes_to_zygotes_map = gametes_to_zygotes_map[_hl_active, :, :][:, _hl_active, :]
            n_hg_effective = n_hl_compressed
            gtype_compressed = True

    # ZType (genotype-axis) compression.
    if ztype_mask.size > 0:
        _z_active = ztype_mask >= 0
        zygotes_to_gametes_map = zygotes_to_gametes_map[:, _z_active, :]
        gametes_to_zygotes_map = gametes_to_zygotes_map[:, :, _z_active]

        ctx.config = compress_config(ctx.config, ztype_mask)
        n_g_compressed = int(ctx.config.n_ztypes)
        ctx.registry.compress(ztype_mask, gtype_mask)

    # ---- apply cytoplasmic preset effects (pre-tensor) ----
    for preset in ctx.presets:
        if isinstance(preset, CytoplasmicPreset):
            n_genotypes = len(ctx.registry.index_to_genotype)
            n_gtypes = len(ctx.registry.index_to_haplo)
            n_glabs = int(ctx.config.n_glabs)
            n_slabs = int(ctx.config.n_slabs)
            CytoplasmicPreset.tag_maternal_gametes(
                zygotes_to_gametes_map, ctx.species.gamete_labels,
                ctx.species.somatic_labels,
                n_genotypes, n_gtypes, n_glabs, n_slabs,
            )
            CytoplasmicPreset.redirect_zygotes(
                gametes_to_zygotes_map, ctx.species.gamete_labels,
                ctx.species.somatic_labels,
                n_genotypes, n_gtypes, n_glabs, n_slabs,
            )

    # ---- recompute offspring probability tensor from the updated maps ----
    offspring_tensor = compute_offspring_probability_tensor(
        meiosis_f=zygotes_to_gametes_map[0],
        meiosis_m=zygotes_to_gametes_map[1],
        haplo_to_genotype_map=gametes_to_zygotes_map,
        n_ztypes=n_g_compressed,
        n_gtypes=n_hg_effective if gtype_compressed else n_hg_effective * n_glabs_effective,
    )

    # ---- write everything back into the config via _replace ----
    overrides: dict[str, Any] = {
        "zygotes_to_gametes_map": zygotes_to_gametes_map,
        "gametes_to_zygotes_map": gametes_to_zygotes_map,
        "offspring_tensor": offspring_tensor,
        "n_ztypes": n_g_compressed,
        "n_gtypes": n_hg_effective if gtype_compressed else n_hg_effective * n_glabs_effective,
    }
    if isinstance(ctx.config, DiscretePopulationConfig):
        # Keep the pre-extracted slices in sync with the source maps.
        overrides["meiosis_f"] = zygotes_to_gametes_map[0]
        overrides["meiosis_m"] = zygotes_to_gametes_map[1]
        overrides["fecundity_f"] = ctx.config.fecundity_fitness[0]
        overrides["fecundity_m"] = ctx.config.fecundity_fitness[1]
        overrides["viability_f"] = ctx.config.viability_fitness[0, 0, :]
        overrides["viability_m"] = ctx.config.viability_fitness[1, 0, :]
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
    desc = _resolve_param(name)  # noqa: F821
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
        sex_patch: Mapping[str, Mapping[str | tuple[Genotype | str, str], float]]
        sex_patch = patch  # type: ignore[assignment]  # Mapping key invariance; narrowed by branch guard
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
                        for f_z in registry.ztype_indices_for(f_geno):
                            for m_geno in matched_m:
                                for m_z in registry.ztype_indices_for(m_geno):
                                    val = float(value)
                                    if mode == "replace":
                                        arr[f_z, m_z] = val
                                    else:
                                        arr[f_z, m_z] *= val
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
                for m_z in registry.ztype_indices_for(m_geno):
                    val = float(value)
                    if mode == "replace":
                        arr[:, m_z] = val        # broadcast: all females × this male
                    else:
                        arr[:, m_z] *= val
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
    patch: Mapping[str | tuple[Genotype | str, str], float],
    mode: str,
    *,
    sex_idx: int,
    species: Species,
    registry: IndexRegistry,
    all_genotypes: list[Genotype],
    age_idx: int | None = None,
) -> None:
    """Write a flat (per-ZType) fitness patch into the correct config array.

    The target array shape depends on *field_name*:

    - ``"viability"`` → ``(n_sexes, n_ages, n_ztypes)`` — writes ``[sex_idx, default_age, zidx]``
    - ``"fecundity"`` → ``(n_sexes, n_ztypes)`` — no age axis
    - ``"sexual_selection"`` → ``(n_ztypes, n_ztypes)`` — no age axis
    - ``"zygote_viability"`` → ``(n_sexes, n_ztypes)`` — no age axis

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

    # Resolve valid slab labels for tuple selectors with @slab suffix.
    raw_slabs: list[str] = getattr(species, "somatic_labels", None) or ["default"]

    for selector, value in patch.items():
        # ── tuple syntax: (Genotype, "slab_label") ──
        if isinstance(selector, tuple):
            if len(selector) != 2:
                raise TypeError(
                    f"fitness tuple selector must have 2 elements "
                    f"(genotype_key, slab_label), got {len(selector)}"
                )
            _genotype_key, _slab = selector
            if _slab not in raw_slabs:
                raise ValueError(
                    f"Unknown slab label '{_slab}'. "
                    f"Available slabs: {raw_slabs}"
                )

            if isinstance(_genotype_key, Genotype):
                matched = [_genotype_key]
            else:
                matched = species.resolve_genotype_selectors(
                    selector=_genotype_key,
                    all_genotypes=all_genotypes,
                    context=f"fitness.{field_name}",
                )

            for genotype in matched:
                age_slice = slice(resolved_age, resolved_age + 1)
                zidx = registry.ztype_index(genotype, _slab)

                if field_name == "viability":
                    arr = config.viability_fitness
                    if mode == "replace":
                        arr[sex_idx, age_slice, zidx] = float(value)
                    else:
                        arr[sex_idx, age_slice, zidx] *= float(value)
                elif field_name == "fecundity":
                    arr = config.fecundity_fitness
                    if mode == "replace":
                        arr[sex_idx, zidx] = float(value)
                    else:
                        arr[sex_idx, zidx] *= float(value)
                elif field_name == "sexual_selection":
                    arr = config.sexual_selection_fitness
                    if mode == "replace":
                        if sex_idx == 0:
                            arr[zidx, :] = float(value)
                        else:
                            arr[:, zidx] = float(value)
                    else:
                        if sex_idx == 0:
                            arr[zidx, :] *= float(value)
                        else:
                            arr[:, zidx] *= float(value)
                elif field_name == "zygote_viability":
                    arr = config.zygote_viability_fitness
                    if mode == "replace":
                        arr[sex_idx, zidx] = float(value)
                    else:
                        arr[sex_idx, zidx] *= float(value)
            continue
        # ── end tuple branch ──

        from natal.patterns import LabPattern, ZygoteTypePattern

        selector_str = str(selector)
        pattern = ZygoteTypePattern.parse(selector_str, species)
        z_indices = registry.resolve_ztype_indices(pattern)

        # For | patterns (not ::), also try :: for unordered matching.
        # Ordered | may only partially match (e.g. *|A → AA but not Aa).
        # Only promote for unordered species (consistent with
        # genetic_structures.Species._resolve_single_genotype_selector).
        if species.unordered and "|" in selector_str and "::" not in selector_str:
            try:
                unordered_str = selector_str.replace("|", "::", 1)
                unordered_pattern = ZygoteTypePattern.parse(unordered_str, species)
                unordered_indices = registry.resolve_ztype_indices(unordered_pattern)
                if len(unordered_indices) >= len(z_indices):
                    z_indices = unordered_indices
            except Exception:
                pass

        if not z_indices:
            # Check for invalid slab first — give a specific error
            if "@" in selector_str:
                _, s_str = selector_str.rsplit("@", 1)
                lab = LabPattern.parse(s_str)
                matching_slabs = [s for s in raw_slabs if lab.matches(s)]
                if not matching_slabs:
                    raise ValueError(
                        f"No slab matches '{s_str}' in fitness.{field_name} "
                        f"selector '{selector_str}'.  Available: {raw_slabs}"
                    )

        if not z_indices:
            raise ValueError(
                f"No zygote type matches '{selector_str}' in fitness.{field_name}"
            )

        age_slice = slice(resolved_age, resolved_age + 1)

        for zidx in z_indices:
            if field_name == "viability":
                arr = config.viability_fitness
                if mode == "replace":
                    arr[sex_idx, age_slice, zidx] = float(value)
                else:
                    arr[sex_idx, age_slice, zidx] *= float(value)
            elif field_name == "fecundity":
                arr = config.fecundity_fitness
                if mode == "replace":
                    arr[sex_idx, zidx] = float(value)
                else:
                    arr[sex_idx, zidx] *= float(value)
            elif field_name == "sexual_selection":
                arr = config.sexual_selection_fitness
                if mode == "replace":
                    if sex_idx == 0:
                        arr[zidx, :] = float(value)
                    else:
                        arr[:, zidx] = float(value)
                else:
                    if sex_idx == 0:
                        arr[zidx, :] *= float(value)
                    else:
                        arr[:, zidx] *= float(value)
            elif field_name == "zygote_viability":
                arr = config.zygote_viability_fitness
                if mode == "replace":
                    arr[sex_idx, zidx] = float(value)
                else:
                    arr[sex_idx, zidx] *= float(value)


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
        # Applied during _rebuild_config_maps (build-time) or
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

    def _make_ctx(self) -> _ConfigContext:
        """Build a :class:`_ConfigContext` seeded from the current state.

        The context receives a shallow copy of the modifier lists so that
        preset / modifier calls can append without mutating the originals
        until :meth:`_sync_from_ctx` explicitly commits them back.

        Flushes both deferred fitness and initial_state before creating
        the context.  The BFS in ``_rebuild_config_maps`` needs the
        initial state as its seed — empty seeds cause over-pruning.
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

        ctx = _ConfigContext(
            self._species, self._config, self._registry,
            compress=False,  # compression is only enabled in build()
        )
        ctx.declared_zygote_types = self._declared_zygote_types
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
        if ctx.compression_applied:
            self._compression_applied = True

    # -- factory ---------------------------------------------------------------

    @classmethod
    def from_species(
        cls,
        species: Species,
        *,
        discrete: bool = False,
    ) -> DiscreteConfigurator | AgeStructuredConfigurator:  # type: ignore[name-defined]  # noqa: F821
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
    def for_discrete(cls, species: Species) -> DiscreteConfigurator:  # type: ignore[name-defined]  # noqa: F821
        """Shorthand for ``from_species(species, discrete=True)``."""
        from natal.configurator.discrete import DiscreteConfigurator as _DC
        return cast(_DC, cls.from_species(species, discrete=True))

    @classmethod
    def for_age_structured(cls, species: Species) -> AgeStructuredConfigurator:  # type: ignore[name-defined]  # noqa: F821
        """Shorthand for ``from_species(species)``."""
        from natal.configurator.age_structured import AgeStructuredConfigurator as _ASC
        return cast(_ASC, cls.from_species(species))

    @staticmethod
    def for_config(
        config: PopulationConfig | DiscretePopulationConfig,
    ) -> DiscreteConfigurator | AgeStructuredConfigurator:  # type: ignore[name-defined]  # noqa: F821
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
    def for_population(pop: BasePopulation[Any]) -> DiscreteConfigurator | AgeStructuredConfigurator:  # type: ignore[name-defined]  # noqa: F821
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
        """Apply genetic presets directly to config arrays.

        When wired to a Population (via ``for_population()``), presets are
        applied directly to the Population — no adapter, no write-back needed.
        Otherwise the ``_ConfigContext`` adapter path is used for build-time.
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
            _rebuild_config_maps(ctx)
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
                _write_fitness_field(
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
        changes.  If a custom ``equilibrium_distribution`` was stored (via
        ``competition(equilibrium_distribution=...)``), it is used as the
        target age structure for the Champer model.  If the user explicitly
        set ``expected_num_adult_females``, external egg counts are computed
        from that value instead of the distribution.
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

        # Compute external_expected_eggs from expected_num_adult_females
        # Only when the user explicitly set it (avoid default config value)
        external_eggs: float | None = None
        if getattr(self, "_has_user_expected_females", False):
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
                expected_num_adult_females=getattr(self, "_user_expected_adult_females", 500.0),
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
            _rebuild_config_maps(ctx)
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


# -- parameter name resolution -----------------------------------------------


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

