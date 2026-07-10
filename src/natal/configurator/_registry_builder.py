"""Registry builder, adapter, and config map rebuild helpers.

Internal helpers shared by the Configurator and the ``ConfigContext``
adapter.

Key components:
  - ``build_registry()`` — create an ``IndexRegistry`` pre-populated
    with all genotypes, haplotypes, and gamete/somatic labels from
    a ``Species``.
  - ``ConfigContext`` — adapter that mimics ``BasePopulation``'s
    attribute surface so that ``apply_preset_to_population()`` and
    modifier functions can operate on config arrays without a live
    Population object.
  - ``rebuild_config_maps()`` — apply gamete/zygote modifiers,
    run optional index compression, and recompute the offspring
    probability tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from natal.data import (
    DiscretePopulationConfig,
    PopulationConfig,
    compress_config,
)
from natal.genetics import Species, build_compression_mask
from natal.presets import CytoplasmicPreset
from natal.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.modifiers.module import GameteModifier, ZygoteModifier
    from natal.presets import GeneticPreset


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


class ConfigContext:
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
        # Compression masks (exposed for spatial post-processing).
        self.ztype_mask: NDArray[np.int32] | None = None
        self.gtype_mask: NDArray[np.int32] | None = None
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
        resolved_id = ConfigContext.next_modifier_id(self.gamete_modifiers) if modifier_id is None else modifier_id
        self.gamete_modifiers.append((resolved_id, name, modifier))
        self.gamete_modifiers.sort(key=lambda x: x[0])
        if refresh:
            rebuild_config_maps(self)

    def refresh_modifier_maps(self) -> None:
        """Rebuild config maps from the current modifier lists.

        Mirrors :meth:`BasePopulation.refresh_modifier_maps` for the
        adapter — required by :func:`apply_preset_to_population` which
        accepts both Population and ConfigContext objects.
        """
        rebuild_config_maps(self)

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
        resolved_id = ConfigContext.next_modifier_id(self.zygote_modifiers) if modifier_id is None else modifier_id
        self.zygote_modifiers.append((resolved_id, name, modifier))
        self.zygote_modifiers.sort(key=lambda x: x[0])
        if refresh:
            rebuild_config_maps(self)

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


def rebuild_config_maps(
    ctx: ConfigContext,
    *,
    override_z2g: NDArray[np.float64] | None = None,
    override_g2z: NDArray[np.float64] | None = None,
) -> None:
    """Apply gamete/zygote modifiers and rebuild ``offspring_tensor``.

    Starts from the species-level Mendelian baseline (cached via
    :meth:`Species.get_config_blueprint`) and applies modifier callables
    in-place, avoiding redundant O(n²) baseline recomputation.

    When *override_z2g* / *override_g2z* are provided, the modifier
    application step is skipped and the override maps are used directly.
    This is used by spatial compression to combine modifier maps from
    multiple demes into a unified BFS adjacency matrix.
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
    if override_z2g is not None and override_g2z is not None:
        zygotes_to_gametes_map = override_z2g.copy()
        gametes_to_zygotes_map = override_g2z.copy()
    else:
        # ---- compile modifier callables from the accumulated modifier lists ----
        gamete_funcs, zygote_funcs = build_modifier_wrappers(
            gamete_modifiers=ctx.gamete_modifiers,
            zygote_modifiers=ctx.zygote_modifiers,
            population=None,
            registry=ctx.registry,
        )

        # ---- fetch the Mendelian baseline from the species cache ----
        bp = ctx.species.get_config_blueprint()
        zygotes_to_gametes_map = bp["zygotes_to_gametes_map"].copy()
        gametes_to_zygotes_map = bp["gametes_to_zygotes_map"].copy()

        # ---- chain modifier callables on top of the baseline ----
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
                declared_zygote_types=declared_ints,
            )
        )
        gtype_mask = _gt_mask
        ztype_mask = _zt_mask
        ctx.gtype_mask = gtype_mask
        ctx.ztype_mask = ztype_mask

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
