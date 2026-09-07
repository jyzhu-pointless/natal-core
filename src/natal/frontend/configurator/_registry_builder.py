"""Registry builder and build-side config map compilation.

Internal helpers for the build-side candidate compile:

  - ``build_registry()`` — create an ``IndexRegistry`` pre-populated
    with all genotypes, haplotypes, and gamete/somatic labels from
    a ``Species``.
  - ``rebuild_config_maps()`` — apply gamete/zygote modifiers to the
    Mendelian baseline through the unified compiler, run optional index
    compression, and return the updated draft.  All inputs and outputs
    are explicit values: there is no adapter object impersonating a
    population.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from natal.frontend.data import (
    ModelDraft,
    compress_config,
)
from natal.frontend.genetics import Species, build_compression_mask
from natal.frontend.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.frontend.genetics.compile import GameteList, ZygoteList

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


# ── Core: rebuild genotype/gamete/zygote maps from modifier lists ─────────────


def rebuild_config_maps(
    species: Species,
    config: ModelDraft,
    registry: IndexRegistry,
    *,
    gamete_modifiers: GameteList,
    zygote_modifiers: ZygoteList,
    compress: bool = False,
    declared_zygote_types: set[str] | set[int] | None = None,
) -> tuple[ModelDraft, bool]:
    """Apply gamete/zygote modifiers and rebuild ``offspring_tensor``.

    Starts from the species-level Mendelian baseline (cached via
    :meth:`Species.get_config_blueprint`) and applies modifier callables
    through the unified compiler
    (:func:`natal.frontend.genetics.compile.compile_modifier_maps`) — the
    same spelling the population-side refresh uses.

    When *compress* is enabled, reachable-index compression runs on the
    result: unreachable gtypes/ztypes are pruned from the maps, the
    config is subsliced via ``compress_config``, and *registry* is
    compressed **in place** (the registry is shared state by design —
    the symbolic name directory must stay index-aligned with the maps).

    Args:
        species: The genetic architecture providing the Mendelian baseline.
        config: The candidate draft whose maps are rebuilt.
        registry: The registry whose active axes the maps address; it is
            compressed in place when *compress* prunes indices.
        gamete_modifiers: ``(id, name, callable)`` triples to chain over
            the meiosis table.
        zygote_modifiers: The zygote-side twin of *gamete_modifiers*.
        compress: Enable both GType and ZType index compression at once.
        declared_zygote_types: Genotypes the user declared (string
            selectors or raw indices) that must survive compression
            pruning even when unreachable from the initial state.

    Returns:
        ``(new_config, compression_applied)`` — the rebuilt draft and
        whether compression ran.  The input *config* is never mutated;
        *registry* is compressed in place when compression applies.
    """
    from natal.frontend.data._engine import recompute_offspring_tensor
    from natal.frontend.genetics.compile import compile_modifier_maps

    # ---- resolve genotype/haplotype lists from the registry ----
    haploid_genotypes = registry.index_to_haplo
    diploid_genotypes = registry.index_to_genotype
    if not haploid_genotypes or not diploid_genotypes:
        # species has no haploid genotypes (no sex chromosomes)
        return config, False

    n_glabs = int(config.n_glabs)

    # ---- the unified compiler: baseline from the species cache,
    # modifier recipes chained, offspring derived — the same spelling
    # the population-side refresh uses ----
    bp = species.get_config_blueprint()
    zygotes_to_gametes_map, gametes_to_zygotes_map, _derived = (
        compile_modifier_maps(
            bp["zygotes_to_gametes_map"],
            bp["gametes_to_zygotes_map"],
            gamete_modifiers=gamete_modifiers,
            zygote_modifiers=zygote_modifiers,
            registry=registry,
            population=None,
        )
    )

    # ---- index compression (optional) ----
    n_g_compressed = int(config.n_ztypes)
    n_hg_effective = int(config.n_gtypes) // n_glabs
    n_glabs_effective = n_glabs
    gtype_mask = np.array([], dtype=np.int32)
    ztype_mask = np.array([], dtype=np.int32)
    compression_applied = False

    if compress:
        compression_applied = True

        # Resolve declared_zygote_types to integer indices for the BFS.
        # Each declared genotype is expanded to all slab variants because
        # the BFS operates in the slab-expanded space (G = G_orig × n_slabs).
        declared_ints: set[int] | None = None
        if declared_zygote_types is not None:
            declared_ints = set()
            n_slabs = int(config.n_slabs)
            for dg in declared_zygote_types:
                if isinstance(dg, str):
                    try:
                        # Use ZygoteTypePattern to properly handle @slab
                        # suffixes (e.g. "Drive|Rescue_Cargo@S").
                        from natal.frontend.patterns.elements.diploid import (
                            ZygoteTypePattern,
                        )
                        pattern = ZygoteTypePattern.parse(dg, species)
                        matched = False
                        for gt in diploid_genotypes:
                            if pattern.genotype.matches(gt):
                                for s in range(n_slabs):
                                    declared_ints.add(
                                        registry.ztype_index(gt, registry.slab_labels[s])
                                    )
                                matched = True
                        if not matched:
                            # Fallback: strip @slab and try exact match
                            dg_clean = dg.split("@")[0]
                            gt = species.get_genotype_from_str(dg_clean)
                            if gt in diploid_genotypes:
                                for s in range(n_slabs):
                                    declared_ints.add(
                                        registry.ztype_index(gt, registry.slab_labels[s])
                                    )
                    except Exception:
                        pass
                else:
                    g_orig = int(dg)
                    for s in range(n_slabs):
                        declared_ints.add(
                            registry.ztype_index(
                                diploid_genotypes[g_orig],
                                registry.slab_labels[s],
                            )
                        )

        _gt_mask, _, _zt_mask, _ = (
            build_compression_mask(
                zygotes_to_gametes_map,
                gametes_to_zygotes_map,
                config.initial_individual_count,
                declared_zygote_types=declared_ints,
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
            return config, compression_applied

    # GType (gamete-axis) compression.
    n_hg_effective = int(config.n_gtypes) // n_glabs
    n_glabs_effective = n_glabs
    gtype_compressed = False
    _hl_active = gtype_mask >= 0
    if gtype_mask.size > 0:
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

        config = compress_config(config, ztype_mask)
        n_g_compressed = int(config.n_ztypes)
        registry.compress(ztype_mask, gtype_mask)

    # ---- recompute offspring probability tensor from the updated maps via
    # the single shared derivation (shape-derived counts, so both the
    # compressed and the full glab-product layouts resolve correctly) ----
    offspring_tensor = recompute_offspring_tensor(
        zygotes_to_gametes_map, gametes_to_zygotes_map
    )

    # ---- write everything back into a fresh draft via _replace ----
    overrides: dict[str, object] = {
        "zygotes_to_gametes_map": zygotes_to_gametes_map,
        "gametes_to_zygotes_map": gametes_to_zygotes_map,
        "offspring_tensor": offspring_tensor,
        "n_ztypes": n_g_compressed,
        "n_gtypes": n_hg_effective if gtype_compressed else n_hg_effective * n_glabs_effective,
    }
    if gtype_compressed:
        # The registry compressed gtypes with the same flat mask; slice the
        # name directory with it so indices stay aligned.
        overrides["gtype_names"] = tuple(
            name
            for name, m in zip(config.gtype_names, gtype_mask.tolist())
            if m >= 0
        )
    return config._replace(**overrides), compression_applied
