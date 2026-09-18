"""Build complete registries and compile complete genetic maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from natal.frontend.genetics import Species
from natal.frontend.model import ModelDraft
from natal.frontend.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.frontend.genetics.compile import GameteList, RecipeHost, ZygoteList


def build_registry(species: Species) -> IndexRegistry:
    """Build a private, complete registry in species product order.

    Args:
        species: Source genetic architecture and label catalogs.

    Returns:
        An unpublished registry containing every ZType and GType.
    """
    registry = IndexRegistry()
    for label in (getattr(species, "gamete_labels", None) or ["default"]):
        registry.register_gamete_label(label)
    for label in (getattr(species, "somatic_labels", None) or ["default"]):
        registry.register_somatic_label(label)
    for genotype in species.get_all_genotypes(unordered=species.unordered):
        registry.register_genotype(genotype)
    for haplotype in species.get_all_haploid_genotypes():
        registry.register_haplogenotype(haplotype)
    return registry


def resolve_declared_ztypes(
    species: Species,
    registry: IndexRegistry,
    declared: set[str] | set[int] | None,
) -> set[int]:
    """Resolve user genotype declarations once onto the complete ZType axis.

    Args:
        species: Source architecture for genotype patterns and canonical order.
        registry: Complete unpublished catalog.
        declared: User genotype patterns or genotype indices. Integer inputs
            are genotype indices, never already expanded ZType indices.

    Returns:
        Complete ZType indices, expanding each matched genotype across slabs.

    Raises:
        RuntimeError: If the registry is published.
        ValueError: If a pattern cannot be parsed or an index is out of range.
    """
    registry.require_unpublished()
    if not declared:
        return set()
    result: set[int] = set()
    from natal.frontend.patterns import GenotypePatternParser
    from natal.frontend.patterns.elements.diploid import ZygoteTypePattern
    for item in declared:
        if isinstance(item, int):
            if item < 0 or item >= len(registry.index_to_genotype):
                raise ValueError(f"Declared genotype index {item} is out of range")
            genotype = registry.index_to_genotype[item]
            result.update(registry.ztype_index(genotype, slab) for slab in registry.slab_labels)
        else:
            pattern = ZygoteTypePattern.parse(item, species)
            matches = [gt for gt in registry.index_to_genotype if pattern.genotype.matches(gt)]
            if not matches and "*" not in item:
                # Exact unordered genotypes may be written in either parental
                # order; the Species parser supplies their canonical identity.
                # ZygoteTypePattern.parse already validated the suffix, so the
                # grammar's own split names the genotype part unambiguously.
                genotype_text, _ = GenotypePatternParser.split_label_suffix(item)
                genotype = species.get_genotype_from_str(genotype_text)
                matches = [genotype]
            for genotype in matches:
                result.update(registry.ztype_index(genotype, slab) for slab in registry.slab_labels)
    return result


def rebuild_config_maps(
    species: Species,
    config: ModelDraft,
    registry: IndexRegistry,
    *,
    gamete_modifiers: GameteList,
    zygote_modifiers: ZygoteList,
    host: RecipeHost | None = None,
) -> ModelDraft:
    """Compile complete baseline maps, leaving offspring derivation to publication.

    Args:
        species: Architecture supplying the Mendelian baseline.
        config: Complete-axis draft; input arrays are not modified.
        registry: Complete unpublished catalog in species order.
        gamete_modifiers: Ordered gamete modifier declarations.
        zygote_modifiers: Ordered zygote modifier declarations.
        host: Isolated compilation context for user recipes.

    Returns:
        Updated complete maps with an empty offspring placeholder.

    Raises:
        ValueError: If the catalog is published or does not match the
            complete species registry (the check lives in
            :func:`natal.frontend.genetics.compile.project_mendelian_maps`,
            which this function delegates to), or a modifier declaration
            is invalid.  An empty catalog is not an error: the function
            returns *config* unchanged.
    """
    from natal.frontend.genetics.compile import (
        compile_modifier_maps,
        project_mendelian_maps,
    )
    if not registry.index_to_haplo or not registry.index_to_genotype:
        return config
    baseline_z2g, baseline_g2z = project_mendelian_maps(species, registry)
    z2g, g2z = compile_modifier_maps(
        baseline_z2g, baseline_g2z,
        gamete_modifiers=gamete_modifiers,
        zygote_modifiers=zygote_modifiers,
        registry=registry,
        population=host,
    )
    return config._replace(
        zygotes_to_gametes_map=z2g,
        gametes_to_zygotes_map=g2z,
        offspring_tensor=np.empty((0, 0, 0), dtype=np.float64),
        n_ztypes=int(z2g.shape[1]),
        n_gtypes=int(z2g.shape[2]),
    )
