"""The single transition from compiled products to runtime products."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import Genotype, HaploidGenotype
from natal.frontend.genetics.structures._helpers import build_compression_mask
from natal.frontend.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.frontend.model.definition_compiler import CompiledProducts
    from natal.frontend.model.draft import ModelDraft


@dataclass(frozen=True)
class IndexProjection:
    """Mapping from a specific complete catalog to runtime index order.

    Attributes:
        ztype_indices: Full ZType indices in runtime order.
        gtype_indices: Full GType indices in runtime order.
        full_ztype_count: Size of the source ZType catalog.
        full_gtype_count: Size of the source GType catalog.
        ztype_keys: Source ZType identities in their original order.
        gtype_keys: Source GType identities in their original order.
    """

    ztype_indices: tuple[int, ...]
    gtype_indices: tuple[int, ...]
    full_ztype_count: int
    full_gtype_count: int
    ztype_keys: tuple[tuple[Genotype, str], ...]
    gtype_keys: tuple[tuple[HaploidGenotype, str], ...]

    @classmethod
    def identity(cls, registry: IndexRegistry) -> IndexProjection:
        """Return an identity projection bound to a complete registry.

        Args:
            registry: Source catalog, including its entry order.

        Returns:
            A projection retaining every source entry.
        """
        return cls(tuple(range(registry.n_ztypes)), tuple(range(registry.n_gtypes)),
                   registry.n_ztypes, registry.n_gtypes,
                   tuple(registry.index_to_ztype), tuple(registry.index_to_gtype))

    @classmethod
    def from_registry(
        cls, full: IndexRegistry, runtime: IndexRegistry,
    ) -> IndexProjection:
        """Resolve existing runtime entries in the complete source catalog.

        Args:
            full: Complete compilation catalog.
            runtime: Target catalog; its entry order is preserved.

        Returns:
            A mapping bound to the complete source identities.

        Raises:
            ValueError: If a target entry is absent or duplicated.
        """
        full_z = full.index_to_ztype
        full_g = full.index_to_gtype
        z_lookup = {key: i for i, key in enumerate(full_z)}
        g_lookup = {key: i for i, key in enumerate(full_g)}
        try:
            z = tuple(z_lookup[key] for key in runtime.index_to_ztype)
            g = tuple(g_lookup[key] for key in runtime.index_to_gtype)
        except KeyError as exc:
            raise ValueError("runtime registry contains an unknown full-axis key") from exc
        if len(set(z)) != len(z) or len(set(g)) != len(g):
            raise ValueError("runtime registry contains duplicate axis keys")
        return cls(z, g, len(full_z), len(full_g), tuple(full_z), tuple(full_g))

    def validate_layout(self, registry: IndexRegistry) -> None:
        """Check source identities as well as dimensions.

        Args:
            registry: Complete catalog about to be projected.

        Raises:
            ValueError: If source counts, identities, or order differ.
        """
        if self.full_ztype_count != registry.n_ztypes or self.full_gtype_count != registry.n_gtypes:
            raise ValueError("projection axis sizes do not match registry")
        if tuple(registry.index_to_ztype) != self.ztype_keys:
            raise ValueError("projection source ZType keys do not match registry")
        if tuple(registry.index_to_gtype) != self.gtype_keys:
            raise ValueError("projection source GType keys do not match registry")

    @property
    def z_full_to_runtime(self) -> NDArray[np.int32]:
        """Array mapping full Z indices to runtime indices (-1 if removed)."""
        result = np.full(self.full_ztype_count, -1, dtype=np.int32)
        result[list(self.ztype_indices)] = np.arange(len(self.ztype_indices))
        return result

    @property
    def g_full_to_runtime(self) -> NDArray[np.int32]:
        """Array mapping full G indices to runtime indices (-1 if removed)."""
        result = np.full(self.full_gtype_count, -1, dtype=np.int32)
        result[list(self.gtype_indices)] = np.arange(len(self.gtype_indices))
        return result


def plan_projection(
    products: CompiledProducts,
    full_ztype_indices: set[int] | None = None,
) -> IndexProjection:
    """Find reachable types without constructing a complete offspring tensor.

    Args:
        products: Unpublished maps and state on complete species axes.
        full_ztype_indices: Additional source ZTypes to retain, already
            resolved from user genotype selectors or spatial union seeds.

    Returns:
        A source-bound projection including initial individuals, both stored
        sperm axes, and their inheritance closure. Empty seeds retain all types.

    Raises:
        RuntimeError: If the input is already published.
        ValueError: If an explicit seed is outside the complete ZType axis.
    """
    config, reg = products.config, products.registry
    reg.require_unpublished()
    n_z = int(config.zygotes_to_gametes_map.shape[1])
    n_g = int(config.zygotes_to_gametes_map.shape[2])
    explicit = set(full_ztype_indices or ())
    if any(i < 0 or i >= n_z for i in explicit):
        raise ValueError("full_ztype_indices contains an out-of-range index")
    seeds = set(explicit)
    counts = config.initial_individual_count
    seeds.update(i for i in range(n_z) if float(counts[:, :, i].sum()) > 0.0)
    sperm = config.initial_sperm_storage
    sperm_positive = np.argwhere(sperm > 0.0)
    for _age, female, male in sperm_positive:
        seeds.add(int(female))
        seeds.add(int(male))
    # Historical behavior retains complete axes when there is no seed.
    if not seeds:
        return IndexProjection.identity(reg)
    gmask, _, zmask, _ = build_compression_mask(
        config.zygotes_to_gametes_map, config.gametes_to_zygotes_map,
        counts, declared_zygote_types=seeds,
    )
    return IndexProjection(
        tuple(int(i) for i in np.flatnonzero(zmask >= 0)),
        tuple(int(i) for i in np.flatnonzero(gmask >= 0)), n_z, n_g,
        tuple(reg.index_to_ztype), tuple(reg.index_to_gtype),
    )


def _project_config(config: ModelDraft, p: IndexProjection, *, genetic: bool = True) -> ModelDraft:
    z = np.asarray(p.ztype_indices, dtype=np.intp)
    g = np.asarray(p.gtype_indices, dtype=np.intp)
    genetic_names = {"viability_fitness", "fecundity_fitness", "sexual_selection_fitness",
                     "zygote_viability_fitness", "female_ztype_compatibility",
                     "male_ztype_compatibility", "female_only_by_sex_chrom",
                     "male_only_by_sex_chrom", "zygotes_to_gametes_map",
                     "gametes_to_zygotes_map", "offspring_tensor"}
    projected_fields = genetic_names | {"initial_individual_count", "initial_sperm_storage"}
    detached = config._replace(
        **{name: value.copy() for name, value in config._asdict().items()
           if isinstance(value, np.ndarray) and name not in projected_fields},
        custom=deepcopy(config.custom),
    )
    return detached._replace(
        n_ztypes=len(z), n_gtypes=len(g),
        ztype_names=tuple(config.ztype_names[int(i)] for i in z) if genetic else config.ztype_names,
        gtype_names=tuple(config.gtype_names[int(i)] for i in g) if genetic else config.gtype_names,
        viability_fitness=config.viability_fitness[:, :, z].copy() if genetic else config.viability_fitness,
        fecundity_fitness=config.fecundity_fitness[:, z].copy() if genetic else config.fecundity_fitness,
        sexual_selection_fitness=config.sexual_selection_fitness[np.ix_(z, z)].copy() if genetic else config.sexual_selection_fitness,
        zygote_viability_fitness=config.zygote_viability_fitness[:, z].copy() if genetic else config.zygote_viability_fitness,
        female_ztype_compatibility=config.female_ztype_compatibility[z].copy() if genetic else config.female_ztype_compatibility,
        male_ztype_compatibility=config.male_ztype_compatibility[z].copy() if genetic else config.male_ztype_compatibility,
        female_only_by_sex_chrom=config.female_only_by_sex_chrom[z].copy() if genetic else config.female_only_by_sex_chrom,
        male_only_by_sex_chrom=config.male_only_by_sex_chrom[z].copy() if genetic else config.male_only_by_sex_chrom,
        initial_individual_count=config.initial_individual_count[:, :, z].copy(),
        initial_sperm_storage=config.initial_sperm_storage[:, z, :][:, :, z].copy(),
        zygotes_to_gametes_map=config.zygotes_to_gametes_map[:, z, :][:, :, g].copy() if genetic else config.zygotes_to_gametes_map,
        gametes_to_zygotes_map=config.gametes_to_zygotes_map[np.ix_(g, g, z)].copy() if genetic else config.gametes_to_zygotes_map,
    )


def _validate_runtime_layout(config: ModelDraft, registry: IndexRegistry) -> None:
    """Validate every published axis field against the runtime registry."""
    z, g = registry.n_ztypes, registry.n_gtypes
    expected = {
        "ztype_names": (z,), "gtype_names": (g,),
        "viability_fitness": (2, config.n_ages, z), "fecundity_fitness": (2, z),
        "sexual_selection_fitness": (z, z), "zygote_viability_fitness": (2, z),
        "female_ztype_compatibility": (z,), "male_ztype_compatibility": (z,),
        "female_only_by_sex_chrom": (z,), "male_only_by_sex_chrom": (z,),
        "initial_individual_count": (config.n_sexes, config.n_ages, z),
        "initial_sperm_storage": (config.n_ages, z, z),
        "zygotes_to_gametes_map": (2, z, g), "gametes_to_zygotes_map": (g, g, z),
        "offspring_tensor": (z, z, z),
    }
    for name, shape in expected.items():
        value = getattr(config, name)
        actual = (len(value),) if name in {"ztype_names", "gtype_names"} else value.shape
        if actual != shape:
            raise ValueError(f"published {name} shape {actual} does not match {shape}")
    from natal.contracts.materialize import (
        gtype_names_from_registry,
        ztype_names_from_registry,
    )

    if tuple(config.ztype_names) != ztype_names_from_registry(registry.index_to_ztype):
        raise ValueError("published ZType names do not match runtime registry")
    if tuple(config.gtype_names) != gtype_names_from_registry(registry.index_to_gtype):
        raise ValueError("published GType names do not match runtime registry")


def publish_products(
    products: CompiledProducts, *, compress: bool = False,
    full_ztype_indices: set[int] | None = None,
    projection: IndexProjection | None = None,
    genetic_template: CompiledProducts | None = None,
) -> CompiledProducts:
    """Project all indexed fields together and seal a detached runtime model.

    The input remains unpublished and reusable. A spatial genetic template
    shares only final genetic arrays; ecology, custom values, and initial
    state are detached. No native session is created here.

    Args:
        products: Complete unpublished compilation products.
        compress: Plan reachable axes when no explicit projection is supplied.
        full_ztype_indices: Extra complete ZType seeds for compression.
        projection: Explicit shared or existing runtime layout.
        genetic_template: Already published genetics for the same spatial
            group, reused instead of deriving another offspring tensor.

    Returns:
        Consistent runtime products with a published registry.

    Raises:
        RuntimeError: If products are already published.
        ValueError: If source axes, template layout, names, or final shapes
            disagree. No candidate state is published on failure.
    """
    from natal.frontend.genetics.matrices import recompute_offspring_tensor
    p = projection or (plan_projection(products, full_ztype_indices) if compress else
                      IndexProjection.identity(products.registry))
    products.registry.require_unpublished()
    p.validate_layout(products.registry)
    if p.full_ztype_count != products.config.n_ztypes or p.full_gtype_count != products.config.n_gtypes:
        raise ValueError("projection does not describe the products' complete axes")
    registry = products.registry
    runtime = IndexRegistry()
    runtime.slab_labels = [*registry.slab_labels]
    runtime.glab_labels = [*registry.glab_labels]
    for i in p.ztype_indices:
        runtime.register_ztype(*registry.index_to_ztype[i])
    for i in p.gtype_indices:
        runtime.register_gtype(*registry.index_to_gtype[i])
    config = _project_config(products.config, p, genetic=genetic_template is None)
    if genetic_template is None:
        config = config._replace(offspring_tensor=recompute_offspring_tensor(
            config.zygotes_to_gametes_map, config.gametes_to_zygotes_map,
        ))
    else:
        if not genetic_template.registry.published:
            raise ValueError("genetic template must already be published")
        if (genetic_template.config.n_ztypes, genetic_template.config.n_gtypes) != (len(p.ztype_indices), len(p.gtype_indices)):
            raise ValueError("published genetic template axes do not match runtime projection")
        template = genetic_template.config
        template_z = tuple(genetic_template.registry.index_to_ztype)
        template_g = tuple(genetic_template.registry.index_to_gtype)
        if tuple(runtime.index_to_ztype) != template_z or tuple(runtime.index_to_gtype) != template_g:
            raise ValueError("genetic template registry does not match published runtime registry")
        genetic_fields = (
            "ztype_names", "gtype_names", "viability_fitness",
            "fecundity_fitness", "sexual_selection_fitness",
            "zygote_viability_fitness", "zygotes_to_gametes_map",
            "gametes_to_zygotes_map", "offspring_tensor",
            "female_ztype_compatibility", "male_ztype_compatibility",
            "female_only_by_sex_chrom", "male_only_by_sex_chrom",
        )
        shared: dict[str, object] = {}
        for field in genetic_fields:
            value = getattr(template, field)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            shared[field] = value
        config = config._replace(**shared)
    _validate_runtime_layout(config, runtime)
    runtime.mark_published()
    return products.__class__(config, runtime, list(products.gamete_modifiers), list(products.zygote_modifiers))


def ensure_layout_closed(products: CompiledProducts, projection: IndexProjection) -> None:
    """Reject positive transitions outside an existing runtime layout.

    Checks every retained type, including types with zero current population,
    so subsequent introductions or restored states remain representable.

    Args:
        products: Newly compiled complete inheritance relations.
        projection: Existing runtime axes expressed in complete coordinates.

    Raises:
        ValueError: If the projection has another source or a retained ZType
            produces an excluded GType, or retained gametes form an excluded
            ZType. Probabilities are never renormalized to hide lost branches.
    """
    projection.validate_layout(products.registry)
    z2g = products.config.zygotes_to_gametes_map
    g2z = products.config.gametes_to_zygotes_map
    retained_z = set(projection.ztype_indices)
    retained_g = set(projection.gtype_indices)
    for z in retained_z:
        for g in np.flatnonzero((z2g[:, z, :] > 0.0).any(axis=0)):
            if int(g) not in retained_g:
                raise ValueError("published layout is not closed: retained ZType produces an external GType")
    for a in retained_g:
        for b in retained_g:
            for z in np.flatnonzero(g2z[a, b, :] > 0.0):
                if int(z) not in retained_z:
                    raise ValueError("published layout is not closed under positive inheritance")
