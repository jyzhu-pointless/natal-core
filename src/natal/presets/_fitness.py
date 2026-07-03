"""Fitness patch construction and application functions.

Private module — not part of the public API.
"""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast

from natal.genetics import Gene, Genotype

from ._types import (
    PresetFitnessPatch,
    _AlleleScalingMode,
    _calculate_allele_effect,
    _coerce_selector,
    _coerce_sex_specifier,
    _count_combined_allele_copies,
    _FecundityScalingConfig,
    _is_effect_scale,
    _is_fecundity_scaling_config,
    _is_sexual_selection_scaling_config,
    _is_simple_age_scale_map,
    _is_viability_age_map,
    _is_viability_scaling_config,
    _is_zygote_viability_scaling_config,
    _normalize_sex_key,
    _SexualSelectionScalingConfig,
    _split_config_mode,
    _ViabilityScalingConfig,
    _ZygoteViabilityScalingConfig,
)

if TYPE_CHECKING:
    from natal.population.base import BasePopulation


def _make_fitness_patch_given_allele_scaling(
    allele_name: Union[str, List[str], Tuple[str, ...]],
    viability_scaling: Optional[_ViabilityScalingConfig] = None,
    fecundity_scaling: Optional[_FecundityScalingConfig] = None,
    sexual_selection_scaling: Optional[_SexualSelectionScalingConfig] = None,
    zygote_viability_scaling: Optional[_ZygoteViabilityScalingConfig] = None,
    viability_mode: _AlleleScalingMode = "multiplicative",
    fecundity_mode: _AlleleScalingMode = "multiplicative",
    sexual_selection_mode: str = "multiplicative",
    zygote_viability_mode: _AlleleScalingMode = "multiplicative",
) -> PresetFitnessPatch:
    """Helper to create a fitness patch dict for a single allele's scaling effects.

    This function supports all four fitness types: viability, fecundity, sexual selection,
    and zygote fitness. Zygote fitness represents the probability that a zygote survives
    to become an individual, applied during reproduction stage before survival and competition.

    Args:
        allele_name: Name or list of allele names to apply scaling to.
        viability_scaling: Viability fitness scaling configuration.
        fecundity_scaling: Fecundity fitness scaling configuration.
        sexual_selection_scaling: Sexual selection scaling configuration.
        zygote_viability_scaling: Zygote fitness scaling configuration.
        viability_mode: Scaling mode for viability fitness.
        fecundity_mode: Scaling mode for fecundity fitness.
        sexual_selection_mode: Scaling mode for sexual selection.
        zygote_viability_mode: Scaling mode for zygote fitness.

    Returns:
        PresetFitnessPatch: Dictionary containing fitness patch configurations.
    """
    # Dictionary keys must be hashable. Lists are not, so we convert to tuple.
    if isinstance(allele_name, list):
        key = tuple(allele_name)
    else:
        key = allele_name

    patch: PresetFitnessPatch = {}

    if viability_scaling is not None:
        patch['viability_per_allele'] = {key: (viability_scaling, viability_mode)}

    if fecundity_scaling is not None:
        patch['fecundity_per_allele'] = {key: (fecundity_scaling, fecundity_mode)}

    if sexual_selection_scaling is not None:
        patch['sexual_selection_per_allele'] = {key: (sexual_selection_scaling, sexual_selection_mode)}

    if zygote_viability_scaling is not None:
        patch['zygote_per_allele'] = {key: (zygote_viability_scaling, zygote_viability_mode)}

    return patch


def _apply_viability_allele_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _ViabilityScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven viability scaling using multiplicative copy-number effect."""
    # viability tensor layout:
    #   viability_fitness[sex_idx, age_idx, ztype_idx]
    # This function multiplies existing values in-place via setter calls,
    # so multiple presets/patches compose multiplicatively.
    viability_arr = population.config.viability_fitness
    default_age = int(population.config.new_adult_age) - 1

    # Resolve one or more alleles
    target_genes: List[Gene] = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in viability_per_allele patch.")
        target_genes.append(gene)

    for genotype in all_genotypes:
        copies = _count_combined_allele_copies(genotype, target_genes)
        if copies == 0:
            # No target allele copies in this genotype: no effect.
            continue

        z_indices = population.index_registry.ztype_indices_for(genotype)
        for z_idx in z_indices:
            if isinstance(config, (int, float, tuple, list)):
                # Scalar/custom tuple branch:
                # apply same factor to both sexes at default viability age.
                factor = _calculate_allele_effect(config, copies, mode)
                for sex_idx in (0, 1):
                    current = float(viability_arr[sex_idx, default_age, z_idx])
                    population.config.set_viability_fitness(sex_idx, z_idx, current * factor, default_age)
                continue

            config_map = cast(Mapping[object, object], config)

            if _is_viability_age_map(config_map):
                # Age-map branch: config treated as {age: scale} for both sexes.
                for age, scale in config_map.items():
                    factor = _calculate_allele_effect(scale, copies, mode)
                    for sex_idx in (0, 1):
                        current = float(viability_arr[sex_idx, age, z_idx])
                        population.config.set_viability_fitness(sex_idx, z_idx, current * factor, age)
                continue

            for sex_key, sex_config in config_map.items():
                # Sex-map branch:
                # sex_config can be either:
                #   - direct scale for default age
                #   - nested {age: scale}
                sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                if _is_effect_scale(sex_config):
                    factor = _calculate_allele_effect(sex_config, copies, mode)
                    current = float(viability_arr[sex_idx, default_age, z_idx])
                    population.config.set_viability_fitness(sex_idx, z_idx, current * factor, default_age)
                elif isinstance(sex_config, Mapping):
                    sex_age_map = cast(Mapping[object, object], sex_config)
                    for age, scale in sex_age_map.items():
                        if not isinstance(age, int):
                            raise TypeError(
                                f"Invalid viability sex-age key for '{allele_name}', sex '{sex_key}': {type(age).__name__}"
                            )
                        if not _is_effect_scale(scale):
                            raise TypeError(
                                f"Invalid viability sex-age scale for '{allele_name}', sex '{sex_key}', age {age}: "
                                f"{type(scale).__name__}"
                            )
                        factor = _calculate_allele_effect(scale, copies, mode)
                        current = float(viability_arr[sex_idx, int(age), z_idx])
                        population.config.set_viability_fitness(sex_idx, z_idx, current * factor, int(age))
                else:
                    raise TypeError(
                        f"Invalid viability allele sex config for '{allele_name}', sex '{sex_key}': "
                        f"{type(sex_config).__name__}"
                    )


def _apply_fecundity_allele_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _FecundityScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven fecundity scaling using multiplicative copy-number effect."""
    # fecundity tensor layout:
    #   fecundity_fitness[sex_idx, ztype_idx]
    # As with viability, this function multiplies current values.
    fecundity_arr = population.config.fecundity_fitness

    # Resolve one or more alleles
    target_genes: List[Gene] = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in fecundity_per_allele patch.")
        target_genes.append(gene)

    for genotype in all_genotypes:
        copies = _count_combined_allele_copies(genotype, target_genes)
        if copies == 0:
            continue

        z_indices = population.index_registry.ztype_indices_for(genotype)
        for z_idx in z_indices:
            if isinstance(config, (int, float, tuple, list)):
                # Global branch (both sexes).
                factor = _calculate_allele_effect(config, copies, mode)
                for sex_idx in (0, 1):
                    current = float(fecundity_arr[sex_idx, z_idx])
                    population.config.set_fecundity_fitness(sex_idx, z_idx, current * factor)
                continue

            config_map = cast(Mapping[object, object], config)
            for sex_key, scale in config_map.items():
                # Sex-specific branch.
                sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                if not _is_effect_scale(scale):
                    raise TypeError(
                        f"Invalid fecundity sex scale for '{allele_name}', sex '{sex_key}': {type(scale).__name__}"
                    )
                factor = _calculate_allele_effect(scale, copies, mode)
                current = float(fecundity_arr[sex_idx, z_idx])
                population.config.set_fecundity_fitness(sex_idx, z_idx, current * factor)


def _apply_sexual_selection_allele_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _SexualSelectionScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven sexual-selection scaling.

    - float: multiplicative by male allele copy-number for all female genotypes.
    - tuple(default, carrier): binary by male carrier status (copy > 0).
    """
    # sexual-selection tensor layout:
    #   sexual_selection_fitness[female_ztype_idx, male_ztype_idx]
    # Effect is computed from male allele copies, then applied per pair.
    sex_sel_arr = population.config.sexual_selection_fitness

    # Resolve one or more alleles
    target_genes: List[Gene] = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in sexual_selection_per_allele patch.")
        target_genes.append(gene)

    for f_genotype in all_genotypes:
        f_z_indices = population.index_registry.ztype_indices_for(f_genotype)
        for m_genotype in all_genotypes:
            m_z_indices = population.index_registry.ztype_indices_for(m_genotype)
            copies = _count_combined_allele_copies(m_genotype, target_genes)

            if isinstance(config, tuple):
                # Binary carrier logic:
                # config[0] for non-carriers, config[1] for carriers.
                if len(config) != 2:
                    raise ValueError(
                        f"sexual_selection allele tuple for '{allele_name}' must have length 2, got {len(config)}"
                    )
                factor = float(config[1] if copies > 0 else config[0])
            else:
                # Copy-number-based logic via mode.
                factor = _calculate_allele_effect(config, copies, mode)

            for f_z in f_z_indices:
                for m_z in m_z_indices:
                    current = float(sex_sel_arr[f_z, m_z])
                    population.config.set_sexual_selection_fitness(f_z, m_z, current * factor)


def _apply_zygote_viability_allele_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _ZygoteViabilityScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven zygote viability scaling using copy-number and scaling mode."""
    # zygote tensor layout:
    #   zygote_viability_fitness[sex_idx, ztype_idx]
    # This function multiplies existing values in-place via setter calls,
    # so multiple presets/patches compose multiplicatively.
    zygote_arr = population.config.zygote_viability_fitness

    # Resolve one or more alleles
    target_genes: List[Gene] = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in zygote_per_allele patch.")
        target_genes.append(gene)

    # Compute scaling factors for each genotype
    for genotype in all_genotypes:
        copy_count: int = _count_combined_allele_copies(genotype, target_genes)

        # Apply scaling based on copy count
        if copy_count == 0:
            continue  # No effect for zero copies

        z_indices = population.index_registry.ztype_indices_for(genotype)
        for z_idx in z_indices:
            # Get scaling factor for this copy count
            if _is_effect_scale(config):
                # Scalar/custom tuple branch for both sexes.
                total_scale = _calculate_allele_effect(config, copy_count, mode)
                for sex_idx in range(2):
                    current = float(zygote_arr[sex_idx, z_idx])
                    population.config.set_zygote_viability_fitness(sex_idx, z_idx, current * total_scale)
            elif isinstance(config, Mapping):
                # Sex-specific config.
                config_map = cast(Mapping[object, object], config)
                for sex_key, sex_config in config_map.items():
                    sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                    if _is_effect_scale(sex_config):
                        total_scale = _calculate_allele_effect(sex_config, copy_count, mode)
                        current = float(zygote_arr[sex_idx, z_idx])
                        population.config.set_zygote_viability_fitness(sex_idx, z_idx, current * total_scale)
                    elif isinstance(sex_config, Mapping):
                        # Age-specific config (not applicable to zygote fitness)
                        raise TypeError(
                            f"Age-specific config not supported for zygote_allele: {sex_config}"
                        )
                    else:
                        raise TypeError(
                            f"Invalid zygote allele sex config for '{allele_name}', sex '{sex_key}': "
                            f"{type(sex_config).__name__}"
                        )
            else:
                raise TypeError(
                    f"Invalid zygote allele config for '{allele_name}': {type(config).__name__}"
                )


# ── Slab-based fitness scaling (per_slab keys) ──────────────────────────


def _apply_viability_slab_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    patch: PresetFitnessPatch,
) -> None:
    """Apply per-slab viability scaling by writing to the (G×S) flat array."""
    for slab_name, factor in patch['viability_per_slab'].items():
        default_age = int(population.config.new_adult_age) - 1
        arr = population.config.viability_fitness
        for genotype in all_genotypes:
            z = population.index_registry.ztype_index(genotype, slab_name)
            for sex in (0, 1):
                current = float(arr[sex, default_age, z])
                arr[sex, default_age, z] = current * float(factor)


def _apply_fecundity_slab_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    patch: PresetFitnessPatch,
) -> None:
    """Apply per-slab fecundity scaling."""
    for slab_name, factor in patch['fecundity_per_slab'].items():
        arr = population.config.fecundity_fitness
        for genotype in all_genotypes:
            z = population.index_registry.ztype_index(genotype, slab_name)
            for sex in (0, 1):
                current = float(arr[sex, z])
                arr[sex, z] = current * float(factor)


def _apply_sexual_selection_slab_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    patch: PresetFitnessPatch,
) -> None:
    """Apply per-slab sexual selection to the (G×S, G×S) matrix.

    Note: only scales the female (row) side — male columns are
    unaffected.  This models asymmetric mate preference where the
    female genotype determines the mating success modifier.
    """
    for slab_name, factor in patch['sexual_selection_per_slab'].items():
        arr = population.config.sexual_selection_fitness
        for genotype in all_genotypes:
            z = population.index_registry.ztype_index(genotype, slab_name)
            # Female side: all male ZTypes paired with this female ZType
            for mz in range(arr.shape[1]):
                current = float(arr[z, mz])
                arr[z, mz] = current * float(factor)


def _apply_zygote_slab_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    patch: PresetFitnessPatch,
) -> None:
    """Apply per-slab zygote viability scaling."""
    for slab_name, factor in patch['zygote_per_slab'].items():
        arr = population.config.zygote_viability_fitness
        for genotype in all_genotypes:
            z = population.index_registry.ztype_index(genotype, slab_name)
            for sex in (0, 1):
                current = float(arr[sex, z])
                arr[sex, z] = current * float(factor)


def apply_preset_fitness_patch(population: 'BasePopulation[Any]', patch: PresetFitnessPatch) -> None:
    """Apply a declarative preset fitness patch to population config tensors.

    Patch schema (all keys optional):
    - viability: Dict[genotype_selector, _ViabilityScalingConfig]
    - fecundity: Dict[genotype_selector, _FecundityScalingConfig]
    - sexual_selection: Dict[female_selector, Union[float, Dict[male_selector, float]]]
    """
    if not patch:
        return

    all_genotypes = population.index_registry.index_to_genotype

    # ----------------------------------------------------------------------
    # 1) Selector-based viability patch
    #
    # Input examples:
    # - {"A1|A1": 0.8}
    # - {"A1|A1": {0: 0.9, 1: 0.8}}
    # - {"A1|A1": {"female": 0.9, "male": {0: 0.95}}}
    # ----------------------------------------------------------------------
    viability_patch = patch.get('viability', {})
    for selector, config in viability_patch.items():
        matched = population.species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context='preset.viability',
        )
        for genotype in matched:
            z_indices = population.index_registry.ztype_indices_for(genotype)

            # scalar: both sexes at default viability age
            if isinstance(config, (int, float)):
                for z_idx in z_indices:
                    population.config.set_viability_fitness(0, z_idx, float(config))
                    population.config.set_viability_fitness(1, z_idx, float(config))
                continue

            # age-specific for both sexes: {age: scale}
            config_map = cast(Mapping[object, object], config)
            if _is_simple_age_scale_map(config_map):
                for z_idx in z_indices:
                    for age, scale in config_map.items():
                        population.config.set_viability_fitness(0, z_idx, float(scale), int(age))
                        population.config.set_viability_fitness(1, z_idx, float(scale), int(age))
                continue

            # sex-specific: {sex: float | {age: scale}}
            for sex_key, sex_config in config_map.items():
                sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                if isinstance(sex_config, (int, float)):
                    for z_idx in z_indices:
                        population.config.set_viability_fitness(sex_idx, z_idx, float(sex_config))
                elif isinstance(sex_config, Mapping):
                    sex_age_map = cast(Mapping[object, object], sex_config)
                    for z_idx in z_indices:
                        for age, scale in sex_age_map.items():
                            if not isinstance(age, int) or not isinstance(scale, (int, float)):
                                raise TypeError(
                                    f"Invalid viability sex-age config for selector '{selector}', sex '{sex_key}'"
                                )
                            population.config.set_viability_fitness(sex_idx, z_idx, float(scale), int(age))
                else:
                    raise TypeError(
                        f"Invalid viability sex config for selector '{selector}', sex '{sex_key}': "
                        f"{type(sex_config).__name__}"
                    )

    # ----------------------------------------------------------------------
    # 2) Selector-based fecundity patch
    #
    # Input examples:
    # - {"A1|A1": 0.8}
    # - {"A1|A1": {"female": 0.9, "male": 0.7}}
    # ----------------------------------------------------------------------
    fecundity_patch = patch.get('fecundity', {})
    for selector, config in fecundity_patch.items():
        matched = population.species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context='preset.fecundity',
        )
        for genotype in matched:
            z_indices = population.index_registry.ztype_indices_for(genotype)

            if isinstance(config, (int, float)):
                for z_idx in z_indices:
                    population.config.set_fecundity_fitness(0, z_idx, float(config))
                    population.config.set_fecundity_fitness(1, z_idx, float(config))
                continue

            config_map = cast(Mapping[object, object], config)
            for sex_key, scale in config_map.items():
                sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                if not isinstance(scale, (int, float)):
                    raise TypeError(
                        f"Invalid fecundity sex scale for selector '{selector}', sex '{sex_key}'"
                    )
                for z_idx in z_indices:
                    population.config.set_fecundity_fitness(sex_idx, z_idx, float(scale))

    # ----------------------------------------------------------------------
    # 3) Selector-based sexual-selection patch
    #
    # Input examples:
    # - {"female_selector": 0.9}  # shorthand for all males
    # - {"female_selector": {"male_selector": 1.2}}
    # ----------------------------------------------------------------------
    sexual_selection_patch = patch.get('sexual_selection', {})
    for female_selector, male_config in sexual_selection_patch.items():
        female_matched = population.species.resolve_genotype_selectors(
            selector=female_selector,
            all_genotypes=all_genotypes,
            context='preset.sexual_selection(female)',
        )

        # Allow shorthand: female_selector -> scalar means all-male targets
        if isinstance(male_config, (int, float)):
            male_map = {'*': float(male_config)}
        else:
            male_map = cast(Mapping[object, object], male_config)

        for male_selector, scale in male_map.items():
            if not isinstance(scale, (int, float)):
                raise TypeError(
                    f"Invalid sexual_selection scale for female selector '{female_selector}'"
                )
            male_matched = population.species.resolve_genotype_selectors(
                selector=_coerce_selector(male_selector),
                all_genotypes=all_genotypes,
                context='preset.sexual_selection(male)',
            )
            for f_genotype in female_matched:
                f_z_indices = population.index_registry.ztype_indices_for(f_genotype)
                for m_genotype in male_matched:
                    m_z_indices = population.index_registry.ztype_indices_for(m_genotype)
                    for f_z in f_z_indices:
                        for m_z in m_z_indices:
                            population.config.set_sexual_selection_fitness(f_z, m_z, float(scale))

    # ----------------------------------------------------------------------
    # 4) Allele-based patches
    #
    # This layer expands allele-centric config into genotype-level writes:
    # 1) resolve allele name(s) to Gene objects
    # 2) count target copies per genotype (0/1/2)
    # 3) convert copies -> factor according to mode
    # 4) multiply corresponding tensor cells
    # ----------------------------------------------------------------------
    viability_per_allele_patch = patch.get('viability_per_allele', {})
    for allele_name, val in viability_per_allele_patch.items():
        config, mode = _split_config_mode(val)
        if not _is_viability_scaling_config(config):
            raise TypeError(f"Invalid viability_per_allele config for '{allele_name}'")
        _apply_viability_allele_scaling(population, all_genotypes, allele_name, config, mode)

    fecundity_per_allele_patch = patch.get('fecundity_per_allele', {})
    for allele_name, val in fecundity_per_allele_patch.items():
        config, mode = _split_config_mode(val)
        if not _is_fecundity_scaling_config(config):
            raise TypeError(f"Invalid fecundity_per_allele config for '{allele_name}'")
        _apply_fecundity_allele_scaling(population, all_genotypes, allele_name, config, mode)

    sexual_selection_per_allele_patch = patch.get('sexual_selection_per_allele', {})
    for allele_name, val in sexual_selection_per_allele_patch.items():
        config, mode = _split_config_mode(val)
        if not _is_sexual_selection_scaling_config(config):
            raise TypeError(f"Invalid sexual_selection_per_allele config for '{allele_name}'")
        _apply_sexual_selection_allele_scaling(population, all_genotypes, allele_name, config, mode)

    # 5) Zygote fitness patch
    zygote_patch = patch.get('zygote', {})
    for selector, config in zygote_patch.items():
        matched = population.species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context='preset.zygote',
        )
        for genotype in matched:
            z_indices = population.index_registry.ztype_indices_for(genotype)
            if isinstance(config, (int, float)):
                for z_idx in z_indices:
                    population.config.set_zygote_viability_fitness(0, z_idx, float(config))
                    population.config.set_zygote_viability_fitness(1, z_idx, float(config))
            elif isinstance(config, Mapping):
                config_map = cast(Mapping[object, object], config)
                for sex_key, sex_config in config_map.items():
                    sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                    if isinstance(sex_config, (int, float)):
                        for z_idx in z_indices:
                            population.config.set_zygote_viability_fitness(sex_idx, z_idx, float(sex_config))

    # 6) Zygote allele-based fitness patch
    zygote_per_allele_patch = patch.get('zygote_per_allele', {})
    for allele_name, val in zygote_per_allele_patch.items():
        config, mode = _split_config_mode(val)
        if not _is_zygote_viability_scaling_config(config):
            raise TypeError(f"Invalid zygote_per_allele config for '{allele_name}'")
        _apply_zygote_viability_allele_scaling(population, all_genotypes, allele_name, config, mode)

    # 7) Slab-based fitness patches (per_slab keys — symmetric with per_allele)
    if patch.get('viability_per_slab'):
        _apply_viability_slab_scaling(population, all_genotypes, patch)
    if patch.get('fecundity_per_slab'):
        _apply_fecundity_slab_scaling(population, all_genotypes, patch)
    if patch.get('sexual_selection_per_slab'):
        _apply_sexual_selection_slab_scaling(population, all_genotypes, patch)
    if patch.get('zygote_per_slab'):
        _apply_zygote_slab_scaling(population, all_genotypes, patch)
