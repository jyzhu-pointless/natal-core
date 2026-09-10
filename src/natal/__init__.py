"""
Genetic Simulation Utilities
============================

Core components for genetic simulation: structures, entities, and population models.
"""

import importlib
from typing import Any, Dict

__version__ = "0.2.0b"

# Maps exported symbol names to the module that defines them.
#
# The package intentionally does not import any child modules during initialization.
# It only builds a name index up front, for Examples:
# {"Sex": "frontend.utils", "AgeStructuredPopulation": "frontend.population"}
# When code first accesses natal.Sex, the matching module is imported on demand.
#
# This index is built from the real (non-shim) package tree only:
# ``frontend.*`` subpackages, ``contracts``, and ``backends.*``.  The legacy
# top-level forwarding shims (``natal.frontend.data``, ``natal.frontend.hooks`` ...) were removed
# when the Phase-0 reorganization completed; each legacy package name below
# still resolves to its relocated module so ``natal.frontend.hooks``-style access keeps
# working, but the physical packages no longer exist.
_lazy_map: Dict[str, str] = {}
_lazy_packages: set[str] = set()




def _scan_unit(module_name: str, names: list[str], allow_legacy_key: bool) -> None:
    """Register one unit's explicitly declared exports in the lazy index.

    Args:
        module_name: Dotted module name of the unit (e.g. ``frontend.hooks``).
        names: The unit's exported names, from ``_PUBLIC_EXPORTS``.
        allow_legacy_key: Whether the legacy short package key (e.g. ``hooks``)
            is registered alongside the exported names.  ``contracts`` keeps its
            own key because the package path did not change; ``frontend`` and
            ``backends`` subpackages keep their pre-Phase-0 keys so that
            ``natal.<legacy-key>`` attribute access keeps resolving.
    """
    if allow_legacy_key:
        short = module_name.rsplit(".", 1)[-1]
        _lazy_map.setdefault(short, module_name)
        if names:
            _lazy_packages.add(short)
    for name in names:
        _lazy_map.setdefault(name, module_name)


# Explicit public export list.
#
# Every public top-level name of ``natal`` is declared here, once, per
# owning unit.  Adding a name to a module's ``__all__`` does NOT publish
# it at the top level: the name becomes public only when it is added to
# this mapping (an intentional act).  The consistency test in
# ``tests/test_phase0_shims.py`` pins this list against the modules'
# literal ``__all__`` so neither side can drift silently; the editor
# stub ``__init__.pyi`` is generated from this list by
# ``scripts/generate_init_pyi.py``.
#
# Lazy loading is unchanged: importing ``natal`` reads this literal and
# never executes child-module code.
_PUBLIC_EXPORTS: dict[str, list[str]] = {
    "contracts": [
        "CONTRACTS_VERSION", "Blueprint", "CustomValue", "Materialized", "Params",
        "format_type_name", "gtype_names_from_registry", "materialize",
        "ztype_names_from_registry",
    ],
    "frontend.builder": [
        "PopulationBuilder", "ROUTES", "ROUTES_BY_METHOD", "RuntimeUpdater",
        "dispatch", "set_param",
    ],
    "frontend.data": [
        "ModelDefinition", "ModelDraft", "NO_COMPETITION", "FIXED", "LOGISTIC",
        "LINEAR", "BEVERTON_HOLT", "PopulationState", "DiscretePopulationState",
        "extract_gamete_frequencies", "extract_gamete_frequencies_by_glab",
        "extract_zygote_frequencies", "build_population_config",
        "build_discrete_engine_config", "build_custom_slots", "initialize_zygote_map",
        "initialize_gamete_map", "compress_hl", "decompress_hl", "compress_config",
        "parse_flattened_state", "parse_flattened_discrete_state",
    ],
    "frontend.fitness": ["apply_preset_fitness_patch", "write_fitness_field"],
    "frontend.genetics": [
        "SexChromosomeType", "Species", "SpeciesConfigBlueprint", "Chromosome",
        "Linkage", "RecombinationMap", "Locus", "Gene", "Allele", "Haplotype",
        "HaploidGenotype", "HaploidGenome", "Genotype", "Genome", "DiploidGenome",
        "DiploidGenotype", "GenomeTemplate", "Karyotype",
        "create_haplotype_from_allele_names", "create_chromosome_from_allele_names",
        "compute_recombinant_haplotypes", "compute_recombinant_haplotypes_with_alleles",
        "build_compression_mask",
    ],
    "frontend.hooks": [
        "OpType", "DemeSelector", "deme_selector_matches", "HookLayout", "HookOp", "Op",
        "CompiledHookPlan", "CompiledHookDescriptor", "HookProgram",
        "empty_hook_program", "hook", "compile_declarative_hook",
        "compile_selector_callback", "TickContext", "TickMetrics", "BlueprintView",
        "HookRunner", "COND_ALWAYS", "COND_TICK_EQ", "COND_TICK_MOD", "ECO_PARAM_NAMES",
        "COND_TICK_GE", "COND_TICK_GT", "COND_TICK_LE", "COND_TICK_LT", "COND_OP_AND",
        "COND_OP_OR", "COND_OP_NOT", "EVENT_FIRST", "EVENT_EARLY", "EVENT_LATE",
        "EVENT_FINISH", "EVENT_NAMES", "EVENT_ID_MAP", "NUM_EVENTS", "RESULT_CONTINUE",
        "RESULT_SKIP", "RESULT_STOP", "parse_condition",
    ],
    "frontend.modifiers": [
        "build_modifier_wrappers", "evaluate_genotype_filter",
        "GameteAlleleConversionRule", "GameteConversionRuleSet",
        "GameteGlabConversionRule", "GameteGtypeConversionRule",
        "GameteHaploidGenomeConversionRule", "GameteModifier", "GenotypeFilter",
        "GlabSelector", "wrap_gamete_modifier", "wrap_zygote_modifier",
        "ZygoteAlleleConversionRule", "ZygoteConversionRuleSet",
        "ZygoteGenotypeConversionRule", "ZygoteGlabRedirectRule", "ZygoteModifier",
        "ZygoteZtypeConversionRule",
    ],
    "frontend.output": [
        "History", "HistorySchema", "Observation", "ObservationMetadata",
        "ObservationResult", "PopulationLayout", "SpatialHistoryLayout", "apply_rule",
        "build_identity_observation", "discrete_population_state_to_dict",
        "discrete_population_state_to_json", "population_history_to_readable_dict",
        "population_history_to_readable_json",
        "population_observation_history_to_readable_dict",
        "population_observation_history_to_readable_json", "population_state_to_dict",
        "population_state_to_json", "population_to_readable_dict",
        "population_to_readable_json", "spatial_population_history_to_readable_dict",
        "spatial_population_history_to_readable_json",
        "spatial_population_observation_history_to_readable_dict",
        "spatial_population_observation_history_to_readable_json",
        "spatial_population_to_observation_dict",
        "spatial_population_to_observation_json", "spatial_population_to_readable_dict",
        "spatial_population_to_readable_json",
    ],
    "frontend.patterns": [
        "GameteTypePattern", "GenotypePatternParser", "GenotypeSelector",
        "IndividualSelector", "LabPattern", "PatternParseError", "ZygoteTypePattern",
        "resolve_zygote_type",
    ],
    "frontend.population": [
        "BasePopulation", "AgeStructuredPopulation", "DiscreteGenerationPopulation",
    ],
    "frontend.presets": [
        "GeneticPreset", "HomingDrive", "ToxinAntidoteDrive", "CytoplasmicPreset",
        "Wolbachia", "TransgenicBackground", "apply_preset_fitness_patch",
        "PresetFitnessPatch", "count_allele_copies", "GameteAlleleConversionRule",
        "GameteConversionRuleSet", "GameteGlabConversionRule",
        "GameteGtypeConversionRule", "GameteHaploidGenomeConversionRule",
        "ZygoteAlleleConversionRule", "ZygoteConversionRuleSet",
        "ZygoteGenotypeConversionRule", "ZygoteGlabRedirectRule",
        "ZygoteZtypeConversionRule",
    ],
    "frontend.registry": ["IndexRegistry"],
    "frontend.spatial": [
        "BatchSetting", "GridTopology", "HexGrid", "MigrationCSR",
        "SpatialPopulationBuilder", "SpatialPopulation", "SquareGrid", "batch_setting",
        "build_adjacency_matrix", "build_gaussian_kernel",
    ],
    "frontend.ui": [
        "Dashboard", "PopulationDashboard", "SpatialDashboard", "get_allele_color",
        "launch", "launch_population", "launch_spatial", "render_cell_svg",
    ],
    "frontend.utils": [
        "Sex", "Age", "GameteLabel", "resolve_sex_label", "validate_name",
        "ALL_PARAMETERS", "PARAM_IDS", "PARAMETERS_BY_DOMAIN", "ParamDescriptor",
    ],
    "frontend.webui": [
        "launch_vue",
    ],
}

# Build the lazy index from the explicit list (deterministic first-wins
# semantics on repeated names; units are listed alphabetically).
for _unit, _names in _PUBLIC_EXPORTS.items():
    _scan_unit(_unit, _names, allow_legacy_key=True)

# Public export list.
#
# This keeps from natal import * aligned with the package's public API and also
# helps dir(natal) and some tooling discover these names.
__all__ = list(_lazy_map)  # type: ignore[reportUnsupportedDunderAll]  # derived from the explicit list above


def __getattr__(name: str) -> Any:
    # When code accesses natal.<name> and that attribute is not present yet,
    # Python calls the module-level __getattr__. Import the owning child module
    # here so the symbol is loaded only on first access.
    if name in _lazy_map:
        module = importlib.import_module(f".{_lazy_map[name]}", __name__)
        if name in _lazy_packages:
            globals()[name] = module
            return module
        value = getattr(module, name)
        # Cache the resolved object in this module's globals so future accesses do
        # not have to go through __getattr__ again.
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    # Expose lazily exported names to dir() and completion tools.
    return sorted(set(globals().keys()) | set(__all__))
