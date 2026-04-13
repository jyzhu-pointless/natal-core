# Changelog

## 2026.4.13
- Add Zygote Fitness support: new fitness type applied during reproduction stage before survival and competition
- Extend PopulationConfig with zygote_fitness field and set_zygote_fitness method
- Update Builder system to support zygote fitness configuration via fitness() method
- Extend Genetic Presets system with zygote allele-based fitness scaling support
- Integrate zygote fitness application in simulation kernels with proper stochastic sampling
- Add comprehensive unit tests for zygote fitness functionality
- Update documentation for PopulationConfig, Builder system, simulation kernels, and genetic presets
- Fix GeneticPattern parsing issues: `enumerate_genotypes_matching_pattern` now correctly recognizes unordered homologous chromosome identifier `::`; fixed parsing of single-character gene syntax with omitted `/`

## 2026.4.10
- Refactor hook dispatch flow: move Python dispatch runners out of population classes into hooks executor helpers, and remove DiscreteGenerationPopulation internal _step_* helpers
- Unify hook execution policy when Numba is disabled: any registered hook type now uses one sequential Python dispatch path
- Rework SpatialPopulation hook aggregation to pin compiled hooks to owning demes and rebuild one consistent aggregate hook registry after set/remove operations
- Simplify spatial wrapper template to run migration-enabled spatial tick kernel directly; keep local lifecycle plus migration responsibilities explicit in kernel/docs
- Add heterogeneous deme-config support on the njit spatial path via per-deme config-bank id routing, while preserving deme-level parallel execution
- Enforce migration-time consistency for `is_stochastic` and `use_continuous_sampling` across demes, and update spatial simulation guides accordingly (EN/ZH)
- Route heterogeneous deme-config execution through the unified hook-aware spatial timeline so hook semantics stay consistent regardless of config heterogeneity

## 2026.4.9
- Correct carrying capacity (equilibrium metrics) handling in population builders
- Enhance sex chromosome handling and genotype compatibility in population dynamics
- Add support for heterogeneous kernel routing in SpatialPopulation
