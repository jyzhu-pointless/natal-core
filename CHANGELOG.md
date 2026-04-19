# Changelog

## 2026.4.19 (v0.1.0-rc.2)
- Remove redundant `parallel=True` decorators from adjacency migration wrapper functions that do not contain `prange`
- Move probability related logic from `algorithms.py` to `numba_compat.py`
- Change the DNA pattern of the logo from left-handed to right-handed helix
- Fix dashboard favicon loading after wheel installation by resolving `natal.svg` from package resources at runtime
- Update the `index` and `quickstart` parts of documentation

## 2026.4.17 (v0.1.0-rc.1)
- Refactor Observation system: make Observation reusable and state-independent by removing dimension coupling from state validation
- Decouple `ObservationFilter` from state-specific logic; dimension validation now occurs at apply-time via `Observation.apply()`
- Refocus API documentation: position `Observation` and state translation output functions as primary user entry points
- Discourage direct user instantiation of `Observation`; recommend population-level convenience methods instead
- Add `output_current_state()` and `output_history()` convenience methods to `BasePopulation` as primary interfaces
- Enhance demo files with observation and translator usage examples: `observation_history_demo.py`, `mosquito.py`, `discrete.py`
- Demonstrate pattern string filtering in demos: use `"Dr::*"` and `"R2|*"` patterns to show flexible genotype matching
- Refactor HexGrid to use parallelogram coordinates instead of odd-r offset coordinates for simpler neighbor calculation
- Update spatial visualization to support parallelogram grid layout with continuous diagonal offset
- Improve colorbar layout: change to horizontal orientation at bottom to avoid overlap with landscape
- Implement dynamic colorbar range adjustment: only update when current max exceeds 110% of historical max
- Enhance user experience: clicking deme no longer automatically switches to selected deme page
- Update spatial dashboard with improved layout and stable visualization ranges

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
