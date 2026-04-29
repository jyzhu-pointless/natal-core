# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

默认使用中文回答。仅在用户明确使用英文提问时用英文回复。

## Commands

The virtual environment is auto-activated for Claude Code agents. Run commands directly — do not prepend `source .venv/bin/activate`.

```bash
# Run all tests
pytest

# Type checking (strict mode)
pyright

# Lint — check and optionally auto-fix
ruff check src demos
ruff check src demos --fix

# If public API exports changed, regenerate stub:
python scripts/generate_init_pyi.py
```

## Validation gates

Before committing any code change, run **all three** of `pytest`, `pyright`, and `ruff check src demos`. Fix **every** issue in modified files before committing — no suppression shortcuts.

### Fix-everything policy

- **Modified files**: All pyright / ruff / pytest failures must be fixed, regardless of whether they pre-existed.
- **Other files affected by the change**: If a signature or import change causes failures elsewhere, those must be fixed too.
- **Pre-existing issues in untouched files**: Explicitly note and analyse them; fixing is encouraged but not required for the current commit.
- **`cast(Any, …)` is forbidden**. Never use it to bypass type checking.
- **`Any` in function parameter lists is forbidden** unless accompanied by a concrete, documented justification.
- **`cast(T, x)`** may be used only when static analysis cannot prove `x: T` at all (e.g., narrowing an `Optional` after a guard) and the error is otherwise unavoidable. Prefer type-narrowing assertions or restructuring before reaching for `cast`.
- **`# type: ignore`** is a last resort. Every ignore must include a short, specific reason on the same line.

### Test coverage

- **New modules**: ≥95% line coverage.
- **New code in existing modules**: ≥95% line coverage.
- **Deterministic simulations** (`stochastic=False`): require exact numerical assertions on counts, frequencies, or derived statistics.
- **Stochastic simulations**: require statistical validation — multiple runs, confidence intervals, or distributional checks. A single passing run is not sufficient.

## Architecture overview

### Package structure

The `natal` package uses **lazy loading** — child modules are never imported at `import natal` time. Instead, `natal/__init__.py` parses each module's `__all__` via AST and builds a name-to-module map. Accessing `natal.SomeSymbol` triggers `__getattr__`, which dynamically imports the owning module. This keeps startup nearly instant.

Key modules and their responsibilities:

```
natal/
├── __init__.py              # Lazy-loading package root
├── type_def.py              # Core type aliases (Sex, Age, IndividualType, GameteType)
├── genetic_structures.py    # Immutable species architecture: Species, Chromosome, Locus
├── genetic_entities.py      # Mutable biological entities (Gene, Genotype, Haplotype, etc.)
├── genetic_presets.py       # Gene drive presets: HomingDrive, ToxinAntidoteDrive
├── genetic_patterns.py      # Genotype/pattern matching and string-based genotype DSL
├── index_registry.py        # Lookup tables for genotypes, haplotypes, gamete labels
├── population_config.py     # PopulationConfig NamedTuple — static simulation parameters
├── population_state.py      # Mutable simulation state (individual counts, gamete pools)
├── population_builder.py    # Fluent builder API for constructing populations
├── base_population.py       # Abstract base for all population models (hook mgmt, modifiers)
├── discrete_generation_population.py  # Wright-Fisher discrete-generation model
├── age_structured_population.py       # Age-structured (non-Wright-Fisher) model
├── spatial_population.py             # Multi-deme spatial simulation
├── spatial_topology.py               # Deme adjacency and migration topology
├── state_translation.py              # Exporting state to history / DataFrame
├── observation.py                    # Simplified observation/filter helpers
├── algorithms.py                     # Core algorithms (competition, sampling, etc.)
├── modifiers.py                      # GameteModifier / ZygoteModifier pipelines
├── gamete_allele_conversion.py       # Haploid-stage allele conversion rules
├── zygote_allele_conversion.py       # Diploid-stage allele conversion rules
├── helpers.py                        # Shared utility functions
├── visualization.py                  # Plotting utilities
├── numba_utils.py                    # Numba enable/disable and cache dir
├── numba_compat.py                   # Numba compatibility shims
├── hooks/                            # Event-driven hook system
│   ├── compiler.py                   # Hook compilation (njit path)
│   ├── declarative.py                # Declarative hooks (Op.add, Op.remove, etc.)
│   ├── executor.py                   # Hook execution engine (CSR-based event arrays)
│   ├── selector.py                   # Deme-aware hook selectors
│   └── types.py                      # Hook enums, opcodes, data structures
├── kernels/                          # Numba-accelerated simulation kernels
│   ├── codegen.py                    # Dynamic kernel wrapper code generation
│   ├── simulation_kernels.py         # Core panmictic simulation loops
│   ├── spatial_simulation_kernels.py # Per-deme simulation for spatial models
│   ├── spatial_migration_kernels.py  # Migration between demes
│   ├── migration/
│   │   ├── adjacency.py              # Adjacency matrix helpers
│   │   └── kernel.py                 # Migration kernel
│   └── templates/                    # Jinja2-style templates for wrapper codegen
├── ui/                               # NiceGUI-based interactive dashboards
│   ├── dashboard.py                  # Panmictic population dashboard
│   ├── dashboard_population.py       # Population display components
│   └── spatial_dashboard.py          # Spatial simulation dashboard
└── hook_dsl.py                       # Legacy compatibility shim (star re-exports)
```

### Key architectural layers

1. **Genetic structure layer** (`genetic_structures.py`, `genetic_entities.py`): Defines the immutable species blueprint (chromosomes, loci, alleles) and concrete entity instances (genes, genotypes, haplotypes). Entities auto-register to structures. The `Species` class provides string-based genotype pattern resolution (e.g., `"WT|Dr"` → `Genotype`).

2. **Configuration & state layer** (`population_config.py`, `population_state.py`): Separates static configuration (`PopulationConfig` NamedTuple — fitness arrays, rates, carrying capacities) from mutable simulation state (`PopulationState` / `DiscretePopulationState` — individual count arrays, gamete pools). Config scalars are immutable (rebuild via `_replace`); NumPy arrays are mutable in-place.

3. **Population model layer** (`base_population.py`, `discrete_generation_population.py`, `age_structured_population.py`, `spatial_population.py`): Concrete population models implementing the simulation loop. The **fluent builder pattern** in `population_builder.py` provides `.setup().initial_state().reproduction().competition().presets().hooks().build()` chaining for construction. Two main models:
   - **DiscreteGenerationPopulation** — Wright-Fisher, non-overlapping generations
   - **AgeStructuredPopulation** — overlapping generations with age classes
   - **SpatialPopulation** — multi-deme, wraps either model per deme with migration

4. **Hook system** (`hooks/`): Event-driven intervention system with five events: `initialization`, `first`, `early`, `late`, `finish`. Hooks can be Python callables (for flexibility) or compiled to Numba njit kernels (for performance). The compiler path uses CSR (Compressed Sparse Row) event arrays and code generation from templates (`kernels/codegen.py`). Two hook styles:
   - **Declarative hooks**: `Op.add(genotypes="...", ages=..., sex="...", delta=..., when="tick == N")`
   - **Python function hooks**: Decorated with `@hook(event="early", priority=0)` to return Op lists or raw operations

5. **Kernel layer** (`kernels/`): Numba-accelerated simulation kernels for the hot loop. Custom codegen (`codegen.py`) dynamically generates optimized wrapper modules from Jinja2 templates, combining user hooks into compiled njit functions. Shared kernel source files handle both panmictic and spatial simulations.

6. **Genetic presets** (`genetic_presets.py`): `HomingDrive` and `ToxinAntidoteDrive` presets that encapsulate gamete modifiers, zygote modifiers, and fitness effects. The `apply_preset_to_population()` function wires them into a population.

### Data flow

```
Species definition → IndexRegistry (all genotype/haplotype/glab mappings)
         ↓
PopulationBuilder → PopulationConfig (static tensors) + PopulationState (mutable arrays)
         ↓
Simulation loop (per tick):
  → Hook events (first/early/late) — modify state via Op arrays or Python
  → Reproduction (gametogenesis → mating → fertilization)
  → Competition / density regulation
  → Survival / aging (age-structured only)
  → Observation recording
  → For spatial: per-deme simulation → migration between demes
```

### Key design decisions

- **Lazy loading**: `natal` package never imports child modules at init. The `__init__.py` builds a name index via AST parsing, then `__getattr__` imports on first access.
- **Numba-first but Python-fallback**: Core simulation can run with or without Numba (controlled by `numba_utils.is_numba_enabled()`). The hook compiler has both njit and pure-Python paths.
- **Fitness is multi-layered**: `viability_fitness`, `fecundity_fitness`, `sexual_selection_fitness`, and `zygote_viability_fitness` are applied at different stages of the life cycle.
- **Builder pattern**: Population construction uses a multi-stage builder with dedicated methods for each configuration domain (initial state, reproduction, competition, presets, hooks).
- **Codegen caching**: Generated kernel wrappers are cached in a Numba cache directory with content-addressed hashing, so repeated simulations reuse compiled modules.
