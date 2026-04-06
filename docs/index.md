# NATAL Core Documentation

**N**umba-**A**ccelerated **T**oolkit for **A**nalysis of **L**ifecycles.

[![GitHub](https://img.shields.io/github/v/release/jyzhu-pointless/natal-core?label=GitHub&color=purple)](https://github.com/jyzhu-pointless/natal-core/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/natal-core.svg?label=PyPI&color=yellow)](https://pypi.org/project/natal-core/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0+-green.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/Numba-0.60.0+-orange.svg)](https://numba.pydata.org/)
[![Docs](https://img.shields.io/readthedocs/natal-core?label=docs)](https://natal-core.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/jyzhu-pointless/natal-core/blob/main/LICENSE)

![NATAL logo](https://raw.githubusercontent.com/jyzhu-pointless/natal-core/main/natal-brand.svg)

NATAL Core is a Python toolkit for simulating population genetic dynamics, with core computational paths accelerated by Numba.

## Key Features

- **🧬 Genetic Architecture:** Support for defining a complete genome–chromosome–locus–gene hierarchical system.
- **🪲 Population Models:** Age‑structured and discrete‑generation populations, supporting various ecological and genetic parameters.
- **🌐 Spatial Simulation:** Spatially‑explicit population simulation using hexagonal and square grids for subpopulations.
- **🚀 High‑Performance Kernels:** Numba JIT‑accelerated simulation kernels, with built‑in parallelism for spatial models.
- **🧩 Extensibility:** Modifier, Preset, and Hook systems for flexible extension; built‑in common gene drives.
- **🔍 Observation & Filtering:** Flexible pattern‑matching filters for defining Presets and observation rules.

## Installation

### 1. Create and activate a virtual environment

It is recommended to use a virtual environment to manage dependencies. Choose one of the following commands.

```bash
# uv (recommended)
uv venv --python 3.12 .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

```bash
# venv
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

```bash
# conda
conda create -n natal-env python=3.12
conda activate natal-env
```

### 2. Install NATAL Core

```bash
uv pip install natal-core
# or
pip install natal-core
```

## Learning Path (in recommended order)

1. [Quick Start: 15 Minutes to NATAL](quickstart.md)
2. [Genetic Structures and Entities](genetic_structures.md)
3. [Builder System Explained](builder_system.md)
4. [IndexRegistry Indexing Mechanism](index_registry.md)
5. [PopulationState and PopulationConfig](population_state_config.md)
6. [Deep Dive into Simulation Kernels](simulation_kernels.md)
7. [Numba Optimization Guide](numba_optimization.md)
8. [Modifier Mechanism](modifiers.md)
9. [Hook System](hooks.md)
10. [Genetic Presets Usage Guide](genetic_presets.md)
11. [Spatial Simulation Guide](spatial_simulation_guide.md)
12. [Population Observation Rules](observation_rules.md)
13. [Pattern Matching and Extensible Configuration](genotype_patterns.md)
14. [Design Your Own Preset (1): Starting from Allele Conversion Rules](allele_conversion_rules.md)
15. [Design Your Own Preset (2): Using genotype_filter to Control Rule Scope](genotype_filter.md)
16. [Design Your Own Preset (3): Encapsulation, Validation, and Pre‑release Checks](preset_encapsulation_and_validation.md)

## API Documentation

- [Genetic Structures API](api/genetic_structures.md)
- [Population Builder API](api/population_builder.md)
- [Simulation Kernels API](api/simulation_kernels.md)
- [Full API Index](api/index.md)

## Source Repository

- [GitHub: jyzhu-pointless/natal-core](https://github.com/jyzhu-pointless/natal-core)
