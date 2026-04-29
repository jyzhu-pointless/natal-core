# NATAL Core Documentation

**N**umba-**A**ccelerated **T**oolkit for **A**nalysis of **L**ifecycles

[![GitHub](https://img.shields.io/github/v/release/jyzhu-pointless/natal-core?label=GitHub&color=purple)](https://github.com/jyzhu-pointless/natal-core/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/natal-core.svg?label=PyPI&color=yellow)](https://pypi.org/project/natal-core/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0+-green.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/Numba-0.60.0+-orange.svg)](https://numba.pydata.org/)
[![Docs](https://img.shields.io/readthedocs/natal-core?label=docs)](https://natal-core.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/jyzhu-pointless/natal-core/blob/main/LICENSE)

![NATAL logo](https://raw.githubusercontent.com/jyzhu-pointless/natal-core/main/natal-brand.svg)

**NATAL Core** is a high-performance forward-time population genetics simulation engine with configurable species lifecycles. It supports age-structured and discrete-generation populations, sperm storage, genetic presets, hook-based interventions, and Numba-accelerated simulation kernels. NATAL Core is especially suited for **modeling gene drive systems in insect populations**, but its flexible architecture also makes it applicable to a wide range of population genetics scenarios.

NATAL Core is part of the NATAL project. The full project also includes **NATAL Inferencer**, a toolkit for inferring population genetics model parameters based on NATAL Core.

## Key Features

- 🪲 Forward-time simulation with flexible population lifecycles (age-structured and discrete-generation populations)
- 🧬 Definable genetic structures including chromosomes, loci, and alleles
- 🚀 Numba-accelerated computation core with excellent performance
- 🧩 Built-in gene drive presets, especially homing drive and toxin-antidote drive
- 🪝 Hook system for inserting custom intervention logic during simulation
- 🔍 Observation and filtering tools for downstream analysis
- 🗺️ Multi-deme (subpopulation) spatial simulation support

## Installation

### 1. Create a Virtual Environment

It is strongly recommended to use a virtual environment to manage dependencies.

Please choose one of the following commands. **Python 3.12 is recommended**, but any Python version >= 3.9 should work.

```bash
uv venv --python 3.12 .venv            # uv (recommended)
python -m venv .venv                   # venv (please ensure Python >= 3.9)
conda create -n natal-env python=3.12  # conda
```

On Windows, you can run `py -3.12 -m venv .venv` to specify Python 3.12 as the interpreter for the virtual environment.

### 2. Activate the Virtual Environment

Linux / macOS:

```bash
source .venv/bin/activate    # uv / venv
conda activate natal-env     # conda
```

Windows:

```powershell
.venv\Scripts\activate       # uv / venv
conda activate natal-env     # conda
```

### 3. Install NATAL Core

```bash
uv pip install natal-core
# or
pip install natal-core
```

## A Minimal Example

```python
import natal as nt
from natal.ui import launch

# 1. Define the species' genetic architecture
sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {"loc1": ["WT", "Dr", "R2", "R1"]}
    },
    gamete_labels=["default", "cas9_deposited"]
)

# 2. Define a drive using built-in presets
drive = nt.HomingDrive(
    name="TestHoming",
    drive_allele="Dr",
    cas9_allele="Dr",
    target_allele="WT",
    resistance_allele="R2",
    functional_resistance_allele="R1",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.9,
    functional_resistance_ratio=0.001,
    embryo_resistance_formation_rate=0.0,
    viability_scaling=1.0,
    fecundity_scaling={"female": 0.0},
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9_deposited"
)

# 3. Define a release event using hooks
@nt.hook(event="first", priority=0)
def release_drive_carriers():
    return [
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=500, when="tick == 10")
    ]

# 4. Build a random-mating population
pop = (nt.DiscreteGenerationPopulation
    .setup(
        species=sp,
        name="TestPop",
        stochastic=True
    )
    .initial_state(
        individual_count={
        "male": {"WT|WT": 50000}, "female": {"WT|WT": 50000}
        }
    )
    .reproduction(
        eggs_per_female=100
    )
    .competition(
        low_density_growth_rate=6.0,
        carrying_capacity=100000,
        juvenile_growth_mode="concave"
    )
    .presets(drive).hooks(release_drive_carriers).build())

# 5. Launch the interactive WebUI and run the simulation
launch(pop)
```

For more ready-to-run examples, see the [demos](https://github.com/jyzhu-pointless/natal-core/tree/main/demos) directory in the GitHub repository.

## Documentation Index

It is recommended to start with Part 1 to get up to speed, then use Part 2 as a project-driven reference, and selectively read Parts 3 and 4 as needed.

### Part 1: Quick Start

> This section introduces the basic concepts and usage of NATAL Core, helping you get started quickly.

1. [Quick Start: NATAL in 15 Minutes](1_quickstart.md)

### Part 2: Practical Components

> This section introduces the main components of NATAL Core, which are the primary features used in daily work.

2. [Genetic Structures and Entities](2_genetics.md)
3. [Population Initialization](2_population_initialization.md)
4. [Random-Mating Population](2_population.md)
5. [Genetic Presets Usage Guide](2_genetic_presets.md)
6. [Hook System](2_hooks.md)
7. [Pattern Matching and Extensible Configuration](2_genotype_patterns.md)
8. [Extracting Population Simulation Data](2_data_output.md)

### Part 3: Advanced Guide

> This section introduces advanced features of NATAL Core, including spatial simulation and more custom configuration.

9. [Spatial Simulation Guide](3_spatial_simulation.md)
10. [Designing Your Own Presets](3_custom_presets.md)
11. [Modifier Mechanism](3_modifiers.md)
12. [Advanced Hook Tutorial](3_advanced_hooks.md)

### Part 4: Internal Implementation

> This section introduces the underlying implementation mechanisms of NATAL Core that are not directly user-facing, helping you understand how NATAL Core works internally.

<!--TODO: numba related content can be moved forward as appropriate-->

13. [IndexRegistry Indexing Mechanism](4_index_registry.md)
14. [PopulationState and PopulationConfig](4_population_state_config.md)
15. [Simulation Kernels in Depth](4_simulation_kernels.md)
16. [Numba Optimization Guide](4_numba_optimization.md)
17. [Observation History Recording Implementation](observation_impl.md)

## API Documentation

- [Complete API Index](api/index.md)

## Links

- GitHub Repository: https://github.com/jyzhu-pointless/natal-core
- PyPI Package: https://pypi.org/project/natal-core/

## License

This project is licensed under the MIT License.
