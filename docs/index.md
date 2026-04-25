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

**NATAL Core** is a high-performance forward-time population genetics simulation engine that supports configurable lifecycles of species. It supports age-structured and discrete-generation populations, sperm storage, genetic presets, hook-based interventions, and simulation kernels accelerated by Numba. NATAL Core is especially useful for **modeling gene drive systems** in insect populations, but its flexible architecture allows it to be applied to a wide range of population genetics scenarios.

NATAL Core is part of the NATAL project. The full project also includes **NATAL Inferencer**, a toolkit for parameter inference in population genetics models based on NATAL Core.

## Key Features

- 🪲 Forward-time configurable population modeling (age-structured and discrete-generation).
- 🧬 Genetic architecture definition with chromosomes, loci, and alleles.
- 🚀 Numba-accelerated kernels for high performance.
- 🧩 Built-in genetic presets, especially for homing drives and toxin-antidote drives.
- 🪝 Hook system for custom interventions during simulation.
- 🔍 Observation and filtering utilities for downstream analysis.
- 🗺️ Spatial simulation support across multiple demes.

## Installation

### 1. Create a virtual environment

It is strongly recommended to use a virtual environment to manage dependencies.

Choose one of the following commands. **Python 3.12** is recommended, but any Python version >= 3.9 should work.

```bash
uv venv --python 3.12 .venv            # uv (recommended)
python -m venv .venv                   # venv (please ensure Python >= 3.9 is used)
conda create -n natal-env python=3.12  # conda
```

On Windows, you can run `py -3.12 -m venv .venv` to specify Python 3.12 as the interpreter for the virtual environment.

### 2. Activate the virtual environment

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

# 1. Define the genetics architecture of a species
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

# 4. Build a panmictic population
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

# 5. Launch interactive WebUI and run simulation
launch(pop)
```

For more ready-to-run examples, see the [demos](https://github.com/jyzhu-pointless/natal-core/tree/main/demos) directory in the GitHub repository.

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

- [Full API Index](api/index.md)

## Links

- GitHub repository: https://github.com/jyzhu-pointless/natal-core
- PyPI package: https://pypi.org/project/natal-core/

## License

This project is licensed under the MIT License.
