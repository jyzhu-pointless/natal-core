# ⚡️ NATAL Core

**N**umerical **A**ggregation **T**oolkit for **A**nalysis of **L**ifecycles

[![GitHub](https://img.shields.io/github/v/release/jyzhu-pointless/natal-core?label=GitHub&color=purple)](https://github.com/jyzhu-pointless/natal-core/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/natal-core.svg?label=PyPI&color=yellow)](https://pypi.org/project/natal-core/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0+-green.svg)](https://numpy.org/)
[![Docs](https://img.shields.io/readthedocs/natal-core?label=docs)](https://natal-core.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/jyzhu-pointless/natal-core/blob/main/LICENSE)

![NATAL logo](https://raw.githubusercontent.com/jyzhu-pointless/natal-core/main/natal-brand.svg)

**NATAL Core** is a high-performance forward-time population genetics simulation engine that supports configurable lifecycles of species. Rather than tracking individuals one by one, it adopts a **numerical aggregation** approach: individuals are grouped by age, sex, and genotype, their dynamics computed mathematically, and randomness introduced through probability distributions — avoiding the overhead of per-individual iteration. It supports age-structured and discrete-generation populations, sperm storage, genetic presets, hook-based interventions, and a Rust-native computation core as its only execution engine. NATAL Core is especially useful for **modeling gene drive systems** in insect populations, but its flexible architecture allows it to be applied to a wide range of population genetics scenarios.

NATAL Core is part of the NATAL project. The full project also includes **NATAL Inferencer**, a toolkit for parameter inference in population genetics models based on NATAL Core.

## Key Features

- 🪲 Forward-time configurable population modeling (age-structured and discrete-generation).
- 🧬 Genetic architecture definition with chromosomes, loci, and alleles.
- 🚀 Numerical aggregation engine, Rust-native.
- 🧩 Built-in genetic presets, especially for homing drives and toxin-antidote drives.
- 🪝 Hook system for custom interventions during simulation.
- 🔍 Observation and filtering utilities for downstream analysis.
- 🗺️ Spatial simulation support across multiple demes.

## Installation

### 1. Create a virtual environment

It is strongly recommended to use a virtual environment to manage dependencies.

Choose one of the following commands. **Python 3.12** is recommended, but any Python version >= 3.10 should work.

```bash
uv venv --python 3.12 .venv            # uv (recommended)
python -m venv .venv                   # venv (please ensure Python >= 3.10 is used)
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
from natal import launch_vue

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

# 3. Build a panmictic population and declare the release event with an Op
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
        juvenile_growth_mode="beverton_holt"
    )
    .presets(drive)
    .hooks(
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=500, when="tick == 10", event="first", priority=0),
    )
    .build())

# 4. Launch interactive WebUI and run simulation
launch_vue(pop)
```

For more ready-to-run examples, see the [demos](https://github.com/jyzhu-pointless/natal-core/tree/main/demos) directory in the GitHub repository.

## Development Checks

Use Python 3.10 or later, a Rust toolchain with rustfmt/clippy, Node.js 24 with Corepack, and a virtual environment:

```bash
python -m pip install -e ".[dev]"
python scripts/ci_full.py
```

Local checks and GitHub Actions share this entry point. Use `--only lint types stubs`,
`--only tests`, `--only baseline`, or `--only rust` to run selected stages.
The default run also builds a fresh release wheel and installs it in a temporary,
isolated environment to run the complex genetic, spatial, and runtime-update tests,
and request the bundled dashboard HTML, assets, and API. End users installing a
release wheel do not need Node.js; it is required only to build the dashboard.
This requires access to the package index for dependencies. Passing locally verifies
the current environment; GitHub checks Python 3.10–3.13 across the supported wheel platforms.
See [development and release checks](docs/en/index.md#development-and-release-checks) for details.

## Documentation and Links

- Documentation (English): https://natal-core.readthedocs.io/en/latest/
- 文档 (中文): https://natal-core.readthedocs.io/zh-cn/latest/
- GitHub repository: https://github.com/jyzhu-pointless/natal-core
- PyPI package: https://pypi.org/project/natal-core/

## License

This project is licensed under the MIT License.
