# NATAL Core Documentation

**N**umerical **A**ggregation **T**oolkit for **A**nalysis of **L**ifecycles

[![GitHub](https://img.shields.io/github/v/release/jyzhu-pointless/natal-core?label=GitHub&color=purple)](https://github.com/jyzhu-pointless/natal-core/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/natal-core.svg?label=PyPI&color=yellow)](https://pypi.org/project/natal-core/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0+-green.svg)](https://numpy.org/)
[![Rust](https://img.shields.io/badge/engine-Rust-red.svg)](https://www.rust-lang.org/)
[![Docs](https://img.shields.io/readthedocs/natal-core?label=docs)](https://natal-core.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/jyzhu-pointless/natal-core/blob/main/LICENSE)

![NATAL logo](https://raw.githubusercontent.com/jyzhu-pointless/natal-core/main/natal-brand.svg)

**NATAL Core** is a high-performance forward-time population genetics simulation engine with configurable species lifecycles. Rather than tracking individuals one by one, it adopts a **numerical aggregation** approach: individuals are grouped by age, sex, and genotype, their dynamics computed mathematically, and randomness introduced through probability distributions — avoiding the overhead of per-individual iteration. It supports age-structured and discrete-generation populations, sperm storage, genetic presets, hook-based interventions, and a Rust-native computation core as its only execution engine. NATAL Core is especially suited for **modeling gene drive systems in insect populations**, but its flexible architecture also makes it applicable to a wide range of population genetics scenarios.

NATAL Core is part of the NATAL project. The full project also includes **NATAL Inferencer**, a toolkit for inferring population genetics model parameters based on NATAL Core.

## Key Features

- 🪲 Forward-time simulation with flexible population lifecycles (age-structured and discrete-generation populations)
- 🧬 Definable genetic structures including chromosomes, loci, and alleles
- 🚀 Numerical aggregation engine, Rust-native accelerated.
- 🧩 Built-in gene drive presets, especially homing drive and toxin-antidote drive
- 🪝 Hook system for inserting custom intervention logic during simulation
- 🔍 Observation and filtering tools for downstream analysis
- 🗺️ Multi-deme (subpopulation) spatial simulation support

## Installation

### 1. Create a Virtual Environment

It is strongly recommended to use a virtual environment to manage dependencies.

Please choose one of the following commands. **Python 3.12 is recommended**, but any Python version >= 3.10 should work.

```bash
uv venv --python 3.12 .venv            # uv (recommended)
python -m venv .venv                   # venv (please ensure Python >= 3.10)
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
from natal import launch_vue

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

# 3. Build a random-mating population and declare the release event with an Op
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
        nt.Op.add(genotypes="WT|Dr", ages=1, sex="male", delta=500, when="tick == 10"),
        event="first",
    )
    .build())

# 4. Launch the interactive WebUI and run the simulation
launch_vue(pop)
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
10. [Runtime Parameter Modification](3_runtime_modification.md)
11. [Designing Your Own Presets](3_custom_presets.md)
12. [Modifier Mechanism](3_modifiers.md)
13. [Advanced Hook Tutorial](3_advanced_hooks.md)

### Part 4: Internal Implementation

> This section introduces the underlying implementation mechanisms of NATAL Core that are not directly user-facing, helping you understand how NATAL Core works internally.


14. [IndexRegistry Indexing Mechanism](4_index_registry.md)
15. [PopulationState and ModelDraft](4_population_state_config.md)
16. [the Simulation Engine in Depth](4_simulation_engine.md)
17. [Observation History Recording Implementation](observation_impl.md)

## API Documentation

- [Complete API Index](api/index.md)

## Development and Release Checks

Run checks from the repository root in a development virtual environment with
Python 3.10 or later and Rust (including rustfmt and clippy):

```bash
python -m pip install -e ".[dev]"
python scripts/ci_full.py
```

The default run checks Ruff, Pyright, the generated public stub, Python tests,
numerical baselines, Rust checks/tests, and a freshly built wheel. Select stages with:

```bash
python scripts/ci_full.py --only lint types stubs
python scripts/ci_full.py --only tests
python scripts/ci_full.py --only baseline
python scripts/ci_full.py --only rust
python scripts/ci_full.py --only wheel
```

Unknown stages and combinations of `--only` with the older `--skip-*` options
are errors. A failed required stage stops execution with a nonzero exit code.
Stub checking never rewrites the file; use `python scripts/generate_init_pyi.py`
to regenerate it after an intentional export change.

Wheel builds require Node.js 24 and Corepack. `python scripts/build_frontend.py`
installs locked frontend dependencies, runs lint and tests, and builds the dashboard
into `src/natal/frontend/webui/dist`. The wheel builder calls this automatically;
CI builds it once and shares the assets across the wheel matrix. Installed wheels
include these assets, so end users do not need Node.js.

Each local wheel build uses the current interpreter and a new output directory
under `rust/target/wheels`. `python scripts/build_rust_wheel.py --out PATH`
requires a directory that does not yet exist. Both local builds and CI use
`python scripts/verify_wheel.py --wheel-dir PATH` to verify exactly one wheel:
its package name, version, interpreter/platform compatibility, metadata, and native
extension must match. Verification installs that exact wheel in a fresh temporary
virtual environment outside the checkout, checks import locations and versions,
and runs the existing complex genetics, spatial population, and runtime-update
E2E tests. It also verifies dashboard HTML, linked JavaScript/CSS, and the API through
HTTP requests against the installed application. It needs package-index access to install dependencies; it does not reuse
an editable installation or the repository's pytest path settings.

GitHub Actions calls the same check stages. Full Python tests run on Linux with
Python 3.10, 3.11, 3.12, and 3.13. The baseline job uses Python 3.13 on Ubuntu 24.04
and dependencies from `uv.lock`; it checks existing digests without updating them.
To reproduce that dependency environment, use `uv sync --locked --python 3.13`
in a separate checkout/environment, then
`uv run --no-sync --python 3.13 python scripts/ci_full.py --only baseline`.
Local baseline checks use the current environment, so record its versions when
investigating a digest difference. Updating a digest requires explaining the
scientific change; it is not a way to bypass a failed check.

The reusable wheel workflow builds and tests all 20 combinations of Python
3.10–3.13 and Linux x86_64/ARM64, macOS Intel/ARM64, and Windows x86_64.
The `ci-success` job succeeds only when all required jobs succeed, including wheel
checks. Configure it as a required status check on `main` after the new workflow
has run on GitHub; editing the YAML does not configure branch protection.

The `wheels` release workflow reuses the complete CI workflow for the selected
commit and uploads those same verified wheel artifacts. It does not rebuild at
publication time. A `v*` tag must match both `pyproject.toml` and
`natal.__version__` (with standard version normalization). Manual runs default to
`dry_run: true`, including when a tag is selected. Publication requires a version
tag and either a tag push or an explicit manual run with `dry_run: false`.
A branch run never publishes. PyPI trusted publishing must be configured for the
`wheels.yml` workflow and its `pypi` environment. A local pass does not verify
GitHub-hosted runners or PyPI permissions.

## Links

- GitHub Repository: https://github.com/jyzhu-pointless/natal-core
- PyPI Package: https://pypi.org/project/natal-core/

## License

This project is licensed under the MIT License.
