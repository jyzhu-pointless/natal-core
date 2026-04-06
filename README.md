# ⚡️ NATAL Core

**N**umba-**A**ccelerated **T**oolkit for **A**nalysis of **L**ifecycles.

[![GitHub](https://img.shields.io/github/v/release/jyzhu-pointless/natal-core?label=GitHub&color=purple)](https://github.com/jyzhu-pointless/natal-core/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/natal-core.svg?label=PyPI&color=yellow)](https://pypi.org/project/natal-core/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.0+-green.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/Numba-0.60.0+-orange.svg)](https://numba.pydata.org/)
[![Docs](https://img.shields.io/readthedocs/natal-core?label=docs)](https://natal-core.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/jyzhu-pointless/natal-core/blob/main/LICENSE)

![NATAL logo](https://raw.githubusercontent.com/jyzhu-pointless/natal-core/main/natal-brand.svg)

**NATAL Core** is a population genetics simulation engine focused on lifecycle-aware modeling. It supports age-structured and discrete-generation populations, sperm storage, gene drive presets, hook-based interventions, and high-performance Numba kernels.

NATAL Core is part of the NATAL project. The full project also includes **NATAL Inferencer**, a toolkit for parameter inference in population genetics models based on NATAL Core.

## Key Features

- 🪲 Lifecycle-aware population modeling (age-structured and discrete-generation).
- 🧬 Genetic architecture definition with chromosomes, loci, and alleles.
- 🚀 Numba-accelerated kernels for high performance.
- 🧩 Built-in genetic presets (for example, homing drives and toxin-antidote drives).
- 🪝 Hook system for custom interventions during simulation.
- 🔍 Observation and filtering utilities for downstream analysis.
- 🌐 Spatial simulation support across multiple demes.

## Installation

```bash
pip install natal-core
```

For development:

```bash
git clone https://github.com/jyzhu-pointless/natal-core.git
cd natal-core
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```python
import natal as nt

species = nt.Species.from_dict(
	name="DemoSpecies",
	structure={"chr1": {"A": ["WT", "Drive"]}},
)

pop = (
	nt.DiscreteGenerationPopulation
	.setup(species=species, stochastic=False)
	.initial_state(individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
	.survival(female_age0_survival=1.0, male_age0_survival=1.0)
	.reproduction(eggs_per_female=10)
	.competition(low_density_growth_rate=2.0, carrying_capacity=1000)
	.build()
)

pop.run(n_steps=10, record_every=1)
print(nt.population_to_readable_dict(pop)["tick"])
```

## Documentation and Links

- Documentation index: https://natal-core.readthedocs.io/en/latest/
- Quickstart guide: https://natal-core.readthedocs.io/en/latest/quickstart/
- API reference overview: https://natal-core.readthedocs.io/en/latest/api/
- PyPI package: https://pypi.org/project/natal-core/
- Source code: https://github.com/jyzhu-pointless/natal-core
- Issue tracker: https://github.com/jyzhu-pointless/natal-core/issues
- Changelog: https://github.com/jyzhu-pointless/natal-core/blob/main/CHANGELOG.md

## Demos

- Discrete model demo: https://github.com/jyzhu-pointless/natal-core/blob/main/demos/discrete.py
- Age-structured demo: https://github.com/jyzhu-pointless/natal-core/blob/main/demos/mosquito.py
- Spatial demo: https://github.com/jyzhu-pointless/natal-core/blob/main/demos/spatial.py

## License

This project is licensed under the MIT License.
