# NATAL Core Documentation

**N**umba-**A**ccelerated **T**oolkit for **A**nalysis of **L**ifecycles

Welcome to the NATAL Core documentation! This is a Python toolkit for simulating population genetic dynamics with Numba acceleration.

## Quick Start

```python
import natal

# Create a species with genetic architecture
species = natal.Species("mosquito")

# Build and run a population simulation
pop = natal.AgeStructuredPopulation.builder(species, ...)
pop.run(100)
```

## Features

- 🧬 **Genetic Architecture**: Define species with chromosomes, loci, and alleles
- 📊 **Population Models**: Age-structured and discrete generation populations
- ⚡ **Numba Acceleration**: High-performance simulation kernels
- 🔧 **Extensible**: Hook system for custom modifications
- 📈 **Visualization**: Built-in plotting and analysis tools

## Installation

```bash
pip install natal-core
```

## Documentation

- [API Reference](api/genetic_structures.md) - Complete API documentation

For detailed user guides and tutorials, please refer to the Markdown files in the `docs/` directory.

## Source Code

Visit the [GitHub repository](https://github.com/jyzhu-pointless/natal-core) for source code, issues, and contributions.