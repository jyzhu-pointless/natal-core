# Developer guide

This guide is for developers who need to understand, maintain, or extend NATAL Core. Start by building and running a population with the [quickstart](../1_quickstart.md). This guide follows that operation down into arrays, engine sessions, and numerical calculations.

The guide explains collaboration and constraints in the current implementation. Complete parameter lists belong in the [API reference](../api/index.md). Private functions mentioned here are source-reading entry points, not promises of public API stability.

## Start with the two sample chapters

Read [A model's complete journey from declaration to results](model_journey.md) to follow one model through type catalogs, probability tables, its runtime session, and outputs. Then read [How one reproduction stage is calculated](reproduction.md) to derive pair counts, eggs, inheritance allocation, and stage-specific count changes.

Both chapters aim to let you understand behavior and evaluate agent proposals without first reading the source. They include complete numerical cases and acceptance questions that distinguish implementations. The topic pages below remain supplementary source guides; not all have yet reached the sample chapters' depth.

## Reading route

| Order | Chapter | Question answered |
| --- | --- | --- |
| 1 | [Architecture and data flow](architecture.md) | Who owns declarations, runtime state, and records? |
| 2 | [Model compilation and index publication](model.md) | Why must genetic tables, initial state, and indices be projected together? |
| 3 | [Sessions and a single tick](runtime.md) | How does execution advance, and where does failure leave it? |
| 4 | [Numerical algorithms](algorithms.md) | How do mathematical quantities map to array axes and kernel functions? |
| 5 | [Hooks and controlled updates](hooks.md) | When do writes become visible, and what can failure undo? |
| 6 | [Spatial execution and migration](spatial.md) | How do demes share layouts, parameters, and a timeline? |
| 7 | [Observation, history, and checkpoints](output.md) | How are results recorded, and which records support restoration? |
| 8 | [Making and validating changes](development.md) | Which neighboring modules need attention when a mechanism changes? |

The main route uses an ordinary staged discrete-generation model: a complete species declaration is compiled and published into a Rust session; age-1 adults produce age-0 offspring, survival acts on those offspring, and aging makes them the next adults. Results are read at session boundaries. Age-structured, spatial, and fused Wright–Fisher paths are distinguished in the relevant chapters.

## Reading alongside the source

Follow the narrative examples through inputs, intermediate state, and results. Source and test links let you or an agent verify implementation when needed; understanding the sample chapters does not require reading the source first. Source and test links target repository `main`. When investigating an older version, use documentation, code, and tests from the same commit.

Each chapter's verification entry points identify behavior protected by existing tests. They do not imply that these tests run automatically whenever documentation is read or built. Function names, axes, and boundary conditions locate concrete facts; explanatory calculations are not runnable Python examples.

The existing user documentation retains its internal-implementation and appendix entries. This guide supplies a continuous developer reading route; historical design pages provide background on earlier implementations.
