# Developer guide

This guide is for developers who need to understand, maintain, or extend NATAL Core. Start by building and running a population with the [quickstart](../1_quickstart.md). This guide follows that operation down into arrays, engine sessions, and numerical calculations.

The guide explains collaboration and constraints in the current implementation. Complete parameter lists belong in the [API reference](../api/index.md). Private functions mentioned here are source-reading entry points, not promises of public API stability.

## Start with the two sample chapters

Read [A model’s complete journey from declaration to results](model_journey.md) to follow one model through type catalogs, probability tables, its runtime session, and outputs. Then read [How one reproduction stage is calculated](reproduction.md) to derive pair counts, eggs, inheritance allocation, and stage-specific count changes.

Both sample chapters and every later chapter aim to let you understand behavior and evaluate agent proposals without first reading the source. They include complete numerical cases and acceptance questions that distinguish implementations. **Every topic chapter is now written to the same depth**: each page puts the algorithm, its boundaries, and its acceptance checks into the prose instead of listing modules.

## Reading route

### Orientation

| Order | Chapter | Question answered |
| --- | --- | --- |
| 0 | [Reading route](index.md) (this page) | Where to start, and how the parts connect |
| 1 | [A model’s complete journey from declaration to results](model_journey.md) | How does one model travel the whole chain? |
| 2 | [Architecture and responsibility boundaries](architecture.md) | Who owns declarations, runtime state, and records? |

### Part 1: how a model is represented and built

| Order | Chapter | Question answered |
| --- | --- | --- |
| 3 | [From biological concepts to genetic objects](genetic_objects.md) | How does a species declaration become enumerable, comparable objects? |
| 4 | [Type catalog, indices, and array coordinates](data_layout.md) | Where does one individual sit in each array, and under which name? |
| 5 | [How patterns and selectors locate data](selectors.md) | How do conditions become coordinates, and when is the index bound? |
| 6 | [From declaration to compiled products](model.md) | How does a chain of declarations become a candidate, and what does failure leave behind? |
| 7 | [How genetic presets and conversion rules compile](genetic_compilation.md) | How do rules rewrite the baseline, and why is nothing applied twice? |
| 8 | [Reachability, index compression, and publication](publication.md) | Which types survive, and how do the coordinates move together? |

### Part 2: how a model executes and computes

| Order | Chapter | Question answered |
| --- | --- | --- |
| 9 | [How Python and Rust exchange model data](contracts.md) | What crosses the boundary, and who validates it? |
| 10 | [How a session advances one simulation](runtime.md) | How does execution advance, and where does failure leave it? |
| 11 | [How one reproduction stage is calculated](reproduction.md) | How are offspring counts and inheritance derived step by step? |
| 12 | [How survival and generation replacement are calculated](survival.md) | How are juveniles filtered, and when do old adults disappear? |
| 13 | [Age structure and long-term sperm storage](age_structure.md) | How do overlapping generations and stored sperm breed? |
| 14 | [How density regulation and equilibrium are computed](density_regulation.md) | How are juveniles scaled by the carrying capacity and the curve? |
| 15 | [Random sampling and reproducibility](randomness.md) | Where does sampling happen, and what does a seed promise? |
| 16 | [The fused Wright-Fisher execution path](wright_fisher.md) | Which stages are merged and which hooks stop running? |
| 17 | [Numerical algorithms (navigation)](algorithms.md) | How do mathematical quantities map to array axes and kernel functions? |

### Part 3: intervening in a running model

| Order | Chapter | Question answered |
| --- | --- | --- |
| 18 | [How runtime parameters are read and updated](runtime_updates.md) | Which route do writes take, and when do they land? |
| 19 | [How hooks compile and are scheduled](hooks.md) | How are the two hook spellings ordered, and when do selectors bind? |
| 20 | [Callback transactions, failure, and stop boundaries](transactions.md) | What do failure and stopping each keep? |

### Part 4: spatial simulation and results

| Order | Chapter | Question answered |
| --- | --- | --- |
| 21 | [How a spatial model is built and shares data](spatial.md) | How do demes share layouts, parameters, and a timeline? |
| 22 | [Spatial lifecycle and migration](migration.md) | How do individuals move between demes, and what is conserved? |
| 23 | [How observation turns state into results](observation.md) | What do grouping, axes, and information loss mean? |
| 24 | [History recording and the parameter timeline](history.md) | How do the plan, modes, and eviction work? |
| 25 | [Checkpoint restoration and experiment replay](checkpoints.md) | What does a restore contain, and what never rolls back? |
| 26 | [Observation, history, and checkpoints (navigation)](output.md) | How do the three correspond? |

### Part 5: development practice for AI agents

| Order | Chapter | Question answered |
| --- | --- | --- |
| 27 | [From a development need to an acceptable change](development.md) | How is a request clarified, located, compared, and accepted? |
| 28 | [How to verify numerical, state, and cross-language behaviour](verification.md) | Where do expected values come from, and what counts as evidence? |

The main line uses an ordinary staged discrete-generation model: a complete species declaration is compiled and published into a Rust session; age-1 adults produce age-0 offspring, survival acts on those offspring, and aging makes them the next adults. Results are read at session boundaries. Age-structured, spatial, and fused Wright–Fisher paths are distinguished in their own chapters, each listing where its behaviour differs from the main line.

## Reading alongside the source

Follow the narrative examples through inputs, intermediate state, and results. Source and test links let you or an agent verify implementation when needed; understanding the chapters does not require reading the source first. Source and test links target repository `main`. When investigating an older version, use documentation, code, and tests from the same commit.

Each chapter's verification entry points identify behavior protected by existing tests. They do not imply that these tests run automatically whenever documentation is read or built. Function names, axes, and boundary conditions locate concrete facts; explanatory calculations are not runnable Python examples.

The existing user documentation retains its internal-implementation and appendix entries. This guide supplies a continuous developer reading route; historical design pages provide background on earlier implementations. The browser interface (the repository-root `ui/` and `src/natal/frontend/webui/`) is outside this guide; see the note at the end of [development practice](development.md).
