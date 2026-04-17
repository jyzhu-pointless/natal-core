# state_translation Module

Human-readable translation helpers for population states.

## Overview

The `state_translation` module converts `PopulationState` and
`DiscretePopulationState` into nested dictionaries or JSON payloads, with
readable labels and optional zero-value filtering. For age-structured states,
translation includes sperm-storage tensors as well.

## History Translation Helpers

Use `population_history_to_readable_dict` and
`population_history_to_readable_json` to convert flattened history snapshots
into readable per-tick state payloads.

### When to use

- Inspect historical trajectories without manually reshaping flattened arrays.
- Export time-series snapshots for logging, debugging, or external tools.

### API behavior summary

- Input `history` can be omitted; when omitted, the function attempts to call
  `population.get_history()`.
- Each row in flattened history is parsed back into either
  `PopulationState` or `DiscretePopulationState` according to the current
  `population.state` type.
- Genotype labels are resolved from `population.index_registry` when
  available.
- Output includes top-level metadata:
  - `state_type`
  - `name`
  - `n_snapshots`
  - `snapshots` (list of translated state dictionaries)

### Example

```python
import natal as nt

hist_payload = nt.population_history_to_readable_dict(
    population=pop,
    include_zero_counts=False,
)

print(hist_payload["n_snapshots"])
print(hist_payload["snapshots"][0]["tick"])

hist_json = nt.population_history_to_readable_json(
    population=pop,
    include_zero_counts=False,
    indent=2,
)
print(hist_json[:200])
```

## Complete Module Reference

::: natal.state_translation
    options:
      heading_level: 3
