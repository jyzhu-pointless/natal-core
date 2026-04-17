"""Human-readable translation helpers for population state objects.

This module converts ``PopulationState`` and ``DiscretePopulationState`` into
structured dictionaries (and JSON strings) that are easier to inspect, log,
or serialize for downstream tooling.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import numpy as np

from natal.population_state import (
    DiscretePopulationState,
    PopulationState,
    parse_flattened_discrete_state,
    parse_flattened_state,
)

if TYPE_CHECKING:
    from natal.base_population import BasePopulation
    from natal.index_registry import IndexRegistry
    from natal.observation import GroupsInput, Observation
    from natal.spatial_population import SpatialPopulation

__all__ = [
    "population_state_to_dict",
    "population_state_to_json",
    "discrete_population_state_to_dict",
    "discrete_population_state_to_json",
    "output_current_state",
    "output_history",
    "population_to_readable_dict",
    "population_to_readable_json",
    "population_history_to_readable_dict",
    "population_history_to_readable_json",
    "spatial_population_to_readable_dict",
    "spatial_population_to_readable_json",
    "spatial_population_to_observation_dict",
    "spatial_population_to_observation_json",
]


def _default_sex_labels(n_sexes: int) -> List[str]:
    base = ["female", "male"]
    if n_sexes <= len(base):
        return base[:n_sexes]
    return base + [f"sex_{idx}" for idx in range(len(base), n_sexes)]


def _resolve_labels(
    count: int,
    labels: Optional[Sequence[str]],
    default_prefix: str,
) -> List[str]:
    if labels is None:
        return [f"{default_prefix}_{idx}" for idx in range(count)]

    resolved = [str(item) for item in labels]
    if len(resolved) != count:
        raise ValueError(
            f"Label count mismatch for {default_prefix}: expected {count}, got {len(resolved)}"
        )
    return resolved


def _genotype_labels_from_registry(
    registry: Optional[IndexRegistry],
    n_genotypes: int,
) -> List[str]:
    if registry is None:
        return _resolve_labels(n_genotypes, None, "genotype")

    if len(registry.index_to_genotype) != n_genotypes:
        raise ValueError(
            "Registry genotype count does not match state shape: "
            f"{len(registry.index_to_genotype)} != {n_genotypes}"
        )
    return [str(item) for item in registry.index_to_genotype]


def _build_individual_count_payload(
    individual_count: np.ndarray,
    sex_labels: Sequence[str],
    genotype_labels: Sequence[str],
    include_zero_counts: bool,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    n_ages = int(individual_count.shape[1])

    for sex_idx, sex_name in enumerate(sex_labels):
        sex_block: Dict[str, Dict[str, float]] = {}
        for age_idx in range(n_ages):
            age_key = f"age_{age_idx}"
            geno_block: Dict[str, float] = {}
            for genotype_idx, genotype_name in enumerate(genotype_labels):
                value = float(individual_count[sex_idx, age_idx, genotype_idx])
                if include_zero_counts or value != 0.0:
                    geno_block[genotype_name] = value
            if include_zero_counts or geno_block:
                sex_block[age_key] = geno_block
        payload[sex_name] = sex_block

    return payload


def _build_sperm_storage_payload(
    sperm_storage: np.ndarray,
    genotype_labels: Sequence[str],
    include_zero_counts: bool,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    n_ages = int(sperm_storage.shape[0])

    for age_idx in range(n_ages):
        age_key = f"age_{age_idx}"
        female_block: Dict[str, Dict[str, float]] = {}
        for female_idx, female_name in enumerate(genotype_labels):
            male_block: Dict[str, float] = {}
            for male_idx, male_name in enumerate(genotype_labels):
                value = float(sperm_storage[age_idx, female_idx, male_idx])
                if include_zero_counts or value != 0.0:
                    male_block[male_name] = value
            if include_zero_counts or male_block:
                female_block[female_name] = male_block
        if include_zero_counts or female_block:
            payload[age_key] = female_block

    return payload


def population_state_to_dict(
    state: PopulationState,
    genotype_labels: Optional[Sequence[str]] = None,
    sex_labels: Optional[Sequence[str]] = None,
    include_zero_counts: bool = False,
) -> Dict[str, Any]:
    """Translate an age-structured ``PopulationState`` to a readable dictionary.

    Args:
        state: State object to translate.
        genotype_labels: Optional labels for genotype axis. If omitted, labels are
            generated as ``genotype_0``, ``genotype_1``, etc.
        sex_labels: Optional labels for sex axis. If omitted, defaults to
            ``female``/``male`` (and ``sex_k`` for additional axes).
        include_zero_counts: Whether to keep zero-valued entries in output blocks.

    Returns:
        A nested dictionary containing tick, dimensions, individual counts, and
        sperm storage.
    """
    n_sexes, n_ages, n_genotypes = state.individual_count.shape
    labels_genotype = _resolve_labels(n_genotypes, genotype_labels, "genotype")
    labels_sex = _resolve_labels(n_sexes, sex_labels or _default_sex_labels(n_sexes), "sex")

    result: Dict[str, Any] = {
        "state_type": "PopulationState",
        "tick": int(state.n_tick),
        "dimensions": {
            "n_sexes": int(n_sexes),
            "n_ages": int(n_ages),
            "n_genotypes": int(n_genotypes),
        },
        "individual_count": _build_individual_count_payload(
            state.individual_count,
            labels_sex,
            labels_genotype,
            include_zero_counts,
        ),
        "sperm_storage": _build_sperm_storage_payload(
            state.sperm_storage,
            labels_genotype,
            include_zero_counts,
        ),
    }
    return result


def population_state_to_json(
    state: PopulationState,
    genotype_labels: Optional[Sequence[str]] = None,
    sex_labels: Optional[Sequence[str]] = None,
    include_zero_counts: bool = False,
    indent: int = 2,
) -> str:
    """Translate an age-structured ``PopulationState`` to a readable JSON string."""
    payload = population_state_to_dict(
        state=state,
        genotype_labels=genotype_labels,
        sex_labels=sex_labels,
        include_zero_counts=include_zero_counts,
    )
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def discrete_population_state_to_dict(
    state: DiscretePopulationState,
    genotype_labels: Optional[Sequence[str]] = None,
    sex_labels: Optional[Sequence[str]] = None,
    include_zero_counts: bool = False,
) -> Dict[str, Any]:
    """Translate a ``DiscretePopulationState`` to a readable dictionary.

    Args:
        state: State object to translate.
        genotype_labels: Optional labels for genotype axis. If omitted, labels are
            generated as ``genotype_0``, ``genotype_1``, etc.
        sex_labels: Optional labels for sex axis. If omitted, defaults to
            ``female``/``male`` (and ``sex_k`` for additional axes).
        include_zero_counts: Whether to keep zero-valued entries in output blocks.

    Returns:
        A nested dictionary containing tick, dimensions, and individual counts.
    """
    n_sexes, n_ages, n_genotypes = state.individual_count.shape
    labels_genotype = _resolve_labels(n_genotypes, genotype_labels, "genotype")
    labels_sex = _resolve_labels(n_sexes, sex_labels or _default_sex_labels(n_sexes), "sex")

    result: Dict[str, Any] = {
        "state_type": "DiscretePopulationState",
        "tick": int(state.n_tick),
        "dimensions": {
            "n_sexes": int(n_sexes),
            "n_ages": int(n_ages),
            "n_genotypes": int(n_genotypes),
        },
        "individual_count": _build_individual_count_payload(
            state.individual_count,
            labels_sex,
            labels_genotype,
            include_zero_counts,
        ),
    }
    return result


def discrete_population_state_to_json(
    state: DiscretePopulationState,
    genotype_labels: Optional[Sequence[str]] = None,
    sex_labels: Optional[Sequence[str]] = None,
    include_zero_counts: bool = False,
    indent: int = 2,
) -> str:
    """Translate a ``DiscretePopulationState`` to a readable JSON string."""
    payload = discrete_population_state_to_dict(
        state=state,
        genotype_labels=genotype_labels,
        sex_labels=sex_labels,
        include_zero_counts=include_zero_counts,
    )
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def population_to_readable_dict(
    population: BasePopulation[Any],
    include_zero_counts: bool = False,
) -> Dict[str, Any]:
    """Translate a population's current state to a readable dictionary.

    This helper automatically pulls genotype labels from
    ``population.index_registry`` so output keys use genotype strings.

    Args:
        population: Population instance containing either an age-structured or
            discrete state object.
        include_zero_counts: Whether to keep zero-valued entries in output blocks.

    Returns:
        A nested dictionary describing the current state.

    Raises:
        TypeError: If the population state type is unsupported.
    """
    state = population.state
    n_genotypes = int(state.individual_count.shape[2])
    genotype_labels = _genotype_labels_from_registry(population.index_registry, n_genotypes)

    if isinstance(state, PopulationState):
        return population_state_to_dict(
            state=state,
            genotype_labels=genotype_labels,
            include_zero_counts=include_zero_counts,
        )
    if isinstance(state, DiscretePopulationState):
        return discrete_population_state_to_dict(
            state=state,
            genotype_labels=genotype_labels,
            include_zero_counts=include_zero_counts,
        )

    raise TypeError(f"Unsupported state type: {type(state).__name__}")


def population_to_readable_json(
    population: BasePopulation[Any],
    include_zero_counts: bool = False,
    indent: int = 2,
) -> str:
    """Translate a population's current state to a readable JSON string."""
    payload = population_to_readable_dict(
        population=population,
        include_zero_counts=include_zero_counts,
    )
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def population_history_to_readable_dict(
    population: BasePopulation[Any],
    history: Optional[np.ndarray] = None,
    include_zero_counts: bool = False,
) -> Dict[str, Any]:
    """Translate flattened history records into readable snapshot dictionaries.

    Args:
        population: Population instance that defines history shape and labels.
        history: Optional flattened history array. When ``None``, uses
            ``population.get_history()``.
        include_zero_counts: Whether to keep zero-valued entries in each
            snapshot payload.

    Returns:
        A dictionary containing metadata and translated snapshot entries.

    Raises:
        TypeError: If the population state type is unsupported.
        ValueError: If ``history`` is not a 2D array.
    """
    state = population.state
    n_sexes, n_ages, n_genotypes = state.individual_count.shape
    genotype_labels = _genotype_labels_from_registry(population.index_registry, int(n_genotypes))

    history_array: np.ndarray
    if history is None:
        get_history = getattr(population, "get_history", None)
        if callable(get_history):
            try:
                history_array = cast(np.ndarray, get_history())
            except ValueError:
                history_array = np.zeros((0, 0), dtype=np.float64)
        else:
            history_array = np.zeros((0, 0), dtype=np.float64)
    else:
        history_array = cast(np.ndarray, np.asarray(history, dtype=np.float64))

    if history_array.ndim != 2:
        raise ValueError(
            f"history must be a 2D array, got shape {history_array.shape}"
        )

    snapshots: List[Dict[str, Any]] = []
    if isinstance(state, PopulationState):
        for idx in range(int(history_array.shape[0])):
            row = history_array[idx, :]
            parsed_state = parse_flattened_state(
                row,
                n_sexes=n_sexes,
                n_ages=n_ages,
                n_genotypes=n_genotypes,
                copy=True,
            )
            snapshots.append(
                population_state_to_dict(
                    state=parsed_state,
                    genotype_labels=genotype_labels,
                    include_zero_counts=include_zero_counts,
                )
            )
    elif isinstance(state, DiscretePopulationState):
        for idx in range(int(history_array.shape[0])):
            row = history_array[idx, :]
            parsed_state = parse_flattened_discrete_state(
                row,
                n_sexes=n_sexes,
                n_ages=n_ages,
                n_genotypes=n_genotypes,
                copy=True,
            )
            snapshots.append(
                discrete_population_state_to_dict(
                    state=parsed_state,
                    genotype_labels=genotype_labels,
                    include_zero_counts=include_zero_counts,
                )
            )
    else:
        raise TypeError(f"Unsupported state type: {type(state).__name__}")

    return {
        "state_type": type(state).__name__,
        "name": str(population.name),
        "n_snapshots": int(history_array.shape[0]),
        "snapshots": snapshots,
    }


def population_history_to_readable_json(
    population: BasePopulation[Any],
    history: Optional[np.ndarray] = None,
    include_zero_counts: bool = False,
    indent: int = 2,
) -> str:
    """Translate flattened history records to a readable JSON string.

    Args:
        population: Population instance that defines history shape and labels.
        history: Optional flattened history array. When ``None``, uses
            ``population.get_history()``.
        include_zero_counts: Whether to keep zero-valued entries.
        indent: Indentation level used by ``json.dumps``.

    Returns:
        JSON string containing history metadata and translated snapshots.
    """
    payload = population_history_to_readable_dict(
        population=population,
        history=history,
        include_zero_counts=include_zero_counts,
    )
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def _build_observation_payload(
    observed: np.ndarray,
    labels: Sequence[str],
    sex_labels: Sequence[str],
    include_zero_counts: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    if observed.ndim == 3:
        n_ages = int(observed.shape[2])
        for group_idx, group_name in enumerate(labels):
            sex_age_block: Dict[str, Dict[str, float]] = {}
            for sex_idx, sex_name in enumerate(sex_labels):
                age_block: Dict[str, float] = {}
                for age_idx in range(n_ages):
                    value = float(observed[group_idx, sex_idx, age_idx])
                    if include_zero_counts or value != 0.0:
                        age_block[f"age_{age_idx}"] = value
                if include_zero_counts or age_block:
                    sex_age_block[sex_name] = age_block
            payload[group_name] = sex_age_block
        return payload

    if observed.ndim == 2:
        for group_idx, group_name in enumerate(labels):
            sex_value_block: Dict[str, float] = {}
            for sex_idx, sex_name in enumerate(sex_labels):
                value = float(observed[group_idx, sex_idx])
                if include_zero_counts or value != 0.0:
                    sex_value_block[sex_name] = value
            payload[group_name] = sex_value_block
        return payload

    raise ValueError(f"Unsupported observed array ndim: {observed.ndim}")


def _write_json_payload(
    payload: Dict[str, Any],
    output_path: Optional[Union[str, Path]],
    indent: int,
) -> None:
    if output_path is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")


def _get_population_observation_payload(
    population: BasePopulation[Any],
    *,
    observation: Optional[Observation],
    groups: Optional[GroupsInput],
    collapse_age: bool,
    include_zero_counts: bool,
) -> Dict[str, Any]:
    state = population.state
    n_sexes = int(state.individual_count.shape[0])
    sex_labels = _default_sex_labels(n_sexes)

    resolved_observation = observation or population.create_observation(
        groups=groups,
        collapse_age=collapse_age,
    )
    observed = resolved_observation.apply(state.individual_count)

    return {
        "state_type": type(state).__name__,
        "tick": int(state.n_tick),
        "collapse_age": bool(resolved_observation.collapse_age),
        "labels": list(resolved_observation.labels),
        "observed": _build_observation_payload(
            observed=observed,
            labels=resolved_observation.labels,
            sex_labels=sex_labels,
            include_zero_counts=include_zero_counts,
        ),
    }


def _get_history_array(
    population: BasePopulation[Any],
    history: Optional[np.ndarray],
) -> np.ndarray:
    if history is None:
        get_history = getattr(population, "get_history", None)
        if callable(get_history):
            try:
                return cast(np.ndarray, get_history())
            except ValueError:
                return np.zeros((0, 0), dtype=np.float64)
        return np.zeros((0, 0), dtype=np.float64)

    return cast(np.ndarray, np.asarray(history, dtype=np.float64))


def _build_history_observation_payload(
    population: BasePopulation[Any],
    *,
    history: Optional[np.ndarray],
    observation: Optional[Observation],
    groups: Optional[GroupsInput],
    collapse_age: bool,
    include_zero_counts: bool,
) -> Dict[str, Any]:
    state = population.state
    n_sexes, n_ages, n_genotypes = state.individual_count.shape
    sex_labels = _default_sex_labels(n_sexes)

    history_array = _get_history_array(population, history)
    if history_array.ndim != 2:
        raise ValueError(f"history must be a 2D array, got shape {history_array.shape}")

    resolved_observation = observation or population.create_observation(
        groups=groups,
        collapse_age=collapse_age,
    )

    snapshots: List[Dict[str, Any]] = []
    for idx in range(int(history_array.shape[0])):
        row = history_array[idx, :]
        if isinstance(state, PopulationState):
            parsed_state = parse_flattened_state(
                row,
                n_sexes=n_sexes,
                n_ages=n_ages,
                n_genotypes=n_genotypes,
                copy=True,
            )
        elif isinstance(state, DiscretePopulationState):
            parsed_state = parse_flattened_discrete_state(
                row,
                n_sexes=n_sexes,
                n_ages=n_ages,
                n_genotypes=n_genotypes,
                copy=True,
            )
        else:
            raise TypeError(f"Unsupported state type: {type(state).__name__}")

        observed = resolved_observation.apply(parsed_state.individual_count)
        snapshots.append(
            {
                "tick": int(parsed_state.n_tick),
                "state_type": type(parsed_state).__name__,
                "labels": list(resolved_observation.labels),
                "observed": _build_observation_payload(
                    observed=observed,
                    labels=resolved_observation.labels,
                    sex_labels=sex_labels,
                    include_zero_counts=include_zero_counts,
                ),
            }
        )

    return {
        "state_type": type(state).__name__,
        "name": str(population.name),
        "n_snapshots": int(history_array.shape[0]),
        "collapse_age": bool(resolved_observation.collapse_age),
        "labels": list(resolved_observation.labels),
        "snapshots": snapshots,
    }


def output_current_state(
    population: BasePopulation[Any],
    *,
    observation: Optional[Observation] = None,
    groups: Optional[GroupsInput] = None,
    collapse_age: bool = False,
    include_zero_counts: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> Dict[str, Any]:
    """Export the current population state with observation rules applied.

    This function integrates ``natal.observation`` with state translation and
    can optionally write the JSON payload to ``output_path``.

    Args:
        population: Population instance to observe.
        observation: Optional prebuilt observation object. When provided,
            ``groups`` and ``collapse_age`` are ignored.
        groups: Observation groups passed to ``ObservationFilter.build_filter``.
            When ``None``, one group per genotype index is used.
        collapse_age: Whether observation rule generation collapses age axis.
        include_zero_counts: Whether to keep zero-valued entries.
        output_path: Optional JSON file path. When provided, the payload is
            written to this file as UTF-8 JSON.
        indent: Indentation used when writing JSON.

    Returns:
        A dictionary with observation metadata and observed counts.
    """
    payload = _get_population_observation_payload(
        population,
        observation=observation,
        groups=groups,
        collapse_age=collapse_age,
        include_zero_counts=include_zero_counts,
    )
    _write_json_payload(payload, output_path, indent)
    return payload


def output_history(
    population: BasePopulation[Any],
    *,
    observation: Optional[Observation] = None,
    groups: Optional[GroupsInput] = None,
    collapse_age: bool = False,
    include_zero_counts: bool = False,
    history: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    indent: int = 2,
) -> Dict[str, Any]:
    """Export the observation history for a population.

    Args:
        population: Population instance to observe.
        observation: Optional prebuilt observation object. When provided,
            ``groups`` and ``collapse_age`` are ignored.
        groups: Observation groups passed to ``ObservationFilter.build_filter``.
            When ``None``, one group per genotype index is used.
        collapse_age: Whether observation rule generation collapses age axis.
        include_zero_counts: Whether to keep zero-valued entries.
        history: Optional flattened history array. When omitted, the population
            history is fetched from ``population.get_history()``.
        output_path: Optional JSON file path. When provided, the payload is
            written to this file as UTF-8 JSON.
        indent: Indentation used when writing JSON.

    Returns:
        A dictionary containing observation metadata and per-snapshot outputs.
    """
    payload = _build_history_observation_payload(
        population,
        history=history,
        observation=observation,
        groups=groups,
        collapse_age=collapse_age,
        include_zero_counts=include_zero_counts,
    )
    _write_json_payload(payload, output_path, indent)
    return payload


def population_to_observation_dict(
    population: BasePopulation[Any],
    *,
    observation: Optional[Observation] = None,
    groups: Optional[GroupsInput] = None,
    collapse_age: bool = False,
    include_zero_counts: bool = False,
) -> Dict[str, Any]:
    """Legacy wrapper for :func:`output_current_state`."""
    return output_current_state(
        population,
        observation=observation,
        groups=groups,
        collapse_age=collapse_age,
        include_zero_counts=include_zero_counts,
    )


def population_to_observation_json(
    population: BasePopulation[Any],
    *,
    observation: Optional[Observation] = None,
    groups: Optional[GroupsInput] = None,
    collapse_age: bool = False,
    include_zero_counts: bool = False,
    indent: int = 2,
) -> str:
    """Legacy wrapper that serializes :func:`output_current_state` to JSON."""
    payload = output_current_state(
        population,
        observation=observation,
        groups=groups,
        collapse_age=collapse_age,
        include_zero_counts=include_zero_counts,
    )
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def spatial_population_to_readable_dict(
    spatial_population: SpatialPopulation,
    include_zero_counts: bool = False,
) -> Dict[str, Any]:
    """Translate a ``SpatialPopulation`` into readable per-deme dictionaries.

    Args:
        spatial_population: Spatial population container.
        include_zero_counts: Whether to keep zero-valued entries.

    Returns:
        Dictionary containing per-deme readable payloads and one aggregate state.
    """
    aggregate_state = spatial_population.aggregate_state()
    n_genotypes = int(aggregate_state.individual_count.shape[2])
    genotype_labels = _genotype_labels_from_registry(
        spatial_population.deme(0).index_registry,
        n_genotypes,
    )

    demes_payload: Dict[str, Any] = {}
    for deme_idx, deme in enumerate(spatial_population.demes):
        demes_payload[f"deme_{deme_idx}"] = population_to_readable_dict(
            population=deme,
            include_zero_counts=include_zero_counts,
        )

    aggregate_payload = population_state_to_dict(
        state=aggregate_state,
        genotype_labels=genotype_labels,
        include_zero_counts=include_zero_counts,
    )

    return {
        "state_type": "SpatialPopulation",
        "name": str(spatial_population.name),
        "tick": int(spatial_population.tick),
        "n_demes": int(spatial_population.n_demes),
        "demes": demes_payload,
        "aggregate": aggregate_payload,
    }


def spatial_population_to_readable_json(
    spatial_population: SpatialPopulation,
    include_zero_counts: bool = False,
    indent: int = 2,
) -> str:
    """Translate a ``SpatialPopulation`` into a readable JSON string."""
    payload = spatial_population_to_readable_dict(
        spatial_population=spatial_population,
        include_zero_counts=include_zero_counts,
    )
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def spatial_population_to_observation_dict(
    spatial_population: SpatialPopulation,
    *,
    groups: Optional[GroupsInput] = None,
    collapse_age: bool = False,
    include_zero_counts: bool = False,
) -> Dict[str, Any]:
    """Apply observation rules for each deme and aggregate in a spatial system.

    Args:
        spatial_population: Spatial population container.
        groups: Observation groups passed to underlying observation filter.
        collapse_age: Whether observation collapses age axis.
        include_zero_counts: Whether to keep zero-valued entries.

    Returns:
        Dictionary with per-deme and aggregate observation payloads.
    """
    from natal.observation import ObservationFilter

    per_deme_payload: Dict[str, Any] = {}
    for deme_idx, deme in enumerate(spatial_population.demes):
        per_deme_payload[f"deme_{deme_idx}"] = population_to_observation_dict(
            population=deme,
            groups=groups,
            collapse_age=collapse_age,
            include_zero_counts=include_zero_counts,
        )

    aggregate_state = spatial_population.aggregate_state()
    n_sexes = int(aggregate_state.individual_count.shape[0])
    sex_labels = _default_sex_labels(n_sexes)

    obs_filter = ObservationFilter(spatial_population.deme(0).index_registry)
    observation = obs_filter.create_observation(
        diploid_genotypes=spatial_population.species,
        groups=groups,
        collapse_age=collapse_age,
    )
    observed = observation.apply(aggregate_state.individual_count)

    aggregate_payload: Dict[str, Any] = {
        "state_type": "PopulationState",
        "tick": int(aggregate_state.n_tick),
        "collapse_age": bool(observation.collapse_age),
        "labels": list(observation.labels),
        "observed": _build_observation_payload(
            observed=observed,
            labels=observation.labels,
            sex_labels=sex_labels,
            include_zero_counts=include_zero_counts,
        ),
    }

    return {
        "state_type": "SpatialPopulationObservation",
        "name": str(spatial_population.name),
        "tick": int(spatial_population.tick),
        "n_demes": int(spatial_population.n_demes),
        "demes": per_deme_payload,
        "aggregate": aggregate_payload,
    }


def spatial_population_to_observation_json(
    spatial_population: SpatialPopulation,
    *,
    groups: Optional[GroupsInput] = None,
    collapse_age: bool = False,
    include_zero_counts: bool = False,
    indent: int = 2,
) -> str:
    """Apply spatial observation export and serialize to JSON string."""
    payload = spatial_population_to_observation_dict(
        spatial_population=spatial_population,
        groups=groups,
        collapse_age=collapse_age,
        include_zero_counts=include_zero_counts,
    )
    return json.dumps(payload, ensure_ascii=False, indent=indent)
