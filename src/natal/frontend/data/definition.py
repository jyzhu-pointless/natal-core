"""Owned model declarations and their normalized compilation inputs.

The journal preserves the user's spelling for replay and diagnostics. The
normalized inputs drive compilation without repeating opaque recipe execution
when cached products already belong to the same declaration. NATAL containers
and arrays are detached; caller-owned recipes and hook resources keep identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, cast

if TYPE_CHECKING:
    from natal.frontend.configurator import Configurator
    from natal.frontend.genetics import Species
    from natal.frontend.genetics.definition_compiler import NormalizedModel
    from natal.frontend.patterns import IndividualSelector
    from natal.frontend.spatial.topology import GridTopology


def copy_declaration_value(value: Any) -> Any:
    """Copy owned arrays/containers while preserving opaque recipes and resources."""
    # Any: declaration values include heterogeneous recipes and user resources;
    # copying is deliberately restricted to NATAL-owned container types.
    import numpy as np

    if isinstance(value, np.ndarray):
        return cast("np.ndarray[Any, Any]", value).copy()
    if isinstance(value, dict):
        return {key: copy_declaration_value(item) for key, item in cast("dict[object, Any]", value).items()}
    if isinstance(value, list):
        return [copy_declaration_value(item) for item in cast("list[Any]", value)]
    if type(value) is tuple:
        return tuple(copy_declaration_value(item) for item in cast("tuple[Any, ...]", value))
    return value


def _copy_journal(
    journal: tuple[tuple[str, dict[str, Any]], ...], species: Species,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Copy declaration containers without copying opaque user resources."""
    return tuple((name, copy_declaration_value(kwargs)) for name, kwargs in journal)


@dataclass(frozen=True)
class SpatialInputs:
    """Normalized spatial controls consumed by the group compiler.

    Batch expressions have already expanded to concrete per-deme values.
    Group calls hold ordinary template declarations; the matching batch
    values replace their first-deme placeholders during group compilation.
    Opaque callbacks and recipe resources retain their identity.
    """

    n_demes: int
    topology: GridTopology | None
    pop_type: Literal["age_structured", "discrete_generation"]
    name: str
    batch_values: tuple[tuple[str, tuple[Any, ...]], ...]  # Any: each batch route has its own scalar/tensor/recipe value type.
    group_calls: tuple[tuple[str, dict[str, Any]], ...]  # Any: normalized Configurator keyword values are heterogeneous.
    migration: Mapping[str, object]
    observation_groups: Mapping[str, IndividualSelector] | None
    observation_collapse_age: bool
    observation_demes: tuple[int, ...]
    observation_deme_mode: Literal["preserve", "aggregate"]
    history_mode: Literal["raw", "observation"]
    history_max_rows: int | None
    compress: bool
    declared_zygote_types: frozenset[str] | frozenset[int] | None


def snapshot_spatial_inputs(inputs: SpatialInputs) -> SpatialInputs:
    """Detach owned spatial values without re-running batches or recipes."""
    from dataclasses import replace

    return replace(
        inputs, batch_values=copy_declaration_value(inputs.batch_values),
        group_calls=copy_declaration_value(inputs.group_calls),
        migration=copy_declaration_value(dict(inputs.migration)),
        observation_groups=None if inputs.observation_groups is None else dict(inputs.observation_groups),
    )


@dataclass(frozen=True, init=False)
class ModelDefinition:
    """The frozen declaration snapshot attached to a built population.

    Attributes:
        species: The genetic architecture the model declares.
        discrete_generation: Whether the declaration targets the
            discrete-generation granularity.
        journal: The ordered declaration entries
            ``(method_name, explicitly_passed_kwargs)`` captured by the
            ``@_declared`` journal — the replayable source of truth.
        build_name: The population name the chain declared.
    """

    species: Species
    discrete_generation: bool
    # Any: declaration arguments include user callables and heterogeneous domain objects.
    _journal: tuple[tuple[str, dict[str, Any]], ...] = field(default=(), repr=False)
    build_name: str | None = None
    _normalized: NormalizedModel | None = field(default=None, repr=False)

    def __init__(
        self, species: Species, discrete_generation: bool,
        journal: tuple[tuple[str, dict[str, Any]], ...] = (),
        build_name: str | None = None,
        normalized: NormalizedModel | None = None,
    ) -> None:
        """Capture an isolated ordered declaration."""
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "discrete_generation", discrete_generation)
        object.__setattr__(self, "_journal", _copy_journal(journal, species))
        object.__setattr__(self, "build_name", build_name)
        from natal.frontend.genetics.definition_compiler import snapshot_inputs

        object.__setattr__(self, "_normalized", None if normalized is None else snapshot_inputs(normalized))

    @property
    def normalized(self) -> NormalizedModel | None:
        """Return a detached normalized declaration for the shared compiler."""
        from natal.frontend.genetics.definition_compiler import snapshot_inputs

        return None if self._normalized is None else snapshot_inputs(self._normalized)

    @property
    def journal(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Return a detached declaration journal."""
        return _copy_journal(self._journal, self.species)

    def replay(
        self, factory: Callable[[], Configurator]
    ) -> Configurator:
        """Rebuild a configurator from this definition's journal.

        Args:
            factory: Zero-argument constructor producing a fresh,
                empty configurator of the right granularity.

        Returns:
            The configurator after replaying every journal entry.
        """
        from natal.frontend.configurator._base import replay_declarations

        return replay_declarations(factory, list(self.journal))

    def entry_names(self) -> tuple[str, ...]:
        """Return the declaration method names in order.

        Returns:
            The ordered ``(method_name, ...)`` sequence of the journal.
        """
        return tuple(name for name, _ in self.journal)
