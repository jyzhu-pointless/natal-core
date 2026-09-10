"""Owned model declarations and their normalized compilation inputs.

``ModelDefinition`` is the single declaration type: it carries the user's
rules and their order (the journal), the normalized compilation inputs
(draft, registry, recipes, fitness patches, observation and history
policies, spatial controls), and the source-identity token that connects
the declaration to its compiled products. NATAL containers and arrays are
detached on construction and on every hand-out; caller-owned recipes and
hook resources keep their identity. Derived genetic matrices are not
declaration state — they live on the builder's draft and are recomputed
from here on a cold rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from natal.frontend.builder import PopulationBuilder
    from natal.frontend.genetics import Species
    from natal.frontend.model.draft import ModelDraft
    from natal.frontend.patterns import IndividualSelector
    from natal.frontend.presets import GeneticPreset
    from natal.frontend.registry.index import IndexRegistry
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
    group_calls: tuple[tuple[str, dict[str, Any]], ...]  # Any: normalized PopulationBuilder keyword values are heterogeneous.
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

    The declaration owns the user-facing inputs only: the replayable
    journal, the normalized compilation inputs, and the compilation key
    connecting them to their products. Expanded genetic matrices are not
    stored here as derived state; the ``draft`` mirrors the declared
    ecology and initial state as captured, and a cold rebuild recomputes
    all derived products from the recipes.

    Attributes:
        species: The genetic architecture the model declares.
        discrete_generation: Whether the declaration targets the
            discrete-generation granularity.
        build_name: The population name the chain declared.
        compilation_key: Identity token connecting this declaration to its
            compiled products; products belonging to a different token are
            stale and must be recomputed.
        presets: Registered user recipes; opaque resources remain caller-owned.
        manual_gamete: Explicit gamete modifier declarations.
        manual_zygote: Explicit zygote modifier declarations.
        observation_collapse_age: Whether observation projections sum the
            age axis.
        history_mode: Raw-state or observation recording policy.
        history_max_rows: Retention bound, or the population default.
        compress: Whether to prune unreachable active genetic types.
        declared_zygote_types: Explicit types retained during pruning.
    """

    species: Species
    discrete_generation: bool
    # Any: declaration arguments include user callables and heterogeneous domain objects.
    _journal: tuple[tuple[str, dict[str, Any]], ...] = field(default=(), repr=False)
    build_name: str | None = None
    presets: tuple[GeneticPreset, ...] = ()
    manual_gamete: tuple[tuple[int, str | None, object], ...] = ()
    manual_zygote: tuple[tuple[int, str | None, object], ...] = ()
    compilation_key: object | None = None
    observation_collapse_age: bool = False
    history_mode: Literal["raw", "observation"] = "raw"
    history_max_rows: int | None = None
    compress: bool = False
    declared_zygote_types: frozenset[str] | frozenset[int] | None = None
    _draft: ModelDraft | None = field(default=None, repr=False)
    _registry: IndexRegistry | None = field(default=None, repr=False)
    _fitness_base: tuple[NDArray[np.float64], ...] = field(default=(), repr=False)
    _fitness_steps: tuple[tuple[int, dict[str, object]], ...] = field(default=(), repr=False)
    _hook_calls: tuple[tuple[tuple[object, ...], dict[str, object]], ...] = field(default=(), repr=False)
    _observation_groups: Mapping[str, IndividualSelector] | None = field(default=None, repr=False)
    _spatial: SpatialInputs | None = field(default=None, repr=False)

    def __init__(
        self, species: Species, discrete_generation: bool,
        journal: tuple[tuple[str, dict[str, Any]], ...] = (),
        build_name: str | None = None,
        *,
        presets: tuple[GeneticPreset, ...] = (),
        manual_gamete: tuple[tuple[int, str | None, object], ...] = (),
        manual_zygote: tuple[tuple[int, str | None, object], ...] = (),
        compilation_key: object | None = None,
        observation_collapse_age: bool = False,
        history_mode: Literal["raw", "observation"] = "raw",
        history_max_rows: int | None = None,
        compress: bool = False,
        declared_zygote_types: frozenset[str] | frozenset[int] | None = None,
        draft: ModelDraft | None = None,
        registry: IndexRegistry | None = None,
        fitness_base: tuple[NDArray[np.float64], ...] = (),
        fitness_steps: tuple[tuple[int, dict[str, object]], ...] = (),
        hook_calls: tuple[tuple[tuple[object, ...], dict[str, object]], ...] = (),
        observation_groups: Mapping[str, IndividualSelector] | None = None,
        spatial: SpatialInputs | None = None,
    ) -> None:
        """Capture an isolated ordered declaration.

        NATAL-owned arrays and containers are detached on capture; user
        recipes, hooks, and other opaque resources keep their identity.
        """
        from copy import deepcopy

        from natal.frontend.genetics.definition_compiler import (
            copy_registry,
            detach_draft,
        )
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "discrete_generation", discrete_generation)
        object.__setattr__(self, "_journal", _copy_journal(journal, species))
        object.__setattr__(self, "build_name", build_name)
        object.__setattr__(self, "presets", tuple(presets))
        object.__setattr__(self, "manual_gamete", tuple(manual_gamete))
        object.__setattr__(self, "manual_zygote", tuple(manual_zygote))
        object.__setattr__(self, "compilation_key", compilation_key)
        object.__setattr__(self, "observation_collapse_age", observation_collapse_age)
        object.__setattr__(self, "history_mode", history_mode)
        object.__setattr__(self, "history_max_rows", history_max_rows)
        object.__setattr__(self, "compress", compress)
        object.__setattr__(self, "declared_zygote_types", declared_zygote_types)
        object.__setattr__(self, "_draft", None if draft is None else detach_draft(draft))
        object.__setattr__(self, "_registry", None if registry is None else copy_registry(registry))
        # Any: declaration values include heterogeneous recipes and user resources.
        object.__setattr__(self, "_fitness_base", tuple(array.copy() for array in fitness_base))
        object.__setattr__(self, "_fitness_steps", deepcopy(fitness_steps))
        object.__setattr__(
            self, "_hook_calls",
            tuple((tuple(items), dict(options)) for items, options in hook_calls),
        )
        object.__setattr__(
            self, "_observation_groups",
            None if observation_groups is None else dict(observation_groups),
        )
        object.__setattr__(self, "_spatial", None if spatial is None else snapshot_spatial_inputs(spatial))

    @property
    def draft(self) -> ModelDraft | None:
        """Return a detached declared draft (ecology, switches, initial state)."""
        from natal.frontend.genetics.definition_compiler import detach_draft

        return None if self._draft is None else detach_draft(self._draft)

    @property
    def registry(self) -> IndexRegistry | None:
        """Return a detached copy of the declared active type layout."""
        from natal.frontend.genetics.definition_compiler import copy_registry

        return None if self._registry is None else copy_registry(self._registry)

    @property
    def fitness_base(self) -> tuple[NDArray[np.float64], ...]:
        """Return detached initial fitness arrays, before any user recipe or patch."""
        return tuple(array.copy() for array in self._fitness_base)

    @property
    def fitness_steps(self) -> tuple[tuple[int, dict[str, object]], ...]:
        """Return detached ordered explicit fitness patches, including their modes."""
        from copy import deepcopy

        return deepcopy(self._fitness_steps)

    @property
    def hook_calls(self) -> tuple[tuple[tuple[object, ...], dict[str, object]], ...]:
        """Return detached event registrations and their default dispatch options."""
        return tuple((tuple(items), dict(options)) for items, options in self._hook_calls)

    @property
    def observation_groups(self) -> Mapping[str, IndividualSelector] | None:
        """Return a detached mapping of the named observation selectors."""
        return None if self._observation_groups is None else dict(self._observation_groups)

    @property
    def spatial(self) -> SpatialInputs | None:
        """Return detached normalized spatial controls, when declared."""
        return None if self._spatial is None else snapshot_spatial_inputs(self._spatial)

    @property
    def journal(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Return a detached declaration journal."""
        return _copy_journal(self._journal, self.species)

    def with_spatial(self, spatial: SpatialInputs | None) -> ModelDefinition:
        """Return a copy of this declaration carrying new spatial controls.

        Args:
            spatial: The concrete spatial controls to attach, or ``None``
                to drop them.

        Returns:
            A new frozen declaration; this snapshot is unchanged.
        """
        return ModelDefinition(
            self.species, self.discrete_generation, self._journal, self.build_name,
            presets=self.presets, manual_gamete=self.manual_gamete,
            manual_zygote=self.manual_zygote, compilation_key=self.compilation_key,
            observation_collapse_age=self.observation_collapse_age,
            history_mode=self.history_mode, history_max_rows=self.history_max_rows,
            compress=self.compress, declared_zygote_types=self.declared_zygote_types,
            draft=self._draft, registry=self._registry,
            fitness_base=self._fitness_base, fitness_steps=self._fitness_steps,
            hook_calls=self._hook_calls, observation_groups=self._observation_groups,
            spatial=spatial,
        )

    def replay(
        self, factory: Callable[[], PopulationBuilder]
    ) -> PopulationBuilder:
        """Rebuild a builder from this definition's journal.

        Args:
            factory: Zero-argument constructor producing a fresh,
                empty builder of the right granularity.

        Returns:
            The builder after replaying every journal entry.
        """
        from natal.frontend.builder._base import replay_declarations

        return replay_declarations(factory, list(self.journal))

    def entry_names(self) -> tuple[str, ...]:
        """Return the declaration method names in order.

        Returns:
            The ordered ``(method_name, ...)`` sequence of the journal.
        """
        return tuple(name for name, _ in self.journal)
