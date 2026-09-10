"""RuntimeUpdater: the single runtime-update handle and its shared cores.

``pop.update()`` and ``ctx.update()`` both return one public type, the
:class:`RuntimeUpdater`, with exactly eight domain methods.  The handle
holds only its commit target — an idle native session, or the current
event transaction guarded by the callback lifetime — and resolves every
model fact (species, registry, presets, manual modifiers, fitness
declarations, parameter values) from the owning population at operation
time.  No parameter snapshot is kept: a retained handle keeps working
across runs and always computes against the live native values, and a
failed update publishes nothing.

The parse/validate/compute cores (route-write parsing, genetic candidate
compilation, candidate commitment) are plain functions with explicit
inputs, shared with the build-side :class:`~natal.frontend.configurator.
Configurator` chain methods and the population's preset/modifier refresh
helpers.  The three former commit-target selections (``_make_writer``,
``custom()`` inline, ``_commit_genetic_candidate``) collapse into
:func:`runtime_writer` plus the two target classes below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.configurator._params import compute_expected_eggs_from_females
from natal.frontend.configurator._routes import lookup_or_none
from natal.frontend.configurator._writers import (
    NATIVE_SCALAR_FIELDS,
    AuditValue,
    CoreConfigWriter,
)
from natal.frontend.data import ModelDraft
from natal.frontend.genetics.compile import next_modifier_id
from natal.frontend.genetics.definition_compiler import (
    FITNESS_FIELDS,
    CompiledProducts,
    compile_definition,
)
from natal.frontend.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.frontend.configurator._writers import SessionChannel
    from natal.frontend.data.definition import ModelDefinition
    from natal.frontend.genetics import Species
    from natal.frontend.genetics.compile import GameteList, ZygoteList
    from natal.frontend.hooks._transaction import EventTransaction
    from natal.frontend.hooks.tick_context import TickContext
    from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
    from natal.frontend.population.base import BasePopulation
    from natal.frontend.presets import GeneticPreset

__all__ = ["RuntimeUpdater"]


# Draft fields a genetic commit publishes; runtime genetic updates may
# rewrite their contents but never the active type layout.
GENETIC_COMMIT_FIELDS: tuple[str, ...] = (
    "viability_fitness",
    "fecundity_fitness",
    "sexual_selection_fitness",
    "zygote_viability_fitness",
    "offspring_tensor",
    "meiosis_map",
    "female_ztype_compatibility",
    "male_ztype_compatibility",
)


# ── Route-write parsing (shared with the build-side chain) ────────────────────


def competition_writes(
    *,
    carrying_capacity: float | None,
    low_density_growth_rate: float | None,
    juvenile_growth_mode: int | str | None,
    growth_mode: int | str | None,
    competition_strength: float | None,
    equilibrium_distribution: NDArray[np.float64] | None,
    age_1_carrying_capacity: float | None,
    old_juvenile_carrying_capacity: float | None,
    draft: ModelDraft | None,
    allow_initial_k_detection: bool,
) -> dict[str, object]:
    """Parse one ``competition()`` call into route-table writes.

    Args:
        carrying_capacity: Equilibrium population at age 1 (K).
        low_density_growth_rate: Per-capita growth at low density (r).
        juvenile_growth_mode: Historical spelling of *growth_mode*.
        growth_mode: Regulation function (string or integer).
        competition_strength: Larval competition weight.
        equilibrium_distribution: Custom Champer equilibrium table.
        age_1_carrying_capacity: Legacy alias for *carrying_capacity*.
        old_juvenile_carrying_capacity: Legacy alias.
        draft: The working draft, read only for the legacy auto-detection
            of K from the declared initial state.
        allow_initial_k_detection: Whether the initial-state auto-detect
            may run (initial build only; runtime updates never re-derive
            K from the frozen declaration).

    Returns:
        The resolved writes dict (possibly empty).
    """
    mode_value = (
        juvenile_growth_mode if juvenile_growth_mode is not None else growth_mode
    )
    k_value = carrying_capacity
    if k_value is None and age_1_carrying_capacity is not None:
        k_value = age_1_carrying_capacity
    if k_value is None and old_juvenile_carrying_capacity is not None:
        k_value = old_juvenile_carrying_capacity
    # Only auto-detect K during initial build (no live Population).
    if k_value is None and allow_initial_k_detection and draft is not None:
        init_ind = draft.initial_individual_count
        if init_ind.size > 0 and init_ind.ndim >= 2 and init_ind.shape[1] >= 2:
            age_1_count = float(init_ind[:, 1, :].sum())
            if age_1_count >= 0.5:
                k_value = age_1_count
            else:
                total = float(init_ind.sum())
                if total >= 0.5:
                    k_value = total
    writes: dict[str, object] = {}
    if k_value is not None:
        writes["carrying_capacity"] = k_value
    if low_density_growth_rate is not None:
        writes["low_density_growth_rate"] = low_density_growth_rate
    if mode_value is not None:
        writes["growth_mode"] = mode_value
    if competition_strength is not None:
        writes["competition_strength"] = competition_strength
    if equilibrium_distribution is not None:
        writes["equilibrium_distribution"] = equilibrium_distribution
    return writes


def reproduction_writes(
    *,
    discrete_generation: bool,
    eggs_per_female: float | None,
    sex_ratio: float | None,
    sperm_displacement_rate: float | None,
    female_age_based_mating_rate: object | None,
    male_age_based_mating_rate: object | None,
    age_based_reproduction_rate: object | None,
    female_age_based_fertility: object | None,
    female_adult_mating_rate: float | None,
    male_adult_mating_rate: float | None,
    fixed_egg_count: bool | None,
) -> dict[str, object]:
    """Parse one ``reproduction()`` call into route-table writes.

    Args:
        discrete_generation: Whether the working draft is a discrete
            (Wright-Fisher) normalized draft; per-age flexible specs are
            rejected there.
        eggs_per_female: Base eggs per reproducing female.
        sex_ratio: Female fraction of offspring (0-1).
        sperm_displacement_rate: Fraction of stored sperm displaced.
        female_age_based_mating_rate: Per-age female mating probability.
        male_age_based_mating_rate: Per-age male mating probability.
        age_based_reproduction_rate: Per-age reproduction participation.
        female_age_based_fertility: Per-age fertility weight.
        female_adult_mating_rate: Adult female mating probability
            (discrete vocabulary; writes the adult cell).
        male_adult_mating_rate: Adult male mating probability.
        fixed_egg_count: Disable Poisson noise.

    Returns:
        The resolved writes dict (possibly empty).

    Raises:
        TypeError: When a per-age parameter is passed on a
            discrete-generation draft.
    """
    # Discrete drafts are normalized to 2 ages where age-0 does not
    # mate: per-age flexible specs are meaningless there and keep the
    # historical rejection of the former DiscreteConfigurator.
    if discrete_generation and (
        female_age_based_mating_rate is not None
        or male_age_based_mating_rate is not None
        or age_based_reproduction_rate is not None
        or female_age_based_fertility is not None
    ):
        raise TypeError(
            "reproduction() rejects per-age parameters on "
            "discrete-generation populations; use the discrete "
            "vocabulary (eggs_per_female, sex_ratio, "
            "female_adult_mating_rate, male_adult_mating_rate)"
        )
    writes: dict[str, object] = {}
    if eggs_per_female is not None:
        writes["eggs_per_female"] = eggs_per_female
    if sex_ratio is not None:
        writes["sex_ratio"] = sex_ratio
    if sperm_displacement_rate is not None:
        writes["sperm_displacement_rate"] = sperm_displacement_rate
    if female_age_based_mating_rate is not None:
        writes["female_age_based_mating_rate"] = female_age_based_mating_rate
    if male_age_based_mating_rate is not None:
        writes["male_age_based_mating_rate"] = male_age_based_mating_rate
    if age_based_reproduction_rate is not None:
        writes["age_based_reproduction_rate"] = age_based_reproduction_rate
    if female_age_based_fertility is not None:
        writes["female_age_based_fertility"] = female_age_based_fertility
    if female_adult_mating_rate is not None:
        writes["female_adult_mating_rate"] = female_adult_mating_rate
    if male_adult_mating_rate is not None:
        writes["male_adult_mating_rate"] = male_adult_mating_rate
    if fixed_egg_count is not None:
        writes["fixed_egg_count"] = fixed_egg_count
    return writes


def survival_writes(
    *,
    female_age_based_survival: object | None,
    male_age_based_survival: object | None,
    female_age0_survival: float | None,
    male_age0_survival: float | None,
) -> dict[str, object]:
    """Parse one ``survival()`` call into route-table writes.

    Args:
        female_age_based_survival: Female survival rates (flexible form).
        male_age_based_survival: Male survival rates (same forms).
        female_age0_survival: Female juvenile (age-0) survival.
        male_age0_survival: Male juvenile (age-0) survival.

    Returns:
        The resolved writes dict (possibly empty).
    """
    writes: dict[str, object] = {}
    if female_age_based_survival is not None:
        writes["female_age_based_survival"] = female_age_based_survival
    if male_age_based_survival is not None:
        writes["male_age_based_survival"] = male_age_based_survival
    if female_age0_survival is not None:
        writes["female_age0_survival"] = female_age0_survival
    if male_age0_survival is not None:
        writes["male_age0_survival"] = male_age0_survival
    return writes


def fitness_writes(
    viability: Mapping[str, float | Mapping[str, float]] | None,
    fecundity: Mapping[str, float | Mapping[str, float]] | None,
    sexual_selection: Mapping[str, float | Mapping[str, float]] | None,
    zygote_viability: Mapping[str, float | Mapping[str, float]] | None,
) -> dict[str, object]:
    """Parse one ``fitness()`` call into pattern-patch writes.

    Args:
        viability: Per-genotype viability fitness patches.
        fecundity: Per-genotype fecundity fitness patches.
        sexual_selection: Per-genotype mating success patches.
        zygote_viability: Per-genotype zygote-stage survival patches.

    Returns:
        The patch writes dict (possibly empty), keyed by patch name.
    """
    writes: dict[str, object] = {}
    for patch_name, patch_dict in (
        ("viability", viability),
        ("fecundity", fecundity),
        ("sexual_selection", sexual_selection),
        ("zygote_viability", zygote_viability),
    ):
        if patch_dict is not None:
            writes[patch_name] = patch_dict
    return writes


def expected_females_eggs(draft: ModelDraft, target_females: float) -> float:
    """Derive the total-egg override equivalent to *target_females*.

    Args:
        draft: The working draft supplying the current demographics.
        target_females: Target number of new adult females.

    Returns:
        The equivalent total egg production for the
        ``external_expected_eggs`` route.
    """
    return compute_expected_eggs_from_females(
        expected_num_new_adult_females=target_females,
        eggs_per_female=float(draft.eggs_per_female),
        age_based_survival_rates=draft.age_based_survival_rates,
        age_based_reproduction_rates=draft.age_based_reproduction_rates,
        female_age_based_fertility=draft.female_age_based_fertility,
        sex_ratio=float(draft.sex_ratio),
        new_adult_age=int(draft.new_adult_age),
        n_ages=int(draft.n_ages),
    )


# ── Commit targets ────────────────────────────────────────────────────────────


class _UpdateTarget(ABC):
    """Resolved commit destination for one runtime update (internal).

    Two concrete targets exist: the idle native session (writes resolve
    the population's live channel at operation time and publish straight
    onto the population) and the event transaction (writes stage into the
    callback's candidate and adopt only when the callback succeeds).
    Both resolve every model fact from the owning population lazily —
    targets never hold a parameter or metadata snapshot.
    """

    def __init__(self, pop: BasePopulation[Any]) -> None:
        """Bind the target to its owning population.

        Args:
            pop: The population whose values and metadata this target
                reads and publishes.
        """
        self.pop = pop

    @property
    def species(self) -> Species:
        """The population's genetic architecture."""
        return self.pop.species

    @property
    def registry(self) -> IndexRegistry:
        """The population's active index registry."""
        return self.pop.index_registry

    @property
    def tick(self) -> int:
        """The tick audit entries are stamped with."""
        return int(self.pop._tick)  # pyright: ignore[reportPrivateUsage]  # same-package owner state

    @property
    @abstractmethod
    def backend(self) -> SessionChannel | EventTransaction | None:
        """The native write channel (transaction or session adapter).

        ``SessionChannel`` serves idle targets, ``EventTransaction``
        serves callback-bound targets, and ``None`` means draft-only
        degradation (reference-path populations).
        """

    @abstractmethod
    def live_draft(self) -> ModelDraft:
        """The current native values as a fresh draft (full pull)."""

    @abstractmethod
    def declaration_draft(self) -> ModelDraft:
        """The declaration-side draft (no native pull)."""

    @abstractmethod
    def adopt(self, draft: ModelDraft) -> None:
        """Publish a committed draft as the target's working state."""

    @abstractmethod
    def audit(self, name: str, old: AuditValue, new: AuditValue, event: str = "update") -> None:
        """Record one committed typed parameter change."""

    @abstractmethod
    def read_metadata(self, name: str) -> object:
        """Read one recipe-metadata attribute at operation time."""

    @abstractmethod
    def publish_metadata(self, name: str, value: object) -> None:
        """Publish one recipe-metadata attribute after a native commit."""

    def publish_definition(self, definition: ModelDefinition) -> None:
        """Publish the current normalized declaration."""
        self.publish_metadata("_current_definition", definition)

    @abstractmethod
    def append_reconfiguration(self, preset_name: str, changes: dict[str, object]) -> None:
        """Record one committed preset reconfiguration."""


class _SessionTarget(_UpdateTarget):
    """Idle-session target: writes commit onto the live population."""

    def __init__(
        self, pop: BasePopulation[Any], backend: SessionChannel | EventTransaction | None,
    ) -> None:
        """Bind the population and its resolved native channel.

        Args:
            pop: The owning population.
            backend: ``_runtime_parameter_writer`` or
                ``_rust_lifecycle_backend`` (either may be ``None`` for
                reference-path populations; draft-only writes degrade).
        """
        super().__init__(pop)
        self._backend = backend

    @property
    def backend(self) -> SessionChannel | EventTransaction | None:
        """The resolved idle session channel."""
        return self._backend  # type: ignore[no-any-return]  # resolved from getattr channels typed object

    def live_draft(self) -> ModelDraft:
        """Pull the session-owned values into a fresh draft."""
        return self.pop.config

    def declaration_draft(self) -> ModelDraft:
        """Read the declaration metadata draft (never a runtime input)."""
        draft = self.pop._config  # pyright: ignore[reportPrivateUsage]  # same-package declaration metadata
        if draft is None:
            raise RuntimeError("Population configuration is not initialized.")
        return draft

    def adopt(self, draft: ModelDraft) -> None:
        """Commit the draft onto the population."""
        self.pop.set_config(draft)

    def audit(self, name: str, old: AuditValue, new: AuditValue, event: str = "update") -> None:
        """Append the typed change to the population's parameter log."""
        self.pop.log_param_value(name, old, new, event)

    def read_metadata(self, name: str) -> object:
        """Read the population's current recipe metadata."""
        return getattr(self.pop, name, None)

    def publish_metadata(self, name: str, value: object) -> None:
        """Write recipe metadata onto the population."""
        setattr(self.pop, name, value)

    def append_reconfiguration(self, preset_name: str, changes: dict[str, object]) -> None:
        """Record the reconfiguration in the population's provenance log.

        The log is attached lazily per instance (never a class-level
        default, which every population would share).
        """
        pop = self.pop
        # The lazily-attached provenance list: created per instance on
        # first reconfiguration (never a class-level default).
        existing = pop.__dict__.get("_reconfiguration_log")
        if isinstance(existing, list):
            log = cast("list[tuple[int, str, dict[str, object]]]", existing)
        else:
            log = []
            pop._reconfiguration_log = log  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]  # the sanctioned runtime provenance attach
        log.append((self.tick, preset_name, dict(changes)))


class _EventTarget(_UpdateTarget):
    """Event-transaction target: writes stage into the callback candidate."""

    def __init__(self, context: TickContext) -> None:
        """Bind the target to one callback's event scope.

        Args:
            context: The live :class:`TickContext` owning the event
                transaction, candidate draft, pending metadata, pending
                parameter log, and rollback actions.
        """
        super().__init__(context.population)
        self._context = context

    @property
    def backend(self) -> SessionChannel | EventTransaction | None:
        """The event's native transaction (``None`` without one)."""
        return self._context.transaction

    def live_draft(self) -> ModelDraft:
        """Return the event's isolated candidate (no session pull).

        Without a native transaction (manually constructed context), the
        population's committed draft is the answer, exactly like the
        candidate-free read path.
        """
        candidate = self._context.materialize_candidate()
        if candidate is not None:
            return candidate
        return self.pop.config

    def declaration_draft(self) -> ModelDraft:
        """Return the event's isolated candidate (already materialized)."""
        candidate = self._context.materialize_candidate()
        if candidate is not None:
            return candidate
        draft = self.pop._config  # pyright: ignore[reportPrivateUsage]  # same-package declaration metadata
        if draft is None:
            raise RuntimeError("Population configuration is not initialized.")
        return draft

    def adopt(self, draft: ModelDraft) -> None:
        """Replace the event's working candidate; adoption is atomic."""
        self._context.adopt_candidate(draft)

    def audit(self, name: str, old: AuditValue, new: AuditValue, event: str = "update") -> None:
        """Append the typed change to the event's pending log."""
        self._context.audit_sink()(name, old, new, event)

    def read_metadata(self, name: str) -> object:
        """Read the event's working metadata (population value until prepared)."""
        return self._context.metadata_value(name)

    def publish_metadata(self, name: str, value: object) -> None:
        """Publish metadata into the event's working copies."""
        self._context.publish_metadata(name, value)

    def append_reconfiguration(self, preset_name: str, changes: dict[str, object]) -> None:
        """Record the reconfiguration in the event's pending provenance."""
        self._context.append_reconfiguration(preset_name, changes)

    def add_rollback(self, action: Callable[[], None]) -> None:
        """Join one undo action to this event's rollback sequence."""
        self._context.add_rollback(action)


def idle_session_target(pop: BasePopulation[Any]) -> _SessionTarget:
    """Resolve the idle-session target for *pop* (population-side entry).

    Args:
        pop: The population to update outside any callback.

    Returns:
        A :class:`_SessionTarget` with the live native channel resolved
        at call time.

    Raises:
        RuntimeError: If a run holds the session borrow, or the
            population has no initialized configuration.
    """
    if getattr(pop, "_running", False) or getattr(pop, "_rust_run_active", False):
        raise RuntimeError("External parameter writes are forbidden during run")
    if pop._config is None:  # pyright: ignore[reportPrivateUsage]  # same-package initialization probe
        raise RuntimeError("Population configuration is not initialized.")
    backend: SessionChannel | EventTransaction | None = getattr(
        pop, "_runtime_parameter_writer", None
    )
    if backend is None:
        backend = getattr(pop, "_rust_lifecycle_backend", None)
    return _SessionTarget(pop, backend)


# ── The one writer-selection function ─────────────────────────────────────────


def runtime_writer(target: _UpdateTarget, writes: Mapping[str, object] | None) -> CoreConfigWriter:
    """Build the runtime writer for *target* (single selection point).

    Scalar-only batches read just their touched native scalars for
    validation and exact logs — unchanged fields stay declaration
    metadata and no genetics tensor is copied.  Every other batch starts
    from the full live draft (the event target's candidate already is
    one, so no extra pull happens there).

    Args:
        target: The resolved commit target.
        writes: Known method writes, used to decide the scalar fast
            path; ``None`` always selects the full-draft base.

    Returns:
        A fresh :class:`CoreConfigWriter` whose ``on_replace`` adopts the
        committed draft through the target.
    """
    backend = target.backend
    entries = [lookup_or_none(name) for name in writes] if writes else []
    read_scalar = getattr(backend, "get_scalar", None)
    if entries and read_scalar is not None and all(
        entry is not None and entry.contract_field in NATIVE_SCALAR_FIELDS
        for entry in entries
    ):
        draft = target.declaration_draft()
        current: dict[str, object] = {}
        for entry in entries:
            assert entry is not None and entry.config_field is not None
            value = float(read_scalar(entry.contract_field))
            current[entry.config_field] = (
                None if entry.contract_field == "external_expected_eggs" and value < 0
                else int(value) if entry.contract_field == "growth_mode" else value
            )
        base = draft._replace(**current)
    else:
        base = target.live_draft()
    return CoreConfigWriter(
        base,
        backend,
        on_replace=target.adopt,
        species=target.species,
        registry=target.registry,
        param_value_log=lambda name, old, new: target.audit(name, old, new),
    )


# ── Recipe metadata carrier and genetic candidate compilation ────────────────


def _as_presets(value: object) -> list[GeneticPreset]:
    """Narrow an untyped metadata read to the presets list."""
    return cast("list[GeneticPreset]", value)


def _as_gamete_list(value: object) -> GameteList:
    """Narrow an untyped metadata read to the gamete declarations list."""
    return cast("GameteList", value)


def _as_zygote_list(value: object) -> ZygoteList:
    """Narrow an untyped metadata read to the zygote declarations list."""
    return cast("ZygoteList", value)


class RuntimeDeclaration:
    """Recipe metadata driving one runtime genetic candidate compile.

    A transient per-operation carrier, not long-lived state: values are
    read from the owning population (or the prepared event metadata) at
    operation time and published back through the same channel after the
    candidate commits.
    """

    def __init__(
        self,
        presets: list[GeneticPreset],
        manual_gamete: list[tuple[int, str | None, GameteModifier]],
        manual_zygote: list[tuple[int, str | None, ZygoteModifier]],
        fitness_base: tuple[NDArray[np.float64], ...],
        fitness_steps: list[tuple[int, dict[str, object]]],
        compilation_key: object,
    ) -> None:
        """Bind the declaration carrier to caller-owned recipe lists.

        Args:
            presets: Registered presets in declaration order.
            manual_gamete: Explicit gamete modifier declarations.
            manual_zygote: Explicit zygote modifier declarations.
            fitness_base: Raw fitness baselines before any patch.
            fitness_steps: Ordered explicit fitness patch declarations.
            compilation_key: Identity token connecting the declaration to
                its compiled products.
        """
        self.presets = presets
        self.manual_gamete = manual_gamete
        self.manual_zygote = manual_zygote
        self.fitness_base = fitness_base
        self.fitness_steps = fitness_steps
        self.compilation_key = compilation_key


def read_declaration(target: _UpdateTarget) -> RuntimeDeclaration:
    """Read the current recipe metadata from *target* at operation time.

    Args:
        target: The resolved commit target.

    Returns:
        A fresh :class:`RuntimeDeclaration` carrying detached copies of
        the presets, manual modifiers, and fitness declarations.
    """
    definition = cast(
        "ModelDefinition | None", target.read_metadata("_current_definition")
    )
    if definition is not None:
        base = definition.fitness_base
        steps = list(definition.fitness_steps)
        key = (
            definition.compilation_key
            if definition.compilation_key is not None
            else object()
        )
    else:
        draft = target.declaration_draft()
        base = tuple(getattr(draft, name).copy() for name in FITNESS_FIELDS)
        steps = []
        key = object()
    return RuntimeDeclaration(
        presets=list(_as_presets(target.read_metadata("_presets"))),
        manual_gamete=list(_as_gamete_list(target.read_metadata("_manual_gamete"))),
        manual_zygote=list(_as_zygote_list(target.read_metadata("_manual_zygote"))),
        fitness_base=base,
        fitness_steps=steps,
        compilation_key=key,
    )


def build_runtime_definition(
    species: Species,
    draft: ModelDraft,
    registry: IndexRegistry,
    declaration: RuntimeDeclaration,
) -> ModelDefinition:
    """Capture a runtime declaration against explicit inputs.

    Runtime updates declare no journal, hooks, observation, or recording
    policies — those belong to the build; the captured definition carries
    the recipe metadata and compilation key only.

    Args:
        species: The genetic architecture.
        draft: The working draft to capture (detached on capture).
        registry: The active index registry (copied on capture).
        declaration: The recipe metadata to carry.

    Returns:
        The frozen declaration handed to the compiler or published as
        ``_current_definition``.
    """
    from natal.frontend.data.definition import ModelDefinition

    return ModelDefinition(
        species,
        bool(draft.discrete_generation),
        (),
        None,
        presets=tuple(declaration.presets),
        manual_gamete=tuple(declaration.manual_gamete),
        manual_zygote=tuple(declaration.manual_zygote),
        compilation_key=declaration.compilation_key,
        draft=draft,
        registry=registry,
        fitness_base=tuple(declaration.fitness_base),
        fitness_steps=tuple(declaration.fitness_steps),
    )


def compile_runtime_candidate(
    species: Species,
    draft: ModelDraft,
    registry: IndexRegistry,
    declaration: RuntimeDeclaration,
    *,
    preserve_fitness: bool,
) -> CompiledProducts:
    """Compile one isolated genetic candidate from runtime declarations.

    The recipes expand against the declaration's own detached copies of
    *draft* and *registry* (``ModelDefinition`` captures detached state),
    so a failed compile publishes nothing and the caller's arrays are
    never touched.

    Args:
        species: The genetic architecture the candidate compiles against.
        draft: The current working draft (compile input, not mutated).
        registry: The active index registry (copied by the capture).
        declaration: The operation's recipe metadata; its
            ``compilation_key`` is replaced with a fresh identity token,
            mirroring every other recompile.
        preserve_fitness: Keep the draft's current fitness arrays (map
            -only updates must not re-derive fitness overrides).

    Returns:
        The compiled candidate products ready for one native commit.
    """
    fitness = (
        {name: getattr(draft, name).copy() for name in FITNESS_FIELDS}
        if preserve_fitness
        else {}
    )
    declaration.compilation_key = object()
    result = compile_definition(
        build_runtime_definition(species, draft, registry, declaration)
    )
    config = result.config._replace(**fitness) if preserve_fitness else result.config
    return CompiledProducts(
        config, result.registry, result.gamete_modifiers, result.zygote_modifiers,
    )


def commit_genetic_update(
    target: _UpdateTarget,
    old: ModelDraft,
    new: ModelDraft,
    declaration: RuntimeDeclaration,
    gamete_modifiers: list[tuple[int, str | None, GameteModifier]],
    zygote_modifiers: list[tuple[int, str | None, ZygoteModifier]],
    *,
    publish_definition: bool = True,
) -> None:
    """Commit one validated genetic candidate to the target.

    Single publication point: the native ``refresh_params`` runs first;
    the draft, recipe metadata, derived modifier lists, declaration, and
    typed audit entries publish only after it succeeds.  A failure above
    this point therefore leaks no candidate state.

    Args:
        target: The resolved commit target.
        old: The draft the candidate was computed from (layout check and
            audit baseline).
        new: The compiled candidate draft.
        declaration: The recipe metadata to publish.
        gamete_modifiers: The candidate's derived gamete modifiers.
        zygote_modifiers: The candidate's derived zygote modifiers.
        publish_definition: Whether to publish the captured declaration;
            preset reconfiguration publishes its final recipe identities
            separately.

    Raises:
        ValueError: If the candidate changes the active type layout.
        RuntimeError: If the target has no native session channel.
    """
    if (
        old.n_ztypes != new.n_ztypes
        or old.n_gtypes != new.n_gtypes
        or old.ztype_names != new.ztype_names
        or old.gtype_names != new.gtype_names
    ):
        raise ValueError("Runtime genetic updates cannot change the active type layout.")
    backend = target.backend
    if backend is None:
        raise RuntimeError("A runtime genetic commit requires a native session")
    from natal.contracts.materialize import materialize_params
    from natal.frontend.configurator._writers import contract_to_draft_field

    backend.refresh_params(list(GENETIC_COMMIT_FIELDS), materialize_params(new))
    target.adopt(new)
    target.publish_metadata("_manual_gamete", list(declaration.manual_gamete))
    target.publish_metadata("_manual_zygote", list(declaration.manual_zygote))
    target.publish_metadata("_presets", list(declaration.presets))
    target.publish_metadata("_gamete_modifiers", list(gamete_modifiers))
    target.publish_metadata("_zygote_modifiers", list(zygote_modifiers))
    if publish_definition:
        target.publish_definition(
            build_runtime_definition(target.species, new, target.registry, declaration)
        )
    for field in GENETIC_COMMIT_FIELDS:
        name = contract_to_draft_field(field)
        target.audit(
            field,
            np.array(getattr(old, name), dtype=np.float64, copy=True),
            np.array(getattr(new, name), dtype=np.float64, copy=True),
            "genetics",
        )


# ── Shared genetic update operations ─────────────────────────────────────────


def apply_runtime_presets(target: _UpdateTarget, presets: Sequence[GeneticPreset]) -> None:
    """Apply genetic presets through *target* (runtime semantics).

    Preset registration is idempotent by object identity.  New presets
    trigger one isolated full compile; the commit always runs (a repeated
    ``presets()`` re-publishes the current candidate unchanged, exactly
    like the historical handle path).

    Args:
        target: The resolved commit target.
        presets: One or more presets to register and apply.
    """
    declaration = read_declaration(target)
    new_presets = [
        preset
        for preset in presets
        if not any(item is preset for item in declaration.presets)
    ]
    old = target.live_draft()
    if new_presets:
        declaration.presets.extend(new_presets)
        products = compile_runtime_candidate(
            target.species, old, target.registry, declaration, preserve_fitness=False,
        )
        commit_genetic_update(
            target, old, products.config, declaration,
            products.gamete_modifiers, products.zygote_modifiers,
        )
        return
    commit_genetic_update(
        target, old, old, declaration,
        list(_as_gamete_list(target.read_metadata("_gamete_modifiers"))),
        list(_as_zygote_list(target.read_metadata("_zygote_modifiers"))),
    )


def append_manual_modifiers(
    declaration: RuntimeDeclaration,
    gamete_modifiers: Sequence[GameteModifier] | None,
    zygote_modifiers: Sequence[ZygoteModifier] | None,
) -> bool:
    """Append manual modifier declarations with auto-assigned ids.

    Args:
        declaration: The operation's recipe metadata (mutated in place).
        gamete_modifiers: New gamete modifiers (may be empty or ``None``).
        zygote_modifiers: New zygote modifiers (same).

    Returns:
        Whether any modifier was appended.
    """
    appended = False
    for modifier in gamete_modifiers or ():
        declaration.manual_gamete.append(
            (next_modifier_id(declaration.manual_gamete), None, modifier)
        )
        appended = True
    for modifier in zygote_modifiers or ():
        declaration.manual_zygote.append(
            (next_modifier_id(declaration.manual_zygote), None, modifier)
        )
        appended = True
    return appended


def add_manual_modifier(
    target: _UpdateTarget,
    side: Literal["gamete", "zygote"],
    modifier: GameteModifier | ZygoteModifier,
    name: str | None,
    modifier_id: int | None,
    *,
    refresh: bool,
) -> int:
    """Register one manual modifier declaration through *target*.

    Args:
        target: The resolved commit target.
        side: Which inheritance stage the modifier affects.
        modifier: The modifier object.
        name: Optional human-readable name.
        modifier_id: Optional explicit ordering id; auto-assigned when
            ``None``.
        refresh: Whether to recompile the modifier maps and commit now
            (``False`` defers to a later batch refresh).

    Returns:
        The resolved modifier id.
    """
    declaration = read_declaration(target)
    old = target.live_draft()

    def compile_and_commit() -> None:
        """Recompile maps (preserving current fitness) and commit."""
        products = compile_runtime_candidate(
            target.species, old, target.registry, declaration, preserve_fitness=True,
        )
        commit_genetic_update(
            target, old, products.config, declaration,
            products.gamete_modifiers, products.zygote_modifiers,
        )

    if side == "gamete":
        gamete_modifier = cast("GameteModifier", modifier)
        declarations = declaration.manual_gamete
        ids = [mid for mid, _, _ in declarations]
        resolved_id = int(modifier_id) if modifier_id is not None else (max(ids) + 1 if ids else 0)
        declarations.append((resolved_id, name, gamete_modifier))
        declarations.sort(key=lambda item: item[0])
        if not refresh:
            # Deferred registration: publish the declaration lists now; the
            # derived lists gain the new modifier without a map recompile.
            target.publish_metadata("_manual_gamete", list(declarations))
            derived = list(_as_gamete_list(target.read_metadata("_gamete_modifiers")))
            target.publish_metadata("_gamete_modifiers", [*derived, (resolved_id, name, gamete_modifier)])
            return resolved_id
        compile_and_commit()
        return resolved_id
    zygote_modifier = cast("ZygoteModifier", modifier)
    declarations = declaration.manual_zygote
    ids = [mid for mid, _, _ in declarations]
    resolved_id = int(modifier_id) if modifier_id is not None else (max(ids) + 1 if ids else 0)
    declarations.append((resolved_id, name, zygote_modifier))
    declarations.sort(key=lambda item: item[0])
    if not refresh:
        target.publish_metadata("_manual_zygote", list(declarations))
        derived = list(_as_zygote_list(target.read_metadata("_zygote_modifiers")))
        target.publish_metadata("_zygote_modifiers", [*derived, (resolved_id, name, zygote_modifier)])
        return resolved_id
    compile_and_commit()
    return resolved_id


def recompile_modifier_maps(target: _UpdateTarget) -> None:
    """Rebuild the modifier maps from the population's derived lists.

    The population-side refresh spelling: the current derived modifier
    lists (presets already expanded) are re-applied to the Mendelian
    baseline and the derived offspring tensor recomputed, then committed
    like any other genetic candidate.

    Args:
        target: The resolved commit target.
    """
    from natal.frontend.configurator._registry_builder import rebuild_config_maps
    from natal.frontend.genetics.definition_compiler import (
        _CompileHost,  # pyright: ignore[reportPrivateUsage]  # canonical isolated recipe host
    )

    pop = target.pop
    declaration = read_declaration(target)
    gamete = list(pop._gamete_modifiers)  # pyright: ignore[reportPrivateUsage]  # the derived lists are the refresh input
    zygote = list(pop._zygote_modifiers)  # pyright: ignore[reportPrivateUsage]
    old = target.live_draft()
    host = _CompileHost(target.species, target.registry, old)
    new, _applied = rebuild_config_maps(
        target.species, old, target.registry,
        gamete_modifiers=gamete, zygote_modifiers=zygote,
        compress=False, host=host,
    )
    commit_genetic_update(target, old, new, declaration, gamete, zygote)


def reset_preset_fitness(target: _UpdateTarget) -> None:
    """Reset fitness to neutral and re-apply only preset fitness.

    Args:
        target: The resolved commit target.
    """
    declaration = read_declaration(target)
    declaration.fitness_base = tuple(
        np.ones_like(array) for array in declaration.fitness_base
    )
    declaration.fitness_steps = []
    old = target.live_draft()
    products = compile_runtime_candidate(
        target.species, old, target.registry, declaration, preserve_fitness=False,
    )
    commit_genetic_update(
        target, old, products.config, declaration,
        products.gamete_modifiers, products.zygote_modifiers,
    )


# ── RuntimeUpdater ────────────────────────────────────────────────────────────


class RuntimeUpdater:
    """Runtime parameter and genetics update handle.

    ``pop.update()`` and ``ctx.update()`` both return this type.  The
    syntax is identical to the build chain's domain methods
    (``pop.update().competition(...)``), but a runtime updater is a
    commit handle, not a builder: exactly eight domain methods exist and
    no build capability (``.build()``, ``.setup()``, ``.hooks()``,
    ``.initial_state()``, observation or recording configuration) is
    present at all.

    The handle stores its commit target only.  An idle updater resolves
    the population's live native channel and publishes straight onto the
    population; an event-bound updater (from ``ctx.update()``) stages
    into the callback's event transaction and adopts atomically when the
    callback succeeds, and every retained event-bound handle rejects use
    once the callback has returned.  No parameter snapshot is kept:
    reads and writes resolve against the session at operation time, so a
    retained handle keeps working across runs.

    Raises:
        RuntimeError: When an operation runs during an active native
            run (idle handles) or after the owning callback expired
            (event-bound handles).
    """

    _DOMAIN_METHODS = (
        "competition",
        "reproduction",
        "survival",
        "custom",
        "presets",
        "modifiers",
        "fitness",
        "reconfigure_preset",
    )

    def __init__(
        self,
        pop: BasePopulation[Any],
        *,
        context: TickContext | None = None,
    ) -> None:
        """Bind the updater to its commit target.

        Args:
            pop: The population whose values this handle updates.
            context: The owning callback's event scope for event-bound
                handles; ``None`` (the idle-session target) for handles
                created by ``pop.update()``.
        """
        self._pop = pop
        self._context = context

    def __repr__(self) -> str:
        """Return a summary naming the commit target."""
        if self._context is not None:
            target = f"event tick={self._context.tick}"
        else:
            target = "session"
        return f"RuntimeUpdater(target={target}, pop={self._pop.name!r})"

    def _resolve_target(self) -> _UpdateTarget:
        """Resolve the commit target and enforce its lifetime guards.

        Returns:
            The event target for event-bound handles (after the callback
            lifetime check), otherwise a freshly resolved idle-session
            target.

        Raises:
            RuntimeError: If the callback expired or a run holds the
                session borrow.
        """
        if self._context is not None:
            self._context.ensure_active()
            return _EventTarget(self._context)
        return idle_session_target(self._pop)

    # -- domain methods --------------------------------------------------------
    #
    # Every method is a thin shell: parse kwargs via the shared plain
    # functions, then one writer apply or one candidate compile+commit
    # against the resolved target.

    def competition(
        self,
        *,
        carrying_capacity: float | None = None,
        low_density_growth_rate: float | None = None,
        juvenile_growth_mode: int | str | None = None,
        growth_mode: int | str | None = None,
        competition_strength: float | None = None,
        expected_num_new_adult_females: float | None = None,
        equilibrium_distribution: NDArray[np.float64] | None = None,
        age_1_carrying_capacity: float | None = None,
        old_juvenile_carrying_capacity: float | None = None,
    ) -> RuntimeUpdater:
        """Update density-dependent competition parameters.

        Args:
            carrying_capacity: Equilibrium population at age 1 (K).
            low_density_growth_rate: Per-capita growth at low density (r).
            juvenile_growth_mode: Historical spelling of *growth_mode*.
            growth_mode: Regulation function (string or integer).
            competition_strength: Larval competition weight.
            expected_num_new_adult_females: Target adult females
                (Champer model); the derived egg override is computed
                from the current live demographics and declared.
            equilibrium_distribution: Custom (2, n_ages) array for the
                Champer equilibrium computation.
            age_1_carrying_capacity: Legacy alias for *carrying_capacity*.
            old_juvenile_carrying_capacity: Legacy alias.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the commit target is unavailable (expired
                callback or active run).
            ValueError: If a value fails route validation (zero writes).
        """
        target = self._resolve_target()
        writes = competition_writes(
            carrying_capacity=carrying_capacity,
            low_density_growth_rate=low_density_growth_rate,
            juvenile_growth_mode=juvenile_growth_mode,
            growth_mode=growth_mode,
            competition_strength=competition_strength,
            equilibrium_distribution=equilibrium_distribution,
            age_1_carrying_capacity=age_1_carrying_capacity,
            old_juvenile_carrying_capacity=old_juvenile_carrying_capacity,
            draft=None,
            allow_initial_k_detection=False,
        )
        if writes:
            writer = runtime_writer(target, writes)
            writer.apply(writes)
        if expected_num_new_adult_females is not None:
            self._declare_expected_females(target, float(expected_num_new_adult_females))
        return self

    def _declare_expected_females(self, target: _UpdateTarget, target_females: float) -> None:
        """Compute and declare the Champer egg override on the live values.

        Args:
            target: The resolved commit target.
            target_females: Target number of new adult females; the
                equivalent total egg production is derived from the
                current live demographics and committed via the
                ``external_expected_eggs`` route.
        """
        eggs = expected_females_eggs(target.live_draft(), target_females)
        writes: dict[str, object] = {"external_expected_eggs": eggs}
        writer = runtime_writer(target, writes)
        writer.apply(writes)

    def reproduction(
        self,
        *,
        eggs_per_female: float | None = None,
        sex_ratio: float | None = None,
        sperm_displacement_rate: float | None = None,
        female_age_based_mating_rate: float
        | list[float]
        | dict[int, float]
        | Callable[[int], float]
        | None = None,
        male_age_based_mating_rate: float
        | list[float]
        | dict[int, float]
        | Callable[[int], float]
        | None = None,
        age_based_reproduction_rate: float
        | list[float]
        | dict[int, float]
        | Callable[[int], float]
        | None = None,
        female_age_based_fertility: float
        | list[float]
        | dict[int, float]
        | Callable[[int], float]
        | None = None,
        female_adult_mating_rate: float | None = None,
        male_adult_mating_rate: float | None = None,
        fixed_egg_count: bool | None = None,
    ) -> RuntimeUpdater:
        """Update reproduction parameters.

        Args:
            eggs_per_female: Base eggs per reproducing female.
            sex_ratio: Female fraction of offspring (0-1).
            sperm_displacement_rate: Fraction of stored sperm displaced.
            female_age_based_mating_rate: Per-age female mating probability.
            male_age_based_mating_rate: Per-age male mating probability.
            age_based_reproduction_rate: Per-age reproduction participation.
            female_age_based_fertility: Per-age fertility weight.
            female_adult_mating_rate: Adult female mating probability
                (discrete vocabulary).
            male_adult_mating_rate: Adult male mating probability.
            fixed_egg_count: Disable Poisson noise.

        Returns:
            Self for chaining.

        Raises:
            TypeError: When a per-age parameter is passed on a
                discrete-generation population.
            RuntimeError: If the commit target is unavailable.
            ValueError: If a value fails route validation (zero writes).
        """
        target = self._resolve_target()
        writes = reproduction_writes(
            discrete_generation=target.declaration_draft().discrete_generation,
            eggs_per_female=eggs_per_female,
            sex_ratio=sex_ratio,
            sperm_displacement_rate=sperm_displacement_rate,
            female_age_based_mating_rate=female_age_based_mating_rate,
            male_age_based_mating_rate=male_age_based_mating_rate,
            age_based_reproduction_rate=age_based_reproduction_rate,
            female_age_based_fertility=female_age_based_fertility,
            female_adult_mating_rate=female_adult_mating_rate,
            male_adult_mating_rate=male_adult_mating_rate,
            fixed_egg_count=fixed_egg_count,
        )
        if writes:
            writer = runtime_writer(target, writes)
            writer.apply(writes)
        return self

    def survival(
        self,
        *,
        female_age_based_survival: float
        | list[float]
        | dict[int, float]
        | Callable[[int], float]
        | None = None,
        male_age_based_survival: float
        | list[float]
        | dict[int, float]
        | Callable[[int], float]
        | None = None,
        female_age0_survival: float | None = None,
        male_age0_survival: float | None = None,
    ) -> RuntimeUpdater:
        """Update survival rates.

        Per-age params accept flexible forms (scalar, list, dict, or
        callable).  The discrete ``female_age0_survival`` /
        ``male_age0_survival`` names keep working on both granularities.

        Args:
            female_age_based_survival: Female survival rates (flexible form).
            male_age_based_survival: Male survival rates (same forms).
            female_age0_survival: Female juvenile (age-0) survival.
            male_age0_survival: Male juvenile (age-0) survival.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the commit target is unavailable.
            ValueError: If a value fails route validation (zero writes).
        """
        target = self._resolve_target()
        writes = survival_writes(
            female_age_based_survival=female_age_based_survival,
            male_age_based_survival=male_age_based_survival,
            female_age0_survival=female_age0_survival,
            male_age0_survival=male_age0_survival,
        )
        if writes:
            writer = runtime_writer(target, writes)
            writer.apply(writes)
        return self

    def custom(self, **kwargs: bool | int | float | NDArray[np.float64]) -> RuntimeUpdater:
        """Write custom named slots on the live configuration.

        Multiple calls accumulate over the live custom values.  The
        complete candidate is validated and the native custom slots
        refresh before the draft and typed audit entries publish; a
        failed update leaves both unchanged.

        Args:
            **kwargs: Name-value pairs for custom slots.  Values must be
                ``bool``, ``int``, ``float``, or ``NDArray[np.float64]``.

        Returns:
            Self for chaining.

        Raises:
            TypeError: If a value's type is unsupported (zero writes).
            RuntimeError: If the commit target is unavailable.
        """
        from natal.frontend.data import build_custom_slots

        target = self._resolve_target()
        current = target.live_draft()
        merged = dict(current.custom)
        merged.update(kwargs)
        normalized = build_custom_slots(merged)
        if target.backend is not None:
            from natal.contracts.materialize import materialize_params

            candidate = current._replace(custom=normalized)
            target.backend.refresh_params(["custom_slots"], materialize_params(candidate))
        target.adopt(current._replace(custom=normalized))
        for name in sorted(set(current.custom) | set(normalized)):
            target.audit(f"custom.{name}", current.custom.get(name), normalized.get(name))
        return self

    def presets(self, *presets: GeneticPreset) -> RuntimeUpdater:
        """Apply genetic presets to the live population.

        Each preset encapsulates modifier callables, fitness patches,
        and optionally a cytoplasmic tag.  Registration is idempotent by
        object identity; new presets compile in one isolated candidate
        that commits only when every recipe succeeds.

        Args:
            *presets: One or more ``GeneticPreset`` instances.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the commit target is unavailable or the
                population has no native session.
            ValueError: If a recipe fails (nothing is published).
        """
        target = self._resolve_target()
        apply_runtime_presets(target, presets)
        return self

    def modifiers(
        self,
        gamete_modifiers: list[GameteModifier] | None = None,
        zygote_modifiers: list[ZygoteModifier] | None = None,
    ) -> RuntimeUpdater:
        """Register gamete / zygote modifiers and rebuild the maps.

        Args:
            gamete_modifiers: List of
                :class:`~natal.frontend.modifiers.GameteModifier`
                instances affecting meiosis.
            zygote_modifiers: List of
                :class:`~natal.frontend.modifiers.ZygoteModifier`
                instances affecting fertilization.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the commit target is unavailable or the
                population has no native session.
        """
        target = self._resolve_target()
        declaration = read_declaration(target)
        appended = append_manual_modifiers(declaration, gamete_modifiers, zygote_modifiers)
        old = target.live_draft()
        if appended:
            products = compile_runtime_candidate(
                target.species, old, target.registry, declaration, preserve_fitness=True,
            )
            commit_genetic_update(
                target, old, products.config, declaration,
                products.gamete_modifiers, products.zygote_modifiers,
            )
            return self
        commit_genetic_update(
            target, old, old, declaration,
            list(_as_gamete_list(target.read_metadata("_gamete_modifiers"))),
            list(_as_zygote_list(target.read_metadata("_zygote_modifiers"))),
        )
        return self

    def fitness(
        self,
        viability: Mapping[str, float | Mapping[str, float]] | None = None,
        fecundity: Mapping[str, float | Mapping[str, float]] | None = None,
        sexual_selection: Mapping[str, float | Mapping[str, float]] | None = None,
        zygote_viability: Mapping[str, float | Mapping[str, float]] | None = None,
        mode: str = "replace",
    ) -> RuntimeUpdater:
        """Write fitness values directly into the live fitness tensors.

        Each dict maps genotype-pattern strings (e.g. ``"WT|WT"``) to
        fitness multipliers.  *mode* can be ``"replace"`` (overwrite) or
        ``"multiply"`` (scale existing values).  Sex-specific fitness
        uses nested dicts; the ``@slab`` suffix writes one slab column
        and a bare pattern writes all slab columns.

        Args:
            viability: Per-genotype viability fitness.
            fecundity: Per-genotype fecundity fitness.
            sexual_selection: Per-genotype mating success fitness.
            zygote_viability: Per-genotype zygote-stage survival fitness.
            mode: ``"replace"`` or ``"multiply"``.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the commit target is unavailable.
            ValueError: If a pattern is malformed (zero writes publish).
        """
        writes = fitness_writes(viability, fecundity, sexual_selection, zygote_viability)
        if not writes:
            return self
        target = self._resolve_target()
        declaration = read_declaration(target)
        writer = runtime_writer(target, None)
        writer.apply(writes, mode=mode)
        new_draft = writer.draft
        step: dict[str, object] = {
            name: value
            for name, value in (
                ("viability", viability),
                ("fecundity", fecundity),
                ("sexual_selection", sexual_selection),
                ("zygote_viability", zygote_viability),
            )
            if value is not None
        }
        step["mode"] = mode
        declaration.fitness_steps.append((len(declaration.presets), deepcopy(step)))
        target.publish_definition(
            build_runtime_definition(target.species, new_draft, target.registry, declaration)
        )
        return self

    def reconfigure_preset(self, preset: GeneticPreset, **changes: object) -> RuntimeUpdater:
        """Modify a registered preset parameter and re-apply.

        Because ``presets()`` appends modifiers cumulatively, calling it
        again after changing a preset attribute would double-apply.
        This method resets the fitness declaration to neutral, then
        re-applies the preset so it writes onto a clean slate.

        Validation happens entirely before any mutation: if the preset
        is not registered or an attribute name is invalid, the exception
        is raised and the preset object is left unchanged (error-path
        state invariant).  Inside a callback, a later failure restores
        the preset's previous attribute values through the event's
        rollback sequence.

        Args:
            preset: A preset previously registered via :meth:`presets`.
            **changes: Attribute name / value pairs to update on *preset*.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If *preset* is not registered on this population.
            AttributeError: If any key in *changes* is not an attribute
                of *preset*.
            RuntimeError: If the commit target is unavailable.
        """
        target = self._resolve_target()
        registered = target.read_metadata("_presets")
        if not isinstance(registered, list) or preset not in registered:
            raise ValueError(
                f"Preset {preset.name!r} is not registered on this "
                f"population. Use presets() to register it first."
            )
        for attr in changes:
            if not hasattr(preset, attr):
                raise AttributeError(
                    f"{type(preset).__name__} {preset.name!r} has no "
                    f"attribute {attr!r}. Cannot reconfigure a non-existent "
                    f"parameter — this would silently create a stray attribute "
                    f"on the preset object."
                )
        updated = copy(preset)
        for attr, value in changes.items():
            setattr(updated, attr, value)
        declaration = read_declaration(target)
        declaration.presets = [
            updated if item is preset else item for item in declaration.presets
        ]
        # Reconfiguration explicitly resets manual fitness under the frozen
        # preset contract; ordinary refresh preserves the ordered declarations.
        declaration.fitness_base = tuple(
            np.ones_like(array) for array in declaration.fitness_base
        )
        declaration.fitness_steps = []
        old = target.live_draft()
        products = compile_runtime_candidate(
            target.species, old, target.registry, declaration, preserve_fitness=False,
        )
        commit_genetic_update(
            target, old, products.config, declaration,
            products.gamete_modifiers, products.zygote_modifiers,
            publish_definition=False,
        )
        if isinstance(target, _EventTarget):
            # Preserve the external preset's identity while allowing the
            # whole callback to fail after this individual reconfiguration
            # succeeds.
            previous = {attr: getattr(preset, attr) for attr in changes}

            def restore_preset() -> None:
                for attr, value in previous.items():
                    setattr(preset, attr, value)

            target.add_rollback(restore_preset)
        for attr, value in changes.items():
            setattr(preset, attr, value)
        # Preserve the external preset's registration identity in the
        # published declaration after the successful commit.
        declaration.presets = [
            preset if item is updated else item for item in declaration.presets
        ]
        target.publish_metadata("_presets", list(declaration.presets))
        target.publish_definition(
            build_runtime_definition(target.species, products.config, target.registry, declaration)
        )
        target.append_reconfiguration(preset.name, dict(changes))
        return self
