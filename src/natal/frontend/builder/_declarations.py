"""Pure projection of declared builder calls onto a draft.

FRONTEND_REFACTOR_PLAN.md §4.2/§4.4: the ordered declaration journal is
the structured record of what the user declared, and this module is the
one interpreter that turns those declarations into draft state.  The
chain methods delegate here for their immediate writes, and the spatial
group compiler feeds each group's concrete declarations through the same
functions — no builder method is re-executed to generate a group.

Every ``apply_*`` function is pure with respect to its inputs: it takes a
draft (and the context the route tables need) and returns the derived
draft.  ``project_declaration_record`` interprets a whole journal into a
``ProjectedDeclarations`` bundle: the draft plus the structured genetic
and execution declarations (presets, manual modifiers, fitness steps,
hooks, observation) that :func:`compile_definition` compiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast

if TYPE_CHECKING:
    from natal.frontend.genetics import Species
    from natal.frontend.model.draft import ModelDraft
    from natal.frontend.model.initial_state import InitialDistributionDeclaration
    from natal.frontend.patterns import IndividualSelector
    from natal.frontend.registry.index import IndexRegistry

from natal.frontend.builder._runtime import (
    competition_writes,
    expected_females_eggs,
    reproduction_writes,
    survival_writes,
)
from natal.frontend.presets import GeneticPreset

__all__ = [
    "ProjectedDeclarations",
    "apply_age_structure",
    "apply_competition",
    "apply_custom",
    "apply_reproduction",
    "apply_setup",
    "apply_survival",
    "project_declaration_record",
]


def _write(
    draft: ModelDraft,
    writes: Mapping[str, object],
    *,
    species: Species | None,
    registry: IndexRegistry | None,
    on_replace: Optional[Callable[[ModelDraft], None]],
) -> ModelDraft:
    """Apply route-table writes through the build-path writer."""
    from natal.frontend.builder._writers import DraftWriter

    writer = DraftWriter(draft, on_replace=on_replace, species=species, registry=registry)
    writer.apply(writes)
    return writer.draft


def _draft_with_initial(
    draft: ModelDraft,
    species: Species | None,
    initial: InitialDistributionDeclaration | None,
) -> ModelDraft:
    """Derive the declared initial arrays into a replacement draft (pure)."""
    from natal.frontend.model.initial_state import (
        resolve_age_structured_initial_individual_count,
        resolve_age_structured_initial_sperm_storage,
        resolve_discrete_initial_individual_count,
    )

    if initial is None or species is None:
        return draft
    if draft.discrete_generation:
        counts = resolve_discrete_initial_individual_count(
            species=species, distribution=initial.individual_count
        )
        return draft._replace(initial_individual_count=counts)
    counts = resolve_age_structured_initial_individual_count(
        species=species, distribution=initial.individual_count,
        n_ages=draft.n_ages, new_adult_age=draft.new_adult_age,
    )
    overrides: dict[str, object] = {"initial_individual_count": counts}
    if initial.sperm_storage is not None:
        overrides["initial_sperm_storage"] = resolve_age_structured_initial_sperm_storage(
            species=species, sperm_storage=initial.sperm_storage,
            n_ages=draft.n_ages, new_adult_age=draft.new_adult_age,
        )
    return draft._replace(**overrides)


def apply_setup(draft: ModelDraft, kwargs: Mapping[str, Any]) -> ModelDraft:
    """Project one ``setup(...)`` declaration's draft flags."""
    overrides: dict[str, object] = {}
    if kwargs.get("stochastic") is not None:
        overrides["stochastic"] = kwargs["stochastic"]
    if kwargs.get("continuous_sampling") is not None:
        overrides["continuous_sampling"] = kwargs["continuous_sampling"]
    if kwargs.get("fixed_egg_count") is not None:
        overrides["fixed_egg_count"] = kwargs["fixed_egg_count"]
    mode = kwargs.get("extreme_speed_mode")
    if mode is not None:
        if mode not in (0, 1, 2, 3):
            raise ValueError(
                "extreme_speed_mode must be one of 0 (off), 1 "
                "(multinomial), 2 (poisson), 3 (multinomial + poisson); "
                f"got {mode!r}"
            )
        overrides["extreme_speed_mode"] = int(mode)
    return draft._replace(**overrides) if overrides else draft


def apply_competition(
    draft: ModelDraft,
    kwargs: Mapping[str, Any],
    *,
    species: Species | None,
    registry: IndexRegistry | None,
    initial: InitialDistributionDeclaration | None,
    on_replace: Optional[Callable[[ModelDraft], None]] = None,
) -> ModelDraft:
    """Project one ``competition(...)`` declaration onto *draft*."""
    writes = competition_writes(
        carrying_capacity=kwargs.get("carrying_capacity"),
        low_density_growth_rate=kwargs.get("low_density_growth_rate"),
        juvenile_growth_mode=kwargs.get("juvenile_growth_mode"),
        growth_mode=kwargs.get("growth_mode"),
        competition_strength=kwargs.get("competition_strength"),
        equilibrium_distribution=kwargs.get("equilibrium_distribution"),
        age_1_carrying_capacity=kwargs.get("age_1_carrying_capacity"),
        old_juvenile_carrying_capacity=kwargs.get("old_juvenile_carrying_capacity"),
        draft=_draft_with_initial(draft, species, initial),
        allow_initial_k_detection=True,
    )
    if writes:
        draft = _write(draft, writes, species=species, registry=registry, on_replace=on_replace)
    expected = kwargs.get("expected_num_new_adult_females")
    if expected is not None:
        eggs = expected_females_eggs(_draft_with_initial(draft, species, initial), float(expected))
        draft = _write(
            draft, {"external_expected_eggs": eggs},
            species=species, registry=registry, on_replace=on_replace,
        )
    return draft


def apply_reproduction(
    draft: ModelDraft,
    kwargs: Mapping[str, Any],
    *,
    species: Species | None,
    registry: IndexRegistry | None,
    on_replace: Optional[Callable[[ModelDraft], None]] = None,
) -> ModelDraft:
    """Project one ``reproduction(...)`` declaration onto *draft*."""
    writes = reproduction_writes(
        discrete_generation=draft.discrete_generation,
        eggs_per_female=kwargs.get("eggs_per_female"),
        sex_ratio=kwargs.get("sex_ratio"),
        sperm_displacement_rate=kwargs.get("sperm_displacement_rate"),
        female_age_based_mating_rate=kwargs.get("female_age_based_mating_rate"),
        male_age_based_mating_rate=kwargs.get("male_age_based_mating_rate"),
        age_based_reproduction_rate=kwargs.get("age_based_reproduction_rate"),
        female_age_based_fertility=kwargs.get("female_age_based_fertility"),
        female_adult_mating_rate=kwargs.get("female_adult_mating_rate"),
        male_adult_mating_rate=kwargs.get("male_adult_mating_rate"),
        fixed_egg_count=kwargs.get("fixed_egg_count"),
    )
    if writes:
        draft = _write(draft, writes, species=species, registry=registry, on_replace=on_replace)
    return draft


def apply_survival(
    draft: ModelDraft,
    kwargs: Mapping[str, Any],
    *,
    species: Species | None,
    registry: IndexRegistry | None,
    on_replace: Optional[Callable[[ModelDraft], None]] = None,
) -> ModelDraft:
    """Project one ``survival(...)`` declaration onto *draft*."""
    writes = survival_writes(
        female_age_based_survival=kwargs.get("female_age_based_survival"),
        male_age_based_survival=kwargs.get("male_age_based_survival"),
        female_age0_survival=kwargs.get("female_age0_survival"),
        male_age0_survival=kwargs.get("male_age0_survival"),
    )
    if writes:
        draft = _write(draft, writes, species=species, registry=registry, on_replace=on_replace)
    return draft


def apply_custom(
    draft: ModelDraft,
    custom_kwargs: dict[str, Any],
    kwargs: Mapping[str, Any],
) -> ModelDraft:
    """Project one ``custom(...)`` declaration onto *draft*."""
    from natal.frontend.model import build_custom_slots

    merged = dict(custom_kwargs)
    merged.update(kwargs)
    normalized = build_custom_slots(merged)
    custom_kwargs.clear()
    custom_kwargs.update(normalized)
    return draft._replace(custom=normalized)


def apply_age_structure(
    draft: ModelDraft,
    species: Species,
    kwargs: Mapping[str, Any],
) -> ModelDraft:
    """Project one ``age_structure(...)`` declaration: rebuild the dimensions."""
    n_ages = int(kwargs["n_ages"])
    new_adult_age = int(kwargs["new_adult_age"])
    generation_time = kwargs.get("generation_time")
    if n_ages <= 1:
        raise ValueError(f"n_ages must be at least 2, got {n_ages}")
    if new_adult_age < 0 or new_adult_age >= n_ages:
        raise ValueError(f"new_adult_age must be in [0, {n_ages}), got {new_adult_age}")
    from natal.frontend.model import build_population_config

    bp = species.get_config_blueprint()
    return build_population_config(
        n_genotypes=bp["n_genotypes"],
        n_gtypes=bp["n_gtypes"],
        n_ages=n_ages,
        n_glabs=draft.n_glabs,
        n_slabs=draft.n_slabs,
        gamete_labels=species.gamete_labels,
        somatic_labels=species.somatic_labels,
        zygotes_to_gametes_map=bp["zygotes_to_gametes_map"],
        gametes_to_zygotes_map=bp["gametes_to_zygotes_map"],
        new_adult_age=new_adult_age,
        generation_time=generation_time,
        stochastic=bool(draft.stochastic),
        continuous_sampling=bool(draft.continuous_sampling),
        fixed_egg_count=bool(draft.fixed_egg_count),
        has_sex_chromosomes=draft.has_sex_chromosomes,
        female_only_by_sex_chrom=bp["female_only_by_sex_chrom"],
        male_only_by_sex_chrom=bp["male_only_by_sex_chrom"],
    )


@dataclass
class ProjectedDeclarations:
    """The draft and structured declarations one journal projects to.

    Everything a group compile needs: the projected draft (ecology,
    flags, initial arrays), the structured genetic declarations
    (:func:`compile_definition` consumes them), and the execution
    declarations carried for publication.
    """

    draft: ModelDraft
    initial_distribution: InitialDistributionDeclaration | None = None
    presets: list[GeneticPreset] = field(default_factory=list[GeneticPreset])
    manual_gamete: list[tuple[int, None, Any]] = field(default_factory=list[tuple[int, None, Any]])
    manual_zygote: list[tuple[int, None, Any]] = field(default_factory=list[tuple[int, None, Any]])
    fitness_steps: list[tuple[int, dict[str, Any]]] = field(default_factory=list[tuple[int, dict[str, Any]]])
    hook_calls: list[tuple[tuple[object, ...], dict[str, Any]]] = field(default_factory=list[tuple[tuple[object, ...], dict[str, Any]]])
    observation_groups: Mapping[str, IndividualSelector] | None = None
    observation_collapse_age: bool = False
    history_mode: str = "raw"
    history_max_rows: Optional[int] = None
    compress: bool = False
    declared_zygote_types: set[str] | set[int] | None = None
    name: Optional[str] = None


def project_declaration_record(
    species: Species,
    journal: list[tuple[str, dict[str, Any]]],
    *,
    base_draft: ModelDraft,
    initial_distribution: InitialDistributionDeclaration | None = None,
    registry: IndexRegistry | None = None,
) -> ProjectedDeclarations:
    """Interpret one declaration journal into draft state and declarations.

    The single interpreter for declared calls (§4.2): the chain methods
    and the spatial group compiler both reduce to it, so a group's config
    is the projection of its concrete declarations — never the re-execution
    of builder methods.  Entries are applied in declaration order; a
    failing entry raises exactly as the original call did.

    Args:
        species: Species defining the genetic architecture.
        journal: Ordered ``(method_name, explicit_kwargs)`` declarations
            with any batch values already resolved to concrete ones.
        base_draft: The draft to project onto (a fresh granularity
            baseline or an existing draft).
        initial_distribution: Initial-distribution declaration carried in
            separately when the caller already holds it.
        registry: Registry for route-table writers, when one is bound.

    Returns:
        The projected bundle (draft plus structured declarations).

    Raises:
        ValueError: If an entry's declaration is invalid, or names a
            method the projector does not interpret.
    """
    from natal.frontend.genetics.compile import next_modifier_id

    out = ProjectedDeclarations(draft=base_draft, initial_distribution=initial_distribution)
    for method_name, kwargs in journal:
        kwargs = {k: v for k, v in kwargs.items() if v is not None} or dict(kwargs)
        if method_name == "setup":
            out.draft = apply_setup(out.draft, kwargs)
            if kwargs.get("name") is not None:
                out.name = kwargs["name"]
            if kwargs.get("compress"):
                out.compress = True
            declared = kwargs.get("declared_zygote_types", kwargs.get("declared_genotypes"))
            if declared is not None:
                out.declared_zygote_types = set(declared)
        elif method_name == "age_structure":
            positional = kwargs.get("__args__", ())
            call = dict(kwargs)
            if positional:
                # Positional spelling journals under "__args__"; restore
                # the parameter names the projector reads.
                for index, name in enumerate(("n_ages", "new_adult_age", "generation_time")):
                    if index < len(positional):
                        call[name] = positional[index]
            out.draft = apply_age_structure(out.draft, species, call)
            # Subsequent resolvers read the initial arrays; re-derive them
            # on the new dimensions, exactly as the chained call does.
            out.draft = _draft_with_initial(out.draft, species, out.initial_distribution)
        elif method_name == "competition":
            out.draft = apply_competition(
                out.draft, kwargs, species=species, registry=registry,
                initial=out.initial_distribution,
            )
        elif method_name == "reproduction":
            out.draft = apply_reproduction(out.draft, kwargs, species=species, registry=registry)
        elif method_name == "survival":
            out.draft = apply_survival(out.draft, kwargs, species=species, registry=registry)
        elif method_name == "custom":
            merged = dict(out.draft.custom)
            out.draft = apply_custom(out.draft, merged, kwargs)
        elif method_name == "initial_state":
            from natal.frontend.model.initial_state import (
                InitialDistributionDeclaration,
            )

            positional = kwargs.get("__args__", ())
            counts = kwargs.get("individual_count")
            sperm = kwargs.get("sperm_storage")
            if positional:
                counts = positional[0]
                sperm = positional[1] if len(positional) > 1 else None
            cast_counts = cast("Any", counts)
            cast_sperm = cast("Any", sperm)
            out.initial_distribution = InitialDistributionDeclaration.capture(cast_counts, cast_sperm)
            out.draft = _draft_with_initial(out.draft, species, out.initial_distribution)
        elif method_name == "presets":
            items = kwargs.get("__args__", ())
            for preset in items:
                if preset is None:
                    continue
                if not any(item is preset for item in out.presets):
                    out.presets.append(preset)
        elif method_name == "modifiers":
            for modifier in kwargs.get("gamete_modifiers") or ():
                out.manual_gamete.append((next_modifier_id(cast("Any", out.manual_gamete)), None, modifier))
            for modifier in kwargs.get("zygote_modifiers") or ():
                out.manual_zygote.append((next_modifier_id(cast("Any", out.manual_zygote)), None, modifier))
        elif method_name == "fitness":
            from copy import deepcopy

            step: dict[str, object] = {
                name: value
                for name, value in (
                    ("viability", kwargs.get("viability")),
                    ("fecundity", kwargs.get("fecundity")),
                    ("sexual_selection", kwargs.get("sexual_selection")),
                    ("zygote_viability", kwargs.get("zygote_viability")),
                )
                if value is not None
            }
            if step:
                step["mode"] = kwargs.get("mode", "replace")
                out.fitness_steps.append((len(out.presets), deepcopy(step)))
        elif method_name == "hooks":
            items = kwargs.get("__args__", ())
            out.hook_calls.append((tuple(items), {
                "event": kwargs.get("event"),
                "priority": kwargs.get("priority"),
                "deme": kwargs.get("deme", "*"),
                "name": kwargs.get("name"),
            }))
        elif method_name == "with_observation":
            out.observation_groups = dict(kwargs["groups"])
            out.observation_collapse_age = bool(kwargs.get("collapse_age", False))
        elif method_name == "record_history":
            out.history_mode = kwargs.get("mode", "raw")
            out.history_max_rows = kwargs.get("max_rows")
        else:
            raise ValueError(
                f"The declaration projector does not interpret {method_name!r}; "
                "its journal entry cannot be projected onto a group draft."
            )
    return out
