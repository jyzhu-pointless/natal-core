"""DemeSlice / Population surface parity contracts.

The internal ``PopulationView`` protocol (finalized decision #3) fixes
the aligned member set structurally, but it only sees members that are
*declared on the protocol*: adding a public member to one side alone
was invisible to it (the disclosed one-sided blind spot).  These tests
close that mechanically — every public instance member of a concrete
population must be classified either as ALIGNED (the protocol surface,
which ``DemeSlice`` mirrors exactly) or as POPULATION_ONLY (lifecycle,
history, import/export, and kind-specific conveniences).  A new member
on either side fails here until it is explicitly classified.
"""

from __future__ import annotations

import inspect

import pytest

from natal.frontend.population.age_structured import AgeStructuredPopulation
from natal.frontend.population.discrete_generation import (
    DiscreteGenerationPopulation,
)
from natal.frontend.spatial.population import DemeSlice

# The 15 aligned members declared by the internal PopulationView protocol.
ALIGNED = frozenset(
    {
        "name", "species", "config", "state", "params", "params_log",
        "index_registry", "presets", "definition",
        "get_total_count", "get_female_count", "get_male_count",
        "export_config", "export_state",
        "update",
    }
)

# Deme-specific writers/reads on the slice only (never on a population).
DEME_ONLY = frozenset({"index", "write_ecology", "write_genetics"})

# Public population members intentionally outside the aligned surface.
# Lifecycle, history/recording, observation, hook introspection, modifier
# and preset mutation, import/export of whole models, legacy conveniences,
# construction entry points, and age-structured-specific queries.
POPULATION_ONLY = frozenset(
    {
        # common (both kinds)
        "add_gamete_modifier", "add_preset", "add_zygote_modifier",
        "apply_preset", "builder", "clear_history",
        "compiled_hook_descriptors", "compute_allele_frequencies",
        "finish_simulation", "gamete_modifiers", "get_compiled_hooks",
        "has_python_callbacks", "has_python_hooks", "history",
        "import_config", "import_state", "is_failed", "is_finished",
        "log_param_change", "log_param_value",
        "observation", "observe", "params_log_details", "reapply_preset_fitness",
        "reconfiguration_log", "record_snapshot", "refresh_modifier_maps",
        "refresh_modifiers", "register_gamete_labels", "registry", "reset",
        "restore_checkpoint", "run", "run_tick", "set_config", "setup",
        "sex_ratio", "step", "tick", "total_females", "total_males",
        "total_population_size", "trigger_event", "zygote_modifiers",
        # age-structured only
        "genotypes_present", "get_adult_count", "get_age_distribution",
        "get_genotype_count", "n_ages", "new_adult_age",
    }
)


def _public_surface(cls: type) -> set[str]:
    """Public properties and callables across a class's MRO (no dunders)."""
    out: set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            continue
        for name, member in inspect.getmembers(klass):
            if name.startswith("_"):
                continue
            if isinstance(member, property) or callable(member):
                out.add(name)
    return out


@pytest.mark.parametrize(
    "cls",
    [DiscreteGenerationPopulation, AgeStructuredPopulation],
    ids=["discrete", "age"],
)
def test_population_members_are_classified(cls: type) -> None:
    """Every public population member is aligned or explicitly local.

    Catches the one-sided blind spot: a new public member appears and
    this failure forces the decision — align it (protocol + DemeSlice)
    or pin it as population-only here.
    """
    surface = _public_surface(cls)
    unclassified = surface - ALIGNED - POPULATION_ONLY
    assert not unclassified, (
        f"unclassified public members on {cls.__name__}: "
        f"{sorted(unclassified)} — add them to the aligned surface "
        f"(PopulationView + DemeSlice) or to POPULATION_ONLY"
    )
    missing_aligned = ALIGNED - surface
    assert not missing_aligned, (
        f"{cls.__name__} lost aligned members: {sorted(missing_aligned)}"
    )


def test_deme_slice_surface_is_exactly_aligned_plus_writers() -> None:
    """``DemeSlice`` exposes exactly the aligned members plus its three.

    Catches a member added to the slice alone (the other direction of
    the blind spot) and any aligned member quietly dropped from it.
    """
    surface = _public_surface(DemeSlice)
    expected = ALIGNED | DEME_ONLY
    assert surface == expected, (
        f"DemeSlice surface drifted: extra={sorted(surface - expected)} "
        f"missing={sorted(expected - surface)}"
    )


def test_classification_tables_do_not_overlap() -> None:
    """The three tables partition: no member classified twice."""
    assert not (ALIGNED & POPULATION_ONLY)
    assert not (ALIGNED & DEME_ONLY)
    assert not (POPULATION_ONLY & DEME_ONLY)
