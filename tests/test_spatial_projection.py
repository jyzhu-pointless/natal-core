"""The spatial group compiler projects declarations; it never replays methods.

FRONTEND_REFACTOR_PLAN.md §4.4/§4.5-4 (P6): each deme's concrete
declarations are projected through the single declaration interpreter
(``builder/_declarations.py``) onto a fresh baseline and handed straight
to the model compiler — the template's consumed first values, builder-
method replay (``_builder_for_group``), and the ``_can_use_replace``
fallback are gone.  These tests pin both halves of the acceptance: no
builder method executes during a batched build, and each deme's compiled
result equals the equivalent single-population declaration.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.builder import SpatialPopulationBuilder, batch_setting


@pytest.fixture(scope="module")
def species() -> nt.Species:
    return nt.Species.from_dict(
        "spatial_projection",
        {"c1": {"l1": ["WT", "Dr"]}},
        somatic_labels=["default"],
    )


def test_batched_build_executes_no_builder_methods(species: nt.Species) -> None:
    """§4.5-4: a batched build must not re-execute builder methods."""
    import natal.frontend.builder._base as base_mod

    names = (
        "competition", "reproduction", "survival", "initial_state",
        "age_structure", "custom", "presets", "modifiers", "fitness",
    )
    calls: list[str] = []
    originals = {name: getattr(base_mod.PopulationBuilder, name) for name in names}

    def make_spy(name, original):
        def wrapper(self, *args, **kwargs):
            calls.append(name)
            return original(self, *args, **kwargs)
        return wrapper

    builder = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
        .setup(stochastic=False)
        .age_structure(3, 1)
        .initial_state(individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 50}})
        .survival(female_age_based_survival=[0.8, 0.9, 0.9], male_age_based_survival=[0.8, 0.9, 0.9])
        .reproduction(eggs_per_female=batch_setting([20.0, 10.0]))
        .competition(carrying_capacity=batch_setting([500.0, 200.0]))
    )
    try:
        for name in names:
            setattr(base_mod.PopulationBuilder, name, make_spy(name, originals[name]))
        before = len(calls)
        pop = builder.build()
        during_build = list(calls[before:])
    finally:
        for name in names:
            setattr(base_mod.PopulationBuilder, name, originals[name])
    assert during_build == [], f"builder methods executed during build: {during_build}"
    assert pop.deme(0).get_total_count() > 0


def test_each_deme_matches_the_equivalent_single_population(species: nt.Species) -> None:
    """§4.5-4: per-deme results equal equivalent single-population compiles."""
    spatial = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
        .setup(stochastic=False)
        .age_structure(3, 1)
        .initial_state(individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 50}})
        .survival(female_age_based_survival=[0.8, 0.9, 0.9], male_age_based_survival=[0.8, 0.9, 0.9])
        .reproduction(eggs_per_female=batch_setting([20.0, 10.0]))
        .competition(carrying_capacity=batch_setting([500.0, 200.0]))
        .build()
    )
    for index, eggs, capacity in ((0, 20.0, 500.0), (1, 10.0, 200.0)):
        single = (
            nt.AgeStructuredPopulation.setup(species, stochastic=False)
            .age_structure(3, 1)
            .initial_state(individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 50}})
            .survival(female_age_based_survival=[0.8, 0.9, 0.9], male_age_based_survival=[0.8, 0.9, 0.9])
            .reproduction(eggs_per_female=eggs)
            .competition(carrying_capacity=capacity)
            .build()
        )
        spatial_cfg = spatial._deme_object(index).config
        assert spatial_cfg.eggs_per_female == pytest.approx(single.config.eggs_per_female)
        assert spatial_cfg.carrying_capacity == pytest.approx(single.config.carrying_capacity)
        np.testing.assert_allclose(
            spatial_cfg.age_based_survival_rates, single.config.age_based_survival_rates
        )
        np.testing.assert_allclose(
            spatial_cfg.initial_individual_count, single.config.initial_individual_count
        )
        np.testing.assert_array_equal(
            spatial_cfg.zygotes_to_gametes_map, single.config.zygotes_to_gametes_map
        )


def test_genetically_distinct_groups_each_match_single_population(species: nt.Species) -> None:
    """§4.5-4 with *different genetics* per group: presets split the groups.

    Two genetic groups (per-deme ``presets(batch_setting(...))``) plus an
    ecology-only variant inside group A: every deme's compiled config —
    ecology scalars, initial arrays, genetic maps, fitness, offspring
    tensor — equals the equivalent single-population declaration.  Recipe
    call counts pin the compile contract: the template's declaration runs
    each preset once, group A inherits that cache (no re-run), and the
    genetically distinct group B compiles exactly once.
    """
    from natal.frontend.presets import PointMutation

    call_log: list[str] = []

    class CountingMutation(PointMutation):
        def gamete_modifier(self, host):  # type: ignore[override]
            call_log.append(self.name)
            return super().gamete_modifier(host)

    pm_a = CountingMutation("pm_a", "WT", target_allele="Dr", mutation_rate=0.1)
    pm_b = CountingMutation("pm_b", "WT", target_allele="Dr", mutation_rate=0.5)
    eggs = [20.0, 10.0, 15.0]
    caps = [500.0, 200.0, 800.0]
    presets = [pm_a, pm_b, pm_a]

    spatial = (
        nt.SpatialPopulation.builder(species, n_demes=3, pop_type="age_structured")
        .setup(stochastic=False)
        .age_structure(3, 1)
        .initial_state(individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 50}})
        .survival(female_age_based_survival=[0.8, 0.9, 0.9], male_age_based_survival=[0.8, 0.9, 0.9])
        .reproduction(eggs_per_female=batch_setting(eggs))
        .competition(carrying_capacity=batch_setting(caps))
        .presets(batch_setting(presets))
        .build()
    )
    # Snapshot before the reference single-population builds below run
    # their own recipes (each ``.presets()`` call compiles once by design).
    spatial_phase_calls = list(call_log)
    for i in range(3):
        single = (
            nt.AgeStructuredPopulation.setup(species, stochastic=False)
            .age_structure(3, 1)
            .initial_state(individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 50}})
            .survival(female_age_based_survival=[0.8, 0.9, 0.9], male_age_based_survival=[0.8, 0.9, 0.9])
            .reproduction(eggs_per_female=eggs[i])
            .competition(carrying_capacity=caps[i])
            .presets(presets[i])
            .build()
        )
        spatial_cfg = spatial._deme_object(i).config
        single_cfg = single.config
        assert spatial_cfg.eggs_per_female == pytest.approx(single_cfg.eggs_per_female)
        assert spatial_cfg.carrying_capacity == pytest.approx(single_cfg.carrying_capacity)
        for field in (
            "age_based_survival_rates", "initial_individual_count",
            "zygotes_to_gametes_map", "gametes_to_zygotes_map",
            "viability_fitness", "offspring_tensor",
        ):
            np.testing.assert_array_equal(
                getattr(spatial_cfg, field), getattr(single_cfg, field),
                err_msg=f"deme {i} field {field}",
            )
    # Template declaration ran each preset once; group A (demes 0, 2)
    # inherited that cache, group B compiled once — no per-deme re-runs.
    assert spatial_phase_calls.count("pm_a") == 1, f"pm_a recipe re-ran: {spatial_phase_calls}"
    assert spatial_phase_calls.count("pm_b") == 1, (
        f"pm_b recipe ran {spatial_phase_calls.count('pm_b')} times: {spatial_phase_calls}"
    )


def test_resolved_group_journal_expands_live_batchsetting_entries(species: nt.Species) -> None:
    """A journal still carrying live ``BatchSetting`` presets resolves per deme.

    ``_definition_for_compile`` normalizes journals before a build, but the
    group journal resolver must also honour a live ``presets`` entry whose
    ``preset_list`` still holds the ``BatchSetting`` itself: the per-deme
    ``_preset_<i>`` value wins, and a missing key falls back to the first
    value (the template's consumed placeholder).
    """
    from natal.frontend.presets import PointMutation

    first = PointMutation("live_first", "WT", target_allele="Dr", mutation_rate=0.1)
    other = PointMutation("live_other", "WT", target_allele="Dr", mutation_rate=0.5)
    builder = nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
    builder._declaration_log.append(  # pyright: ignore[reportPrivateUsage]  # journal-shaped live entry under test.
        ("presets", {"preset_list": (batch_setting([first, other]),)})
    )
    resolved = builder._resolved_group_journal(  # pyright: ignore[reportPrivateUsage]  # the resolver under contract test.
        {"_preset_0": other}
    )
    assert resolved == [("presets", {"__args__": (other,)})]
    # No per-deme value staged: the placeholder first value is used.
    fallback = builder._resolved_group_journal({})  # pyright: ignore[reportPrivateUsage]
    assert fallback == [("presets", {"__args__": (first,)})]


def test_replay_machinery_is_gone() -> None:
    """The retired mechanisms stay retired (§5.6-style inaccessibility)."""
    assert not hasattr(SpatialPopulationBuilder, "_builder_for_group")
    assert not hasattr(SpatialPopulationBuilder, "_can_use_replace")
    assert not hasattr(SpatialPopulationBuilder, "_build_variant_config")


def test_projector_rejects_unknown_methods(species: nt.Species) -> None:
    """An uninterpretable journal entry fails instead of being skipped."""
    from natal.frontend.builder._declarations import project_declaration_record
    from natal.frontend.builder._base import PopulationBuilder

    baseline = PopulationBuilder.for_age_structured(species)._config  # pyright: ignore[reportPrivateUsage]  # test-local baseline.
    with pytest.raises(ValueError, match="does not interpret"):
        project_declaration_record(
            species, [("no_such_method", {"x": 1})], base_draft=baseline
        )


class TestDeclarationProjectorBranches:
    """Cover the projector's validation and accumulation branches."""

    @staticmethod
    def _baseline(species: nt.Species):
        from natal.frontend.builder._base import PopulationBuilder

        return PopulationBuilder.for_age_structured(species)._config  # pyright: ignore[reportPrivateUsage]  # test-local baseline.

    @pytest.fixture(scope="class")
    def species(self) -> nt.Species:
        return nt.Species.from_dict(
            "projector_branches", {"c1": {"l1": ["WT", "Dr"]}}, somatic_labels=["default"]
        )

    def _project(self, species, journal, **kwargs):
        from natal.frontend.builder._declarations import project_declaration_record

        return project_declaration_record(
            species, journal, base_draft=self._baseline(species), **kwargs
        )

    def test_setup_flags_and_mode_validation(self, species) -> None:
        out = self._project(species, [
            ("setup", {"stochastic": False, "extreme_speed_mode": 2, "name": "sp", "compress": True}),
        ])
        assert out.draft.stochastic is False
        assert out.draft.extreme_speed_mode == 2
        assert out.name == "sp" and out.compress is True
        from natal.frontend.builder._declarations import apply_setup

        import pytest as _pytest

        with _pytest.raises(ValueError, match="extreme_speed_mode"):
            apply_setup(out.draft, {"extreme_speed_mode": 9})

    def test_age_structure_validation(self, species) -> None:
        with pytest.raises(ValueError, match="n_ages"):
            self._project(species, [("age_structure", {"n_ages": 1, "new_adult_age": 0})])
        with pytest.raises(ValueError, match="new_adult_age"):
            self._project(species, [("age_structure", {"n_ages": 3, "new_adult_age": 5})])

    def test_positional_journal_forms(self, species) -> None:
        out = self._project(species, [
            ("age_structure", {"__args__": (3, 1)}),
            ("initial_state", {"__args__": ({"female": {"WT|WT": {1: 4}}},)}),
        ])
        assert out.draft.n_ages == 3
        assert out.initial_distribution is not None
        assert out.draft.initial_individual_count.sum() == pytest.approx(4.0)

    def test_genetic_and_execution_accumulation(self, species) -> None:
        from natal.frontend.presets import PointMutation

        preset = PointMutation("pm_probe", "WT", target_allele="Dr", mutation_rate=0.1)
        selector = nt.IndividualSelector(ztype="WT|WT")
        out = self._project(species, [
            ("presets", {"__args__": (None, preset, preset)}),
            ("fitness", {"viability": {"WT|WT": 0.5}, "mode": "multiply"}),
            ("with_observation", {"groups": {"het": selector}, "collapse_age": True}),
            ("record_history", {"mode": "observation", "max_rows": 7}),
        ])
        assert out.presets == [preset]  # None skipped, identity deduped
        assert out.fitness_steps == [(1, {"viability": {"WT|WT": 0.5}, "mode": "multiply"})]
        assert out.observation_groups is not None and "het" in out.observation_groups
        assert out.observation_collapse_age is True
        assert out.history_mode == "observation" and out.history_max_rows == 7


def test_modifiers_journal_projects_into_manual_lists(species=None) -> None:
    """The projector accumulates declared modifiers with fresh ids."""
    from natal.frontend.builder._declarations import project_declaration_record
    from natal.frontend.builder._base import PopulationBuilder

    sp = nt.Species.from_dict(
        "projector_modifiers", {"c1": {"l1": ["WT", "Dr"]}}, somatic_labels=["default"]
    )

    class _GameteStub:
        pass

    class _ZygoteStub:
        pass

    gamete, zygote = _GameteStub(), _ZygoteStub()
    out = project_declaration_record(
        sp,
        [
            ("age_structure", {"__args__": (3, 1)}),
            ("modifiers", {"gamete_modifiers": [gamete], "zygote_modifiers": [zygote]}),
        ],
        base_draft=PopulationBuilder.for_age_structured(sp)._config,  # pyright: ignore[reportPrivateUsage]  # test-local baseline.
    )
    assert [entry[2] for entry in out.manual_gamete] == [gamete]
    assert [entry[2] for entry in out.manual_zygote] == [zygote]
    assert out.manual_gamete[0][0] == 0 and out.manual_zygote[0][0] == 0
