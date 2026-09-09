"""Query-path lightweighting contracts (plan phase P1).

The three pinned behaviors:

- ``pop.params`` field reads (scalars, slots, vector rows, tensors) return
  exactly the values of the full ``pop.config`` snapshot path — after
  build, after pause-phase ``pop.update()`` writes, and inside a hook
  with a pending ``ctx.update()`` write (the event candidate).
- Field reads and per-sex count queries no longer pull full native
  snapshots: spying ``config_snapshot``/``state_snapshot`` must observe
  zero calls.
- ``observe()`` compiles its projection mask once instead of per query,
  with results identical to the per-query rebuild.

Each test names the concrete regression it would catch: a silent value
divergence between the fast path and the snapshot path, a reintroduced
full-state pull on a small read, or a stale/rebuilt observation mask.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.frontend.hooks.tick_context import TickContext
from natal.frontend.output.observation import Observation
from natal.frontend.patterns import IndividualSelector

if TYPE_CHECKING:
    from natal.backends.rust.rust_backend import RustLifecycleBackend
    from natal.frontend.data import ModelDraft
    from natal.frontend.population.age_structured import AgeStructuredPopulation
    from natal.frontend.population.discrete_generation import (
        DiscreteGenerationPopulation,
    )

AnyPopulation: TypeAlias = "AgeStructuredPopulation | DiscreteGenerationPopulation"
AnyBuilder: TypeAlias = "Callable[[str], AnyPopulation]"


def _species(name: str) -> nt.Species:
    """Return a fresh two-allele species (unique name per call site)."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _build_age(name: str) -> AgeStructuredPopulation:
    """Return a deterministic age-structured population (fractional counts)."""
    return (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 40.0, 0.0], "WT|Dr": [0.0, 10.0, 0.0]},
                "male": {"WT|WT": [0.0, 30.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.85, 0.7],
        )
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(juvenile_growth_mode=1, carrying_capacity=800.0)
        .build()
    )


def _build_discrete(name: str) -> DiscreteGenerationPopulation:
    """Return a deterministic discrete-generation population."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), stochastic=False
        )
        .initial_state(individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 40}})
        .survival(female_age0_survival=0.9, male_age0_survival=0.8)
        .reproduction(eggs_per_female=5, sex_ratio=0.5)
        .competition(carrying_capacity=400.0, low_density_growth_rate=2.0)
        .build()
    )


def _assert_params_equal_config(pop: AnyPopulation) -> None:
    """Assert every read shape matches the full-snapshot path exactly.

    Covers one representative of each route kind plus the contract-name
    tensor reads: a mismatch means the field-level fast path diverged
    from ``pop.config`` (the regression this guards).
    """
    snapshot: ModelDraft = pop.config
    params = pop.params
    # Plain scalars (session-resident and declaration-resident).
    assert params.carrying_capacity == snapshot.carrying_capacity
    assert params.eggs_per_female == snapshot.eggs_per_female
    assert params.sex_ratio == snapshot.sex_ratio
    assert params.growth_mode == snapshot.juvenile_growth_mode
    assert params.n_ages == snapshot.n_ages
    assert params.generation_time == snapshot.generation_time
    assert params.external_expected_eggs == snapshot.external_expected_eggs
    assert isinstance(params.carrying_capacity, float)
    # Slot and vector kinds.
    assert params.female_age0_survival == snapshot.age_based_survival_rates[0, 0]
    assert (
        params.competition_strength
        == snapshot.age_based_relative_competition_strength[1]
    )
    np.testing.assert_array_equal(
        np.asarray(params.female_age_based_survival),
        snapshot.age_based_survival_rates[0],
    )
    np.testing.assert_array_equal(
        np.asarray(params.age_based_reproduction_rate),
        snapshot.age_based_reproduction_rates,
    )
    np.testing.assert_array_equal(
        np.asarray(params.female_age_based_fertility),
        snapshot.female_age_based_fertility,
    )
    # Genetics tensors by route name and by contract name.
    np.testing.assert_array_equal(np.asarray(params.viability), snapshot.viability_fitness)
    np.testing.assert_array_equal(
        np.asarray(params.offspring_tensor), snapshot.offspring_tensor
    )
    np.testing.assert_array_equal(
        np.asarray(params.meiosis_map), snapshot.zygotes_to_gametes_map
    )
    np.testing.assert_array_equal(
        np.asarray(params.survival_rates), snapshot.age_based_survival_rates
    )
    # Undeclared equilibrium reads as None on both paths.
    eq_params = params.equilibrium_distribution
    eq_snapshot = snapshot.equilibrium_individual_distribution
    if eq_params is None or eq_snapshot is None:
        assert eq_params is None and eq_snapshot is None
    else:
        np.testing.assert_array_equal(np.asarray(eq_params), eq_snapshot)
    # Blueprint flags read from the draft on both paths.
    assert params.stochastic == snapshot.stochastic


def _build_age_declared(name: str) -> AgeStructuredPopulation:
    """Return a deterministic age population with a declared equilibrium."""
    return (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"WT|WT": [0.0, 40.0, 0.0]}, "male": {"WT|WT": [0.0, 30.0, 0.0]}}
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.85, 0.7],
        )
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(
            juvenile_growth_mode=1,
            carrying_capacity=800.0,
            equilibrium_distribution=[[50.0, 40.0, 30.0], [50.0, 40.0, 30.0]],
        )
        .build()
    )


def test_params_field_reads_equal_full_snapshot_age_structured() -> None:
    """Field reads equal the snapshot after build and after update writes.

    Catches a fast path that bypasses committed session values (stale
    scalar/tensor after a pause-phase ``pop.update()``).
    """
    pop = _build_age("QLWAgeEquivalence")
    _assert_params_equal_config(pop)
    declared = _build_age_declared("QLWAgeEquivalenceDeclared")
    _assert_params_equal_config(declared)
    np.testing.assert_array_equal(
        np.asarray(declared.params.equilibrium_distribution),
        declared.config.equilibrium_individual_distribution,
    )
    pop.update().competition(carrying_capacity=1234.5)
    pop.update().reproduction(eggs_per_female=9, sex_ratio=0.4)
    pop.update().survival(female_age0_survival=0.77)
    assert pop.params.carrying_capacity == 1234.5
    assert pop.params.female_age0_survival == 0.77
    _assert_params_equal_config(pop)


def test_params_field_reads_equal_full_snapshot_discrete() -> None:
    """Discrete field reads equal the snapshot after build and updates."""
    pop = _build_discrete("QLWDiscreteEquivalence")
    _assert_params_equal_config(pop)
    pop.update().competition(carrying_capacity=250.0)
    pop.update().survival(male_age0_survival=0.6)
    assert pop.params.carrying_capacity == 250.0
    assert pop.params.male_age0_survival == 0.6
    _assert_params_equal_config(pop)


@pytest.mark.parametrize(
    "builder", [_build_age, _build_discrete], ids=["age_structured", "discrete"]
)
def test_custom_slots_visible_through_config_after_update(
    builder: AnyBuilder,
) -> None:
    """Custom slot writes land in the config projection for both kinds.

    Catches a divergence between the write channel and the custom-slots
    section of the native snapshot (the read path for user slots).
    """
    pop = builder("QLWCustomSlots")
    assert pop.config.custom == {}
    pop.update().custom(flag=True, weight=2.5)
    assert pop.config.custom["flag"] is True
    assert pop.config.custom["weight"] == 2.5


@pytest.mark.parametrize(
    "builder", [_build_age, _build_discrete], ids=["age_structured", "discrete"]
)
def test_in_hook_reads_return_event_candidate(
    builder: AnyBuilder,
) -> None:
    """In-hook field reads see the pending ctx.update() write, in value.

    The pending write must be visible through the field-level read
    immediately (candidate values), through the in-hook public config
    projection, and through the committed config after the run — any
    divergence means the fast path bypasses staged transaction values.
    """
    pop = builder("QLWInHookReads")
    observed: dict[str, Any] = {}

    def hook(ctx: TickContext) -> int:
        """Read parameters around a pending event write."""
        observed["before"] = ctx.params.carrying_capacity
        observed["before_tensor"] = np.array(ctx.params.survival_rates, copy=True)
        ctx.update().competition(carrying_capacity=650.0)
        ctx.update().survival(female_age0_survival=0.5)
        # Field-level reads must see the staged candidate...
        observed["after"] = ctx.params.carrying_capacity
        observed["after_slot"] = ctx.params.female_age0_survival
        observed["after_tensor"] = np.array(ctx.params.survival_rates, copy=True)
        # ...exactly like the public config projection (the candidate).
        config = pop.config
        observed["config_after"] = config.carrying_capacity
        observed["config_tensor_after"] = np.array(
            config.age_based_survival_rates, copy=True
        )
        # Blueprint flags have no session value: they read the candidate.
        observed["flag"] = ctx.params.stochastic
        return 0

    pop.register_hooks(hook, event="early")
    before_config = pop.config.carrying_capacity
    before_cell = float(pop.config.age_based_survival_rates[0, 0])
    pop.run(1, record_every=0)
    assert observed["before"] == before_config
    assert observed["after"] == 650.0
    assert observed["config_after"] == 650.0
    assert observed["after_slot"] == 0.5
    # The staged tensor candidate matches the candidate projection, and
    # the pending survival write is visible in both.
    np.testing.assert_array_equal(observed["after_tensor"], observed["config_tensor_after"])
    assert observed["after_tensor"][0, 0] == pytest.approx(0.5)
    assert observed["before_tensor"][0, 0] == pytest.approx(before_cell)
    assert observed["flag"] is pop.config.stochastic
    # The candidate committed: the pause-phase snapshot carries the write.
    assert pop.config.carrying_capacity == 650.0
    assert pop.params.carrying_capacity == 650.0
    assert pop.params.female_age0_survival == 0.5


@pytest.mark.parametrize(
    "builder", [_build_age, _build_discrete], ids=["age_structured", "discrete"]
)
def test_param_reads_and_counts_avoid_full_snapshot_pulls(
    builder: AnyBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Small reads never pull the full config or state snapshots.

    This is the negative contract of the phase: with the cache stale
    (right after a run), the retired path pulled ``config_snapshot`` for
    any field read and ``state_snapshot`` for any count. Zero calls
    proves the fast path; a nonzero count means the full-pull regression
    came back.
    """
    pop = builder("QLWNoFullPull")
    pop.run(1, record_every=0)  # mark the state cache stale
    backend = pop._rust_lifecycle_backend
    assert backend is not None
    calls = {"config": 0, "state": 0}
    backend_cls = type(backend)
    original_config = backend_cls.config_snapshot
    original_state = backend_cls.state_snapshot

    def spy_config(session: Any, draft: ModelDraft) -> ModelDraft:
        calls["config"] += 1
        return original_config(session, draft)

    def spy_state(session: Any) -> Any:
        calls["state"] += 1
        return original_state(session)

    monkeypatch.setattr(backend_cls, "config_snapshot", spy_config)
    monkeypatch.setattr(backend_cls, "state_snapshot", spy_state)

    # Field-level reads.
    _ = pop.params.carrying_capacity
    _ = pop.params.female_age0_survival
    _ = np.asarray(pop.params.survival_rates)
    _ = np.asarray(pop.params.viability_fitness)
    assert calls == {"config": 0, "state": 0}

    # Per-sex counts.
    _ = pop.get_total_count()
    _ = pop.get_female_count()
    _ = pop.get_male_count()
    assert calls == {"config": 0, "state": 0}

    # Sanity: the spy instruments do observe pulls when they happen, so
    # the zero assertions above cannot pass vacuously.
    _ = pop.config
    assert calls["config"] == 1
    _ = pop.state  # stale cache -> the retired path pulls state_snapshot
    assert calls["state"] == 1


@pytest.mark.parametrize(
    "builder,survival_change",
    [
        (_build_age, {"female_age_based_survival": [1.0, 0.5, 0.5]}),
        (_build_discrete, {"female_age0_survival": 0.5}),
    ],
    ids=["age_structured", "discrete"],
)
def test_native_counts_match_numpy_reductions_across_states(
    builder: AnyBuilder,
    survival_change: dict[str, object],
) -> None:
    """Native counts stay exactly equal to the numpy-sum semantics.

    Deterministic runs produce fractional counts, so any summation-order
    divergence between the native reduction and the retired Python sums
    over state-snapshot arrays would fail exact equality here. Checks
    the fresh build, mid-run ticks, and a post-update state.
    """
    pop = builder("QLWCountEquivalence")

    def assert_counts() -> None:
        # Pin each kind's exact retired semantics: age returns raw float
        # sums, discrete returns int(round(sum)) — Python banker's
        # rounding included.
        ic = pop.state.individual_count
        if isinstance(pop, nt.DiscreteGenerationPopulation):
            assert pop.get_total_count() == int(round(float(ic.sum())))
            assert pop.get_female_count() == int(round(float(ic[0].sum())))
            assert pop.get_male_count() == int(round(float(ic[1].sum())))
        else:
            assert float(pop.get_total_count()) == float(ic.sum())
            assert float(pop.get_female_count()) == float(ic[0].sum())
            assert float(pop.get_male_count()) == float(ic[1].sum())

    assert_counts()  # fresh build
    pop.run(1, record_every=0)
    assert_counts()  # mid-run
    pop.run(5, record_every=0)
    assert_counts()  # after several ticks
    pop.update().survival(**survival_change)
    pop.run(2, record_every=0)
    assert_counts()  # after an update that changed survival


@pytest.mark.parametrize(
    "builder", [_build_age, _build_discrete], ids=["age_structured", "discrete"]
)
def test_counts_fall_back_to_local_container_without_session(
    builder: AnyBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a session the counts still come from the local container.

    Catches a wiring that unconditionally requires a backend and crashes
    for populations that have not initialized a session.
    """
    pop = builder("QLWCountFallback")
    ic = pop.state.individual_count
    if isinstance(pop, nt.DiscreteGenerationPopulation):
        expected = (
            int(round(float(ic.sum()))),
            int(round(float(ic[0].sum()))),
            int(round(float(ic[1].sum()))),
        )
    else:
        expected = (float(ic.sum()), float(ic[0].sum()), float(ic[1].sum()))
    monkeypatch.setattr(pop, "_rust_lifecycle_backend", None)
    assert pop.get_total_count() == expected[0]
    assert pop.get_female_count() == expected[1]
    assert pop.get_male_count() == expected[2]


@pytest.mark.parametrize(
    "builder", [_build_age, _build_discrete, _build_age_declared],
    ids=["age_structured", "discrete", "age_declared_equilibrium"],
)
def test_params_reads_fall_back_to_draft_without_session(
    builder: AnyBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a native channel the reads use the draft, same values.

    Catches a fallback that would diverge from the snapshot values or
    crash for populations whose session has not been created (declared
    equilibrium, slots, vectors, tensors, and flags included).
    """
    pop = builder("QLWParamsFallback")
    monkeypatch.setattr(pop, "_rust_lifecycle_backend", None)
    _assert_params_equal_config(pop)


def test_reads_during_active_run_fall_back_to_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While a run holds the session borrow, reads use the draft.

    Catches a fast path that would call into the borrowed session during
    a run (native RuntimeError) instead of matching ``pop.config``'s
    draft behavior.
    """
    pop = _build_age("QLWRunGuard")
    monkeypatch.setattr(pop, "_rust_run_active", True)
    assert pop.params.carrying_capacity == pop.config.carrying_capacity
    assert pop.params.n_ages == pop.config.n_ages


def test_uninitialized_config_still_raises_canonical_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structural read on an uninitialized config raises AttributeError.

    Catches a fast path that would crash with TypeError/None instead of
    the documented "config has not been initialized" error.
    """
    pop = _build_age("QLWUninitialized")
    monkeypatch.setattr(pop, "_config", None)
    with pytest.raises(AttributeError, match="config has not been initialized"):
        _ = pop.params.n_ages


def test_transaction_without_tensor_reads_degrades_to_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transaction-like object without tensor reads falls back safely.

    Catches an unconditional attribute access on the event channel that
    would crash instead of degrading to the draft path.
    """
    pop = _build_age("QLWDegradedChannel")
    expected = pop.config.carrying_capacity
    monkeypatch.setattr(pop, "_event_transaction", object(), raising=False)
    assert pop.params.carrying_capacity == expected


def test_context_without_transaction_reads_the_draft() -> None:
    """A TickContext without a native transaction reads the live draft."""
    pop = _build_age("QLWCtxNoTransaction")
    ctx = TickContext(pop, tick=0, deme_id=0, state=None)
    assert ctx.params.carrying_capacity == pop.config.carrying_capacity
    assert ctx.params.female_age0_survival == pop.config.age_based_survival_rates[0, 0]


def _build_observed_age(name: str, groups: OrderedDict[str, IndividualSelector]) -> Any:
    """Return a labelled age-structured population with an explicit rule."""
    return (
        nt.AgeStructuredPopulation.setup(species=_species(name), stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 40.0, 0.0], "WT|Dr": [0.0, 10.0, 0.0]},
                "male": {"WT|WT": [0.0, 30.0, 0.0]},
            }
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.85, 0.7],
        )
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(juvenile_growth_mode=1, carrying_capacity=800.0)
        .with_observation(groups=groups, collapse_age=False)
        .build()
    )


def _reference_projection(pop: AnyPopulation) -> NDArray[np.float64]:
    """Project the current state through an explicit mask rebuild."""
    obs = pop.observation
    layout = pop.history.schema.population
    mask = obs.build_mask(layout.n_sexes, layout.n_ages, layout.n_ztypes)
    backend: RustLifecycleBackend | None = pop._rust_lifecycle_backend
    assert backend is not None
    _tick, values = backend.observe_current(mask, [0], obs.collapse_age, False)
    shape = (obs.n_groups, layout.n_sexes)
    if not obs.collapse_age:
        shape += (layout.n_ages,)
    return values.reshape(shape)


def test_observe_reuses_one_compiled_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """observe() compiles the mask once; results match the rebuild path.

    A per-query rebuild (the retired behavior) would increment the
    ``build_mask`` spy on every call; a stale cache would produce values
    different from a fresh ``build_mask`` projection.
    """
    groups: OrderedDict[str, IndividualSelector] = OrderedDict(
        (
            ("wild", IndividualSelector(ztype="WT|WT")),
            ("drive", IndividualSelector(ztype="WT|Dr")),
        )
    )
    pop = _build_observed_age("QLWObserveMask", groups)
    reference = _reference_projection(pop)

    calls = {"n": 0}
    original_build_mask = Observation.build_mask

    def spy_build_mask(self: Observation, n_sexes: int, n_ages: int, n_ztypes: int) -> Any:
        calls["n"] += 1
        return original_build_mask(self, n_sexes, n_ages, n_ztypes)

    monkeypatch.setattr(Observation, "build_mask", spy_build_mask)

    results = [pop.observe() for _ in range(4)]
    assert calls["n"] == 1  # compiled once, reused for every later query
    for result in results:
        assert result.tick == pop.tick
        np.testing.assert_array_equal(result.values, reference)
    assert results[0].axes == pop.observation.axes
    assert results[0].labels["group"] == pop.observation.labels


def test_observe_values_stable_across_runs() -> None:
    """observe() after runs equals an explicit projection of the same state."""
    groups: OrderedDict[str, IndividualSelector] = OrderedDict(
        (("wild", IndividualSelector(ztype="WT|WT")),)
    )
    pop = _build_observed_age("QLWObserveStable", groups)
    first = pop.observe()
    pop.run(2, record_every=0)
    second = pop.observe()

    assert second.tick == pop.tick == first.tick + 2
    np.testing.assert_array_equal(second.values, _reference_projection(pop))
    assert first.values.sum() < second.values.sum()
