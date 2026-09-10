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
    from natal.frontend.population._params_view import ParamsView
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


def _build_age(name: str, hook_calls: list | None = None) -> AgeStructuredPopulation:
    """Return a deterministic age-structured population (fractional counts)."""
    builder = (
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
    )
    for items, kwargs in hook_calls or []:
        builder = builder.hooks(*items, **kwargs)
    return builder.build()


def _build_discrete(
    name: str, hook_calls: list | None = None
) -> DiscreteGenerationPopulation:
    """Return a deterministic discrete-generation population."""
    builder = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(name), stochastic=False
        )
        .initial_state(individual_count={"female": {"WT|WT": 50}, "male": {"WT|WT": 40}})
        .survival(female_age0_survival=0.9, male_age0_survival=0.8)
        .reproduction(eggs_per_female=5, sex_ratio=0.5)
        .competition(carrying_capacity=400.0, low_density_growth_rate=2.0)
    )
    for items, kwargs in hook_calls or []:
        builder = builder.hooks(*items, **kwargs)
    return builder.build()


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
    holder: dict[str, object] = {}
    observed: dict[str, Any] = {}

    def hook(ctx: TickContext) -> int:
        """Read parameters around a pending event write."""
        pop = holder["pop"]
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

    pop = builder("QLWInHookReads", hook_calls=[((hook,), {"event": "early"})])
    holder["pop"] = pop
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


def test_transaction_without_tensor_reads_degrades_to_draft() -> None:
    """An event channel without tensor reads falls back safely.

    Catches an unconditional attribute access on the event channel that
    would crash instead of degrading to the draft path.  The channel is
    now handed to the view explicitly, so the degraded double binds
    directly to a :class:`ParamsView`.
    """
    from natal.frontend.population._params_view import ParamsView

    pop = _build_age("QLWDegradedChannel")
    expected = pop.config.carrying_capacity
    view = ParamsView(pop, validate=lambda: None, channel=object())
    assert view.carrying_capacity == expected
    assert view.female_age0_survival == float(
        pop.config.age_based_survival_rates[0, 0]
    )


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


# ── evaluator strengthening: exhaustive and boundary coverage ─────────────────


def _expected_snapshot_value(snapshot: ModelDraft, entry: Any) -> Any:
    """The snapshot-path value one route entry must reproduce."""
    field = getattr(snapshot, entry.config_field)
    if entry.kind == "bool":
        return bool(field)
    if entry.kind in ("scalar", "mode_enum", "slot"):
        if isinstance(field, np.ndarray):
            raw = field[entry.config_path] if entry.config_path else field[()]
            return int(raw) if entry.dtype is int else float(raw)
        if field is None:
            return None
        return float(field)
    if entry.kind == "age_vec":
        return None if field is None else field.copy()
    if entry.kind == "sex_row":
        if not entry.config_path:
            return None if field is None else field.copy()
        return field[entry.config_path].copy()
    return np.asarray(field)


def _assert_read_matches_snapshot(params: ParamsView, snapshot: ModelDraft, entry: Any) -> None:
    """Compare one field-level read against the snapshot-path value."""
    got = getattr(params, entry.name)
    expected = _expected_snapshot_value(snapshot, entry)
    if isinstance(expected, np.ndarray) or isinstance(got, np.ndarray):
        if expected is None or got is None:
            assert (got is None) == (expected is None)
        else:
            got_arr, exp_arr = np.asarray(got), np.asarray(expected)
            assert got_arr.shape == exp_arr.shape, (entry.name, got_arr.shape, exp_arr.shape)
            assert np.array_equal(got_arr, exp_arr), entry.name
    else:
        assert got == expected, (entry.name, got, expected)
        assert (got is None) == (expected is None)


@pytest.mark.parametrize(
    "builder", [_build_age, _build_discrete], ids=["age_structured", "discrete"]
)
def test_every_route_read_equals_the_snapshot_path(builder: AnyBuilder) -> None:
    """Every route entry resolves natively to the snapshot-path value.

    The shipped equivalence tests sample representative kinds; this sweep
    pins all of them, so a future route-table row whose contract field is
    misclassified (unreadable natively, or shape-mismatched after a
    rename) fails here instead of surfacing as a user-facing KeyError or
    a silently stale draft read. Checked after build and again after a
    run (committed-state reads).
    """
    from natal.frontend.configurator._routes import ROUTES

    pop = builder("QLWAllRoutes")
    for phase in ("fresh", "after-run"):
        snapshot = pop.config
        params = pop.params
        for name, entry in ROUTES.items():
            if entry.config_field is None:
                # Spatial-only rows reject reads by contract.
                with pytest.raises(AttributeError):
                    getattr(params, name)
                continue
            _assert_read_matches_snapshot(params, snapshot, entry)
        if phase == "fresh":
            pop.run(1, record_every=0)


def test_native_counts_bit_exact_through_the_recursive_split() -> None:
    """Counts stay bit-identical to numpy sums when the plane exceeds 128.

    The pairwise reduction splits recursively only above NumPy's 128-wide
    block; small fixtures never reach that branch against real NumPy
    (the Rust unit reference is the same algorithm, not independent).
    This test imports a crafted 240-element per-sex plane with
    magnitude-mixed counts (huge and small cells interleaved, an
    order-sensitive payload for pairwise summation — verified by a
    reversed-order numpy sum differing bitwise) and asserts the native
    per-sex and total counts equal the real ``numpy`` reductions exactly.
    """
    n_ages = 80
    rng = np.random.default_rng(777)
    flat = np.empty(2 * n_ages * 3, dtype=np.float64)
    flat[0::2] = 1e15 * (1.0 + 1e-7 * rng.random(flat[0::2].size))
    flat[1::2] = 1e4 * (1.0 + 1e-3 * rng.random(flat[1::2].size))
    ic = flat.reshape(2, n_ages, 3)
    # Data precheck: summation order is observable on this payload.
    assert float(ic.sum()) != float(ic.reshape(-1)[::-1].sum())

    pop = (
        nt.AgeStructuredPopulation.setup(species=_species("QLWBigPlaneCounts"), stochastic=False)
        .age_structure(n_ages=n_ages, new_adult_age=1)
        .initial_state(
            individual_count={"female": {"WT|WT": [10.0] * n_ages}, "male": {"WT|WT": [10.0] * n_ages}}
        )
        .survival(female_age_based_survival=[1.0] * n_ages, male_age_based_survival=[1.0] * n_ages)
        .reproduction(eggs_per_female=6, sex_ratio=0.5)
        .competition(juvenile_growth_mode=0, carrying_capacity=8000.0)
        .build()
    )
    pop.run(1, record_every=0)  # the session owns the state
    imported = pop.state._replace(individual_count=ic)  # pyright: ignore[reportAttributeAccessIssue]  # NamedTuple state container
    pop.import_state(imported)
    np.testing.assert_array_equal(pop.state.individual_count, ic)
    plane = int(ic.shape[1]) * int(ic.shape[2])
    assert plane > 128, "fixture must exercise the recursive split branch"
    assert pop.get_total_count() == float(ic.sum())
    assert pop.get_female_count() == float(ic[0].sum())
    assert pop.get_male_count() == float(ic[1].sum())


@pytest.mark.parametrize(
    "builder", [_build_age, _build_discrete], ids=["age_structured", "discrete"]
)
def test_bare_params_read_in_callback_matches_config_projection(builder: AnyBuilder) -> None:
    """A held ``pop.params`` view read inside a callback stays in sync.

    The mid-run draft fallback skips the lazy transaction projection, so
    it must still return the values the full ``pop.config`` projection
    returns at the same moment — including a change committed by an
    earlier callback in the same run (the draft-sync invariant the
    fallback relies on).
    """
    holder: dict[str, object] = {}
    observed: list[tuple[float, float]] = []

    def first_hook(ctx: TickContext) -> int:
        ctx.update().competition(carrying_capacity=650.0)
        return 0

    def held_view_hook(ctx: TickContext) -> int:
        # Bare view (no ctx binding), read before anything prepares the
        # callback's candidate projection.
        pop = holder["pop"]
        observed.append((pop.params.carrying_capacity, pop.config.carrying_capacity))
        return 0

    pop = builder(
        "QLWHeldViewInHook",
        hook_calls=[((first_hook,), {"event": "first"}), ((held_view_hook,), {"event": "first"})],
    )
    holder["pop"] = pop
    pop.run(1, record_every=0)
    assert observed == [(650.0, 650.0)]
    assert pop.params.carrying_capacity == 650.0
