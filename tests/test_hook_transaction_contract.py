"""Independent event-candidate and sampler boundary regressions."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from natal.frontend.hooks._transaction import EventTransaction
from natal.frontend.hooks.tick_context import TickContext
from tests.test_review_runtime_regressions import _population


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_samplers_support_limits_and_invalid_arguments(model: Literal["age", "discrete"]) -> None:
    """Degenerate distributions are exact; nondegenerate draws obey support.

    These are sampler support and dispatch checks, not a distribution-fit claim.
    Six-sigma mean checks separately reject incorrect normal location/scale:
    N=4096 independent normal draws have standard error 2/sqrt(4096).
    """
    def sample(ctx: TickContext) -> int:
        np.testing.assert_array_equal(ctx.rng.uniform(7.0, 7.0, (2, 3)), np.full((2, 3), 7.0))
        np.testing.assert_array_equal(ctx.rng.normal(3.0, 0.0, 3), np.full(3, 3.0))
        assert ctx.rng.normal(3.0, 0.0) == 3.0
        assert ctx.rng.integers(7, 7, endpoint=True) == 7
        integer_values = np.asarray(ctx.rng.integers(3, size=(4, 3)))
        assert np.all((integer_values >= 0) & (integer_values < 3))
        assert ctx.rng.binomial(8, 0.0) == 0
        np.testing.assert_array_equal(ctx.rng.binomial(8, 1.0, 3), np.full(3, 8))
        np.testing.assert_array_equal(ctx.rng.binomial(np.array([2, 5]), np.array([0.0, 1.0])), [0, 5])
        np.testing.assert_array_equal(ctx.rng.binomial(np.array([2, 5]), 1.0, (2, 2)), [[2, 5], [2, 5]])
        values = np.asarray(ctx.rng.uniform(-2.0, 5.0, 32))
        assert np.all((values >= -2) & (values < 5))
        counts = np.asarray(ctx.rng.binomial(8, 0.5, 32))
        assert np.all((counts >= 0) & (counts <= 8))
        normal = np.asarray(ctx.rng.normal(3.0, 2.0, 4096))
        assert abs(normal.mean() - 3.0) < 6 * 2 / np.sqrt(4096)
        # Under the normal null, Var(s^2)=2*sigma^4/(N-1).
        assert abs(normal.var(ddof=1) - 4.0) < 6 * 4 * np.sqrt(2 / 4095)
        for bad in (
            lambda: ctx.rng.uniform(1, 0),
            lambda: ctx.rng.uniform(0, np.inf),
            lambda: ctx.rng.normal(0, -1),
            lambda: ctx.rng.binomial(-1, .5),
            lambda: ctx.rng.binomial(2**63, .5),
            lambda: ctx.rng.binomial(3, 1.1),
            lambda: ctx.rng.random(-1),
        ):
            with pytest.raises(ValueError):
                bad()
        assert np.asarray(ctx.rng.random((0, 2))).shape == (0, 2)
        return 0
    _population(f"SamplerContract_{model}", model, callback=sample).run(1)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_prior_callback_commit_survives_later_failure(model: Literal["age", "discrete"]) -> None:
    """Each callback commits independently, including genetic/custom candidates."""
    def commit(ctx: TickContext) -> int:
        ctx.params.tensor_write("viability_fitness", np.full_like(ctx.params.viability_fitness.array, .5))
        ctx.update().custom(marker=31)
        return 0
    def fail(ctx: TickContext) -> int:
        ctx.params.tensor_write("viability_fitness", np.full_like(ctx.params.viability_fitness.array, .2))
        ctx.update().custom(marker=99)
        raise ValueError("later callback")
    pop = _population(
        f"EarlierCallback_{model}",
        model,
        stochastic=False,
        hook_calls=[((commit,), {"event": "first"}), ((fail,), {"event": "first"})],
    )
    with pytest.raises(ValueError, match="later callback"):
        pop.run(1)
    np.testing.assert_array_equal(pop.params.viability_fitness.array, .5)
    assert pop.config.custom["marker"] == 31


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_transaction_rejects_expired_channels(model: Literal["age", "discrete"]) -> None:
    """A retained native handle cannot bypass Python lifetime validation."""
    retained: list[EventTransaction] = []
    def callback(ctx: TickContext) -> int:
        transaction = ctx._transaction
        assert transaction is not None
        retained.append(transaction)
        assert transaction.get_scalar("carrying_capacity") == 100000
        transaction.apply({"carrying_capacity": 345.0})
        # The public read/write surface observes this same transaction.
        assert transaction.get_scalar("carrying_capacity") == 345
        values = transaction.get_custom_slots()
        values["native"] = True
        transaction.set_custom_slots(values)
        tensor = transaction.get_tensor("viability_fitness")
        transaction.tensor_write("viability_fitness", tensor)
        return 0
    pop = _population(f"NativeLifetime_{model}", model, callback=callback)
    pop.run(1)
    transaction = retained[0]
    for operation in (
        transaction.ecology,
        transaction.state_arrays,
        transaction.validate_state,
        transaction.get_custom_slots,
        lambda: transaction.set_custom_slots({}),
        lambda: transaction.get_scalar("carrying_capacity"),
        lambda: transaction.get_tensor("viability_fitness"),
        lambda: transaction.apply({"carrying_capacity": 7.0}),
        lambda: transaction.tensor_write("viability_fitness", np.ones(1)),
        lambda: transaction.refresh_params([], object()),
        lambda: transaction.sample("uniform", 0.0, 1.0),
    ):
        with pytest.raises(RuntimeError, match="expired"):
            operation()


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_parameter_only_callback_does_not_materialize_python_state(
    model: Literal["age", "discrete"], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writable NumPy state boundary is entered only for state or metrics."""
    import natal.frontend.hooks.tick_context as context_module
    def reject_state(*args: object, **kwargs: object) -> None:
        raise AssertionError("parameter-only callback requested state")
    monkeypatch.setattr(context_module, "state_view_for", reject_state)
    def callback(ctx: TickContext) -> int:
        ctx.update().competition(carrying_capacity=90.0)
        ctx.rng.random()
        return 0
    pop = _population(f"LazyState_{model}", model, callback=callback)
    pop.run(1)
    assert pop.params.carrying_capacity == 90.0


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 2, 3), (1, 2, 2, 3)])
def test_custom_arrays_preserve_rank_and_isolation(shape: tuple[int, ...]) -> None:
    """Custom shapes survive native storage and checkpoint recovery exactly."""
    pop = _population(f"CustomShape_{len(shape)}", "discrete")
    values = np.arange(int(np.prod(shape)) or 1, dtype=np.float64).reshape(shape)
    expected = values.copy()
    pop.update().custom(payload=values)
    values[...] = -7
    np.testing.assert_array_equal(pop.config.custom["payload"], expected)
    pop.record_snapshot()
    pop.update().custom(payload=np.zeros(shape))
    pop.restore_checkpoint(0)
    np.testing.assert_array_equal(pop.config.custom["payload"], expected)


@pytest.mark.parametrize("value", [{1: "bad"}, {4: 1}, {1: -1}, ["bad"], [-1], ("bad",), (-1,), -1])
def test_direct_age_constructor_rejects_invalid_sperm(value: object) -> None:
    """Invalid counts or out-of-range ages cannot initialize sperm state."""
    from natal import AgeStructuredPopulation
    template = _population("InvalidDirectSperm", "age", stochastic=False)
    with pytest.raises((TypeError, ValueError)):
        AgeStructuredPopulation(template.species, template.config, initial_sperm_storage={"WT|WT": {"WT|WT": value}})


@pytest.mark.parametrize("value", [[True], {1: "bad"}, 3])
def test_direct_age_constructor_rejects_invalid_individual_distribution(value: object) -> None:
    """Malformed per-age counts fail before a population can run."""
    from natal import AgeStructuredPopulation
    template = _population("InvalidDirectIndividual", "age", stochastic=False)
    with pytest.raises(TypeError):
        AgeStructuredPopulation(template.species, template.config, initial_individual_count={"female": {"WT|WT": value}})


def test_direct_age_constructor_tuple_sperm_and_lazy_session() -> None:
    """Constructor inputs fill exact age buckets and lazily initialize a session."""
    from natal import AgeStructuredPopulation
    template = _population("DirectTupleSperm", "age", stochastic=False)
    pop = AgeStructuredPopulation(template.species, template.config, initial_sperm_storage={"WT|WT": {"WT|WT": (0, 7, 4, 99)}})
    np.testing.assert_array_equal(pop.state.sperm_storage[:, 0, 0], [0, 7, 4])
    pop.run(1, record_every=0)
    assert pop.tick == 1


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_reentrant_run_rejected_without_partial_callback_commit(model: Literal["age", "discrete"]) -> None:
    """A callback cannot borrow its own running native session through run()."""
    holder: dict[str, object] = {}
    def callback(ctx: TickContext) -> int:
        with pytest.raises(RuntimeError):
            holder["pop"].run(1)
        ctx.update().custom(reentrant_rejected=True)
        return 0
    pop = _population(
        f"Reentrant_{model}", model, hook_calls=[((callback,), {"event": "first"})]
    )
    holder["pop"] = pop
    pop.run(1)
    assert pop.tick == 1
    assert pop.config.custom["reentrant_rejected"] is True


def test_spatial_migration_rate_forms_and_shape_rejection() -> None:
    """Both supported rate shapes broadcast equally without changing other axes."""
    from tests.test_spatial_session_ownership import _build
    pop = _build("MigrationRateForms", 9, n_demes=2)
    pop.params.tensor_write("migration_rate", {"female": .2, "male": .4})
    expected = np.array([[0, .2, .2], [0, .4, .4]])
    np.testing.assert_array_equal(pop.params.migration_rate, np.tile(expected, (2, 1, 1)))
    matrix = np.array([[0, .1, .3], [0, .2, .4]])
    pop.params.tensor_write("migration_rate", matrix)
    np.testing.assert_array_equal(pop.params.migration_rate, np.tile(matrix, (2, 1, 1)))
    before = pop.params.migration_rate.copy()
    with pytest.raises(ValueError, match="shape"):
        pop.params.tensor_write("migration_rate", np.zeros((3, 3)))
    np.testing.assert_array_equal(pop.params.migration_rate, before)
    assert pop.get_female_count() == 200
    assert pop.get_male_count() == 200


def test_spatial_range_selector_matches_exact_demes() -> None:
    """A range selector updates only its selected native ecology columns."""
    from natal.frontend.hooks import Op
    from tests.test_spatial_session_ownership import _build
    deme_op = Op.set_param("carrying_capacity", 321.0)
    deme_op.event = "first"
    deme_op.every = 1
    pop = _build(
        "RangeSelector",
        9,
        n_demes=3,
        stochastic=False,
        rate=0,
        hook_calls=[((deme_op,), {"deme": range(1, 3)})],
    )
    pop.run(1, record_every=0)
    np.testing.assert_array_equal(pop.params.carrying_capacity, [100000, 321, 321])


def test_direct_spatial_population_can_record_before_running() -> None:
    """A direct container initializes its sole session at a manual snapshot."""
    from natal import SpatialPopulation
    first = _population("DirectSpatialFirst", "discrete", stochastic=False)
    pop = SpatialPopulation([first], migration_rate=0)
    with pytest.raises(AttributeError, match="declaration"):
        _ = pop.definition
    pop.record_snapshot()
    assert len(pop.history) == 1


def test_spatial_custom_values_restore_each_deme_independently() -> None:
    """Distinct custom types and shapes are scoped and restored per deme."""
    from tests.test_spatial_session_ownership import _build
    pop = _build("CustomDemeCheckpoint", 19, stochastic=False, n_demes=2, rate=0)
    pop.demes[0].update().custom(flag=True, grid=np.arange(6.).reshape(2, 3))
    pop.demes[1].update().custom(flag=False, grid=np.arange(3.))
    pop.record_snapshot()
    pop.demes[0].update().custom(flag=False, grid=np.zeros((2, 3)))
    assert pop.demes[1].config.custom["flag"] is False
    np.testing.assert_array_equal(pop.demes[1].config.custom["grid"], np.arange(3.))
    pop.restore_checkpoint(0)
    assert pop.demes[0].config.custom["flag"] is True
    assert pop.demes[1].config.custom["flag"] is False
    np.testing.assert_array_equal(pop.demes[0].config.custom["grid"], np.arange(6.).reshape(2, 3))
    np.testing.assert_array_equal(pop.demes[1].config.custom["grid"], np.arange(3.))


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_external_parameter_handle_cannot_bypass_callback_transaction(model: Literal["age", "discrete"]) -> None:
    """A preexisting population writer cannot mutate the borrowed native owner."""
    holder = {}
    def callback(ctx: TickContext) -> int:
        with pytest.raises(RuntimeError, match="External parameter writes"):
            holder["external"].carrying_capacity = 99.0
        assert ctx.params.carrying_capacity == 100000.0
        return 0
    pop = _population(
        f"ExternalParameterGuard_{model}",
        model,
        hook_calls=[((callback,), {"event": "first"})],
    )
    holder["external"] = pop.params
    pop.run(1)
    assert pop.params.carrying_capacity == 100000.0


def test_spatial_reentrant_and_failed_run_guards() -> None:
    """Nested runs fail before borrowing state; callback errors require recovery."""
    from tests.test_spatial_session_ownership import _build
    holder = {}
    def callback(ctx: TickContext) -> int:
        with pytest.raises(RuntimeError, match="Nested run"):
            holder["pop"].run(1)
        ctx.update().custom(uncommitted=True)
        raise ValueError("spatial callback failure")
    pop = _build(
        "SpatialExecutionGuards",
        11,
        n_demes=2,
        stochastic=False,
        rate=0,
        hook_calls=[((callback,), {"event": "first"})],
    )
    holder["pop"] = pop
    with pytest.raises(ValueError, match="spatial callback failure"):
        pop.run(1)
    with pytest.raises(RuntimeError, match="failed"):
        pop.run(1)
    assert "uncommitted" not in pop.demes[0].config.custom
    assert pop.tick == 0


def test_native_spatial_parameter_updates_are_atomic_and_deme_scoped() -> None:
    """Native entry points independently validate updates without Python writers."""
    from natal.contracts.materialize import materialize
    from tests.test_spatial_session_ownership import _build
    pop = _build("NativeDemeParameters", 5, n_demes=2, stochastic=False, rate=0)
    session = pop._rust_spatial_backend._session
    before = session.get_deme_scalar(0, "carrying_capacity")
    with pytest.raises(KeyError):
        session.apply_deme(0, {"carrying_capacity": 7., "no_such_parameter": 1.})
    assert session.get_deme_scalar(0, "carrying_capacity") == before
    session.apply_deme(0, {"carrying_capacity": 71.})
    assert session.get_deme_scalar(0, "carrying_capacity") == 71
    assert session.get_deme_scalar(1, "carrying_capacity") == before
    original = session.get_deme_tensor(0, "viability_fitness")
    with pytest.raises(ValueError):
        session.tensor_write_deme(0, "viability_fitness", [0.5])
    np.testing.assert_array_equal(session.get_deme_tensor(0, "viability_fitness"), original)
    reduced = np.full_like(original, .25)
    session.tensor_write_deme(0, "viability_fitness", reduced)
    np.testing.assert_array_equal(session.get_deme_tensor(0, "viability_fitness"), reduced)
    np.testing.assert_array_equal(session.get_deme_tensor(1, "viability_fitness"), original)
    assert session.n_variants() == 2
    session.tensor_write_deme(1, "viability_fitness", reduced)
    assert session.n_variants() == 2
    source = materialize(pop.demes[0].config).params
    source.viability_fitness = np.full_like(original, .75)
    session.refresh_deme_parameters(0, ["viability_fitness"], source)
    np.testing.assert_array_equal(session.get_deme_tensor(0, "viability_fitness"), .75)
    for invalid in (
        lambda: session.apply_deme(2, {}),
        lambda: session.tensor_write_deme(2, "viability_fitness", reduced),
        lambda: session.refresh_deme_parameters(2, [], source),
        lambda: session.get_deme_scalar(2, "carrying_capacity"),
        lambda: session.get_deme_tensor(2, "viability_fitness"),
        lambda: session.get_deme_custom_slots(2),
        lambda: session.set_deme_custom_slots(2, {}),
        lambda: session.trigger_deme_event(2, 0),
        lambda: session.trigger_deme_event(0, 9),
        lambda: session.set_migration_rate(np.zeros(1)),
    ):
        with pytest.raises(ValueError):
            invalid()


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_native_tick_cannot_resume_stopped_session(model: Literal["age", "discrete"]) -> None:
    """The native tick entry enforces the same Ready requirement as run."""
    pop = _population(f"NativeStopped_{model}", model)
    session = pop._rust_lifecycle_backend._session
    session.stop()
    assert session.execution_state()[0] == "Stopped"
    before = pop.export_state().copy()
    with pytest.raises(RuntimeError, match="not Ready"):
        session.tick(False if model == "discrete" else 0)
    np.testing.assert_array_equal(pop.export_state(), before)


def test_native_spatial_explicit_callback_commits_and_preserves_original_error() -> None:
    """Manual events use transactions and native failure status without a run."""
    from tests.test_spatial_session_ownership import _build
    def commit(ctx: TickContext) -> int:
        ctx.state.individual_count[0, 1, 0] += 7
        ctx.params.tensor_write("viability_fitness", np.full_like(ctx.params.viability_fitness.array, .4))
        return 0
    pop = _build(
        "SpatialManualTransaction",
        17,
        n_demes=2,
        stochastic=False,
        rate=0,
        hook_calls=[((commit,), {"event": "first"})],
    )
    before = pop.demes[1].state.individual_count.copy()
    pop.trigger_event("first", deme_id=1)
    after = before.copy()
    after[0, 1, 0] += 7
    np.testing.assert_array_equal(pop.demes[1].state.individual_count, after)
    np.testing.assert_array_equal(pop.demes[0].state.individual_count, before)
    np.testing.assert_array_equal(pop.demes[1].params.viability_fitness.array, .4)
    session = pop._rust_spatial_backend._session
    def invalid(ind: np.ndarray, sperm: np.ndarray, tick: int, deme: int) -> int:
        ind[:] = -1
        raise LookupError("original native callback")
    session.set_python_callbacks([invalid], [], [], [])
    with pytest.raises(LookupError, match="original native callback"):
        session.trigger_deme_event(1, 0)
    assert session.execution_state()[0] == "Failed"
    np.testing.assert_array_equal(session.state_snapshot()[1].reshape(2, *before.shape)[1], after)
    with pytest.raises(RuntimeError, match="not Ready"):
        session.run_tick()


def test_native_spatial_stop_and_malformed_state_candidates() -> None:
    """Raw native callbacks cannot commit invalid state; explicit stop is final."""
    from tests.test_spatial_session_ownership import _build
    pop = _build("SpatialNativeStop", 21, n_demes=2, stochastic=False, rate=0)
    session = pop._rust_spatial_backend._session
    def stop(ind: np.ndarray, sperm: np.ndarray, tick: int, deme: int) -> int:
        return 1
    session.set_python_callbacks([stop], [], [], [])
    assert session.trigger_deme_event(0, 0) == 1
    assert session.execution_state()[0] == "Stopped"
    with pytest.raises(RuntimeError, match="not Ready"):
        session.run_tick()
    session.clear_python_callbacks()
    session.clear_hook_program()
    _, individuals, sperm = session.state_snapshot()
    session.set_state(individuals.reshape(2, 2, 3, -1), sperm.reshape(2, 3, 3, 3), 0)
    original = session.state_snapshot()[1].copy()
    def bad_state(ind: np.ndarray, sperm: np.ndarray, tick: int, deme: int) -> int:
        ind[:] = np.nan
        return 0
    session.set_python_callbacks([bad_state], [], [], [])
    with pytest.raises(ValueError, match="nonnegative"):
        session.trigger_deme_event(0, 0)
    np.testing.assert_array_equal(session.state_snapshot()[1], original)


def _mixed_equilibrium_population(name: str, declarations: list[object]):
    """Isolate demes so undeclared equilibrium mode has an independent control."""
    import natal as nt
    species = nt.Species.from_dict(name=name, structure={"chr": {"loc": ["WT"]}})
    return (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
        .setup(stochastic=False)
        .age_structure(n_ages=3, new_adult_age=1)
        .initial_state(individual_count={"female": {"WT|WT": [0, 100, 0]}, "male": {"WT|WT": [0, 100, 0]}})
        .reproduction(eggs_per_female=4, female_age_based_mating_rate=[0, 1, 0], male_age_based_mating_rate=[0, 1, 0])
        .survival(female_age_based_survival=[1, .9, 0], male_age_based_survival=[1, .9, 0])
        .competition(carrying_capacity=100000, low_density_growth_rate=2, equilibrium_distribution=nt.batch_setting(declarations))
        .migration(migration_rate=0)
        .build()
    )


@pytest.mark.parametrize("declared", [np.array([[0., 100., 0.], [0., 100., 0.]]), np.zeros((2, 3))])
def test_mixed_initial_equilibrium_presence_preserves_dynamic_derive(declared: np.ndarray) -> None:
    """An explicit zero or nonzero declaration never replaces a sibling's sentinel."""
    pop = _mixed_equilibrium_population("MixedInitialEquilibrium", [None, declared])
    control = _mixed_equilibrium_population("MixedInitialControl", [None, None])
    native = pop._rust_spatial_backend._session
    assert native.get_deme_tensor(0, "equilibrium_distribution").size == 0
    assert pop.demes[0].config.equilibrium_individual_distribution is None
    np.testing.assert_array_equal(pop.demes[1].config.equilibrium_individual_distribution, declared)
    for capacity in (100000., 1700.):
        pop.demes[0].params.carrying_capacity = capacity
        control.demes[0].params.carrying_capacity = capacity
        pop.run(1)
        control.run(1)
        np.testing.assert_array_equal(pop.demes[0].export_state(), control.demes[0].export_state())


def test_equilibrium_presence_clear_and_checkpoint_restore() -> None:
    """Clearing a native declaration is observable and full checkpoints restore it."""
    declared = np.array([[0., 100., 0.], [0., 100., 0.]])
    pop = _mixed_equilibrium_population("PresenceRestore", [None, declared])
    control = _mixed_equilibrium_population("PresenceRestoreControl", [None, None])
    native = pop._rust_spatial_backend._session
    pop.record_snapshot()
    columns = pop._rust_spatial_backend.ecology_columns_snapshot()
    np.testing.assert_array_equal(columns["equilibrium_declared"], [0, 1])
    assert columns["equilibrium_declared"].dtype == np.int64
    native.tensor_write_deme(1, "equilibrium_distribution", np.zeros(0))
    np.testing.assert_array_equal(
        pop._rust_spatial_backend.ecology_columns_snapshot()["equilibrium_declared"], [0, 0]
    )
    assert pop.demes[1].config.equilibrium_individual_distribution is None
    pop.run(1)
    control.run(1)
    np.testing.assert_array_equal(pop.demes[1].export_state(), control.demes[1].export_state())
    pop.restore_checkpoint(0)
    np.testing.assert_array_equal(pop._ecology_columns["equilibrium_declared"], [0, 1])
    assert native.get_deme_tensor(0, "equilibrium_distribution").size == 0
    assert pop.demes[0].config.equilibrium_individual_distribution is None
    np.testing.assert_array_equal(pop.demes[1].config.equilibrium_individual_distribution, declared)
    reference = _mixed_equilibrium_population("PresenceRestoredReference", [None, declared])
    pop.run(1, record_every=0)
    reference.run(1, record_every=0)
    for deme in range(2):
        np.testing.assert_array_equal(pop.demes[deme].export_state(), reference.demes[deme].export_state())


@pytest.mark.parametrize("model,wf", [("age", False), ("discrete", False), ("discrete", True)])
def test_unbound_native_batch_checkpoints_replay_state_rng_and_ecology(model: str, wf: bool) -> None:
    """Native batches capture complete boundaries even without a bound HistoryStore.

    Replay is exact because the same kernel starts from the same full boundary;
    this checks persistent RNG restoration, not a distributional approximation.
    """
    from natal import _engine_rs
    from natal.contracts.materialize import materialize
    pop = _population(f"UnboundCheckpoint_{model}_{wf}", model)
    contracts = materialize(pop.config)
    session_type = _engine_rs.EngineSession if model == "age" else _engine_rs.DiscreteEngineSession
    native = session_type(contracts.blueprint, contracts.params, 73)
    if wf:
        native.set_execution_flags(True, False, False, 1)
    def run(count: int) -> tuple:
        if model == "age":
            return native.run(count, 1, checkpoint_every=1)
        return native.run(count, 1, wf, checkpoint_every=1)
    tick, rows, stopped = run(3)
    assert tick == 3 and not stopped
    assert rows.shape[0] == 4
    final = native.snapshot_state()
    for checkpoint, steps in ((1, 2), (0, 3)):
        native.apply({"carrying_capacity": 37.0, "eggs_per_female": 1.0})
        native.set_custom_slots({"wrong": 19})
        assert native.restore_from_checkpoint(checkpoint)[0] == checkpoint
        assert native.get_scalar("carrying_capacity") == 100000.0
        assert native.get_custom_slots()["cohort"] == 7
        assert "wrong" not in native.get_custom_slots()
        assert run(steps)[0] == 3
        replay = native.snapshot_state()
        assert replay[0] == final[0]
        for actual, expected in zip(replay[1:-2], final[1:-2]):
            np.testing.assert_array_equal(actual, expected)
        assert replay[-2] == final[-2]
        assert replay[-1].keys() == final[-1].keys()
        for key in replay[-1]:
            if key == "custom_slots":
                for name, value in final[-1][key].items():
                    np.testing.assert_array_equal(replay[-1][key][name], value)
            else:
                np.testing.assert_array_equal(replay[-1][key], final[-1][key])


@pytest.mark.parametrize("event,phase", [("early", 2), ("late", 4)])
def test_discrete_later_callback_failure_preserves_committed_boundary(event: str, phase: int) -> None:
    """A failing early/late callback retains the preceding stage, never its writes."""
    before: list[np.ndarray] = []
    def fail(ctx: TickContext) -> int:
        before.append(ctx.state.individual_count.copy())
        ctx.state.individual_count[:] = 0
        ctx.params.carrying_capacity = 31
        raise LookupError("later event failure")
    pop = _population(
        f"LaterFailure_{event}",
        "discrete",
        stochastic=False,
        hook_calls=[((fail,), {"event": event})],
    )
    with pytest.raises(LookupError, match="later event failure"):
        pop.run(1)
    native = pop._rust_lifecycle_backend._session
    assert native.execution_state() == ("Failed", phase)
    assert native.get_scalar("carrying_capacity") == 100000
    np.testing.assert_array_equal(native.state_snapshot()[1].reshape(before[0].shape), before[0])


@pytest.mark.parametrize("access", ["counter", "state", "rng", "metrics"])
def test_callbacks_materialize_parameters_only_when_requested(access: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """State, counters and persistent RNG need no Python parameter projection."""
    import natal.backends.rust.rust_backend as backend_module
    original_snapshot = backend_module.config_snapshot_from_session
    projected: list[int] = []
    def snapshot(session: object, draft: object) -> object:
        if getattr(pop, "_active_event", None) is not None:
            projected.append(1)
        return original_snapshot(session, draft)
    monkeypatch.setattr(backend_module, "config_snapshot_from_session", snapshot)
    visits: list[int] = []
    def callback(ctx: TickContext) -> int:
        visits.append(ctx.tick)
        if access == "state":
            assert ctx.state.individual_count.shape == (2, 2, 3)
        elif access == "rng":
            assert ctx.rng.uniform(3., 3.) == 3.
        elif access == "metrics":
            assert ctx.metrics.total == 200.
        return 0
    pop = _population(
        f"LazyParameter_{access}",
        "discrete",
        stochastic=False,
        hook_calls=[((callback,), {"event": "first"})],
    )
    pop.run(1)
    assert visits == [0]
    assert projected == []


@pytest.mark.parametrize("access", ["counter", "rng", "metrics"])
def test_completed_callbacks_release_context_without_cyclic_gc(access: str) -> None:
    """Optional samplers and metric readers cannot retain whole native candidates."""
    import gc
    import weakref
    references: list[weakref.ReferenceType[TickContext]] = []
    def callback(ctx: TickContext) -> int:
        references.append(weakref.ref(ctx))
        if access == "rng":
            ctx.rng.random()
        elif access == "metrics":
            assert ctx.metrics.total >= 0
        return 0
    pop = _population(f"ContextRelease_{access}", "discrete", callback=callback)
    enabled = gc.isenabled()
    gc.disable()
    try:
        pop.run(3)
        assert len(references) == 3
        assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_retained_scalar_updater_reads_only_touched_native_fields(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scalar updates preserve unrelated live values without copying tensor columns."""
    pop = _population(f"ScalarNativeReads_{model}", model)
    updater = pop.update()
    backend = pop._rust_lifecycle_backend
    native = backend._session
    native.apply({"carrying_capacity": 1234., "eggs_per_female": 9.})
    original = backend.get_scalar
    reads: list[str] = []
    def read(name: str) -> float:
        reads.append(name)
        return original(name)
    def forbidden_snapshot(*args: object) -> object:
        raise AssertionError("a scalar patch must not snapshot unrelated tensors")
    monkeypatch.setattr(backend, "get_scalar", read)
    with monkeypatch.context() as scoped:
        scoped.setattr(backend, "config_snapshot", forbidden_snapshot)
        updater.competition(carrying_capacity=5678.)
        mark = pop._params_log.mark()
        with pytest.raises(ValueError):
            updater.competition(carrying_capacity=2000., growth_mode=-1)
        assert pop._params_log.mark() == mark
        updater.reproduction(eggs_per_female=5., sex_ratio=.75)
    assert reads == ["carrying_capacity", "carrying_capacity", "growth_mode", "eggs_per_female", "sex_ratio"]
    assert pop.config.carrying_capacity == 5678.
    assert pop.config.eggs_per_female == 5.
    rows = pop._params_log.details()
    assert any(row[3:] == ("carrying_capacity", 1234., 5678.) for row in rows)
    assert any(row[3:] == ("eggs_per_female", 9., 5.) for row in rows)


def test_retained_deme_scalar_writer_preserves_native_sibling_values() -> None:
    """The same selective scalar path scopes every query and commit to its deme."""
    from tests.test_spatial_session_ownership import _build
    pop = _build("SelectiveDemeScalar", 11, n_demes=2, stochastic=False, rate=0)
    updater = pop.demes[1].update()
    native = pop._rust_spatial_backend._session
    native.apply_deme(1, {"carrying_capacity": 1234., "eggs_per_female": 9.})
    updater.competition(carrying_capacity=5678.)
    assert pop.demes[0].config.carrying_capacity == 100000.
    assert pop.demes[1].config.carrying_capacity == 5678.
    assert pop.demes[1].config.eggs_per_female == 9.


@pytest.mark.parametrize("model", ["age_structured", "discrete_generation"])
@pytest.mark.parametrize("operation", ["step", "record_snapshot", "clear_history", "finish_simulation", "import_config", "tick"])
def test_managed_deme_rejects_independent_control_without_mutation(model: str, operation: str) -> None:
    """Facade calls cannot revive the released standalone owner or alter its clock."""
    import natal as nt
    species = nt.Species.from_dict(name=f"ManagedControl_{model}_{operation}", structure={"chr": {"locus": ["WT"]}})
    pop = nt.SpatialPopulation.builder(species, n_demes=2, pop_type=model).build()
    deme = pop.demes[0]
    pop.record_snapshot()
    before = deme.export_state().copy()
    with pytest.raises(RuntimeError, match="managed deme"):
        if operation == "import_config":
            deme.import_config(deme.config)
        elif operation == "tick":
            deme.tick = 7
        else:
            getattr(deme, operation)()
    assert pop.tick == deme.tick == 0
    assert not deme.is_finished
    np.testing.assert_array_equal(deme.export_state(), before)
    assert len(pop.history.ticks) == 1
    assert pop._demes[0]._rust_lifecycle_backend is None
    deme.update().competition(carrying_capacity=77.)
    assert deme.params.carrying_capacity == 77.


def test_container_finish_and_reset_preserve_single_owner_and_replay_rng() -> None:
    """The owner can reset stopped state and reproduce the initial stochastic run."""
    from tests.test_spatial_session_ownership import _build
    pop = _build("ManagedResetReplay", 44, n_demes=2, stochastic=True, rate=.2)
    pop.run(2)
    expected = [deme.export_state().copy() for deme in pop.demes]
    pop.run(0, finish=True)
    assert all(deme.is_finished for deme in pop.demes)
    assert pop._rust_spatial_backend.execution_state()[0] == "Stopped"
    pop.reset()
    assert pop.tick == 0 and all(not deme.is_finished for deme in pop.demes)
    assert pop.history.is_empty
    pop.run(2)
    for deme, state in zip(pop.demes, expected):
        np.testing.assert_array_equal(deme.export_state(), state)
        assert deme._rust_lifecycle_backend is None


@pytest.mark.parametrize("model", ["age", "discrete"])
@pytest.mark.parametrize("entry", ["native", "import", "restore"])
@pytest.mark.parametrize("invalid", ["tick", "negative", "nan", "inf", "shape"])
def test_invalid_state_replacement_preserves_complete_boundary(model: str, entry: str, invalid: str) -> None:
    """Rejected replacement cannot revive Stopped, alter RNG, cache, or history."""
    pop = _population(f"RejectedState_{model}_{entry}_{invalid}", model)
    pop.run(1)
    native = pop._rust_lifecycle_backend._session
    live = pop.state
    counts = live.individual_count.copy()
    tick = -1 if invalid == "tick" else live.n_tick + 1
    if invalid in {"negative", "nan", "inf"}:
        counts.flat[0] = {"negative": -1., "nan": np.nan, "inf": np.inf}[invalid]
    elif invalid == "shape":
        counts = counts[:, :, :-1]
    native.stop()
    before = native.snapshot_state()
    phase = native.execution_state()
    cache = pop._state
    history = pop.history._to_numpy().copy()
    with pytest.raises(ValueError):
        if entry == "native":
            if model == "age":
                native.set_state(counts.ravel(), live.sperm_storage.ravel(), tick)
            else:
                native.set_state(counts.ravel(), tick)
        elif entry == "restore":
            if model == "age":
                native.restore_state(tick, counts.ravel(), live.sperm_storage.ravel(), before[-2], before[-1])
            else:
                native.restore_state(tick, counts.ravel(), before[-2], before[-1])
        else:
            payload = {"n_tick": tick, "individual_count": counts}
            if model == "age":
                payload["sperm_storage"] = live.sperm_storage
            pop.import_state(payload)
    after = native.snapshot_state()
    assert native.execution_state() == phase
    assert after[0] == before[0] == pop.tick
    assert after[-2] == before[-2]
    for actual, expected in zip(after[1:-2], before[1:-2]):
        np.testing.assert_array_equal(actual, expected)
    assert pop._state is cache
    np.testing.assert_array_equal(pop.history._to_numpy(), history)


@pytest.mark.parametrize("entry", ["native", "import", "restore"])
@pytest.mark.parametrize("invalid", [-1., np.nan, np.inf])
def test_invalid_sperm_replacement_is_atomic(entry: str, invalid: float) -> None:
    """Invalid sperm cannot publish otherwise valid changed individual counts."""
    pop = _population(f"RejectedSperm_{entry}_{invalid}", "age")
    pop.run(1)
    native = pop._rust_lifecycle_backend._session
    live = pop.state
    sperm = live.sperm_storage.copy()
    sperm.flat[0] = invalid
    counts = live.individual_count + 10
    before = native.snapshot_state()
    phase = native.execution_state()
    cache = pop._state
    history = pop.history._to_numpy().copy()
    with pytest.raises(ValueError):
        if entry == "native":
            native.set_state(counts.ravel(), sperm.ravel(), 2)
        elif entry == "restore":
            native.restore_state(2, counts.ravel(), sperm.ravel(), before[-2], before[-1])
        else:
            pop.import_state({"n_tick": 2, "individual_count": counts, "sperm_storage": sperm})
    after = native.snapshot_state()
    assert after[0] == before[0] == pop.tick
    assert after[-2] == before[-2]
    assert native.execution_state() == phase
    for actual, expected in zip(after[1:-2], before[1:-2]):
        np.testing.assert_array_equal(actual, expected)
    assert pop._state is cache
    np.testing.assert_array_equal(pop.history._to_numpy(), history)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_import_rejected_by_native_does_not_publish_python_candidate(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even native-side failure after Python validation leaves every local view intact."""
    pop = _population(f"RejectedNativeCommit_{model}", model)
    pop.run(1)
    current = pop.state
    cache = pop._state
    history = pop.history._to_numpy().copy()
    def reject(_candidate: object) -> None:
        raise ValueError("native candidate rejected")
    monkeypatch.setattr(pop._rust_lifecycle_backend, "set_state", reject)
    candidate = current._replace(n_tick=2, individual_count=current.individual_count + 10)
    with pytest.raises(ValueError, match="native candidate rejected"):
        pop.import_state(candidate)
    assert pop.tick == 1 and pop._state is cache
    np.testing.assert_array_equal(pop.state.individual_count, current.individual_count)
    np.testing.assert_array_equal(pop.history._to_numpy(), history)


def test_discrete_spatial_callback_genetics_commit_survives_following_ticks() -> None:
    """An early callback's zero viability kills only its isolated deme's recruits."""
    from tests.test_spatial_update import _build_two_allele_discrete
    def remove_recruits(ctx: TickContext) -> int:
        if ctx.tick == 0:
            ctx.params.tensor_write("viability_fitness", np.zeros_like(ctx.params.viability_fitness.array))
        return 0
    pop = _build_two_allele_discrete(
        "SpatialDiscreteCandidate",
        hook_calls=[((remove_recruits,), {"event": "early", "deme": 1})],
    )
    control = _build_two_allele_discrete("SpatialDiscreteCandidateControl")
    for _ in range(2):
        pop.run(1)
        control.run(1)
        np.testing.assert_array_equal(pop.demes[1].state.individual_count, 0.)
        np.testing.assert_array_equal(pop.demes[1].params.viability_fitness.array, 0.)
        for index in (0, 2, 3):
            np.testing.assert_array_equal(pop.demes[index].export_state(), control.demes[index].export_state())
            np.testing.assert_array_equal(pop.demes[index].params.viability_fitness.array, 1.)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_raw_restore_ecology_rejection_preserves_state_rng_and_phase(model: str) -> None:
    """A late ecology error cannot partially commit an earlier state/RNG restore."""
    pop = _population(f"RawRestoreAtomic_{model}", model)
    pop.run(1)
    native = pop._rust_lifecycle_backend._session
    native.stop()
    before = native.snapshot_state()
    phase = native.execution_state()
    ecology = dict(before[-1])
    ecology["carrying_capacity"] = np.nan
    with pytest.raises(ValueError):
        if model == "age":
            native.restore_state(2, before[1] + 10, before[2], [1, 2, 3, 4], ecology)
        else:
            native.restore_state(2, before[1] + 10, [1, 2, 3, 4], ecology)
    after = native.snapshot_state()
    assert native.execution_state() == phase
    assert after[0] == before[0]
    assert after[-2] == before[-2]
    assert native.get_scalar("carrying_capacity") == before[-1]["carrying_capacity"]
    for actual, expected in zip(after[1:-2], before[1:-2]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_population_tick_assignment_cannot_fork_native_clock(model: str) -> None:
    """Public clock assignment rejects before changing any owning state."""
    pop = _population(f"ReadOnlyClock_{model}", model)
    pop.run(1)
    native = pop._rust_lifecycle_backend._session
    before = native.snapshot_state()
    phase = native.execution_state()
    history = pop.history._to_numpy().copy()
    with pytest.raises(RuntimeError, match="tick is read-only"):
        pop.tick = 42
    assert pop.tick == native.state_snapshot()[0] == before[0] == 1
    after = native.snapshot_state()
    assert after[-2] == before[-2]
    assert native.execution_state() == phase
    for actual, expected in zip(after[1:-2], before[1:-2]):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(pop.history._to_numpy(), history)
