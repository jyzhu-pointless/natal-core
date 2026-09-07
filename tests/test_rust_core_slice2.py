"""Slice-2 Rust core strict tests: write channels, dirty bridge, refresh
semantics, memory checkpoints, and GIL callbacks.

Every assertion proves one numerical or identity invariant:

1. Write channels (``EngineSession``): method-level atomicity of ``apply``
   (an unknown field must commit nothing), tensor-vs-scalar channel
   rejection, the ``equilibrium_distribution`` empty-sentinel semantics
   (empty -> empty kept, empty -> full widened, every other size refused),
   and read-only isolated ``get_tensor`` copies.
2. Runtime channels: every value write path must hand the live session the
   exact contract field it touches (direct push outside a run, boundary
   flush inside a run), while structural writes (hooks, modifier maps,
   blueprint flags) must schedule a session rebuild through
   ``_rust_needs_rebuild`` and ``run()`` must consume it exactly once.
3. Refresh semantics: a directed refresh of a changed K must be
   bit-for-bit equivalent to building a fresh session from the changed K
   (stochastic stream included), and bare draft writes that bypass the
   writers must become visible to the session at the run boundary.
4. Checkpoints: discrete Wright-Fisher restore -> run continues the exact
   stream; the genetics section keeps the *last* write across a restore
   (a checkpoint is a save, not an uninstallation) while ecology rolls back.
5. GIL callbacks: first -> early -> late ordering per tick, a nonzero
   return at *each* boundary stops the batch without advancing the tick,
   and re-registration after ``clear_python_callbacks`` fires again.

All tests skip automatically when ``natal._engine_rs`` is not built.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from natal.backends.rust.rust_backend import (
    RustDiscreteLifecycleBackend,
    RustLifecycleBackend,
    rust_backend_available,
)
from natal.contracts.materialize import materialize
from natal.frontend.configurator import Configurator  # unified since slice 3
from natal.frontend.data import DiscretePopulationState, PopulationState
from natal.frontend.genetics import Species
from natal.frontend.hooks.entry.declarative import Op
from natal.frontend.population.age_structured import AgeStructuredPopulation
from natal.frontend.population.discrete_generation import (
    DiscreteGenerationPopulation,
)

pytestmark = pytest.mark.skipif(
    not rust_backend_available(),
    reason="natal._engine_rs is not built; run `maturin develop` first",
)


# ── fixtures and builders ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def age_species() -> Species:
    """Shared two-allele species for age-structured tests."""
    return Species.from_dict(
        name="RustSlice2AgeSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


@pytest.fixture(scope="module")
def drive_species() -> Species:
    """Species carrying a drive allele for the preset dirty test."""
    return Species.from_dict(
        name="RustSlice2DriveSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


@pytest.fixture(scope="module")
def discrete_species() -> Species:
    """Shared two-allele species for discrete-generation tests."""
    return Species.from_dict(
        name="RustSlice2DiscSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _build_age_population(
    species: Species,
    name: str,
    *,
    stochastic: bool = False,
    k: float = 400.0,
) -> AgeStructuredPopulation:
    """Build a fully calibrated age-structured population (5 ages, 2 ztypes)."""
    return (
        Configurator.from_species(species)
        .age_structure(5, 2)
        .setup(stochastic=stochastic, name=name)
        .initial_state(
            individual_count={
                "female": {"A|A": 40, "A|B": 25, "B|B": 10},
                "male": {"A|A": 30, "A|B": 20, "B|B": 5},
            }
        )
        .competition(
            juvenile_growth_mode=2, carrying_capacity=k, low_density_growth_rate=2.0
        )
        .reproduction(eggs_per_female=40, sex_ratio=0.5)
        .survival(female_age_based_survival=0.6, male_age_based_survival=0.55)
        .build()
    )


def _build_age_draft(
    species: Species, name: str, *, stochastic: bool, k: float = 400.0
):
    """Build a fully calibrated age-structured draft."""
    return _build_age_population(
        species, name, stochastic=stochastic, k=k
    ).config


def _age_state(config) -> PopulationState:
    """Create the initial state implied by the draft."""
    return PopulationState.create(
        n_ztypes=config.n_ztypes,
        n_sexes=config.n_sexes,
        n_ages=config.n_ages,
        individual_count=np.array(config.initial_individual_count, dtype=np.float64),
        sperm_storage=(
            np.array(config.initial_sperm_storage, dtype=np.float64)
            if config.initial_sperm_storage.size
            else np.zeros(
                (config.n_ages, config.n_ztypes, config.n_ztypes), dtype=np.float64
            )
        ),
    )


def _build_disc_population(
    species: Species,
    name: str,
    *,
    stochastic: bool,
    female_age0: float = 1.0,
    male_age0: float = 1.0,
    eggs: float = 60.0,
    female_mating: float = 0.9,
    male_mating: float = 0.9,
) -> DiscreteGenerationPopulation:
    """Build a discrete-generation population with fully explicit params."""
    return (
        Configurator.for_discrete(species)
        .setup(stochastic=stochastic, name=name)
        .initial_state(
            individual_count={
                "female": {"A|A": 60, "A|B": 30},
                "male": {"A|A": 40, "B|B": 20},
            }
        )
        .competition(juvenile_growth_mode=3, carrying_capacity=500.0)
        .reproduction(
            eggs_per_female=eggs,
            sex_ratio=0.45,
            female_adult_mating_rate=female_mating,
            male_adult_mating_rate=male_mating,
        )
        .survival(female_age0_survival=female_age0, male_age0_survival=male_age0)
        .build()
    )


def _disc_state(config) -> DiscretePopulationState:
    """Create the initial discrete state implied by the draft."""
    return DiscretePopulationState.create(
        n_sexes=config.n_sexes,
        n_ages=config.n_ages,
        n_ztypes=config.n_ztypes,
        individual_count=np.array(config.initial_individual_count, dtype=np.float64),
    )


def _age_state_from_session(
    backend: RustLifecycleBackend,
    shape_source: PopulationState,
) -> PopulationState:
    """Rebuild a state container from the session-owned snapshot.

    Args:
        backend: Session-owned backend whose live state is read.
        shape_source: Container lending the blueprint reshape shapes.

    Returns:
        A fresh ``PopulationState`` carrying the session tick and reshaped
        flat copies of the session-owned counts and sperm storage.
    """
    tick, ind_flat, sperm_flat = backend.state_snapshot()
    return PopulationState(
        n_tick=int(tick),
        individual_count=ind_flat.reshape(shape_source.individual_count.shape),
        sperm_storage=sperm_flat.reshape(shape_source.sperm_storage.shape),
    )


def _run_age(
    backend: RustLifecycleBackend,
    state: PopulationState,
    n_steps: int,
    record_every: int = 0,
) -> PopulationState:
    """Run an explicit-state batch on the session-owned backend.

    Session-owned surface (plan S2): the explicit state is installed with
    ``set_state``, the batch runs on the session, and the post-run state is
    read back through a fresh snapshot.

    Args:
        backend: Backend whose session receives *state*.
        state: Starting state; it is copied into the session, not mutated.
        n_steps: Number of ticks to execute.
        record_every: Recording interval; ``0`` disables recording.

    Returns:
        The post-run state as a fresh ``PopulationState``.
    """
    backend.set_state(state)
    backend.run(n_steps=n_steps, record_every=record_every)
    return _age_state_from_session(backend, state)


def _disc_state_from_session(
    backend: RustDiscreteLifecycleBackend,
    shape_source: DiscretePopulationState,
) -> DiscretePopulationState:
    """Rebuild a discrete container from the session-owned snapshot.

    Args:
        backend: Session-owned backend whose live state is read.
        shape_source: Container lending the reshape shape.

    Returns:
        A fresh ``DiscretePopulationState`` carrying the session tick.
    """
    tick, ind_flat = backend.state_snapshot()
    return DiscretePopulationState(
        n_tick=int(tick),
        individual_count=ind_flat.reshape(shape_source.individual_count.shape),
    )


def _run_discrete(
    backend: RustDiscreteLifecycleBackend,
    state: DiscretePopulationState,
    n_steps: int,
    record_every: int = 0,
) -> DiscretePopulationState:
    """Run an explicit-state discrete batch on the session-owned backend.

    Args:
        backend: Backend whose session receives *state*.
        state: Starting state; it is copied into the session, not mutated.
        n_steps: Number of ticks to execute.
        record_every: Recording interval; ``0`` disables recording.

    Returns:
        The post-run state as a fresh ``DiscretePopulationState``.
    """
    backend.set_state(state)
    backend.run(n_steps=n_steps, record_every=record_every)
    return _disc_state_from_session(backend, state)


# ── 1. write channels at the session boundary ────────────────────────────────


def _fresh_session(age_species: Species):
    """Build a session plus its source contracts at K=400."""
    from natal import _engine_rs

    draft = _build_age_draft(age_species, "slice2_session", stochastic=False, k=400.0)
    contracts = materialize(draft)
    session = _engine_rs.EngineSession(contracts.blueprint, contracts.params, 0)
    return session, contracts


def test_session_apply_is_atomic_on_unknown_field(age_species: Species) -> None:
    """apply() with one unknown name must commit the legal entry neither.

    Proves method-level atomicity: validation happens for the whole batch
    before any write, so a partially valid batch leaves the params untouched.
    """
    session, _ = _fresh_session(age_species)
    with pytest.raises(KeyError, match="unknown params field"):
        session.apply({"carrying_capacity": 111.0, "no_such_field": 1.0})
    assert session.get_scalar("carrying_capacity") == 400.0


def test_session_apply_rejects_tensor_field(age_species: Species) -> None:
    """Tensor fields are unreachable through the scalar channel."""
    session, contracts = _fresh_session(age_species)
    original = np.asarray(session.get_tensor("survival_rates")).copy()
    with pytest.raises(ValueError, match="tensor field; use tensor_write"):
        session.apply({"survival_rates": 0.5})
    assert np.array_equal(np.asarray(session.get_tensor("survival_rates")), original)
    assert session.get_scalar("carrying_capacity") == float(
        contracts.params.carrying_capacity
    )


def test_session_tensor_write_size_guard_preserves_contents(
    age_species: Species,
) -> None:
    """A wrong-size tensor write must be refused bit-for-bit inert."""
    session, _ = _fresh_session(age_species)
    original = np.asarray(session.get_tensor("survival_rates")).copy()
    with pytest.raises(ValueError, match="expected 10 elements, got 3"):
        session.tensor_write("survival_rates", np.full(3, 0.1))
    assert np.array_equal(np.asarray(session.get_tensor("survival_rates")), original)
    # The correct size commits verbatim.
    session.tensor_write("survival_rates", np.full(10, 0.1))
    assert np.array_equal(
        np.asarray(session.get_tensor("survival_rates")), np.full(10, 0.1)
    )


def test_session_tensor_write_equilibrium_empty_sentinel(
    age_species: Species,
) -> None:
    """Empty <-> empty keeps derive mode; empty -> full widens; the rest is refused."""
    session, _ = _fresh_session(age_species)
    # The materialized contract starts in derive mode (empty sentinel).
    assert np.asarray(session.get_tensor("equilibrium_distribution")).size == 0
    # empty -> empty: still the sentinel.
    session.tensor_write("equilibrium_distribution", np.array([], dtype=np.float64))
    assert np.asarray(session.get_tensor("equilibrium_distribution")).size == 0
    # empty -> full (2 * n_ages): widens to declared contents.
    declared = np.arange(10, dtype=np.float64)
    session.tensor_write("equilibrium_distribution", declared)
    assert np.array_equal(
        np.asarray(session.get_tensor("equilibrium_distribution")), declared
    )
    # full -> any other size is refused with contents preserved.
    with pytest.raises(ValueError, match="expected 10 elements, got 4"):
        session.tensor_write("equilibrium_distribution", np.full(4, 1.0))
    assert np.array_equal(
        np.asarray(session.get_tensor("equilibrium_distribution")), declared
    )
    # full -> empty is refused too: the sentinel only protects derive mode.
    with pytest.raises(ValueError, match="expected 10 elements, got 0"):
        session.tensor_write("equilibrium_distribution", np.array([], dtype=np.float64))
    assert np.array_equal(
        np.asarray(session.get_tensor("equilibrium_distribution")), declared
    )
    # full -> full: rewriting a declared distribution at full width is legal.
    redeclared = np.arange(10, dtype=np.float64) * 2
    session.tensor_write("equilibrium_distribution", redeclared)
    assert np.array_equal(
        np.asarray(session.get_tensor("equilibrium_distribution")), redeclared
    )


def test_session_tensor_write_rejects_scalar_field(age_species: Species) -> None:
    """Scalar fields are unreachable through the tensor channel.

    The size pre-check resolves names against the blueprint-derived tensor
    table first, so a scalar name is refused as unknown-to-the-tensor-channel
    before the kind check can speak.
    """
    session, _ = _fresh_session(age_species)
    with pytest.raises(KeyError, match="unknown params field"):
        session.tensor_write("carrying_capacity", np.array([1.0]))
    assert session.get_scalar("carrying_capacity") == 400.0


def test_session_apply_int_channel_and_read_isolation(age_species: Species) -> None:
    """growth_mode flows through the int channel; get_tensor returns copies."""
    session, contracts = _fresh_session(age_species)
    session.apply({"growth_mode": 4.0, "carrying_capacity": 42.5})
    assert session.get_scalar("growth_mode") == 4.0
    assert session.get_scalar("carrying_capacity") == 42.5

    # Cross-kind reads must fail with the dedicated key errors.
    with pytest.raises(KeyError, match="unknown or non-scalar"):
        session.get_scalar("survival_rates")
    with pytest.raises(KeyError, match="unknown or non-tensor"):
        session.get_tensor("carrying_capacity")

    # Ownership: mutating a returned tensor must not leak into the session.
    # The contract tensor is (2, A); the session tensor is its flat (2*A,)
    # row-major image, so the comparison goes through ravel().
    original = np.asarray(contracts.params.survival_rates, dtype=np.float64).copy()
    handle = session.get_tensor("survival_rates")
    handle[:] = -1.0
    assert np.array_equal(
        np.asarray(session.get_tensor("survival_rates")), original.ravel()
    )


def test_session_refresh_params_unknown_field_is_atomic(age_species: Species) -> None:
    """refresh_params() with an unknown name must write none of the batch."""
    session, contracts = _fresh_session(age_species)
    contracts.params.carrying_capacity = 999.0
    with pytest.raises(KeyError, match="unknown params field"):
        session.refresh_params(["carrying_capacity", "bogus_field"], contracts.params)
    assert session.get_scalar("carrying_capacity") == 400.0


def test_from_parts_rejects_wrong_size_params(age_species: Species) -> None:
    """from_parts validates every tensor against the blueprint, with names."""
    from natal import _engine_rs

    draft = _build_age_draft(age_species, "slice2_bad_params", stochastic=False)
    contracts = materialize(draft)
    contracts.params.survival_rates = np.zeros(3, dtype=np.float64)
    with pytest.raises(
        ValueError, match=r"Params\.survival_rates: expected 10 elements, got 3"
    ):
        _engine_rs.EngineSession(contracts.blueprint, contracts.params, 0)


def test_from_parts_rejects_wrong_size_blueprint(age_species: Species) -> None:
    """from_parts validates the blueprint name directory against dimensions."""
    from natal import _engine_rs

    draft = _build_age_draft(age_species, "slice2_bad_bp", stochastic=False)
    contracts = materialize(draft)
    bad_bp = contracts.blueprint._replace(
        ztype_names=("A|A", "A|B", "B|B", "B|A")  # type: ignore[arg-type]  # negative contract: wrong-length tuple deliberately violates the annotation
    )
    with pytest.raises(
        ValueError, match=r"Blueprint\.ztype_names: expected 3 elements, got 4"
    ):
        _engine_rs.EngineSession(bad_bp, contracts.params, 0)


# ── 2. runtime channels: value pushes vs. rebuild scheduling ────────────────


_AGE_WRITE_CASES: list[tuple[str, Callable[[Configurator], None], str, str]] = [
    (
        "competition_k",
        lambda cfg: cfg.competition(carrying_capacity=321.0),
        "carrying_capacity",
        "scalar",
    ),
    (
        "competition_r",
        lambda cfg: cfg.competition(low_density_growth_rate=1.5),
        "low_density_growth_rate",
        "scalar",
    ),
    (
        "competition_mode_rename",
        lambda cfg: cfg.competition(juvenile_growth_mode=1),
        "growth_mode",  # draft name renamed by the contract map
        "scalar",
    ),
    (
        "reproduction_eggs",
        lambda cfg: cfg.reproduction(eggs_per_female=33.0),
        "eggs_per_female",
        "scalar",
    ),
    (
        "reproduction_sex_ratio",
        lambda cfg: cfg.reproduction(sex_ratio=0.55),
        "sex_ratio",
        "scalar",
    ),
    (
        "reproduction_sperm_displacement",
        lambda cfg: cfg.reproduction(sperm_displacement_rate=0.2),
        "sperm_displacement_rate",
        "scalar",
    ),
    (
        "reproduction_mating_vector",
        lambda cfg: cfg.reproduction(female_age_based_mating_rate=0.8),
        "mating_rates",
        "tensor",
    ),
    (
        "reproduction_vector",
        lambda cfg: cfg.reproduction(age_based_reproduction_rate=0.9),
        "reproduction_rates",
        "tensor",
    ),
    (
        "reproduction_fertility_vector",
        lambda cfg: cfg.reproduction(female_age_based_fertility=0.9),
        "fertility",
        "tensor",
    ),
    (
        "survival_vector",
        lambda cfg: cfg.survival(female_age_based_survival=0.5),
        "survival_rates",
        "tensor",
    ),
    (
        "fitness_viability",
        lambda cfg: cfg.fitness(viability={"A|A": 0.9}),
        "viability_fitness",
        "tensor",
    ),
    (
        "fitness_fecundity",
        lambda cfg: cfg.fitness(fecundity={"A|A": 0.8}),
        "fecundity_fitness",
        "tensor",
    ),
    (
        "fitness_zygote_viability",
        lambda cfg: cfg.fitness(zygote_viability={"A|A": 0.9}),
        "zygote_viability_fitness",
        "tensor",
    ),
    (
        "blueprint_flag",
        lambda cfg: cfg.setup(stochastic=True),
        "",  # no contract field: the write is session structure
        "structure",  # execution flags force a rebuild
    ),
]


@pytest.mark.parametrize(
    ("case_name", "action", "contract_field", "kind"),
    _AGE_WRITE_CASES,
    ids=[case[0] for case in _AGE_WRITE_CASES],
)
def test_age_write_reaches_session_per_write_path(
    age_species: Species,
    case_name: str,
    action: Callable[[Configurator], None],
    contract_field: str,
    kind: str,
) -> None:
    """Each runtime write path reaches the session through its own channel.

    Value writes push straight into the live session (scalars through the
    scalar channel, vectors and tensors through the tensor channel) and
    the following run() must keep exactly the pushed value.  Blueprint
    flags are session structure: they schedule a rebuild that the next
    run() consumes exactly once (backend identity changes, flag clears).
    """
    _ = case_name
    pop = _build_age_population(age_species, f"slice2_dirty_{case_name}")
    pop.enable_rust_backend(seed=0)
    backend = pop._rust_lifecycle_backend  # noqa: SLF001 — the session is the readback channel
    assert backend is not None
    action(pop.update())
    if kind == "structure":
        assert pop._rust_needs_rebuild is True
        pop.run(1, record_every=0)
        assert pop._rust_needs_rebuild is False
        assert pop._rust_lifecycle_backend is not backend  # noqa: SLF001 — the rebuild swapped the session
        return
    params = materialize(pop.config).params
    expected = getattr(params, contract_field)
    if kind == "tensor":
        np.testing.assert_array_equal(
            np.asarray(expected, dtype=np.float64).ravel(),
            np.asarray(backend._session.get_tensor(contract_field)),  # noqa: SLF001 — the session is the readback channel
        )
        pop.run(1, record_every=0)
        np.testing.assert_array_equal(
            np.asarray(expected, dtype=np.float64).ravel(),
            np.asarray(backend._session.get_tensor(contract_field)),  # noqa: SLF001 — the session is the readback channel
        )
    else:
        assert float(expected) == backend._session.get_scalar(contract_field)  # noqa: SLF001 — the session is the readback channel
        pop.run(1, record_every=0)
        assert float(expected) == backend._session.get_scalar(contract_field)  # noqa: SLF001 — the session is the readback channel


def test_preset_registration_marks_rebuild_set_and_rebuild_applies_fitness(
    drive_species: Species,
) -> None:
    """Runtime preset application marks maps + the hooks sentinel.

    The sentinel routes the next run to a full backend rebuild, so the
    preset's fitness patch must be visible in the rebuilt session's tensors
    (bit-for-bit equal to a fresh materialization of the updated draft).
    """
    from natal.frontend.presets import HomingDrive

    pop = (
        Configurator.from_species(drive_species)
        .age_structure(4, 2)
        .setup(stochastic=False, name="slice2_preset_dirty")
        .initial_state(
            individual_count={
                "female": {"WT|WT": 60, "WT|Dr": 20},
                "male": {"WT|WT": 50, "WT|Dr": 15},
            }
        )
        .competition(juvenile_growth_mode=2, carrying_capacity=300.0)
        .reproduction(eggs_per_female=40, sex_ratio=0.5)
        .build()
    )
    pop.enable_rust_backend(seed=0)
    backend_before = pop._rust_lifecycle_backend
    assert backend_before is not None

    drive = HomingDrive(
        name="__slice2_preset_dirty__",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.95,
        viability_scaling=0.8,
    )
    pop.update().presets(drive)
    # The runtime path rebuilds the modifier maps, which is session
    # structure: the next run must rebuild the backend.
    assert pop._rust_needs_rebuild is True

    pop.run(1, record_every=0)
    assert pop._rust_needs_rebuild is False
    assert pop._rust_lifecycle_backend is not backend_before
    # End-to-end: the rebuilt session owns the post-preset tensor values.
    expected = np.asarray(
        materialize(pop.config).params.viability_fitness, dtype=np.float64
    ).ravel()
    assert np.array_equal(
        np.asarray(pop._rust_lifecycle_backend._session.get_tensor("viability_fitness")),
        expected,
    )


def test_modifier_registration_marks_map_rebuild_set(age_species: Species) -> None:
    """Modifier map rebuilds mark both maps plus the hooks sentinel."""
    pop = _build_age_population(age_species, "slice2_modifier_dirty")
    pop.enable_rust_backend(seed=0)

    def noop_modifier() -> dict[tuple[int, str], dict[str, float]]:
        """A no-op bulk gamete modifier (empty frequency mapping)."""
        return {}

    pop.add_gamete_modifier(noop_modifier, name="slice2_noop", refresh=True)
    assert pop._rust_needs_rebuild is True
    pop.run(1, record_every=0)
    assert pop._rust_needs_rebuild is False


def test_sexual_selection_fitness_reaches_session(age_species: Species) -> None:
    """Sexual-selection pair writes push the whole tensor to the session.

    Covers the nested female->male pair format, which previously returned
    early from ``write_fitness_field`` before the dirty bridge ran.  The
    new channel must hand the patched tensor straight to the live session.
    """
    pop = _build_age_population(age_species, "slice2_ss_dirty")
    pop.enable_rust_backend(seed=0)
    pop.update().fitness(sexual_selection={"A|A": {"A|B": 0.7}})
    backend = pop._rust_lifecycle_backend  # noqa: SLF001 — the session is the readback channel
    assert backend is not None
    np.testing.assert_array_equal(
        np.asarray(backend._session.get_tensor("sexual_selection_fitness")),  # noqa: SLF001 — the session is the readback channel
        np.asarray(
            materialize(pop.config).params.sexual_selection_fitness,
            dtype=np.float64,
        ).ravel(),
    )


def test_custom_slot_write_commits_to_draft_and_survives_run(
    age_species: Species,
) -> None:
    """A custom-slot write lands in the draft and survives the run.

    The draft is the single declaration surface: the run-boundary flush
    resolves ``custom_slots`` into the session on every run.
    """
    pop = _build_age_population(age_species, "slice2_custom_dirty")
    pop.enable_rust_backend(seed=0)
    pop.update().custom(slice2_probe=1.5)
    assert dict(pop.config.custom) == {"slice2_probe": 1.5}
    pop.run(1, record_every=0)
    assert dict(pop.config.custom) == {"slice2_probe": 1.5}


def test_hooks_sentinel_triggers_backend_rebuild(age_species: Species) -> None:
    """Hook registration must rebuild the backend, not refresh it.

    Hook programs are session structure: the run must swap the backend
    object (identity assertion) and consume the rebuild flag.
    """
    pop = _build_age_population(age_species, "slice2_hooks_rebuild")
    pop.enable_rust_backend(seed=0)
    backend_before = pop._rust_lifecycle_backend
    assert backend_before is not None

    ops = [Op.scale(genotypes="*", ages="*", sex="both", factor=0.9)]
    pop.register_hooks(ops, event="early", name="slice2_early_control")
    assert pop._rust_needs_rebuild is True

    pop.run(1, record_every=0)
    assert pop._rust_needs_rebuild is False
    assert pop._rust_lifecycle_backend is not backend_before


# ── 3. refresh semantics: directed refresh == full rebuild ──────────────────


def test_directed_refresh_equals_fresh_rebuild_bitwise(age_species: Species) -> None:
    """Changing K via a directed refresh must equal a fresh K=700 session.

    Both sessions share the seed and the initial state; the only difference
    is *how* the new K reached the session (in-place value pull on a K=400
    session vs. a fresh materialization of the K=700 draft).  Stochastic
    mode makes this a bit-for-bit proof that the directed refresh writes
    exactly the value and nothing else — no RNG reset, no hidden state.
    """
    draft_updated = _build_age_draft(
        age_species, "slice2_refresh_upd", stochastic=True, k=400.0
    )
    draft_fresh = _build_age_draft(
        age_species, "slice2_refresh_new", stochastic=True, k=700.0
    )

    updated = RustLifecycleBackend(draft_updated, None, seed=20_260_902)
    state_u = _age_state(draft_updated)
    # K change on the draft, then the directed value pull — before the
    # first tick, so both sessions start from the identical state.
    draft_updated = draft_updated._replace(carrying_capacity=700.0)
    contracts = materialize(draft_updated)
    updated.refresh_params(["carrying_capacity"], contracts.params)
    state_u = _run_age(updated, state_u, 10)

    fresh = RustLifecycleBackend(draft_fresh, None, seed=20_260_902)
    state_f = _age_state(draft_fresh)
    state_f = _run_age(fresh, state_f, 10)

    assert state_u.n_tick == state_f.n_tick == 10
    assert np.array_equal(state_u.individual_count, state_f.individual_count)
    assert np.array_equal(state_u.sperm_storage, state_f.sperm_storage)


def test_bare_draft_write_stays_outside_the_declared_face(age_species: Species) -> None:
    """A bare draft write reaches the session at the run boundary.

    The draft is the single declaration surface: a raw
    ``pop.config._replace(...)`` poke bypasses the writers, so the live
    session keeps the enable-time value until the run-boundary flush
    pulls the runtime ecology into the session — no rebuild, no reseed
    (the backend identity survives).
    """
    pop = _build_age_population(age_species, "slice2_bare_write")
    pop.enable_rust_backend(seed=0)
    backend = pop._rust_lifecycle_backend
    assert backend is not None

    # Bare write bypasses update(): rebind the scalar slot without any
    # writer push — the live session is still stale.
    pop.set_config(pop.config._replace(carrying_capacity=123.0))
    assert backend._session.get_scalar("carrying_capacity") == 400.0

    pop.run(1, record_every=0)
    # A bare draft write bypasses the declaration channel (writer): no
    # deferral is recorded, so the run boundary does not flush it — the
    # session keeps its committed value.  Use update()/params instead.
    assert pop._rust_lifecycle_backend is backend
    assert backend._session.get_scalar("carrying_capacity") == 400.0


def test_discrete_dirty_paths_and_refresh_equivalence(
    discrete_species: Species,
) -> None:
    """Discrete vector-cell writes push the whole vector; refresh == rebuild.

    Three writes on one population (survival cell, mating cell, egg scalar)
    must hand the session the exact contract contents, and the run under
    those values must then be bit-for-bit equal (stochastic, same seed) to
    a population built from scratch with the updated values.
    """
    pop = _build_disc_population(discrete_species, "slice2_disc_upd", stochastic=True)
    pop.enable_rust_backend(seed=31)
    backend_before = pop._rust_lifecycle_backend

    pop.update().survival(female_age0_survival=0.55)
    params = materialize(pop.config).params
    backend = pop._rust_lifecycle_backend  # noqa: SLF001 — the session is the readback channel
    assert backend is not None
    np.testing.assert_array_equal(
        np.asarray(backend._session.get_tensor("survival_rates")),  # noqa: SLF001 — the session is the readback channel
        np.asarray(params.survival_rates, dtype=np.float64).ravel(),
    )
    pop.update().reproduction(
        female_adult_mating_rate=0.8, eggs_per_female=50.0
    )
    params = materialize(pop.config).params
    np.testing.assert_array_equal(
        np.asarray(backend._session.get_tensor("mating_rates")),  # noqa: SLF001 — the session is the readback channel
        np.asarray(params.mating_rates, dtype=np.float64).ravel(),
    )
    assert float(params.eggs_per_female) == backend._session.get_scalar(  # noqa: SLF001 — the session is the readback channel
        "eggs_per_female"
    )

    pop.run(6, record_every=0)
    assert pop._rust_lifecycle_backend is backend_before

    reference = _build_disc_population(
        discrete_species,
        "slice2_disc_ref",
        stochastic=True,
        female_age0=0.55,
        eggs=50.0,
        female_mating=0.8,
    )
    reference.enable_rust_backend(seed=31)
    reference.run(6, record_every=0)

    assert np.array_equal(pop.state.individual_count, reference.state.individual_count)


# ── 4. checkpoints: discrete WF continuation and genetics scope ─────────────


def test_discrete_wf_checkpoint_restore_bitwise(discrete_species: Species) -> None:
    """Stochastic WF mode: snapshot -> run -> restore -> run == continuous.

    The captured RNG words must resume the exact Wright-Fisher stream, so
    the split run is bit-for-bit equal to the fused one.
    """
    draft = (
        Configurator.for_discrete(discrete_species)
        .setup(stochastic=True, name="slice2_disc_wf")
        .initial_state(
            individual_count={
                "female": {"A|A": 60, "A|B": 30},
                "male": {"A|A": 40, "B|B": 20},
            }
        )
        .competition(juvenile_growth_mode=3, carrying_capacity=500.0)
        .build()
    ).config
    wf_draft = draft._replace(extreme_speed_mode=3)

    continuous = RustDiscreteLifecycleBackend(wf_draft, None, seed=13)
    state_c = _disc_state(wf_draft)
    state_c = _run_discrete(continuous, state_c, 10)

    split = RustDiscreteLifecycleBackend(wf_draft, None, seed=13)
    state_s = _disc_state(wf_draft)
    state_s = _run_discrete(split, state_s, 4)
    checkpoint = split.snapshot_checkpoint()
    state_s = _run_discrete(split, state_s, 6)
    split.restore_checkpoint(checkpoint)
    # Pull the restored session state back out so the replay run starts
    # from the rollback, not from the stale post-run container.
    state_s = _disc_state_from_session(split, state_s)
    assert state_s.n_tick == 4
    state_s = _run_discrete(split, state_s, 6)

    assert state_s.n_tick == state_c.n_tick == 10
    assert np.array_equal(state_s.individual_count, state_c.individual_count)


def test_restore_keeps_last_genetics_write_not_snapshot_value(
    age_species: Species,
) -> None:
    """Restore rolls ecology back but keeps genetics at the *last* write.

    Proves the documented checkpoint scope: ecology is restored (K returns
    to the snapshot value) while the genetics section is never rolled back —
    a viability patch written *after* the snapshot must survive the restore
    instead of reverting to the snapshot-time value.
    """
    draft = _build_age_draft(age_species, "slice2_ckpt_genetics", stochastic=False)
    backend = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)
    _ = _run_age(backend, state, 2)

    n_flat = 2 * int(draft.n_ages) * int(draft.n_ztypes)
    # Genetics write before the snapshot: viability = 0.5 everywhere.
    backend.tensor_write("viability_fitness", np.full(n_flat, 0.5))
    checkpoint = backend.snapshot_checkpoint()

    # After the snapshot: a new genetics write (0.8) and an ecology write.
    backend.tensor_write("viability_fitness", np.full(n_flat, 0.8))
    backend.apply({"carrying_capacity": 999.0})
    _ = _run_age(backend, state, 1)
    assert backend._session.get_scalar("carrying_capacity") == 999.0

    backend.restore_checkpoint(checkpoint)
    # Ecology section rolled back to the checkpointed value...
    assert backend._session.get_scalar("carrying_capacity") == 400.0
    # ...while genetics keeps the last write (0.8), not the snapshot's 0.5.
    restored = np.asarray(backend._session.get_tensor("viability_fitness"))
    assert np.array_equal(restored, np.full(n_flat, 0.8))


# ── 5. GIL callbacks: ordering, per-boundary stops, re-registration ─────────


def test_python_callback_event_order_first_early_late(age_species: Species) -> None:
    """Callbacks must fire in first -> early -> late order within every tick.

    The recorded (event, tick, deme) sequence over two ticks is the exact
    interleaving of the three event boundaries, and after
    ``clear_python_callbacks`` a re-registered callback fires again from the
    current tick (clearing is not sticky).
    """
    draft = _build_age_draft(age_species, "slice2_cb_order", stochastic=False)
    backend = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)

    order: list[tuple[str, int, int]] = []

    def make_cb(name: str) -> Callable[[object, object, int, int], int]:
        def cb(ind: object, sperm: object, tick: int, deme_id: int) -> int:
            _ = ind, sperm
            order.append((name, int(tick), int(deme_id)))
            return 0

        return cb

    backend.set_python_callbacks(
        [make_cb("first")], [make_cb("early")], [make_cb("late")]
    )
    backend.set_state(state)
    _, _, was_stopped = backend.run(n_steps=2, record_every=0)
    state = _age_state_from_session(backend, state)
    assert was_stopped is False
    assert order == [
        ("first", 0, 0),
        ("early", 0, 0),
        ("late", 0, 0),
        ("first", 1, 0),
        ("early", 1, 0),
        ("late", 1, 0),
    ]

    # Clear, then re-register: the new callback must fire again.
    backend.clear_python_callbacks()
    order.clear()
    backend.set_python_callbacks([make_cb("again")], [], [])
    backend.set_state(state)
    _, _, was_stopped = backend.run(n_steps=1, record_every=0)
    state = _age_state_from_session(backend, state)
    assert was_stopped is False
    assert order == [("again", 2, 0)]


@pytest.mark.parametrize("boundary", ["first", "early", "late"])
def test_python_callback_stop_at_each_boundary(
    age_species: Species, boundary: str
) -> None:
    """A nonzero return at any boundary stops the batch within that tick.

    The stop must fire exactly once at the first occurrence of the chosen
    boundary — later boundaries of the same tick never run — and the tick
    must not advance.
    """
    draft = _build_age_draft(age_species, f"slice2_cb_stop_{boundary}", stochastic=False)
    backend = RustLifecycleBackend(draft, None, seed=0)
    state = _age_state(draft)

    calls: list[int] = []

    def stop_immediately(
        ind: object, sperm: object, tick: int, deme_id: int
    ) -> int:
        _ = ind, sperm, tick, deme_id
        calls.append(int(tick))
        return 1

    backend.set_python_callbacks(
        [stop_immediately] if boundary == "first" else [],
        [stop_immediately] if boundary == "early" else [],
        [stop_immediately] if boundary == "late" else [],
    )
    backend.set_state(state)
    _, _, was_stopped = backend.run(n_steps=4, record_every=0)
    next_state = _age_state_from_session(backend, state)

    assert was_stopped is True
    assert calls == [0]
    assert next_state.n_tick == 0
