"""Machine-checkable contract ledger for the Rust-only refactor (S0 exit).

RUST_ONLY_REFACTOR_PLAN.md section 11 (S0) requires three checkable
ledgers — must-exist, must-not-exist, and invariants — that distinguish
pre-existing known defects from new regressions.  This module encodes
the ledgers as data and checks them mechanically:

- ``MUST_EXIST``: every frozen user surface and recorded rule maps to
  concrete test files and test names; a missing file or sample breaks
  this suite.
- ``MUST_NOT_EXIST``: items already removed are asserted unreachable
  right now; items slated for later stages are registered with their
  owning stage and asserted *still reachable* — the registry flips to
  unreachability assertions when the owning stage deletes them, so a
  silent partial deletion cannot go unnoticed.
- ``INVARIANTS``: every verification face from plan section 12.1 maps
  to owning tests; a violated invariant can be registered as a known
  violation.  All S0 red-light repros (R1-R5) have graduated into the
  pytest suite next to their owning stage's fix, and the construction
  script ``scripts/known_defect_repro.py`` is retired (C2/C4 remain
  documented-only pending a spec decision).

Any pytest failure that is NOT in the known-violation list is by
construction a new regression.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Ledger 1: must-exist ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurfaceEntry:
    """One frozen surface area and the tests that pin it.

    Attributes:
        area: Human-readable surface name.
        test_file: Test module (relative to ``tests/``) holding the samples.
        samples: Test function names that must exist in that module.
    """

    area: str
    test_file: str
    samples: tuple[str, ...]


MUST_EXIST: tuple[SurfaceEntry, ...] = (
    SurfaceEntry(
        area="chained configuration API (plan 2.1 #1)",
        test_file="test_frozen_chained_api.py",
        samples=(
            "test_age_structured_full_chain_builds_and_runs",
            "test_discrete_generation_chain_reproduces_exactly",
            "test_spatial_chain_with_batch_setting_and_migration",
            "test_spatial_batch_setting_matrix_form",
            "test_runtime_update_chain_changes_params_and_custom",
            "test_custom_slots_keep_scalar_types_and_array_shapes",
        ),
    ),
    SurfaceEntry(
        area="species structure registration syntax (plan 2.1 #2)",
        test_file="test_frozen_species_syntax.py",
        samples=(
            "test_locus_name_list_form_builds_skeleton_loci",
            "test_locus_allele_map_form_expands_genotypes",
            "test_extended_spec_sex_types_register",
            "test_xy_species_produce_sex_specific_genotypes",
            "test_gamete_and_somatic_labels_round_trip",
            "test_set_recombination_records_adjacent_rate",
            "test_genotype_string_syntax_parses_and_round_trips",
        ),
    ),
    SurfaceEntry(
        area="preset rules (plan 2.1 #3)",
        test_file="test_frozen_preset_semantics.py",
        samples=(
            "test_bind_species_then_conflicting_rebind_raises",
            "test_registration_idempotent_by_object_identity",
            "test_priority_orders_runtime_modifier_application",
            "test_manual_modifiers_append_after_preset_modifiers",
            "test_homing_conversion_probability_table",
            "test_reconfigure_preset_updates_probability_table",
            "test_preset_reconfiguration_rebuilds_fitness_over_manual_writes",
        ),
    ),
    SurfaceEntry(
        area="declarative hook format (plan 2.1 #4)",
        test_file="test_frozen_hook_format.py",
        samples=(
            "test_every_action_op_constructs_registers_and_runs",
            "test_event_order_first_once_early_late_per_tick_finish_once",
            "test_priority_orders_callbacks_within_one_event",
            "test_when_condition_false_leaves_dynamics_untouched",
            "test_set_param_every_start_schedule",
            "test_decorator_selector_callback_receives_pop_and_resolved_index",
            "test_deme_scoped_hook_touches_only_selected_demes",
        ),
    ),
    SurfaceEntry(
        area="event/stop/reset/import/restore rules (plan 7.4, 8.3, 9)",
        test_file="test_frozen_lifecycle_rules.py",
        samples=(
            "test_early_stop_short_circuits_before_late_and_aging",
            "test_stopped_population_requires_reset_before_running_again",
            "test_finish_run_closes_the_population",
            "test_restore_checkpoint_rolls_back_state_and_resumes",
            "test_import_state_resumes_from_exported_tick_with_cleared_history",
            "test_clear_history_keeps_population_and_allows_new_records",
        ),
    ),
)


def test_must_exist_ledger_files_and_samples_exist() -> None:
    """Every registered surface sample exists in its registered file."""
    for entry in MUST_EXIST:
        path = Path(__file__).parent / entry.test_file
        assert path.is_file(), f"{entry.area}: missing {entry.test_file}"
        text = path.read_text(encoding="utf-8")
        for sample in entry.samples:
            assert f"def {sample}(" in text, (
                f"{entry.area}: sample {sample} not found in {entry.test_file}"
            )


# ── Ledger 2: must-not-exist ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RemovalEntry:
    """One interface that must (eventually) not exist.

    Attributes:
        item_id: Short identifier used in messages.
        description: What must be unreachable.
        owner_stage: Plan stage that performs the deletion.
        status: ``"removed"`` (assert unreachable now) or ``"pending"``
            (still reachable today; flip the assertion with the owner stage).
    """

    item_id: str
    description: str
    owner_stage: str
    status: str


MUST_NOT_EXIST: tuple[RemovalEntry, ...] = (
    RemovalEntry(
        item_id="output.record",
        description="dead module natal.frontend.output.record (removed in S0 batch 1)",
        owner_stage="S0",
        status="removed",
    ),
    RemovalEntry(
        item_id="build_observation_row_panmictic",
        description="dead panmictic row encoder function",
        owner_stage="S0",
        status="removed",
    ),
    RemovalEntry(
        item_id="backends.reference",
        description="production pure-Python reference engine package",
        owner_stage="S6",
        status="removed",
    ),
    RemovalEntry(
        item_id="backend-selector",
        description=(
            "backend= selection (removed at S6 batch A), enable/disable "
            "facades, and their exports/stubs (facades removed at S6 batch B)"
        ),
        owner_stage="S6",
        status="removed",
    ),
    RemovalEntry(
        item_id="python-authoritative-state",
        description="_state/_tick/_config authoritative copies, journal-to-draft replay",
        owner_stage="S2",
        status="pending",
    ),
    RemovalEntry(
        item_id="rust-dirty-bridge",
        description="_rust_dirty set and the _contract_params mirror (S2 batch 24)",
        owner_stage="S2",
        status="removed",
    ),
    RemovalEntry(
        item_id="configcontext-population-clone",
        description=(
            "ConfigContext Population mimicry, apply_preset_to_population, "
            "and clone-to-validate preset transactions"
        ),
        owner_stage="S1",
        status="removed",
    ),
    RemovalEntry(
        item_id="per-run-session-rebuild",
        description="the directed-pull _sync_rust_backend bridge ahead of every run",
        owner_stage="S2",
        status="removed",
    ),
    RemovalEntry(
        item_id="discrete-spatial-hookless-backend",
        description=(
            "separate hook-less discrete-spatial backend construction path "
            "(S3b: one session and Program for every model)"
        ),
        owner_stage="S3",
        status="removed",
    ),
    RemovalEntry(
        item_id="python-history-append-store",
        description="Python History as the runtime append-only numeric store",
        owner_stage="S4",
        status="pending",
    ),
    RemovalEntry(
        item_id="same-path-parity-tests",
        description="parity tests that compare a path against itself",
        owner_stage="S6",
        status="removed",
    ),
)


def _module_importable(module_name: str) -> bool:
    """Return whether *module_name* imports successfully right now."""
    try:
        importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return False
    return True


# Reachability probes.  Every ledger entry — removed or pending — must map
# to exactly one probe.  A removed entry without an unreachability probe
# would silently pass (the branch simply never runs), so the dispatch is
# exhaustive by construction: unknown ids fail the test that uses them.


def _removed_probe_output_record() -> None:
    """The dead output.record module must not import."""
    assert not _module_importable("natal.frontend.output.record"), (
        "natal.frontend.output.record imports again — the dead module is back"
    )


def _removed_probe_build_observation_row_panmictic() -> None:
    """The dead panmictic row encoder must not hang off the output package."""
    import natal.frontend.output as output_pkg

    assert not hasattr(output_pkg, "build_observation_row_panmictic"), (
        "build_observation_row_panmictic is reachable again"
    )


def _removed_probe_configcontext_population_clone() -> None:
    """The Population-mimicry adapter surface must stay unreachable.

    Slice 6 removed the ConfigContext adapter, the
    ``apply_preset_to_population`` dual-target helper, and the
    Configurator's ``_make_ctx`` / ``_sync_from_ctx`` write-back pair:
    build-time preset application now compiles against the Configurator
    itself (RecipeHost protocol) through explicit-data rebuilds.
    """
    import natal as nt
    from natal.frontend import presets as presets_pkg
    from natal.frontend.configurator import _registry_builder as rb

    assert not _module_importable(
        "natal.frontend.configurator._registry_builder.ConfigContext"
    ), "ConfigContext is importable again — the Population mimicry is back"
    assert not hasattr(rb, "ConfigContext"), (
        "ConfigContext is getattr-reachable on _registry_builder again"
    )
    assert not hasattr(presets_pkg, "apply_preset_to_population"), (
        "apply_preset_to_population is reachable on the presets package again"
    )
    assert not hasattr(nt, "apply_preset_to_population"), (
        "apply_preset_to_population is reachable on the top-level package again"
    )
    from natal.frontend.configurator._base import Configurator

    for attr in ("_make_ctx", "_sync_from_ctx"):
        assert not hasattr(Configurator, attr), (
            f"Configurator.{attr} is back — the adapter round-trip returned"
        )


def _removed_probe_rust_dirty_bridge() -> None:
    """The Rust dirty-set bridge and the contract-params mirror are gone."""
    from natal.frontend.population.age_structured import AgeStructuredPopulation
    from natal.frontend.population.discrete_generation import (
        DiscreteGenerationPopulation,
    )

    for cls in (AgeStructuredPopulation, DiscreteGenerationPopulation):
        for attr in ("_rust_dirty", "_contract_params"):
            assert not hasattr(cls, attr), (
                f"{cls.__name__}.{attr} is reachable again — the dirty-set "
                f"bridge returned"
            )


def _removed_probe_per_run_session_rebuild() -> None:
    """The directed-pull ``_sync_rust_backend`` bridge must stay gone."""
    from natal.frontend.population.age_structured import AgeStructuredPopulation
    from natal.frontend.population.discrete_generation import (
        DiscreteGenerationPopulation,
    )

    for cls in (AgeStructuredPopulation, DiscreteGenerationPopulation):
        assert not hasattr(cls, "_sync_rust_backend"), (
            f"{cls.__name__}._sync_rust_backend is reachable again — the "
            "directed-pull bridge returned"
        )


def _removed_probe_backends_reference() -> None:
    """The reference engine package must not exist, statically or dynamically."""
    assert importlib.util.find_spec("natal.backends.reference") is None, (
        "natal.backends.reference resolves to an import spec — the package "
        "(or a stale bytecode directory) is back on disk"
    )
    assert not _module_importable("natal.backends.reference"), (
        "natal.backends.reference imports again — the reference engine returned"
    )


def _removed_probe_backend_selector() -> None:
    """Backend selection and the enable/disable facades are gone for good.

    The three ``setup()`` signatures carry no ``backend`` kwarg, the
    ``disable_rust_backend`` / ``refresh_rust_backend`` facades and the
    ``using_rust_backend`` property are absent from every population class,
    and ``_initialize_session`` remains as the engine initialization entry.
    """
    from natal.frontend.configurator._base import Configurator
    from natal.frontend.population.age_structured import AgeStructuredPopulation
    from natal.frontend.population.discrete_generation import (
        DiscreteGenerationPopulation,
    )
    from natal.frontend.spatial.population import SpatialPopulation

    for cls in (AgeStructuredPopulation, DiscreteGenerationPopulation):
        assert "backend" not in inspect.signature(cls.setup).parameters, (
            f"{cls.__name__}.setup carries the retired backend= selector kwarg"
        )
    assert "backend" not in inspect.signature(Configurator.setup).parameters, (
        "Configurator.setup carries the retired backend= selector kwarg"
    )
    for cls in (
        AgeStructuredPopulation,
        DiscreteGenerationPopulation,
        SpatialPopulation,
    ):
        for attr in ("disable_rust_backend", "refresh_rust_backend", "using_rust_backend"):
            assert not hasattr(cls, attr), (
                f"{cls.__name__}.{attr} is reachable again — backend "
                "selection returned"
            )
        assert hasattr(cls, "_initialize_session"), (
            f"{cls.__name__}._initialize_session must remain as the engine "
            "initialization entry"
        )


def _removed_probe_same_path_parity_tests() -> None:
    """The same-path parity suites are physically deleted."""
    for name in (
        "test_rust_lifecycle.py",
        "test_sampling_consistency.py",
        "test_random_sampling_consistency.py",
        "test_improved_sampling.py",
        "test_lifecycle_unified.py",
    ):
        assert not (REPO_ROOT / "tests" / name).is_file(), (
            f"{name} is back — the same-path parity suite must stay deleted"
        )


def _removed_probe_discrete_spatial_hookless_backend() -> None:
    """The hook-less discrete-spatial construction is unreachable."""
    from natal.frontend.spatial.population import SpatialPopulation

    # The container no longer builds per-bank hook-less discrete backends:
    # both models share the one heterogeneous session.
    assert not hasattr(SpatialPopulation, "_run_rust_discrete_spatial_tick"), (
        "the legacy discrete per-bank tick path is back"
    )
    import natal.backends.rust.rust_backend as rb

    source = (
        REPO_ROOT / "src" / "natal" / "frontend" / "spatial" / "population.py"
    ).read_text(encoding="utf-8")
    assert "RustDiscreteLifecycleBackend(cfg, None" not in source, (
        "the hook-less discrete-spatial construction is back"
    )
    assert hasattr(rb, "RustDiscreteLifecycleBackend"), (
        "the plain-model discrete session must remain (spatial per-bank use is gone)"
    )


REMOVED_PROBES: dict[str, Callable[[], None]] = {
    "output.record": _removed_probe_output_record,
    "build_observation_row_panmictic": _removed_probe_build_observation_row_panmictic,
    "backends.reference": _removed_probe_backends_reference,
    "backend-selector": _removed_probe_backend_selector,
    "same-path-parity-tests": _removed_probe_same_path_parity_tests,
    "configcontext-population-clone": _removed_probe_configcontext_population_clone,
    "rust-dirty-bridge": _removed_probe_rust_dirty_bridge,
    "per-run-session-rebuild": _removed_probe_per_run_session_rebuild,
    "discrete-spatial-hookless-backend": _removed_probe_discrete_spatial_hookless_backend,
}


def test_removed_entries_are_unreachable() -> None:
    """Entries with status "removed" must be dynamically unreachable.

    The dispatch over ``REMOVED_PROBES`` is exhaustive: a removed entry
    whose id has no probe fails here instead of silently passing.
    """
    for entry in MUST_NOT_EXIST:
        if entry.status != "removed":
            continue
        probe = REMOVED_PROBES.get(entry.item_id)
        assert probe is not None, (
            f'{entry.item_id}: status is "removed" but REMOVED_PROBES has no '
            "unreachability probe — the removal would be silently unverified"
        )
        probe()


def _pending_probe_python_authoritative_state() -> None:
    """The private authoritative copies must still live on the population."""
    import natal as nt

    species = nt.Species.from_dict(
        name="LedgerProbeSpecies",
        structure={"chr1": {"loc": ["WT"]}},
        gamete_labels=["default"],
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="LedgerProbePop", stochastic=False
        )
        .initial_state(
            individual_count={"female": {"WT|WT": 10.0}, "male": {"WT|WT": 10.0}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1000.0, low_density_growth_rate=2.0)
        .build()
    )
    for attr in ("_state", "_tick", "_config"):
        assert hasattr(pop, attr), (
            f"private authoritative field {attr} is gone — flip this entry "
            'to "removed" in the same batch as the deletion'
        )


def _pending_probe_python_history_append_store() -> None:
    """The Python History append-store must still exist."""
    from natal.frontend.output.history import History

    assert hasattr(History, "_append"), (
        "History._append (the Python append-only numeric store) is gone"
    )


PENDING_PROBES: dict[str, Callable[[], None]] = {
    "python-authoritative-state": _pending_probe_python_authoritative_state,
    "python-history-append-store": _pending_probe_python_history_append_store,
}


def test_pending_entries_are_registered_with_owner_stage() -> None:
    """Pending deletions carry an owning stage, a probe, and stay reachable.

    The reachability half is deliberate: it pins the registry to reality.
    When an owning stage deletes an interface, this assertion fails and
    forces the entry's status to flip to "removed" in the same batch.
    """
    for entry in MUST_NOT_EXIST:
        if entry.status != "pending":
            continue
        assert entry.owner_stage, f"{entry.item_id}: pending entry lacks owner stage"
        probe = PENDING_PROBES.get(entry.item_id)
        assert probe is not None, (
            f"{entry.item_id}: pending entry has no reachability probe — "
            "its claimed reachability would be silently unverified"
        )
        probe()


# ── Ledger 3: invariants ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class InvariantEntry:
    """One verification face from plan section 12.1.

    Attributes:
        area: Verification face name.
        owning_tests: Test files that prove the invariant today.
        known_violations: Defect ids (C2/C4 pending spec decisions) that
            currently break the face; everything else graduates into the
            owning tests once its stage lands.
    """

    area: str
    owning_tests: tuple[str, ...]
    known_violations: tuple[str, ...] = field(default=())


INVARIANTS: tuple[InvariantEntry, ...] = (
    InvariantEntry(
        area="issue #43 Mendelian meiosis (0.495/0.495/0.005/0.005)",
        owning_tests=("test_genetic_entities.py",),
    ),
    InvariantEntry(
        area="offspring tensor P[i,j,k] = sum(meiosis_f * meiosis_m * fusion)",
        owning_tests=("test_offspring_alignment.py", "test_frozen_preset_semantics.py"),
    ),
    InvariantEntry(
        area="preset/fitness dose, sex, label, priority, reconfiguration",
        owning_tests=("test_frozen_preset_semantics.py", "test_modifiers.py"),
    ),
    InvariantEntry(
        area="density curves g(1)=1, monotone non-increasing, compensated g(0)=r",
        owning_tests=("test_config_slice3.py",),  # curve properties pinned by rust curves::tests
    ),
    InvariantEntry(
        area="random streams advance; split run == single run; restore replays",
        owning_tests=(
            "test_spatial_session_ownership.py",
            "test_restore_checkpoint_semantics.py",
        ),
    ),
    InvariantEntry(
        area="migration conserves counts; empty-neighbor/edge/zero-rate behavior",
        owning_tests=("test_spatial_slice5.py",),
    ),
    InvariantEntry(
        area="parameter transactions: all-or-nothing apply, no partial writes",
        owning_tests=("test_config_slice3.py", "test_routes_slice3.py"),
    ),
    InvariantEntry(
        area="ownership: returned arrays are snapshots; hook refs detach",
        owning_tests=("test_rust_session_bridge.py", "test_ownership_snapshots.py"),
    ),
    InvariantEntry(
        area="lifecycle: restore->run, finish->snapshot, import->run, clear->record",
        owning_tests=(
            "test_frozen_lifecycle_rules.py",
            "test_restore_checkpoint_semantics.py",
        ),
    ),
    InvariantEntry(
        area="hook x model x space combinations run uniformly (plan 12.2)",
        owning_tests=(
            "test_frozen_hook_format.py",
            "test_spatial_session_ownership.py",
        ),
    ),
    InvariantEntry(
        area="observation: current query == history recompute == direct record",
        owning_tests=(
            "test_observation_phase2.py",
            "test_history_observation_contract.py",
        ),
    ),
)

# Audit findings recorded by the S0 reviewers (not in the plan's R/T list):
# C1 pop.params.meiosis_map raised AttributeError — FIXED in S1 batch 4 by
#     adding the meiosis_map -> zygotes_to_gametes_map rename.
# C2 plain vs spatial discrete default growth semantics diverge — still open.
# C3 tensor_write("meiosis_map") reached storage but not dynamics — FIXED in
#     S1 batch 5: the write now recomputes the derived offspring tensor in
#     the same transaction (and rejects non-distribution rows atomically).
# C4 deme.update().fitness(...) still writes the shared viability tables
#     in place, leaking into every other deme (pre-existing sibling of the
#     P2 leak closed in batch 5; needs a spec decision whether to refuse
#     like tensor_write or route through write_genetics).
EXTRA_FINDINGS: tuple[InvariantEntry, ...] = (
    InvariantEntry(
        area="C1: public meiosis_map params route resolves",
        owning_tests=("test_routes_slice3.py",),
    ),
    InvariantEntry(
        area="C2: plain vs spatial discrete density semantics agree",
        owning_tests=(),
        known_violations=("C2",),
    ),
    InvariantEntry(
        area="C3: meiosis_map writes recompute the derived offspring tensor",
        owning_tests=("test_routes_slice3.py",),
    ),
    InvariantEntry(
        area="C4: deme-level fitness writes do not leak across shared tables",
        owning_tests=(),
        known_violations=("C4",),
    ),
)

KNOWN_DEFECT_IDS = {
    violation
    for entry in (*INVARIANTS, *EXTRA_FINDINGS)
    for violation in entry.known_violations
}


def test_invariant_ledger_covers_all_plan_faces() -> None:
    """All ten verification faces of plan section 12.1 are registered.

    The hook-combination face from section 12.2 is registered on top of
    the ten 12.1 faces, hence the lower bound.  Every face is matched by
    a distinctive substring so dropping one entry breaks this test.
    """
    assert len(INVARIANTS) >= 10
    faces = {entry.area for entry in INVARIANTS}
    for face_fragment in (
        "issue #43",
        "offspring tensor",
        "preset/fitness",
        "density curves",
        "random streams",
        "migration",
        "parameter transactions",
        "ownership",
        "lifecycle",
        "observation",
        "hook x model x space",
    ):
        assert any(face_fragment in f for f in faces), (
            f"plan verification face {face_fragment!r} missing from the ledger"
        )


def test_invariant_owning_test_files_exist() -> None:
    """Every owning test named by the invariant ledger exists under tests/."""
    for entry in (*INVARIANTS, *EXTRA_FINDINGS):
        for test_file in entry.owning_tests:
            path = Path(__file__).parent / test_file
            assert path.is_file(), (
                f"{entry.area}: owning test file {test_file} does not exist"
            )


def test_offspring_derivation_has_a_single_spelling() -> None:
    """The offspring-tensor derivation is spelled exactly once (S1).

    ``compute_offspring_probability_tensor`` is the numeric kernel; it
    may be called only from the shared wrapper
    (:func:`natal.frontend.data._engine.recompute_offspring_tensor`).
    Any second call site reintroduces the four-way drift the S1
    batches collapsed (writer channel, modifier refresh, registry
    compression, species blueprint, build-time maps).
    """
    allowed = {
        "natal/frontend/data/_engine.py",  # single wrapper (Rust kernel)
    }
    src_root = REPO_ROOT / "src"
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        rel = str(path.relative_to(src_root))
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "compute_offspring_probability_tensor" in text:
            offenders.append(rel)
    assert not offenders, (
        f"offspring derivation re-spelled outside the single wrapper: {offenders}"
    )


def test_graduated_defects_keep_their_promoted_tests() -> None:
    """Every graduated red-light repro has a live pytest home.

    R4/R5 graduated in S2 batch 22a, R3 in batch 22b, and R1/R2 in S3
    (batches 95544b4 and the discrete unification).  Deleting a promoted
    file silently would otherwise go unnoticed by this ledger.
    """
    for test_file in (
        "test_ownership_snapshots.py",
        "test_restore_checkpoint_semantics.py",
        "test_spatial_session_ownership.py",
    ):
        path = REPO_ROOT / "tests" / test_file
        assert path.is_file(), (
            f"graduated repro home {test_file} is missing — the defect's "
            "regression evidence must stay in the suite"
        )
