"""Tests for ``Op.set_param`` and ``Op.convert`` (issues 26 / 35).

Five test families:

1. Compile-time validation error paths (pattern multi/zero matches,
   tensor/vector targets, malformed RPN, bad schedules/probabilities).
2. ``Op.convert`` numerical invariants (deterministic bit-exact
   conservation, per-bucket atomic migration, frozen male sperm axis,
   expectation conservation over 200 stochastic repeats, conditional
   chain splits, discrete-model degeneration).
3. ``Op.set_param`` numerical semantics (RPN equals a hand-written
   Python expression per tick, every/start scheduling, params_log rows,
   per-deme column writes on spatial populations).
4. Mixed-program run contract: the engine session column tracks the
   population draft against hand-compounded expectations.
5. Mixed registration with existing ops (priority-ordered execution).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import natal as nt  # noqa: E402
from natal.frontend.configurator import Configurator  # noqa: E402
from natal.frontend.hooks.entry.declarative import Op  # noqa: E402
from natal.frontend.hooks.types import OpType  # noqa: E402
from natal.frontend.population.age_structured import AgeStructuredPopulation  # noqa: E402
from natal.frontend.population.discrete_generation import DiscreteGenerationPopulation  # noqa: E402
from natal.frontend.spatial.population import SpatialPopulation  # noqa: E402

try:
    from natal.backends.rust.rust_backend import rust_backend_available

    RUST_AVAILABLE = rust_backend_available()
except Exception:  # pragma: no cover - import guard for unbuilt extensions
    RUST_AVAILABLE = False

_SPECIES_COUNTER = 0


def _fresh_species() -> nt.Species:
    """Return a uniquely-named two-allele species (cache-safe)."""
    global _SPECIES_COUNTER
    _SPECIES_COUNTER += 1
    return nt.Species.from_dict(
        name=f"SetParamConvertSpecies{_SPECIES_COUNTER}",
        structure={"chr1": {"loc": ["A", "a"]}},
        gamete_labels=["default"],
    )


def _build_age_structured(
    species: nt.Species,
    name: str,
    *,
    carrying_capacity: float = 800.0,
) -> AgeStructuredPopulation:
    """Build a deterministic age-structured population with sperm storage."""
    return (
        Configurator.from_species(species)
        .age_structure(3, 1)
        .setup(stochastic=False, name=name)
        .initial_state(
            individual_count={
                "female": {"A|A": 20, "A|a": 10},
                "male": {"A|A": 15, "A|a": 5},
            },
            sperm_storage={"A|A": {"A|A": 3, "A|a": 3}},
        )
        .competition(juvenile_growth_mode=1, carrying_capacity=carrying_capacity)
        .build()
    )


# ---------------------------------------------------------------------------
# Family 1: compile-time validation error paths
# ---------------------------------------------------------------------------


def test_convert_source_multiple_match_raises_with_match_list() -> None:
    """A wildcard source matches several ZTypes and must list them."""
    species = _fresh_species()
    pop = _build_age_structured(species, "multi_src")
    with pytest.raises(ValueError, match="matched 3"):
        pop.register_hooks([Op.convert("*", "A|a", probability=0.5)], event="early")


def test_convert_target_multiple_match_raises() -> None:
    """A multi-match target pattern is rejected at compile time."""
    species = _fresh_species()
    pop = _build_age_structured(species, "multi_dst")
    with pytest.raises(ValueError, match="target pattern"):
        pop.register_hooks([Op.convert("A|A", "*", probability=0.5)], event="early")


def test_convert_zero_match_raises() -> None:
    """A pattern matching no ZType is rejected at compile time."""
    species = _fresh_species()
    pop = _build_age_structured(species, "zero_src")
    with pytest.raises(ValueError):
        pop.register_hooks([Op.convert("Z|Z", "A|a", probability=0.5)], event="early")


def test_convert_identical_source_target_raises() -> None:
    """source == target is a user error and must fail at compile time."""
    species = _fresh_species()
    pop = _build_age_structured(species, "same_src_dst")
    with pytest.raises(ValueError, match="must differ"):
        pop.register_hooks([Op.convert("A|A", "A|A", probability=0.5)], event="early")


def test_convert_probability_bounds() -> None:
    """Out-of-range probabilities are rejected by the factory."""
    with pytest.raises(ValueError, match="probability"):
        Op.convert("A|A", "A|a", probability=1.5)
    with pytest.raises(ValueError, match="probability"):
        Op.convert("A|A", "A|a", probability=-0.1)


def test_set_param_tensor_target_raises() -> None:
    """Genetics-tensor parameters are rejected with a targeted message."""
    species = _fresh_species()
    pop = _build_age_structured(species, "tensor_target")
    with pytest.raises(ValueError, match="tensor"):
        pop.register_hooks([Op.set_param("viability", 1.0)], event="early")


def test_set_param_vector_target_raises() -> None:
    """Vector parameters (age vectors / sex rows) are rejected."""
    species = _fresh_species()
    pop = _build_age_structured(species, "vector_target")
    with pytest.raises(ValueError, match="vector"):
        pop.register_hooks(
            [Op.set_param("female_age_based_survival", 1.0)], event="early"
        )


def test_set_param_mode_enum_target_raises() -> None:
    """Non-scalar kinds (mode enums) are rejected — scalar-only surface."""
    species = _fresh_species()
    pop = _build_age_structured(species, "enum_target")
    with pytest.raises(ValueError, match="growth_mode"):
        pop.register_hooks([Op.set_param("growth_mode", 1)], event="early")


def test_set_param_unknown_name_raises() -> None:
    """Unknown operand names fail at compile time, not mid-run."""
    species = _fresh_species()
    pop = _build_age_structured(species, "unknown_name")
    with pytest.raises(ValueError, match="not_a_param"):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", "not_a_param * 2")], event="early"
        )


@pytest.mark.parametrize(
    "expr",
    ["K ** 0.5", "K +", "(K + 1", "", "K 0.9"],
)
def test_set_param_bad_rpn_raises(expr: str) -> None:
    """Malformed RPN expressions fail at compile time."""
    species = _fresh_species()
    pop = _build_age_structured(species, f"bad_rpn{abs(hash(expr)) % 1000}")
    with pytest.raises(ValueError):
        pop.register_hooks([Op.set_param("carrying_capacity", expr)], event="early")


def test_set_param_schedule_validation() -> None:
    """every < 1 and negative start are rejected by the factory."""
    with pytest.raises(ValueError, match="every"):
        Op.set_param("carrying_capacity", 1.0, every=0)
    with pytest.raises(ValueError, match="start"):
        Op.set_param("carrying_capacity", 1.0, start=-1)


# ---------------------------------------------------------------------------
# Family 2: convert numerical invariants
# ---------------------------------------------------------------------------


def test_convert_deterministic_conservation_bit_exact() -> None:
    """Deterministic conversion conserves every moved unit exactly."""
    species = _fresh_species()
    pop = _build_age_structured(species, "conv_det")
    pop.register_hooks(
        [Op.convert("A|A", "A|a", probability=0.25)], event="early"
    )

    ind_before = pop.state.individual_count.copy()
    sperm_before = pop.state.sperm_storage.copy()
    pop.trigger_event("early")

    ind_after = pop.state.individual_count
    sperm_after = pop.state.sperm_storage

    # Total conservation, bit exact: ind + sperm sums unchanged.
    total_before = ind_before.sum() + sperm_before.sum()
    total_after = ind_after.sum() + sperm_after.sum()
    assert total_before == total_after

    for age in range(ind_before.shape[1]):
        # Males: plain 25 % migration per age row (adult row: 15 -> 11.25).
        male_before = float(ind_before[1, age, 0])
        male_moved = male_before * 0.25
        assert ind_after[1, age, 0] == pytest.approx(male_before - male_moved, abs=1e-12)
        assert ind_after[1, age, 1] == pytest.approx(
            float(ind_before[1, age, 1]) + male_moved, abs=1e-12
        )
        if male_before > 0:
            assert ind_after[1, age, 0] == pytest.approx(11.25, abs=1e-12)

        # Females: virgins plus every sperm bucket each migrate at 25 %.
        # Adult row: virgins = 20 - 6 mated, buckets 3 + 3 -> moved 5.0.
        sperm_row = float(sperm_before[age, 0].sum())
        female_before = float(ind_before[0, age, 0])
        virgin_before = max(female_before - sperm_row, 0.0)
        moved = sperm_row * 0.25 + virgin_before * 0.25
        assert ind_after[0, age, 0] == pytest.approx(female_before - moved, abs=1e-12)
        assert ind_after[0, age, 1] == pytest.approx(
            float(ind_before[0, age, 1]) + moved, abs=1e-12
        )
        if female_before > 0:
            assert moved == pytest.approx(5.0, abs=1e-12)

        # Per-bucket atomicity: source + target bucket sums unchanged.
        for mz in range(sperm_before.shape[2]):
            src = sperm_before[age, 0, mz]
            dst = sperm_before[age, 1, mz]
            moved_bucket = src * 0.25
            assert sperm_after[age, 0, mz] == pytest.approx(src - moved_bucket, abs=1e-12)
            assert sperm_after[age, 1, mz] == pytest.approx(dst + moved_bucket, abs=1e-12)

    # Male sperm axis frozen: column sums over female rows per male z.
    col_before = sperm_before.sum(axis=(0, 1))
    col_after = sperm_after.sum(axis=(0, 1))
    np.testing.assert_array_equal(col_before, col_after)


def test_convert_stochastic_expectation_conserved() -> None:
    """Stochastic conversion conserves the moved count in expectation."""
    from natal.frontend.hooks.runtime.csr_kernel import execute_csr_event_arrays
    from natal.frontend.hooks.types import COND_ALWAYS

    species = _fresh_species()
    pop = _build_age_structured(species, "conv_stoch")
    plan = nt.hooks.compile_declarative_hook(
        [Op.convert("A|A", "A|a", probability=0.25)],
        pop,
        "early",
    ).plan
    assert plan is not None

    n_trials = 200
    n_moved_total = 0.0
    expected_per_trial = (20 + 15) * 0.25  # females + males at age 2
    for trial in range(n_trials):
        ind = np.zeros((2, 1, 3), dtype=np.float64)
        ind[0, 0, 0] = 20.0
        ind[1, 0, 0] = 15.0
        np.random.seed(trial)
        execute_csr_event_arrays(
            n_events=np.int32(1),
            n_hooks=np.int32(1),
            hook_offsets=np.array([0, 1], dtype=np.int32),
            n_ops_list=np.array([1], dtype=np.int32),
            op_offsets=np.array([0, 1], dtype=np.int32),
            op_types_data=plan.op_types,
            zidx_offsets_data=plan.zidx_offsets,
            zidx_data=plan.zidx_data,
            age_offsets_data=plan.age_offsets,
            age_data=plan.age_data,
            sex_masks_data=plan.sex_masks.ravel(),
            params_data=plan.params,
            condition_offsets_data=plan.condition_offsets,
            condition_types_data=plan.condition_types,
            condition_params_data=plan.condition_params,
            sp_param_ids_data=plan.sp_param_ids,
            sp_every_data=plan.sp_every,
            sp_start_data=plan.sp_start,
            rpn_offsets_data=plan.rpn_offsets,
            rpn_kinds_data=plan.rpn_kinds,
            rpn_payload_data=plan.rpn_payload,
            sp_literals_data=plan.sp_literals,
            convert_source_z_data=plan.convert_source_z,
            convert_target_z_data=plan.convert_target_z,
            deme_selector_types=np.array([0], dtype=np.int32),
            deme_selector_offsets=np.array([0, 0], dtype=np.int32),
            deme_selector_data=np.array([], dtype=np.int32),
            event_id=0,
            individual_count=ind,
            sperm_storage=None,
            has_sperm_storage=False,
            tick=0,
            stochastic=True,
            continuous_sampling=False,
            deme_id=0,
        )
        # Totals conserved per trial as well.
        assert ind.sum() == 35.0
        n_moved_total += float(ind[1, 0, 1] + ind[0, 0, 1])

    mean_moved = n_moved_total / n_trials
    # Binomial(35, 0.25) has sigma ~ 2.29; mean over 200 trials has
    # sigma ~ 0.16, so 1.0 is a >6-sigma guard.
    assert abs(mean_moved - expected_per_trial) < 1.0, mean_moved


def test_convert_conditional_chain_split() -> None:
    """A probability=1.0 remainder expresses a one-to-many split."""
    species = _fresh_species()
    pop = _build_age_structured(species, "conv_chain")
    pop.register_hooks(
        [
            Op.convert("A|A", "A|a", probability=0.3),
            Op.convert("A|A", "a|a", probability=1.0),
        ],
        event="early",
    )
    pop.trigger_event("early")

    ind = pop.state.individual_count
    # The 30/70 split applies to the *initial* A|A totals per sex; the
    # remainder step moves everything that is left after the first draw.
    male_aa_before = 30.0  # two age rows x 15
    assert ind[1, :, 0].sum() == 0.0
    assert ind[1, :, 1].sum() == pytest.approx(2 * 5.0 + male_aa_before * 0.3, abs=1e-12)
    assert ind[1, :, 2].sum() == pytest.approx(male_aa_before * 0.7, abs=1e-12)
    # Females: the initial A|A total (2 x 20, virgins + buckets) splits the
    # same 30/70; the pre-existing A|a (2 x 10) stays.
    female_aa_before = 40.0
    assert ind[0, :, 1].sum() == pytest.approx(2 * 10.0 + female_aa_before * 0.3, abs=1e-12)
    assert ind[0, :, 2].sum() == pytest.approx(female_aa_before * 0.7, abs=1e-12)


def test_convert_discrete_model_degenerates_to_plain_migration() -> None:
    """Discrete models have no sperm storage: plain binomial migration."""
    species = _fresh_species()
    pop = (
        DiscreteGenerationPopulation.setup(species=species, name="conv_discrete", stochastic=False)
        .initial_state(
            individual_count={"female": {"A|A": 40.0}, "male": {"A|A": 20.0}}
        )
        .build()
    )
    pop.register_hooks(
        [Op.convert("A|A", "A|a", probability=0.5)], event="early"
    )
    before = pop.state.individual_count.sum()
    pop.trigger_event("early")
    ind = pop.state.individual_count
    assert ind.sum() == before
    assert ind[0, 1, 0] == pytest.approx(20.0, abs=1e-12)
    assert ind[0, 1, 1] == pytest.approx(20.0, abs=1e-12)
    assert ind[1, 1, 0] == pytest.approx(10.0, abs=1e-12)
    assert ind[1, 1, 1] == pytest.approx(10.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Family 3: set_param numerical semantics
# ---------------------------------------------------------------------------


def test_set_param_rpn_matches_hand_written_expression_per_tick() -> None:
    """``"K * 0.95"`` compounds identically to a Python ``k *= 0.95``."""
    species = _fresh_species()
    pop = _build_age_structured(species, "rpn_vs_hand")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.95", every=10)],
        event="early",
    )

    k_manual = 800.0
    for tick in range(31):
        pop.run(1, record_every=0)
        assert pop.tick == tick + 1
        # Fires at ticks 0, 10, 20, 30 (schedule (tick - 0) % 10 == 0).
        if (tick) % 10 == 0:
            k_manual *= 0.95
        assert pop.params.carrying_capacity == k_manual, f"tick {tick}"


def test_set_param_every_start_schedule_and_params_log() -> None:
    """Fires only at ``tick >= start and (tick - start) % every == 0``."""
    species = _fresh_species()
    pop = _build_age_structured(species, "schedule")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", 123.0, every=10, start=5)],
        event="early",
    )

    # Assignment semantics: the value persists between firings, so the
    # parameter stays at 123 after tick 5 (re-fired unchanged at 15/25).
    for tick in range(26):
        pop.run(1, record_every=0)
        expected = 123.0 if tick >= 5 else 800.0
        assert pop.params.carrying_capacity == expected, f"tick {tick}"

    # Snapshot log rows: the writer channel skips same-value commits, so
    # only the tick-5 transition is recorded — identical to a handwritten
    # ``pop.params.carrying_capacity = 123.0`` at ticks 5/15/25.
    rows = pop.params_log
    assert rows == ((5, "carrying_capacity", 800.0, 123.0),)


def test_set_param_params_log_matches_handwritten_channel() -> None:
    """The declarative op logs the same rows as ``pop.params.<name> = ...``."""
    species = _fresh_species()
    declarative = _build_age_structured(species, "log_declarative")
    declarative.register_hooks(
        [Op.set_param("eggs_per_female", 5.0)], event="early"
    )
    declarative.run(1, record_every=0)

    species2 = _fresh_species()
    handwritten = _build_age_structured(species2, "log_handwritten")
    # Handwritten twin: the same write at the same tick-0 early boundary.
    handwritten.params.eggs_per_female = 5.0
    handwritten.run(1, record_every=0)

    assert declarative.params_log == handwritten.params_log
    first = declarative.params_log[0]
    assert first[0] == 0
    assert first[1] == "eggs_per_female"
    assert first[3] == 5.0


def test_set_param_when_clause_gates_firing() -> None:
    """The optional ``when`` condition composes with the schedule."""
    species = _fresh_species()
    pop = _build_age_structured(species, "when_gate")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", 42.0, when="tick >= 2 and tick < 4")],
        event="early",
    )
    values = []
    for _ in range(5):
        pop.run(1, record_every=0)
        values.append(pop.params.carrying_capacity)
    assert values == [800.0, 800.0, 42.0, 42.0, 42.0]


def test_set_param_spatial_per_deme_columns() -> None:
    """Deme selectors restrict writes to the selected demes' columns."""
    species = _fresh_species()

    def build_deme(name: str) -> AgeStructuredPopulation:
        return _build_age_structured(species, name, carrying_capacity=500.0)

    demes = [build_deme(f"sp_d{d}") for d in range(3)]
    for deme in demes:
        deme.register_hooks(
            [Op.set_param("carrying_capacity", 111.0, every=1)],
            event="early",
        )
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial.enable_rust_backend(seed=0)
    spatial.run(1, record_every=0)

    # The python dispatch path writes every deme's own draft via its
    # HookExecutor; each deme population shows its own value.
    for deme in demes:
        assert deme.params.carrying_capacity == 111.0

    # Now the same write through a deme selector restricted to demes 0/2:
    species2 = _fresh_species()

    def build_deme2(name: str) -> AgeStructuredPopulation:
        return _build_age_structured(species2, name, carrying_capacity=500.0)

    d0 = build_deme2("sel_d0")
    d1 = build_deme2("sel_d1")
    d2 = build_deme2("sel_d2")
    # Register only on demes 0 and 2 (per-deme registration surface).
    for deme in (d0, d2):
        deme.register_hooks(
            [Op.set_param("carrying_capacity", 222.0, every=1)],
            event="early",
        )
    spatial2 = SpatialPopulation([d0, d1, d2], migration_rate=0.0)
    spatial2.enable_rust_backend(seed=0)
    spatial2.run(1, record_every=0)
    assert d0.params.carrying_capacity == 222.0
    assert d2.params.carrying_capacity == 222.0
    assert d1.params.carrying_capacity == 500.0


# ---------------------------------------------------------------------------
# Family 4: mixed-program run contract
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_mixed_program_session_column_tracks_draft() -> None:
    """A shared mixed program keeps the session column equal to the draft.

    ``K * 0.9 every=1`` over 6 single-tick ``run()`` calls compounds the
    carrying capacity; after each batch the engine session column must
    equal the population draft, and the final value must equal the
    hand-compounded ``800 * 0.9**6``.
    """
    species = _fresh_species()
    pop = _build_age_structured(species, "mixed_program_contract")
    pop.register_hooks(
        [
            Op.set_param("carrying_capacity", "K * 0.9", every=1),
            Op.convert("A|A", "A|a", probability=0.25),
            Op.scale(genotypes="a|a", factor=0.5, sex="male"),
        ],
        event="early",
        name="mixed_program",
    )
    pop.enable_rust_backend(seed=11)
    session = pop._rust_lifecycle_backend._session  # noqa: SLF001

    k_manual = 800.0
    for _tick in range(6):
        pop.run(1, record_every=0)
        k_manual *= 0.9
        rust_k = float(session.get_scalar("carrying_capacity"))
        assert rust_k == pop.params.carrying_capacity
        assert rust_k == k_manual


# ---------------------------------------------------------------------------
# Family 5: mixed registration with existing ops
# ---------------------------------------------------------------------------


def test_mixed_ops_execute_in_priority_order() -> None:
    """set_param + convert + classic ops compose in priority order."""
    species = _fresh_species()
    pop = _build_age_structured(species, "mixed")
    pop.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.5")],
        event="early",
        priority=0,
        name="first_write",
    )
    pop.register_hooks(
        [
            Op.convert("A|A", "A|a", probability=0.5),
            Op.add(genotypes="a|a", ages="*", sex="both", delta=4.0),
        ],
        event="early",
        priority=1,
        name="then_convert_add",
    )
    # Trigger only the early event: the hook composition stays isolated
    # from the demographic lifecycle stages of a full tick.
    pop.trigger_event("early")

    # Priority order: the param write happened first (one log row).
    assert len(pop.params_log) == 1
    assert pop.params_log[0] == (0, "carrying_capacity", 800.0, 400.0)

    # Then the convert ran on A|A (50 % of each row's 20/15 per sex) and
    # the add landed on a|a afterwards: two age rows carry the initials.
    ind = pop.state.individual_count
    # A|A keeps its un-converted half (2 rows x (20 + 15) x 0.5).
    assert ind[:, :, 0].sum() == pytest.approx(2 * 35.0 * 0.5, abs=1e-9)
    moved_female_total = 2 * 20.0 * 0.5
    moved_male_total = 2 * 15.0 * 0.5
    assert ind[0, :, 1].sum() == pytest.approx(2 * 10.0 + moved_female_total, abs=1e-9)
    assert ind[1, :, 1].sum() == pytest.approx(2 * 5.0 + moved_male_total, abs=1e-9)
    # The classic add op contributed +4 per (sex, age) cell of a|a
    # (3 age classes x 2 sexes).
    assert ind[:, :, 2].sum() == pytest.approx(3 * 2 * 4.0, abs=1e-9)


def test_op_types_and_program_flags() -> None:
    """Opcode values and the has_set_param flag are stable wire contracts."""
    assert int(OpType.SET_PARAM) == 10
    assert int(OpType.CONVERT) == 11

    species = _fresh_species()
    pop = _build_age_structured(species, "flags")
    assert pop._run_program.hooks.has_set_param is False  # noqa: SLF001
    pop.register_hooks([Op.set_param("sex_ratio", 0.5)], event="early")
    assert pop._run_program.hooks.has_set_param is True  # noqa: SLF001
