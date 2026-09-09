"""Regressions for the three evaluator hard-blockers on ``Op.set_param``.

HB-1 — spatial runs must apply ``Op.set_param`` writes at *event*
granularity (per-deme local ecology copies): a first-event write is
visible to the same tick's reproduction and survival.  Locked by
same-tick hand math on a demography whose juvenile regulation actually
binds.

HB-2 — the Rust run path must merge its parameter writes back into the
population's audit trail: ``params_log`` rows under their own commit
ticks and final draft values, with the same jsonc bounds the Python flush
channel enforces (``"K / 0"`` fails loudly mid-run).

HB-3 — a ``None``/``bool`` value expression is a type error, an invalid
token is a value error, and a missing parameter name stays a value error.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import natal as nt  # noqa: E402
from natal.frontend.configurator import Configurator  # noqa: E402
from natal.frontend.hooks.entry.declarative import Op  # noqa: E402
from natal.frontend.hooks.types import HookOp, OpType  # noqa: E402
from natal.frontend.population.age_structured import AgeStructuredPopulation  # noqa: E402
from natal.frontend.population.discrete_generation import (  # noqa: E402
    DiscreteGenerationPopulation,
)
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
        name=f"RustChannelSpecies{_SPECIES_COUNTER}",
        structure={"chr1": {"loc": ["A", "a"]}},
        gamete_labels=["default"],
    )


def _build_viable(
    species: nt.Species,
    name: str,
    *,
    carrying_capacity: float = 900.0,
    hook_calls: Optional[List] = None,
) -> AgeStructuredPopulation:
    """Build a demography whose juvenile regulation actually binds.

    Reproduction parameters are complete (mating, fertility, eggs) and
    juveniles survive their birth tick, so the age-0 output far exceeds
    the FIXED carrying capacity and the cap — not egg production —
    decides the visible age-0 total.  This is the scenario where
    tick-granular (pre-HB-1) writes were observable: the cap value used
    by the same tick's density regulation differs between event-level and
    tick-level semantics.

    Args:
        species: Genetic architecture.
        name: Population name.
        carrying_capacity: Declared capacity.
        hook_calls: Optional ``(items, kwargs)`` pairs declared through
            ``.hooks()`` in the build chain.
    """
    chain = (
        Configurator.from_species(species)
        .age_structure(4, 1)
        .setup(stochastic=False, name=name)
        .initial_state(
            individual_count={
                "female": {"A|A": 60, "A|a": 30},
                "male": {"A|A": 40, "A|a": 20},
            },
            sperm_storage={"A|A": {"A|A": 5, "A|a": 5}},
        )
        .reproduction(eggs_per_female=60.0, sex_ratio=0.5)
        .survival(female_age0_survival=0.5, male_age0_survival=0.5)
        .competition(juvenile_growth_mode=1, carrying_capacity=carrying_capacity)
    )
    for items, kwargs in hook_calls or []:
        chain = chain.hooks(*items, **kwargs)
    return chain.build()


def _build_spatial(
    event: str,
    *,
    carrying_capacity: float = 900.0,
) -> Tuple[SpatialPopulation, List[AgeStructuredPopulation]]:
    """Build one spatial population whose demes share a set_param program.

    Every deme carries ``K * 0.5 every=1`` at *event*; the session runs
    the per-deme lifecycle plus migration.
    """
    species = _fresh_species()
    demes = [
        _build_viable(
            species,
            f"hb1{d}",
            carrying_capacity=carrying_capacity,
            hook_calls=[
                (
                    ([Op.set_param("carrying_capacity", "K * 0.5", every=1, event=event)],),
                    {"event": event},
                )
            ],
        )
        for d in range(2)
    ]
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)
    return spatial, demes


# ---------------------------------------------------------------------------
# Section HB-1 — spatial event-level EcoCtx parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
@pytest.mark.parametrize("event", ["first", "early", "late"])
def test_spatial_set_param_schedules_fire_per_deme_per_tick(event: str) -> None:
    """Every event slot fires the write each tick in every deme.

    ``K * 0.5 every=1`` over four single-tick ``run()`` calls compounds
    the carrying capacity per deme; after each tick every deme's draft
    value equals the hand-compounded ``K0 * 0.5**(tick+1)`` and its log
    carries exactly one new ``(tick, name, old, new)`` row.
    """
    spatial, demes = _build_spatial(event)

    k_expected = 900.0
    expected_rows: List[Tuple[int, str, float, float]] = []
    for tick in range(4):
        spatial.run(1, record_every=0)
        old_k = k_expected
        k_expected *= 0.5
        expected_rows.append((tick, "carrying_capacity", old_k, k_expected))
        for d in range(2):
            assert demes[d].params.carrying_capacity == k_expected, (
                f"deme {d} K at tick {tick} (event={event})"
            )
            assert demes[d].params_log == tuple(expected_rows), (
                f"deme {d} log at tick {tick} (event={event})"
            )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_spatial_first_event_write_binds_same_tick_density_regulation() -> None:
    """The halved K caps the *same tick's* age-0 output, proving the fix.

    With FIXED growth mode the recruited cohort equals
    ``min(produced, K_effective)`` scaled by the age-0 survival (0.5); a
    full tick's aging then moves it to age 1.  The first-event ``K * 0.5``
    fires before reproduction, so the tick-0 age-1 cohort must equal
    ``900 * 0.5 * 0.5 = 225`` exactly (egg production far exceeds the
    cap); the pre-fix Rust path kept the full K for the whole tick and
    would place 450.0 there.  The per-deme log row lands at tick 0.
    """
    spatial, demes = _build_spatial("first")

    spatial.run(1, record_every=0)

    for d in range(2):
        cohort = float(demes[d].state.individual_count[:, 1, :].sum())
        expected_cohort = 900.0 * 0.5 * 0.5
        # The deterministic recruit distributes 112.5 per sex over genotype
        # proportions, so the cap shows up to last-ulp float error; the
        # broken tick-level semantics would land near 450.0 instead.
        assert cohort == pytest.approx(expected_cohort, abs=1e-9), (
            "FIXED cap must bind at the halved K within the same tick"
        )
        assert abs(cohort - 450.0) > 1.0, (
            "tick-level deferral would cap at the un-halved K (450.0)"
        )
        assert demes[d].params.carrying_capacity == 450.0
        assert demes[d].params_log == ((0, "carrying_capacity", 900.0, 450.0),)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_spatial_heterogeneous_columns_split_across_demes() -> None:
    """Distinct per-deme ecology keeps event-level writes deme-local.

    Deme 0 and deme 1 carry different carrying capacities; the write
    compounds each deme's own column (K0 -> K0/2 -> K0/4) without
    cross-talk.
    """
    species = _fresh_species()
    demes = [
        _build_viable(
            species,
            "het0",
            carrying_capacity=800.0,
            hook_calls=[
                (
                    ([Op.set_param("carrying_capacity", "K * 0.5", every=1, event="early")],),
                    {"event": "early"},
                )
            ],
        ),
        _build_viable(
            species,
            "het1",
            carrying_capacity=400.0,
            hook_calls=[
                (
                    ([Op.set_param("carrying_capacity", "K * 0.5", every=1, event="early")],),
                    {"event": "early"},
                )
            ],
        ),
    ]
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)

    for tick in range(2):
        spatial.run(1, record_every=0)
        # Each deme compounds its own column: 800 -> 400 -> 200 and
        # 400 -> 200 -> 100, with one log row per fire per deme.
        assert demes[0].params.carrying_capacity == 800.0 * 0.5 ** (tick + 1)
        assert demes[1].params.carrying_capacity == 400.0 * 0.5 ** (tick + 1)
    assert demes[0].params_log == (
        (0, "carrying_capacity", 800.0, 400.0),
        (1, "carrying_capacity", 400.0, 200.0),
    )
    assert demes[1].params_log == (
        (0, "carrying_capacity", 400.0, 200.0),
        (1, "carrying_capacity", 200.0, 100.0),
    )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_spatial_session_applies_event_writes_raw_rows() -> None:
    """The spatial session applies event writes and journals raw rows.

    With set_param, the per-deme local ecology copies commit at event
    granularity; the raw journal rows are ``(deme, tick, param_id, old,
    new)``.  Driven through the container backend so the session owns
    the stacked state and both deme columns.
    """
    species = _fresh_species()
    demes = [
        _build_viable(
            species,
            f"hom{d}",
            hook_calls=[
                (
                    ([Op.set_param("carrying_capacity", "K * 0.5", every=1, event="first")],),
                    {"event": "first"},
                )
            ],
        )
        for d in range(2)
    ]
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial._initialize_session(seed=5)

    backend = spatial._rust_spatial_backend  # noqa: SLF001 — test drives the backend directly
    assert backend is not None
    backend.run_tick()
    # Raw journal rows: (deme, tick, param_id, old, new) — both demes
    # halved their own (identical) column at tick 0.
    assert backend._session.drain_eco_journal() == [  # noqa: SLF001
        (0, 0, 0, 900.0, 450.0),
        (1, 0, 0, 900.0, 450.0),
    ]
    # Draining is destructive.
    assert backend._session.drain_eco_journal() == []  # noqa: SLF001
    # The event-level write bound the same tick: each deme's recruited
    # cohort (aged to slot 1 by the tick's aging stage) sits at the halved
    # cap times the age-0 survival — far from the un-halved 450.0.  The
    # session-owned state reads back through the backend snapshot.
    snap_tick, ind_flat, _ = backend.state_snapshot()
    assert snap_tick == 1
    ind_all = np.asarray(ind_flat).reshape(2, 2, 4, -1)
    for deme_id in range(2):
        cohort = float(ind_all[deme_id, :, 1, :].sum())
        assert cohort == pytest.approx(900.0 * 0.5 * 0.5, abs=1e-9)
        assert abs(cohort - 450.0) > 1.0


# ---------------------------------------------------------------------------
# Section HB-2 — rust run-path write-channel merge
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_run_merges_journal_into_draft_and_params_log() -> None:
    """After ``run(3)`` the draft and the audit log match hand math.

    ``K * 0.9 every=1`` over one batch call produces three fired ticks;
    each contributes its own ``(tick, name, old, new)`` row (multi-fire,
    multi-row) and the final draft value equals the compounded
    ``900 * 0.9**3`` — the docstring contract "draft synced with the
    session + params_log audit" holds on the engine run path.
    """
    species_rs = _fresh_species()
    pop_rs = _build_viable(
        species_rs,
        "mergers",
        hook_calls=[(([Op.set_param("carrying_capacity", "K * 0.9", every=1)],), {"event": "early"})],
    )
    pop_rs._initialize_session(seed=11)
    pop_rs.run(3, record_every=0)

    expected_rows: List[Tuple[int, str, float, float]] = []
    k = 900.0
    for tick in range(3):
        new_k = k * 0.9
        expected_rows.append((tick, "carrying_capacity", k, new_k))
        k = new_k
    assert pop_rs.params_log == tuple(expected_rows)
    assert pop_rs.params.carrying_capacity == pytest.approx(900.0 * 0.9**3)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_run_inf_expression_raises_value_error_with_param_name() -> None:
    """``"K / 0"`` fails the Rust bounds gate mid-run with a loud error.

    The pre-fix path silently wrote inf into the session columns; the
    commit gate now rejects non-finite values, the adapter converts the
    failure to ``ValueError`` (the Python channel's exception type), and
    the message names the parameter, bounds, value, and tick.
    """
    species = _fresh_species()
    pop = _build_viable(
        species,
        "infrun",
        hook_calls=[(([Op.set_param("carrying_capacity", "K / 0")],), {"event": "early"})],
    )
    pop._initialize_session(seed=13)
    with pytest.raises(ValueError, match="carrying_capacity") as excinfo:
        pop.run(3, record_every=0)
    message = str(excinfo.value)
    assert "inf" in message
    assert "tick 0" in message


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_discrete_run_merges_journal_into_draft_and_params_log() -> None:
    """Discrete populations merge the journal the same way."""
    species_rs = _fresh_species()
    pop_rs = DiscreteGenerationPopulation.setup(
        species=species_rs, name="discmergers", stochastic=False
    ).initial_state(
        individual_count={"female": {"A|A": 40.0}, "male": {"A|A": 20.0}}
    ).reproduction(eggs_per_female=4.0).hooks(
        [Op.set_param("eggs_per_female", "eggs_per_female * 0.5", every=1)],
        event="early",
    ).build()
    pop_rs._initialize_session(seed=19)
    pop_rs.run(3, record_every=0)

    expected_rows = [
        (0, "eggs_per_female", 4.0, 2.0),
        (1, "eggs_per_female", 2.0, 1.0),
        (2, "eggs_per_female", 1.0, 0.5),
    ]
    assert pop_rs.params_log == tuple(expected_rows)
    assert pop_rs.params.eggs_per_female == pytest.approx(0.5)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_spatial_drain_presentation_uses_deme_prefix() -> None:
    """The spatial drain API presents rows as ``deme{i}:{name}``.

    ``params_log`` rows have no deme dimension, so the spatial journal's
    log presentation carries the deme as a name prefix (documented in the
    drain docstring); the population splits it back into per-deme plain
    rows (asserted by the HB-1 tests above).
    """
    species = _fresh_species()
    demes = [
        _build_viable(
            species,
            f"prefix{d}",
            hook_calls=[
                (
                    ([Op.set_param("carrying_capacity", "K * 0.5", every=1, event="first")],),
                    {"event": "first"},
                )
            ],
        )
        for d in range(2)
    ]
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial._initialize_session(seed=0)
    spatial._initialize_session(seed=29)
    backend = spatial._rust_spatial_backend  # noqa: SLF001 — drive the session directly
    assert backend is not None
    backend.run_tick()
    rows = backend.drain_eco_journal()
    assert rows == [
        (0, "deme0:carrying_capacity", 900.0, 450.0),
        (0, "deme1:carrying_capacity", 900.0, 450.0),
    ]
    # Draining is destructive: a second drain returns nothing.
    assert backend.drain_eco_journal() == []


# ---------------------------------------------------------------------------
# Section HB-3 — None -> TypeError error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [None, True, False])
def test_set_param_none_and_bool_value_raise_type_error(bad_value: object) -> None:
    """None / bool values are type errors raised by the value compiler.

    The check lives in ``_compile_value_expr`` (the isinstance branch), so
    the exception names the offending type; registration stays atomic.
    """
    species = _fresh_species()
    with pytest.raises(TypeError, match="set_param value must be a string expression"):
        _build_viable(
            species,
            "badtype",
            hook_calls=[
                (
                    ([Op.set_param("carrying_capacity", bad_value)],),  # type: ignore[arg-type]  # deliberately invalid input
                    {"event": "early"},
                )
            ],
        )


@pytest.mark.parametrize("expr", ["K @ 2", "K $ 2", "K # 2"])
def test_set_param_invalid_character_raises_value_error(expr: str) -> None:
    """Lexically invalid expressions are value errors with the position."""
    species = _fresh_species()
    with pytest.raises(ValueError, match="Unsupported set_param value syntax"):
        _build_viable(
            species,
            f"badlex{abs(hash(expr)) % 10000}",
            hook_calls=[(([Op.set_param("carrying_capacity", expr)],), {"event": "early"})],
        )


def test_set_param_missing_param_name_stays_value_error() -> None:
    """A missing target name remains a value error (not a type error)."""
    species = _fresh_species()
    raw_op = HookOp(
        OpType.SET_PARAM,
        "*",
        "*",
        "both",
        0.0,
        None,
        param_name=None,
        value_expr=1.0,
    )
    with pytest.raises(ValueError, match="requires a parameter name"):
        _build_viable(
            species,
            "noname",
            hook_calls=[(([raw_op],), {"event": "early"})],
        )
