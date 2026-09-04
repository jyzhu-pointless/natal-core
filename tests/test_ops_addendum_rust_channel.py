"""Regressions for the three evaluator hard-blockers on ``Op.set_param``.

HB-1 — Rust spatial runs must apply ``Op.set_param`` writes at *event*
granularity (per-deme local ecology copies), matching the Python per-deme
lifecycle: a first-event write is visible to the same tick's reproduction
and survival.  Locked by python-vs-rust bitwise parity per tick on a
demography whose juvenile regulation actually binds.

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
) -> AgeStructuredPopulation:
    """Build a demography whose juvenile regulation actually binds.

    Reproduction parameters are complete (mating, fertility, eggs) and
    juveniles survive their birth tick, so the age-0 output far exceeds
    the FIXED carrying capacity and the cap — not egg production —
    decides the visible age-0 total.  This is the scenario where
    tick-granular (pre-HB-1) writes were observable: the cap value used
    by the same tick's density regulation differs between event-level and
    tick-level semantics.
    """
    return (
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
        .build()
    )


def _build_spatial_pair(
    event: str,
    *,
    carrying_capacity: float = 900.0,
) -> Tuple[SpatialPopulation, SpatialPopulation, List[AgeStructuredPopulation], List[AgeStructuredPopulation]]:
    """Build python-dispatch and rust spatial twins with one shared program.

    Both carry ``K * 0.5 every=1`` at *event*; the python twin runs the
    per-deme python lifecycle (the reference channel), the rust twin runs
    the Rust spatial backend.
    """
    species_py = _fresh_species()
    demes_py = [_build_viable(species_py, f"hb1py{d}", carrying_capacity=carrying_capacity) for d in range(2)]
    for deme in demes_py:
        deme.register_hooks(
            [Op.set_param("carrying_capacity", "K * 0.5", every=1, event=event)],
            event=event,
        )
    spatial_py = SpatialPopulation(demes_py, migration_rate=0.0)

    species_rs = _fresh_species()
    demes_rs = [_build_viable(species_rs, f"hb1rs{d}", carrying_capacity=carrying_capacity) for d in range(2)]
    for deme in demes_rs:
        deme.register_hooks(
            [Op.set_param("carrying_capacity", "K * 0.5", every=1, event=event)],
            event=event,
        )
    spatial_rs = SpatialPopulation(demes_rs, migration_rate=0.0)
    spatial_rs.enable_rust_backend(seed=17)
    return spatial_py, spatial_rs, demes_py, demes_rs


# ---------------------------------------------------------------------------
# Section HB-1 — spatial event-level EcoCtx parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
@pytest.mark.parametrize("event", ["first", "early", "late"])
def test_spatial_set_param_event_granularity_python_vs_rust_bitwise(event: str) -> None:
    """Rust spatial applies set_param writes at event granularity.

    Per tick over four ticks, every deme's individuals and sperm storage
    are bit-identical between the python dispatch path and the Rust
    backend — only possible when the first/early event write re-enters the
    same tick's reproduction and density regulation (the pre-fix Rust path
    deferred the write to the next tick entry and diverged on tick 0).
    """
    spatial_py, spatial_rs, demes_py, demes_rs = _build_spatial_pair(event)

    for tick in range(4):
        spatial_py.run(1, record_every=0)
        spatial_rs.run(1, record_every=0)
        for d in range(2):
            np.testing.assert_array_equal(
                demes_py[d].state.individual_count,
                demes_rs[d].state.individual_count,
                err_msg=f"deme {d} ind mismatch at tick {tick} (event={event})",
            )
            np.testing.assert_array_equal(
                demes_py[d].state.sperm_storage,
                demes_rs[d].state.sperm_storage,
                err_msg=f"deme {d} sperm mismatch at tick {tick} (event={event})",
            )
            # Write-channel parity per deme: same K value, same log rows.
            assert demes_py[d].params.carrying_capacity == demes_rs[d].params.carrying_capacity
            assert demes_py[d].params_log == demes_rs[d].params_log


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
    spatial_py, spatial_rs, demes_py, demes_rs = _build_spatial_pair("first")

    spatial_py.run(1, record_every=0)
    spatial_rs.run(1, record_every=0)

    for d in range(2):
        cohort_py = float(demes_py[d].state.individual_count[:, 1, :].sum())
        cohort_rs = float(demes_rs[d].state.individual_count[:, 1, :].sum())
        expected_cohort = 900.0 * 0.5 * 0.5
        # The deterministic recruit distributes 112.5 per sex over genotype
        # proportions, so the cap shows up to last-ulp float error; the
        # broken tick-level semantics would land near 450.0 instead.
        assert cohort_py == pytest.approx(expected_cohort, abs=1e-9), (
            "python reference: FIXED cap binds at halved K"
        )
        assert cohort_rs == cohort_py, "rust cohort must equal the python cohort bit-for-bit"
        assert abs(cohort_rs - 450.0) > 1.0, (
            "tick-level deferral would cap at the un-halved K (450.0)"
        )
        assert demes_rs[d].params.carrying_capacity == 450.0
        assert demes_rs[d].params_log == ((0, "carrying_capacity", 900.0, 450.0),)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_spatial_heterogeneous_columns_split_across_demes() -> None:
    """Distinct per-deme ecology keeps event-level writes deme-local.

    Deme 0 and deme 1 carry different carrying capacities; the write
    compounds each deme's own column (K0 -> K0/2 -> K0/4) without
    cross-talk, and both backends stay bitwise identical.
    """
    species = _fresh_species()
    demes_py = [
        _build_viable(species, "hetpy0", carrying_capacity=800.0),
        _build_viable(species, "hetpy1", carrying_capacity=400.0),
    ]
    for deme in demes_py:
        deme.register_hooks(
            [Op.set_param("carrying_capacity", "K * 0.5", every=1, event="early")],
            event="early",
        )
    spatial_py = SpatialPopulation(demes_py, migration_rate=0.0)

    species_rs = _fresh_species()
    demes_rs = [
        _build_viable(species_rs, "hetrs0", carrying_capacity=800.0),
        _build_viable(species_rs, "hetrs1", carrying_capacity=400.0),
    ]
    for deme in demes_rs:
        deme.register_hooks(
            [Op.set_param("carrying_capacity", "K * 0.5", every=1, event="early")],
            event="early",
        )
    spatial_rs = SpatialPopulation(demes_rs, migration_rate=0.0)
    spatial_rs.enable_rust_backend(seed=23)

    for _ in range(2):
        spatial_py.run(1, record_every=0)
        spatial_rs.run(1, record_every=0)

    assert demes_rs[0].params.carrying_capacity == 200.0  # 800 -> 400 -> 200
    assert demes_rs[1].params.carrying_capacity == 100.0  # 400 -> 200 -> 100
    for d in range(2):
        np.testing.assert_array_equal(
            demes_py[d].state.individual_count, demes_rs[d].state.individual_count
        )
        assert demes_py[d].params_log == demes_rs[d].params_log


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_homogeneous_spatial_session_applies_event_writes() -> None:
    """The homogeneous session (``run_spatial_tick``) applies writes too.

    The homogeneous Rust session shares one config across demes; with
    set_param its per-deme local ecology copies must commit at event
    granularity like the heterogeneous session does.  Driven directly at
    the session level (raw ``(deme, tick, param_id, old, new)`` rows),
    using the spatial blueprint so the session owns two deme columns.
    """
    from natal import _engine_rs
    from natal.contracts.materialize import materialize

    species = _fresh_species()
    demes = [_build_viable(species, f"hom{d}") for d in range(2)]
    for deme in demes:
        deme.register_hooks(
            [Op.set_param("carrying_capacity", "K * 0.5", every=1, event="first")],
            event="first",
        )
    spatial = SpatialPopulation(demes, migration_rate=0.0)

    contracts = materialize(demes[0].config)
    # The panmictic contract carries a (1, 2, A) migration column; widen it
    # to the spatial blueprint's extent so the session validates.
    contracts.params.migration_rate = np.zeros((2, 2, 4), dtype=np.float64).ravel()
    session = _engine_rs.SpatialEngineSession(spatial._blueprint, contracts.params, 5)  # noqa: SLF001
    session.set_hook_program(demes[0]._build_hook_program())  # noqa: SLF001 — test drives the session directly
    ind_all, sperm_all = spatial._stack_deme_state_arrays()  # noqa: SLF001
    session.run(ind_all, sperm_all, 0)
    # Raw journal rows: (deme, tick, param_id, old, new) — both demes
    # halved their own (identical) column at tick 0.
    assert session.drain_eco_journal() == [
        (0, 0, 0, 900.0, 450.0),
        (1, 0, 0, 900.0, 450.0),
    ]
    # Draining is destructive.
    assert session.drain_eco_journal() == []
    # The event-level write bound the same tick: each deme's recruited
    # cohort (aged to slot 1 by the tick's aging stage) sits at the halved
    # cap times the age-0 survival — far from the un-halved 450.0.
    for deme in range(2):
        cohort = float(ind_all[deme, :, 1, :].sum())
        assert cohort == pytest.approx(900.0 * 0.5 * 0.5, abs=1e-9)
        assert abs(cohort - 450.0) > 1.0


# ---------------------------------------------------------------------------
# Section HB-2 — rust run-path write-channel merge
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_run_merges_journal_into_draft_and_params_log() -> None:
    """After ``run(3)`` the draft and the audit log match the python twin.

    ``K * 0.9 every=1`` over one batch call produces three fired ticks;
    each contributes its own ``(tick, name, old, new)`` row (multi-fire,
    multi-row) and the final draft value equals the compounded python
    value — the docstring contract "draft synced with the session +
    params_log audit" now holds on the Rust run path.
    """
    species_py = _fresh_species()
    pop_py = _build_viable(species_py, "mergepy")
    pop_py.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.9", every=1)], event="early"
    )
    pop_py._python_backend = True  # noqa: SLF001 — forcing the reference path
    pop_py.run(3, record_every=0)

    species_rs = _fresh_species()
    pop_rs = _build_viable(species_rs, "mergers")
    pop_rs.register_hooks(
        [Op.set_param("carrying_capacity", "K * 0.9", every=1)], event="early"
    )
    pop_rs.enable_rust_backend(seed=11)
    pop_rs.run(3, record_every=0)

    assert pop_rs.params.carrying_capacity == pop_py.params.carrying_capacity
    expected_rows: List[Tuple[int, str, float, float]] = []
    k = 900.0
    for tick in range(3):
        new_k = k * 0.9
        expected_rows.append((tick, "carrying_capacity", k, new_k))
        k = new_k
    assert pop_rs.params_log == tuple(expected_rows)
    assert pop_rs.params_log == pop_py.params_log
    np.testing.assert_array_equal(
        pop_py.state.individual_count, pop_rs.state.individual_count
    )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_run_inf_expression_raises_value_error_with_param_name() -> None:
    """``"K / 0"`` fails the Rust bounds gate mid-run with a loud error.

    The pre-fix path silently wrote inf into the session columns; the
    commit gate now rejects non-finite values, the adapter converts the
    failure to ``ValueError`` (the Python channel's exception type), and
    the message names the parameter, bounds, value, and tick.
    """
    species = _fresh_species()
    pop = _build_viable(species, "infrun")
    pop.register_hooks([Op.set_param("carrying_capacity", "K / 0")], event="early")
    pop.enable_rust_backend(seed=13)
    with pytest.raises(ValueError, match="carrying_capacity") as excinfo:
        pop.run(3, record_every=0)
    message = str(excinfo.value)
    assert "inf" in message
    assert "tick 0" in message


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_rust_discrete_run_merges_journal_into_draft_and_params_log() -> None:
    """Discrete populations merge the journal the same way."""
    species_py = _fresh_species()
    pop_py = DiscreteGenerationPopulation.setup(
        species=species_py, name="discmergepy", stochastic=False
    ).initial_state(
        individual_count={"female": {"A|A": 40.0}, "male": {"A|A": 20.0}}
    ).reproduction(eggs_per_female=4.0).build()
    pop_py.register_hooks(
        [Op.set_param("eggs_per_female", "eggs_per_female * 0.5", every=1)],
        event="early",
    )
    pop_py._python_backend = True  # noqa: SLF001
    pop_py.run(3, record_every=0)

    species_rs = _fresh_species()
    pop_rs = DiscreteGenerationPopulation.setup(
        species=species_rs, name="discmergers", stochastic=False
    ).initial_state(
        individual_count={"female": {"A|A": 40.0}, "male": {"A|A": 20.0}}
    ).reproduction(eggs_per_female=4.0).build()
    pop_rs.register_hooks(
        [Op.set_param("eggs_per_female", "eggs_per_female * 0.5", every=1)],
        event="early",
    )
    pop_rs.enable_rust_backend(seed=19)
    pop_rs.run(3, record_every=0)

    assert pop_rs.params.eggs_per_female == pop_py.params.eggs_per_female
    expected_rows = [
        (0, "eggs_per_female", 4.0, 2.0),
        (1, "eggs_per_female", 2.0, 1.0),
        (2, "eggs_per_female", 1.0, 0.5),
    ]
    assert pop_rs.params_log == tuple(expected_rows)
    assert pop_rs.params_log == pop_py.params_log
    np.testing.assert_array_equal(
        pop_py.state.individual_count, pop_rs.state.individual_count
    )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="rust extension not built")
def test_spatial_drain_presentation_uses_deme_prefix() -> None:
    """The spatial drain API presents rows as ``deme{i}:{name}``.

    ``params_log`` rows have no deme dimension, so the spatial journal's
    log presentation carries the deme as a name prefix (documented in the
    drain docstring); the population splits it back into per-deme plain
    rows (asserted by the HB-1 parity tests above).
    """
    species = _fresh_species()
    demes = [_build_viable(species, f"prefix{d}") for d in range(2)]
    for deme in demes:
        deme.register_hooks(
            [Op.set_param("carrying_capacity", "K * 0.5", every=1, event="first")],
            event="first",
        )
    spatial = SpatialPopulation(demes, migration_rate=0.0)
    spatial.enable_rust_backend(seed=29)
    backend = spatial._rust_spatial_backend  # noqa: SLF001 — drive the session directly
    assert backend is not None
    ind_all, sperm_all = spatial._stack_deme_state_arrays()  # noqa: SLF001
    backend.run(ind_all, sperm_all, 0)
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
    pop = _build_viable(species, "badtype")
    with pytest.raises(TypeError, match="set_param value must be a string expression"):
        pop.register_hooks(
            [Op.set_param("carrying_capacity", bad_value)],  # type: ignore[arg-type]  # deliberately invalid input
            event="early",
        )
    assert len(pop.compiled_hook_descriptors) == 0


@pytest.mark.parametrize("expr", ["K @ 2", "K $ 2", "K # 2"])
def test_set_param_invalid_character_raises_value_error(expr: str) -> None:
    """Lexically invalid expressions are value errors with the position."""
    species = _fresh_species()
    pop = _build_viable(species, f"badlex{abs(hash(expr)) % 10000}")
    with pytest.raises(ValueError, match="Unsupported set_param value syntax"):
        pop.register_hooks([Op.set_param("carrying_capacity", expr)], event="early")


def test_set_param_missing_param_name_stays_value_error() -> None:
    """A missing target name remains a value error (not a type error)."""
    species = _fresh_species()
    pop = _build_viable(species, "noname")
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
        pop.register_hooks([raw_op], event="early")
