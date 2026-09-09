"""strict tests: the config domain attack suite.

Complements ``test_routes_slice3.py`` (which proves the table shapes and
build-path contracts) with runtime-side attacks.  Every assertion proves
one numerical or identity invariant:

1. **Method-level atomicity**: one invalid entry in a writer batch must
   leave *zero* writes on both sides — the draft AND the live Rust
   session — regardless of dict insertion order.  A failed batch must
   also leave a valid recovery batch fully functional.
2. **Sensitive-driven sync**: writes flagged ``sensitive`` recompute the
   equilibrium caches to *exactly* what
   ``compute_equilibrium_metrics`` returns for the new inputs;
   non-sensitive writes leave the caches bit-identical; a cleared
   Champer egg override is pushed to the session as ``-1.0`` and the
   derive-mode equilibrium declaration is *not* pushed at all.
3. **Mode-enum parsing matrix**: string aliases resolve case-
   insensitively to canonical integers; integers pass through only
   inside ``[0, 4]``; bools and floats are type-rejected.
4. ``pop.params`` **surface against a live session**: rejected setters
   leave draft + session + rebuild flag untouched, pattern reads
   aggregate exactly like a manual sum over the resolved ztype
   indices, reads are independent copies on both sides, and
   ``tensor_write`` is visible in the session immediately.
5. **Vocabulary preservation (axis combos)**: the (age x discrete)
   granularity x method-name matrix writes exact unified-vector cells;
   per-age reproduction names stay rejected on discrete drafts;
   build-time and runtime vocabulary produce identical deterministic
   trajectories.
6. **Route-table data integrity**: exactly 44 rows in the seven shapes
   with the documented counts; ``geno_tensor`` <=> ``genetics`` section;
   the sensitive set is exactly the five documented parameters; every
   alias resolves to its owning entry and stays writable.
7. **Ricker engine path (mode 4)**: a deterministic Rust discrete
   population with ``growth_mode="ricker"`` reproduces the hand-derived
   Ricker recursion tick-by-tick, and mode 4 diverges from mode 3 —
   proving the Rust density kernel implements mode 4 end-to-end.
8. **HookConfigWriter**: the in-hook writer talks to the session and to
   nothing else — no refresh, no rebuild, no draft, no rebuild
   scheduling — and its direct writes survive into the next ``run()``.
9. **Negative contracts**: frozen route records, replace-field
   classification, species-context guard for pattern patches, and
   pattern strings rejected outside the last axis.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal.backends.rust.rust_backend import rust_backend_available
from natal.contracts.params import Params
from natal.frontend.configurator import Configurator
from natal.frontend.configurator._routes import (
    ROUTES,
    commit_write,
    dispatch,
    is_replace_field,
    lookup,
    plan_write,
)
from natal.frontend.configurator._writers import (
    CoreConfigWriter,
    DraftWriter,
    HookConfigWriter,
)
from natal.frontend.data import ModelDraft, build_population_config
from natal.frontend.data._engine import derive_equilibrium_metrics_from_draft

# ── markers and shared builders ───────────────────────────────────────────────

RUST = pytest.mark.skipif(
    not rust_backend_available(), reason="rust extension not built"
)


class RecordingSession:
    """Structural SessionChannel fake recording every pushed write."""

    def __init__(self) -> None:
        self.applied: list[dict[str, float]] = []
        self.tensors: list[tuple[str, NDArray[np.float64]]] = []
        self.other_calls: list[str] = []
        self.refreshed: list[tuple[list[str], Params]] = []

    def apply(self, writes: dict[str, float]) -> None:
        self.applied.append(dict(writes))

    def tensor_write(self, field: str, values: NDArray[np.float64]) -> None:
        self.tensors.append((field, np.asarray(values, dtype=np.float64).copy()))

    def refresh_params(self, fields: list[str], source: Params) -> None:
        self.refreshed.append((list(fields), source))

    def rebuild(self) -> None:
        self.other_calls.append("rebuild")


@pytest.fixture(scope="module")
def age_species() -> nt.Species:
    return nt.Species.from_dict(
        name="__slice3cfg_age__",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


@pytest.fixture(scope="module")
def discrete_species() -> nt.Species:
    return nt.Species.from_dict(
        name="__slice3cfg_discrete__",
        structure={"auto": {"A": ["WT", "Var"]}},
    )


def _fresh_age_species() -> nt.Species:
    """A standalone age-structured Species (builders must not share state)."""
    return nt.Species.from_dict(
        name="__slice3cfg_age__",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _fresh_discrete_species() -> nt.Species:
    return nt.Species.from_dict(
        name="__slice3cfg_discrete__",
        structure={"auto": {"A": ["WT", "Var"]}},
    )


def _age_draft() -> ModelDraft:
    return build_population_config(
        n_genotypes=4,
        n_gtypes=4,
        n_glabs=1,
        n_ages=3,
        new_adult_age=1,
    )


def _age_pop() -> nt.AgeStructuredPopulation:
    return (
        nt.AgeStructuredPopulation.setup(
            _fresh_age_species(),
            stochastic=False,
        )
        .age_structure(4, 2)
        .initial_state(
            individual_count={
                "female": {"A|A": 2000, "A|B": 1000},
                "male": {"A|A": 1600, "B|B": 400},
            }
        )
        .competition(carrying_capacity=500.0, juvenile_growth_mode=2)
        .reproduction(eggs_per_female=20.0, sex_ratio=0.5)
        .build()
    )


def _discrete_pop(
    mode: str | int,
    *,
    r: float = 3.5,
    k: float = 400.0,
    eggs: float = 6.0,
) -> nt.DiscreteGenerationPopulation:
    return (
        nt.DiscreteGenerationPopulation.setup(
            _fresh_discrete_species(),
            stochastic=False,
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=eggs, sex_ratio=0.5)
        .competition(
            carrying_capacity=k,
            low_density_growth_rate=r,
            growth_mode=mode,
        )
        .initial_state(
            individual_count={"female": {"WT|WT": 900}, "male": {"WT|WT": 900}},
        )
        .build()
    )


def _pop_writer(pop: nt.AgeStructuredPopulation) -> CoreConfigWriter:
    """A CoreConfigWriter wired exactly like ParamsView._writer does."""
    return CoreConfigWriter(
        pop.config,
        getattr(pop, "_rust_lifecycle_backend", None),
        on_replace=pop.set_config,
        species=pop.species,
        registry=pop.index_registry,
    )


# ── 1. method-level atomicity (dual side, order attacks) ──────────────────────


class TestMethodLevelAtomicity:
    @staticmethod
    def _orderings() -> list[dict[str, object]]:
        """The same legal/illegal batch in every insertion order.

        ``sex_ratio = 7.0`` violates bounds [0, 1]; the two other entries
        are legal scalars.  The illegal entry moves through every
        position to prove planning happens wholly before committing.
        """
        legal_k = ("carrying_capacity", 555.0)
        illegal = ("sex_ratio", 7.0)
        legal_r = ("low_density_growth_rate", 2.0)
        return [
            dict([legal_k, illegal]),
            dict([illegal, legal_k]),
            dict([legal_k, legal_r, illegal]),
            dict([illegal, legal_r, legal_k]),
        ]

    def test_core_writer_rejected_batch_zero_writes_both_sides(self):
        pop = _age_pop()
        session = pop._rust_lifecycle_backend._session  # noqa: SLF001 — readback channel
        writer = _pop_writer(pop)
        draft_k0 = float(pop.config.carrying_capacity)
        session_k0 = session.get_scalar("carrying_capacity")
        for writes in self._orderings():
            with pytest.raises(ValueError):
                writer.apply(writes)
            # Draft side: bit-identical scalar.
            assert float(pop.config.carrying_capacity) == draft_k0
            # Session side: the live Rust params never saw the batch.
            assert session.get_scalar("carrying_capacity") == session_k0

    def test_fake_session_rejected_batch_pushes_nothing(self):
        cfg = _age_draft()
        session = RecordingSession()
        writer = CoreConfigWriter(cfg, session)
        with pytest.raises(ValueError):
            writer.apply({"carrying_capacity": 555.0, "sex_ratio": 7.0})
        assert session.applied == []
        assert session.tensors == []
        assert session.refreshed == []

    def test_mixed_kind_batch_rejected_zero_writes(self):
        # The illegal entry is a geno_tensor shape mismatch inside a batch
        # that also carries a scalar: plans are all resolved before any
        # commit, so neither side may move.
        pop = _age_pop()
        session = pop._rust_lifecycle_backend._session  # noqa: SLF001
        writer = _pop_writer(pop)
        rates0 = np.asarray(pop.config.age_based_survival_rates).copy()
        k0 = session.get_scalar("carrying_capacity")
        with pytest.raises(ValueError, match="requires an array of shape"):
            writer.apply(
                {
                    "carrying_capacity": 555.0,
                    "fecundity": np.ones((5, 5)),
                }
            )
        assert float(pop.config.carrying_capacity) != 555.0
        np.testing.assert_array_equal(
            np.asarray(pop.config.age_based_survival_rates), rates0
        )
        assert session.get_scalar("carrying_capacity") == k0

    def test_tensor_write_size_mismatch_zero_writes_both_sides(self):
        pop = _age_pop()
        session = pop._rust_lifecycle_backend._session  # noqa: SLF001
        rates0 = np.asarray(pop.config.age_based_survival_rates).copy()
        session0 = np.asarray(session.get_tensor("survival_rates")).copy()
        with pytest.raises(ValueError, match="expected"):
            pop.params.tensor_write("survival_rates", np.ones(7))
        np.testing.assert_array_equal(
            np.asarray(pop.config.age_based_survival_rates), rates0
        )
        np.testing.assert_array_equal(
            np.asarray(session.get_tensor("survival_rates")), session0
        )

    def test_draft_writer_rejected_batch_commits_nothing(self):
        cfg = _age_draft()
        writer = DraftWriter(cfg)
        writer.apply({"carrying_capacity": 400.0})
        with pytest.raises(ValueError):
            writer.apply({"eggs_per_female": 33.0, "sex_ratio": -0.5})
        # A rejected batch commits no field — atomic, nothing derived moved.
        assert float(writer.draft.carrying_capacity) == 400.0
        assert float(writer.draft.eggs_per_female) != 33.0
        assert float(writer.draft.sex_ratio) >= 0.0

    def test_recovery_after_rejected_batch(self):
        # Atomicity must not poison the writer: the same writer commits a
        # fully-valid batch right after the rejected one.
        cfg = _age_draft()
        writer = DraftWriter(cfg)
        with pytest.raises(ValueError):
            writer.apply({"carrying_capacity": 555.0, "sex_ratio": 7.0})
        writer.apply({"carrying_capacity": 556.0})
        assert float(writer.draft.carrying_capacity) == 556.0


# ── 2. sensitive-driven equilibrium sync ──────────────────────────────────────


class TestSensitiveDrivenSync:
    def test_sensitive_write_moves_the_derived_metrics(self):
        # A sensitive write must refresh the derived metrics: K 100 -> 400
        # strictly lowers the expected competition strength C*.
        writer = DraftWriter(_age_draft())
        comp0, _ = derive_equilibrium_metrics_from_draft(writer.draft)
        writer.apply({"carrying_capacity": 400.0})
        comp1, _ = derive_equilibrium_metrics_from_draft(writer.draft)
        assert comp1 < comp0

    def test_declared_distribution_switches_the_derived_metrics(self):
        writer = DraftWriter(_age_draft())
        derive0 = derive_equilibrium_metrics_from_draft(writer.draft)
        declared = np.array([[30.0, 30.0, 0.0], [70.0, 70.0, 0.0]])
        writer.apply({"equilibrium_distribution": declared})
        derive1 = derive_equilibrium_metrics_from_draft(writer.draft)
        # Declaring the equilibrium distribution switches the kernel
        # branch (declared column instead of the derived one).
        assert derive1 != derive0

    def test_non_sensitive_write_changes_only_its_own_field(self):
        writer = DraftWriter(_age_draft())
        writer.apply({"carrying_capacity": 400.0})
        comp0, surv0 = derive_equilibrium_metrics_from_draft(writer.draft)
        writer.apply({"sperm_displacement_rate": 0.3})
        got_comp, got_surv = derive_equilibrium_metrics_from_draft(writer.draft)
        # sperm_displacement_rate does not enter the equilibrium formulas,
        # so the derived metrics are bit-identical.
        assert got_comp == comp0
        assert got_surv == surv0
        assert float(writer.draft.sperm_displacement_rate) == 0.3

    def test_cleared_champer_override_pushed_as_minus_one(self):
        session = RecordingSession()
        writer = CoreConfigWriter(_age_draft(), session)
        writer.apply({"external_expected_eggs": 123.0})
        assert session.applied[0] == {"external_expected_eggs": 123.0}
        writer.apply({"external_expected_eggs": None})
        # The cleared declaration materializes as the -1.0 sentinel.
        assert session.applied[-1] == {"external_expected_eggs": -1.0}
        assert writer.draft.external_expected_eggs is None

    def test_derive_mode_equilibrium_declaration_not_pushed(self):
        session = RecordingSession()
        writer = CoreConfigWriter(_age_draft(), session)
        declared = np.array([[10.0, 5.0, 1.0], [10.0, 5.0, 1.0]])
        writer.apply({"equilibrium_distribution": declared})
        assert len(session.refreshed) == 1
        fields, params = session.refreshed[0]
        assert fields == ["equilibrium_distribution"]
        np.testing.assert_array_equal(params.equilibrium_distribution, declared)
        writer.apply({"equilibrium_distribution": None})
        # Clearing is an explicit native commit of the derive-mode sentinel.
        assert len(session.refreshed) == 2
        assert session.refreshed[-1][0] == ["equilibrium_distribution"]
        assert session.refreshed[-1][1].equilibrium_distribution.size == 0
        assert writer.draft.equilibrium_individual_distribution is None


# ── 3. mode-enum parsing matrix ───────────────────────────────────────────────


class TestModeEnumMatrix:
    @pytest.mark.parametrize(
        ("text", "canonical"),
        [
            ("RICKER", 4),
            ("Beverton_Holt", 3),
            ("NO_COMPETITION", 0),
            ("Logistic", 2),
            ("FIXED", 1),
        ],
    )
    def test_case_insensitive_aliases(self, text: str, canonical: int):
        live = dispatch(_age_draft(), "growth_mode", text)
        assert int(live.juvenile_growth_mode) == canonical

    @pytest.mark.parametrize("bad", [5, -1, 100])
    def test_integer_out_of_bounds_rejected(self, bad: int):
        with pytest.raises(ValueError, match="requires a value in"):
            dispatch(_age_draft(), "growth_mode", bad)

    def test_bool_and_float_rejected(self):
        with pytest.raises(TypeError, match="mode string or an integer"):
            dispatch(_age_draft(), "growth_mode", True)
        with pytest.raises(TypeError, match="mode string or an integer"):
            dispatch(_age_draft(), "growth_mode", 2.0)

    def test_numpy_integer_passthrough(self):
        live = dispatch(_age_draft(), "growth_mode", np.int64(3))
        assert int(live.juvenile_growth_mode) == 3

    def test_route_bounds_are_zero_to_four(self):
        assert lookup("growth_mode").bounds == (0.0, 4.0)


# ── 4. pop.params surface against a live session ──────────────────────────────


@RUST
class TestParamsSessionSurface:
    def _rust_age_pop(self) -> nt.AgeStructuredPopulation:
        return _age_pop()

    def _session(self, pop: nt.AgeStructuredPopulation) -> object:
        return pop._rust_lifecycle_backend._session  # noqa: SLF001

    def test_rejected_setter_leaves_three_sides_unchanged(self):
        pop = self._rust_age_pop()
        session = self._session(pop)
        draft0 = float(pop.config.carrying_capacity)
        session0 = session.get_scalar("carrying_capacity")  # type: ignore[attr-defined]  # rust session readback
        with pytest.raises(ValueError, match="requires a value in"):
            pop.params.carrying_capacity = -1.0
        assert float(pop.config.carrying_capacity) == draft0
        assert session.get_scalar("carrying_capacity") == session0  # type: ignore[attr-defined]
        assert pop._rust_needs_rebuild is False

    def test_pattern_read_aggregates_exactly_like_manual_sum(self):
        pop = self._rust_age_pop()
        arr = np.asarray(pop.config.viability_fitness)
        for pattern in ("A|*", "*", "B|B"):
            parsed = nt.ZygoteTypePattern.parse(pattern, pop.species)
            indices = pop.index_registry.resolve_ztype_indices(parsed)
            assert indices, pattern
            expected = float(arr[0, 1, tuple(indices)].sum())
            assert pop.params.viability_fitness[0, 1, pattern] == expected

    def test_tensor_read_copy_is_independent_on_both_sides(self):
        pop = self._rust_age_pop()
        session = self._session(pop)
        draft0 = np.asarray(pop.config.age_based_survival_rates).copy()
        session0 = np.asarray(session.get_tensor("survival_rates")).copy()  # type: ignore[attr-defined]
        copy = pop.params.survival_rates.copy()
        copy[:] = 0.0
        np.testing.assert_array_equal(
            np.asarray(pop.config.age_based_survival_rates), draft0
        )
        np.testing.assert_array_equal(
            np.asarray(session.get_tensor("survival_rates")),
            session0,  # type: ignore[attr-defined]
        )

    def test_direct_subscript_write_is_rejected(self):
        pop = self._rust_age_pop()
        with pytest.raises(TypeError, match="tensor_write"):
            pop.params.viability[0, 0, 0] = 0.5

    def test_tensor_write_is_session_visible_immediately(self):
        pop = self._rust_age_pop()
        session = self._session(pop)
        rates = np.full((2, 4), 0.77)
        pop.params.tensor_write("survival_rates", rates)
        np.testing.assert_allclose(
            np.asarray(pop.config.age_based_survival_rates), rates
        )
        np.testing.assert_allclose(
            np.asarray(session.get_tensor("survival_rates")).reshape(2, 4),  # type: ignore[attr-defined]
            rates,
        )

    def test_scalar_write_is_session_visible_immediately(self):
        pop = self._rust_age_pop()
        session = self._session(pop)
        pop.params.carrying_capacity = 900.0
        assert pop.params.carrying_capacity == 900.0
        assert session.get_scalar("carrying_capacity") == 900.0  # type: ignore[attr-defined]

    def test_bool_write_marks_rebuild_and_run_drains_it(self):
        pop = self._rust_age_pop()
        pop.params.fixed_egg_count = True
        assert pop.params.fixed_egg_count is True
        # Boolean rows are frozen Blueprint flags: rebuild sentinel only.
        assert pop._rust_needs_rebuild is True
        pop.run(n_steps=1)
        assert pop._rust_needs_rebuild is False
        assert pop.state.individual_count.sum() > 0.0

    def test_spatial_and_unknown_names_raise_attribute_error(self):
        pop = self._rust_age_pop()
        with pytest.raises(AttributeError, match="spatial container"):
            _ = pop.params.migration_rate
        with pytest.raises(AttributeError, match="has no parameter"):
            _ = pop.params.definitely_not_a_parameter


# ── 5. vocabulary preservation (axis combos) ─────────────────────────────────


class TestVocabularyAxisCombos:
    # Default unified vectors carry 0.0 in the juvenile column and 1.0 in
    # the adult column (age drafts n_ages=3; discrete drafts n_ages=2).
    @pytest.mark.parametrize(
        ("method", "kwargs", "field", "expected"),
        [
            (
                "survival",
                {"female_age_based_survival": [0.1, 0.2, 0.3]},
                "age_based_survival_rates",
                [[0.1, 0.2, 0.3], [0.0, 1.0, 1.0]],
            ),
            (
                "survival",
                {"male_age_based_survival": 0.6},
                "age_based_survival_rates",
                [[0.0, 1.0, 1.0], [0.6, 0.6, 0.6]],
            ),
            (
                "survival",
                {"female_age0_survival": 0.05},
                "age_based_survival_rates",
                [[0.05, 1.0, 1.0], [0.0, 1.0, 1.0]],
            ),
            (
                "survival",
                {"male_age0_survival": 0.04},
                "age_based_survival_rates",
                [[0.0, 1.0, 1.0], [0.04, 1.0, 1.0]],
            ),
            (
                "reproduction",
                {"female_age_based_mating_rate": 0.8},
                "age_based_mating_rates",
                [[0.8, 0.8, 0.8], [0.0, 1.0, 1.0]],
            ),
            (
                "reproduction",
                {"male_age_based_mating_rate": [0.2, 0.3, 0.4]},
                "age_based_mating_rates",
                [[0.0, 1.0, 1.0], [0.2, 0.3, 0.4]],
            ),
            (
                "reproduction",
                {"female_adult_mating_rate": 0.33},
                "age_based_mating_rates",
                [[0.0, 0.33, 1.0], [0.0, 1.0, 1.0]],
            ),
            (
                "reproduction",
                {"male_adult_mating_rate": 0.44},
                "age_based_mating_rates",
                [[0.0, 1.0, 1.0], [0.0, 0.44, 1.0]],
            ),
            (
                "reproduction",
                {"age_based_reproduction_rate": 0.9},
                "age_based_reproduction_rates",
                [0.9, 0.9, 0.9],
            ),
            (
                "reproduction",
                {"female_age_based_fertility": [0.5, 0.6, 0.7]},
                "female_age_based_fertility",
                [0.5, 0.6, 0.7],
            ),
        ],
    )
    def test_age_vocabulary_writes_exact_cells(
        self,
        method: str,
        kwargs: dict[str, object],
        field: str,
        expected: object,
    ):
        cfg = Configurator.from_species(_fresh_age_species()).age_structure(3, 1)
        getattr(cfg, method)(**kwargs)
        np.testing.assert_allclose(
            np.asarray(getattr(cfg.config, field)), np.asarray(expected)
        )

    @pytest.mark.parametrize(
        ("method", "kwargs", "field", "expected"),
        [
            # Discrete drafts normalize to 2 ages; the survival vocabulary
            # keeps working on the unified (2, 2) vectors.
            (
                "survival",
                {"female_age0_survival": 0.8},
                "age_based_survival_rates",
                [[0.8, 0.0], [1.0, 0.0]],
            ),
            (
                "survival",
                {"male_age0_survival": 0.6},
                "age_based_survival_rates",
                [[1.0, 0.0], [0.6, 0.0]],
            ),
            (
                "survival",
                {"female_age_based_survival": 0.7},
                "age_based_survival_rates",
                [[0.7, 0.7], [1.0, 0.0]],
            ),
            (
                "survival",
                {"male_age_based_survival": [0.4, 0.5]},
                "age_based_survival_rates",
                [[1.0, 0.0], [0.4, 0.5]],
            ),
            # Discrete mating / reproduction vocabulary.
            (
                "reproduction",
                {"female_adult_mating_rate": 0.9},
                "age_based_mating_rates",
                [[0.0, 0.9], [0.0, 1.0]],
            ),
            (
                "reproduction",
                {"male_adult_mating_rate": 0.7},
                "age_based_mating_rates",
                [[0.0, 1.0], [0.0, 0.7]],
            ),
        ],
    )
    def test_discrete_vocabulary_writes_exact_cells(
        self,
        discrete_species: nt.Species,
        method: str,
        kwargs: dict[str, object],
        field: str,
        expected: object,
    ):
        cfg = Configurator.for_discrete(discrete_species)
        getattr(cfg, method)(**kwargs)
        np.testing.assert_allclose(
            np.asarray(getattr(cfg.config, field)), np.asarray(expected)
        )

    def test_reproduction_rate_slot_cell_on_both_granularities(self):
        # ``reproduction_rate`` is a route-table name (slot into the adult
        # reproduction cell), exposed by dispatch rather than as a
        # reproduction() keyword.  The unified-vector cell must land
        # exactly, on both granularities.
        cfg = Configurator.from_species(_fresh_age_species()).age_structure(3, 1)
        live = dispatch(cfg.config, "reproduction_rate", 0.66)
        np.testing.assert_allclose(
            np.asarray(live.age_based_reproduction_rates),
            [0.0, 0.66, 1.0],
        )
        discrete = Configurator.for_discrete(_fresh_discrete_species())
        live_d = dispatch(discrete.config, "reproduction_rate", 0.5)
        np.testing.assert_allclose(
            np.asarray(live_d.age_based_reproduction_rates), [0.0, 0.5]
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"female_age_based_mating_rate": 0.8},
            {"male_age_based_mating_rate": 0.8},
            {"age_based_reproduction_rate": 0.9},
            {"female_age_based_fertility": 0.9},
        ],
    )
    def test_discrete_rejects_per_age_reproduction_names(
        self,
        discrete_species: nt.Species,
        kwargs: dict[str, object],
    ):
        # Negative axis-combo contract: per-age specs are meaningless on a
        # 2-age discrete draft, and the rejection must not write anything.
        cfg = Configurator.for_discrete(discrete_species)
        mating0 = np.asarray(cfg.config.age_based_mating_rates).copy()
        repro0 = np.asarray(cfg.config.age_based_reproduction_rates).copy()
        fert0 = np.asarray(cfg.config.female_age_based_fertility).copy()
        with pytest.raises(TypeError, match="discrete"):
            cfg.reproduction(**kwargs)
        np.testing.assert_array_equal(
            np.asarray(cfg.config.age_based_mating_rates), mating0
        )
        np.testing.assert_array_equal(
            np.asarray(cfg.config.age_based_reproduction_rates), repro0
        )
        np.testing.assert_array_equal(
            np.asarray(cfg.config.female_age_based_fertility), fert0
        )

    def test_build_time_and_runtime_vocabulary_identical_trajectories(
        self,
        discrete_species: nt.Species,
    ):
        # Setting the same demographic values at build time vs through
        # pop.update() must yield bit-identical deterministic dynamics.
        def demographics() -> dict[str, float]:
            return {
                "carrying_capacity": 400.0,
                "low_density_growth_rate": 2.0,
                "growth_mode": 2,
            }

        initial = {
            "female": {"WT|WT": 300, "WT|Var": 50},
            "male": {"WT|WT": 300, "Var|Var": 20},
        }
        pop_a = (
            nt.DiscreteGenerationPopulation.setup(
                discrete_species,
                stochastic=False,
            )
            .survival(female_age0_survival=0.8, male_age0_survival=0.6)
            .reproduction(eggs_per_female=6.0, sex_ratio=0.5)
            .competition(**demographics())
            .initial_state(individual_count=initial)
            .build()
        )
        pop_b = (
            nt.DiscreteGenerationPopulation.setup(
                discrete_species,
                stochastic=False,
            )
            .initial_state(individual_count=initial)
            .build()
        )
        pop_b.update().survival(
            female_age0_survival=0.8,
            male_age0_survival=0.6,
        )
        pop_b.update().reproduction(eggs_per_female=6.0, sex_ratio=0.5)
        pop_b.update().competition(**demographics())
        for _ in range(5):
            pop_a.run(n_steps=1)
            pop_b.run(n_steps=1)
            np.testing.assert_array_equal(
                np.asarray(pop_a.state.individual_count),
                np.asarray(pop_b.state.individual_count),
            )


# ── 6. route-table data integrity ─────────────────────────────────────────────


class TestRouteTableIntegrity:
    SEVEN_KINDS = {
        "scalar",
        "geno_tensor",
        "slot",
        "sex_row",
        "bool",
        "age_vec",
        "mode_enum",
    }
    EXPECTED_COUNTS = {
        "scalar": 14,
        "geno_tensor": 8,
        "slot": 7,
        "sex_row": 5,
        "bool": 4,
        "age_vec": 2,
        "mode_enum": 1,
    }
    EXPECTED_SENSITIVE = {
        "carrying_capacity",
        "eggs_per_female",
        "sex_ratio",
        "external_expected_eggs",
        "equilibrium_distribution",
    }

    def _unique_entries(self) -> list[object]:
        return list({e.name: e for e in ROUTES.values()}.values())

    def test_exactly_forty_one_rows_in_seven_shapes(self):
        entries = self._unique_entries()
        assert len(entries) == 41
        counts = Counter(e.kind for e in entries)  # type: ignore[attr-defined]  # entries are RouteEntry records
        assert dict(counts) == self.EXPECTED_COUNTS
        for entry in entries:
            assert entry.kind in self.SEVEN_KINDS  # type: ignore[attr-defined]
            assert entry.section in ("ecology", "genetics")  # type: ignore[attr-defined]
            assert entry.bounds[0] <= entry.bounds[1]  # type: ignore[attr-defined]

    def test_geno_tensor_rows_are_exactly_the_genetics_section(self):
        # The six fitness tensors are the genetics section; the two
        # initial-state tensors are genotype-indexed but ecological.
        geno = [
            e
            for e in self._unique_entries()
            if e.kind == "geno_tensor"  # type: ignore[attr-defined]
        ]
        genetics = [
            e.name  # type: ignore[attr-defined]
            for e in self._unique_entries()
            if e.section == "genetics"  # type: ignore[attr-defined]
        ]
        assert {e.name for e in geno} == {  # type: ignore[attr-defined]
            "initial_individual_count",
            "initial_sperm_storage",
            "viability",
            "fecundity",
            "sexual_selection",
            "zygote_viability",
            "female_ztype_compatibility",
            "male_ztype_compatibility",
        }
        assert set(genetics) == {
            "viability",
            "fecundity",
            "sexual_selection",
            "zygote_viability",
            "female_ztype_compatibility",
            "male_ztype_compatibility",
        }
        for entry in geno:
            if entry.name not in ("initial_individual_count", "initial_sperm_storage"):  # type: ignore[attr-defined]
                assert entry.section == "genetics"  # type: ignore[attr-defined]

    def test_sensitive_set_is_exactly_the_documented_five(self):
        sensitive = {
            e.name  # type: ignore[attr-defined]
            for e in self._unique_entries()
            if e.sensitive  # type: ignore[attr-defined]
        }
        assert sensitive == self.EXPECTED_SENSITIVE

    def test_every_alias_resolves_to_its_owning_entry(self):
        for entry in self._unique_entries():
            canonical = lookup(entry.name)  # type: ignore[attr-defined]
            for alias in entry.aliases:  # type: ignore[attr-defined]
                assert lookup(alias) is canonical

    def test_full_key_lookup_matches_short_name(self):
        assert lookup("competition.carrying_capacity") is lookup("carrying_capacity")

    @pytest.mark.parametrize(
        ("alias", "field", "value"),
        [
            ("old_juvenile_carrying_capacity", "carrying_capacity", 321.0),
            ("age_1_carrying_capacity", "carrying_capacity", 322.0),
            ("expected_eggs_per_female", "eggs_per_female", 41.0),
            ("juvenile_growth_mode", "juvenile_growth_mode", 1),
            ("relative_competition_factor", "competition_strength", 2.5),
        ],
    )
    def test_legacy_aliases_still_writable(
        self,
        alias: str,
        field: str,
        value: float,
    ):
        cfg = _age_draft()
        live = dispatch(cfg, alias, value)
        if field == "juvenile_growth_mode":
            assert int(live.juvenile_growth_mode) == 1
        elif field == "competition_strength":
            assert float(live.age_based_relative_competition_strength[1]) == 2.5
        else:
            assert float(getattr(live, field)) == value


# ── 7. ricker engine path (mode 4) ────────────────────────────────────────────


def _ricker_scaling(x: float, r: float) -> float:
    """g_ricker from rust/src/kernels/density_regulation.rs: r ** (1 - x)."""
    return r ** (1.0 - x)


def _manual_discrete_ricker_trajectory(
    pop: nt.DiscreteGenerationPopulation,
    n_ticks: int,
) -> list[NDArray[np.float64]]:
    """Hand-derive the deterministic discrete tick with ricker regulation.

    Replicates rust/src/kernels/discrete_generation.rs (mate -> fertilize -> survival ->
    aging) for the deterministic path, reading every driver from the
    population's own config so the only assumed fact under test is the
    regulation curve itself.
    """
    cfg = pop.config
    ss = np.asarray(cfg.sexual_selection_fitness, dtype=np.float64)
    ot = np.asarray(cfg.offspring_tensor, dtype=np.float64)
    fec = np.asarray(cfg.fecundity_fitness, dtype=np.float64)
    fec_f, fec_m = fec[0], fec[1]
    via = np.asarray(cfg.viability_fitness, dtype=np.float64)
    via_f, via_m = via[0, 0, :], via[1, 0, :]
    eggs = float(cfg.eggs_per_female)
    sex_ratio = float(cfg.sex_ratio)
    mating_f = float(cfg.age_based_mating_rates[0, 1])
    mating_m = float(cfg.age_based_mating_rates[1, 1])
    repr_rate = float(cfg.age_based_reproduction_rates[1])
    s_f = float(cfg.age_based_survival_rates[0, 0])
    s_m = float(cfg.age_based_survival_rates[1, 0])
    r = float(cfg.low_density_growth_rate)
    exp_comp = float(pop.params.expected_competition_strength)
    exp_surv = float(pop.params.expected_survival_rate)

    ind = np.asarray(pop.state.individual_count, dtype=np.float64).copy()
    trajectory: list[NDArray[np.float64]] = []
    for _ in range(n_ticks):
        adult_f = ind[0, 1, :].copy()
        adult_m = ind[1, 1, :].copy()
        # Mating probability: row-normalized selection x effective males.
        weights = ss * (adult_m * mating_m)[None, :]
        row_sums = weights.sum(axis=1, keepdims=True)
        probs = np.divide(
            weights,
            row_sums,
            out=np.zeros_like(weights),
            where=row_sums > 0.0,
        )
        pairs = (adult_f * mating_f)[:, None] * probs
        # Fertilization: pairs * p_reproduce * eggs * fec_f * fec_m,
        # distributed over offspring genotypes by the offspring tensor.
        coef = pairs * (repr_rate * eggs) * (fec_f[:, None] * fec_m[None, :])
        off = (coef[:, :, None] * ot).sum(axis=(0, 1))
        off_f = off * sex_ratio
        off_m = off * (1.0 - sex_ratio)
        # Density regulation: ricker at x = juveniles / C*, times s*.
        total = off_f.sum() + off_m.sum()
        ratio = total / exp_comp if exp_comp > 0.0 else 1.0
        scaling = _ricker_scaling(ratio, r) * exp_surv
        off_f = off_f * scaling
        off_m = off_m * scaling
        # Juvenile survival, then aging (age 0 -> age 1).
        ind[0, 1, :] = off_f * (s_f * via_f)
        ind[1, 1, :] = off_m * (s_m * via_m)
        ind[0, 0, :] = 0.0
        ind[1, 0, :] = 0.0
        trajectory.append(ind.copy())
    return trajectory


def _age_pop_rust(mode: str) -> nt.AgeStructuredPopulation:
    """An age-structured rust population seeded far above equilibrium.

    Juvenile survival must be positive (the default draft zeroes age-0
    survival), otherwise no juvenile cohort ever exists for the density
    kernel to regulate.
    """
    return (
        nt.AgeStructuredPopulation.setup(
            _fresh_age_species(),
            stochastic=False,
        )
        .age_structure(4, 2)
        .survival(female_age_based_survival=0.8, male_age_based_survival=0.8)
        .initial_state(
            individual_count={
                "female": {"A|A": 5000, "A|B": 2000},
                "male": {"A|A": 5000, "B|B": 2000},
            },
        )
        .competition(
            carrying_capacity=500.0,
            low_density_growth_rate=3.5,
            growth_mode=mode,
        )
        .reproduction(eggs_per_female=20.0, sex_ratio=0.5)
        .build()
    )


@RUST
class TestRickerEnginePath:
    def test_rust_discrete_ricker_matches_manual_recursion(self):
        # growth_mode="ricker" parses at the route layer to 4 and the Rust
        # kernel must run the exact Ricker recursion, tick by tick.
        pop = _discrete_pop("ricker")
        assert pop.params.growth_mode == 4
        manual = _manual_discrete_ricker_trajectory(pop, n_ticks=10)
        assert len(manual) == 10
        for tick in range(10):
            pop.run(n_steps=1)
            np.testing.assert_allclose(
                np.asarray(pop.state.individual_count),
                manual[tick],
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"ricker trajectory diverged at tick {tick}",
            )

    def test_rust_mode4_overcompensates_where_mode3_compensates(self):
        # Above equilibrium the ricker curve must under-retain relative to
        # beverton_holt (overcompensation): the trajectories must differ.
        pop_ricker = _discrete_pop("ricker")
        pop_bh = _discrete_pop("beverton_holt")
        traj_ricker: list[float] = []
        traj_bh: list[float] = []
        for _ in range(10):
            pop_ricker.run(n_steps=1)
            pop_bh.run(n_steps=1)
            traj_ricker.append(float(pop_ricker.state.individual_count.sum()))
            traj_bh.append(float(pop_bh.state.individual_count.sum()))
        assert traj_ricker[0] < traj_bh[0]  # overcompensation crash
        assert traj_ricker != traj_bh

    def test_rust_age_structured_mode4_differs_from_mode3(self):
        pop_ricker = _age_pop_rust("ricker")
        pop_bh = _age_pop_rust("beverton_holt")
        for _ in range(6):
            pop_ricker.run(n_steps=1)
            pop_bh.run(n_steps=1)
            traj_r = np.asarray(pop_ricker.state.individual_count)
            traj_b = np.asarray(pop_bh.state.individual_count)
            if not np.allclose(traj_r, traj_b):
                return
        pytest.fail("age-structured rust mode 4 is indistinguishable from mode 3")


# ── 8. HookConfigWriter ───────────────────────────────────────────────────────


class TestHookConfigWriterDirect:
    def test_touches_only_the_session_channel(self):
        session = RecordingSession()
        writer = HookConfigWriter(session)  # type: ignore[arg-type]  # structural fake of the runtime session protocol
        writer.apply({"carrying_capacity": 300.0, "eggs_per_female": 7.0})
        writer.apply({"growth_mode": 4.0}, mode="multiply")  # mode ignored
        writer.tensor_write("survival_rates", np.arange(6, dtype=np.float64))
        assert session.applied == [
            {"carrying_capacity": 300.0, "eggs_per_female": 7.0},
            {"growth_mode": 4.0},
        ]
        assert len(session.tensors) == 1
        field, values = session.tensors[0]
        assert field == "survival_rates"
        np.testing.assert_array_equal(values, np.arange(6, dtype=np.float64))
        # No refresh, no rebuild: the writer never touches session structure.
        assert session.other_calls == []

    def test_binds_no_draft_and_schedules_no_rebuild(self):
        pop = _age_pop()
        backend = pop._rust_lifecycle_backend  # noqa: SLF001
        writer = HookConfigWriter(backend)
        assert getattr(writer, "draft", None) is None
        old_snapshot = pop.config
        writer.apply({"carrying_capacity": 300.0})
        assert float(pop.config.carrying_capacity) == 300.0
        assert float(old_snapshot.carrying_capacity) == 500.0
        # No rebuild is scheduled: direct session writes are values only.
        assert pop._rust_needs_rebuild is False

    def test_direct_write_survives_into_the_next_run(self):
        """Session-only direct writes survive and drive the next run.

        HookConfigWriter (the in-hook path) bypasses the draft on
        purpose: it is the emergency push channel.  A run only flushes
        the draft at the boundary when an in-run write deferred through a
        writer — a bare session push leaves no deferral and is therefore
        not overwritten.
        """
        pop = _discrete_pop(0)
        session = pop._rust_lifecycle_backend._session  # noqa: SLF001
        writer = HookConfigWriter(pop._rust_lifecycle_backend)  # noqa: SLF001
        writer.apply({"carrying_capacity": 300.0})
        rates = np.full(4, 0.5)
        writer.tensor_write("survival_rates", rates)
        assert session.get_scalar("carrying_capacity") == 300.0
        pop.run(n_steps=1)
        assert session.get_scalar("carrying_capacity") == 300.0
        np.testing.assert_allclose(
            np.asarray(session.get_tensor("survival_rates")), rates
        )
        assert pop.state.individual_count.sum() > 0.0


# ── 9. pop.params read surface (TensorView + per-kind reads) ─────────────────


class TestParamsViewReadSurface:
    """Read-path coverage of the public ``pop.params`` surface.

    Every read must return an independent copy (or a Python scalar) and
    every unsupported access must fail with the documented error.
    """

    def test_tensor_view_metadata_and_repr(self):
        pop = _age_pop()
        view = pop.params.viability
        arr = np.asarray(pop.config.viability_fitness)
        assert view.shape == arr.shape
        assert view.dtype == arr.dtype
        assert "TensorView(shape=" in repr(view)
        assert str(arr.dtype) in repr(view)

    def test_tensor_view_array_conversion_with_dtype(self):
        pop = _age_pop()
        view = pop.params.viability
        narrowed = np.asarray(view, dtype=np.float32)
        assert narrowed.dtype == np.float32
        np.testing.assert_allclose(
            narrowed,
            np.asarray(pop.config.viability_fitness),
            rtol=1e-6,
        )

    def test_tensor_view_plain_index_reads_return_copies(self):
        pop = _age_pop()
        arr = np.asarray(pop.config.viability_fitness)
        # Non-tuple index (whole first axis slice).
        row = pop.params.viability[0]
        assert isinstance(row, np.ndarray)
        np.testing.assert_array_equal(row, arr[0])
        row[:] = 99.0
        np.testing.assert_array_equal(np.asarray(pop.config.viability_fitness), arr)
        # Tuple index whose last element is not a pattern string.
        cell = pop.params.viability[0, 1, 0]
        assert cell == float(arr[0, 1, 0])

    def test_scalar_reads_per_python_type(self):
        pop = _age_pop()
        # 0-d ndarray float scalar.
        assert pop.params.carrying_capacity == float(pop.config.carrying_capacity)
        # int dtype scalar (mode_enum row).
        assert pop.params.growth_mode == int(pop.config.juvenile_growth_mode)
        # age_vec row read returns a copy.
        vec0 = np.asarray(pop.config.age_based_reproduction_rates).copy()
        vec = pop.params.age_based_reproduction_rate
        np.testing.assert_array_equal(vec, vec0)
        vec[:] = 0.0
        np.testing.assert_array_equal(
            np.asarray(pop.config.age_based_reproduction_rates), vec0
        )

    def test_sex_row_reads_return_copies(self):
        pop = _age_pop()
        draft0 = np.asarray(pop.config.age_based_survival_rates).copy()
        row = pop.params.female_age_based_survival
        np.testing.assert_array_equal(row, draft0[0])
        row[:] = 0.0
        np.testing.assert_array_equal(
            np.asarray(pop.config.age_based_survival_rates), draft0
        )

    def test_equilibrium_declaration_read_none_and_declared(self):
        pop = _age_pop()
        # Derive mode reads back as None (nothing is shared or invented).
        assert pop.params.equilibrium_distribution is None
        declared = np.array(
            [[10.0, 5.0, 1.0, 0.5], [10.0, 5.0, 1.0, 0.5]],
        )
        pop.update().competition(equilibrium_distribution=declared)
        table = pop.params.equilibrium_distribution
        assert table is not None
        np.testing.assert_array_equal(
            table, np.asarray(pop.config.equilibrium_individual_distribution)
        )
        table[:] = 0.0
        np.testing.assert_array_equal(
            np.asarray(pop.config.equilibrium_individual_distribution), declared
        )

    def test_external_eggs_scalar_reads_none_and_declared(self):
        pop = _age_pop()
        # Undeclared override reads back as None (derive mode).
        assert pop.params.external_expected_eggs is None
        # Declaring a Champer target persists the egg override as a plain
        # Python float, and the read converts it exactly.
        pop.update().competition(expected_num_new_adult_females=50.0)
        declared = pop.config.external_expected_eggs
        assert isinstance(declared, float)
        assert pop.params.external_expected_eggs == declared

    def test_dir_lists_routes_and_tensors(self):
        pop = _age_pop()
        names = dir(pop.params)
        assert "carrying_capacity" in names
        assert "growth_mode" in names
        assert "viability_fitness" in names
        assert "survival_rates" in names

    def test_contract_to_draft_field_default_is_identity(self):
        from natal.frontend.configurator._writers import (
            contract_to_draft_field,
        )

        assert contract_to_draft_field("survival_rates") == ("age_based_survival_rates")
        assert contract_to_draft_field("carrying_capacity") == ("carrying_capacity")


# ── 10. negative contracts ───────────────────────────────────────────


def _registry_probe_entry(
    config_field: str,
    *,
    name: str,
) -> object:
    """A minimal descriptor for route-table validation tests."""
    from natal.frontend.utils.parameters import ParamDescriptor

    return ParamDescriptor(
        domain="competition",
        name=name,
        method="competition",
        kind="scalar",
        section="ecology",
        config_field=config_field,
        config_path=(),
        dtype=float,
        bounds=(0.0, 10.0),
        sensitive=False,
    )


class TestNegativeContractsSlice3:
    def test_lookup_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown parameter"):
            lookup("definitely_not_a_parameter")

    def test_spatial_row_contract_field_falls_back_to_name(self):
        assert lookup("migration_rate").contract_field == "migration_rate"

    def test_bool_kind_rejects_non_bool_value(self):
        with pytest.raises(TypeError, match="requires a bool"):
            dispatch(_age_draft(), "fixed_egg_count", "yes")

    def test_sex_row_bounds_rejected_on_whole_table_and_row(self):
        cfg = _age_draft()
        # Whole-table declaration with an out-of-bounds cell.
        with pytest.raises(ValueError, match="requires all values in"):
            dispatch(
                cfg,
                "equilibrium_distribution",
                np.array([[10.0, 5.0, 1.0], [10.0, 5.0, 1e13]]),
            )
        # Single per-sex row with an out-of-bounds element.
        with pytest.raises(ValueError, match="requires all values in"):
            dispatch(cfg, "female_age_based_survival", [0.5, 1.5, 0.5])

    def test_geno_tensor_pattern_dict_rejected_outside_writer_patches(self):
        # A Mapping for a geno_tensor entry is only meaningful inside a
        # writer batch (pattern patch); a raw route write must reject it.
        with pytest.raises(TypeError, match="pattern patches"):
            plan_write(_age_draft(), lookup("fecundity"), {"A|A": 0.5})

    def test_bool_row_with_config_path_rejected_at_build(self):
        from natal.frontend.configurator import _routes
        from natal.frontend.utils.parameters import ParamDescriptor

        entry = ParamDescriptor(
            domain="setup",
            name="probe",
            method="setup",
            kind="bool",
            section="ecology",
            config_field="fixed_egg_count",
            config_path=(1,),
            dtype=bool,
            bounds=(0.0, 1.0),
            sensitive=False,
        )
        with pytest.raises(ValueError, match="bool rows must have"):
            _routes._build_routes({"setup.probe": entry})

    def test_route_level_name_collision_rejected_at_build(self):
        from natal.frontend.configurator import _routes

        # Both rows carry the same user-facing name: the second must fail
        # table construction instead of silently shadowing the first.
        first = _registry_probe_entry("carrying_capacity", name="probe")
        second = _registry_probe_entry("eggs_per_female", name="probe")
        with pytest.raises(ValueError, match="collides"):
            _routes._build_routes(
                {
                    "competition.probe": first,
                    "reproduction.probe": second,
                }
            )

    def test_route_entry_is_frozen(self):
        entry = lookup("carrying_capacity")
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.name = "mutated"  # type: ignore[misc]  # frozen contract check

    def test_resolved_write_is_frozen(self):
        plan = plan_write(_age_draft(), lookup("carrying_capacity"), 10.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.scalar = 0.0  # type: ignore[misc]  # frozen contract check

    def test_replace_field_classification(self):
        for field in (
            "stochastic",
            "continuous_sampling",
            "fixed_egg_count",
            "has_sex_chromosomes",
            "external_expected_eggs",
            "equilibrium_individual_distribution",
            "initial_individual_count",
            "initial_sperm_storage",
        ):
            assert is_replace_field(field)
        for field in (
            "carrying_capacity",
            "eggs_per_female",
            "sex_ratio",
            "sperm_displacement_rate",
            "low_density_growth_rate",
            "juvenile_growth_mode",
        ):
            assert is_replace_field(field)
        for field in ("age_based_survival_rates",):
            assert not is_replace_field(field)

    def test_pattern_patch_without_species_context_rejected(self):
        writer = DraftWriter(_age_draft())
        with pytest.raises(RuntimeError, match="species context"):
            writer.apply({"viability": {"A|A": 0.5}})

    def test_pattern_string_outside_last_axis_rejected(self):
        pop = _age_pop()
        with pytest.raises(TypeError, match="last axis"):
            _ = pop.params.viability_fitness["A|A", 1, 0]

    def test_plan_write_tensor_payload_size_guard(self):
        # plan_write validates the whole tensor against the live shape
        # before any commit buffer exists: a wrong-shape array plans
        # nothing and leaves the draft bit-identical.
        cfg = _age_draft()
        before = np.asarray(cfg.fecundity_fitness).copy()
        with pytest.raises(ValueError, match="requires an array of shape"):
            plan_write(cfg, lookup("fecundity"), np.ones((5, 5)))
        np.testing.assert_array_equal(np.asarray(cfg.fecundity_fitness), before)
        # A valid plan commits the exact payload afterwards.
        fresh = np.full_like(np.asarray(cfg.fecundity_fitness), 0.25)
        plan = plan_write(cfg, lookup("fecundity"), fresh)
        commit_write(cfg, plan)
        np.testing.assert_array_equal(np.asarray(cfg.fecundity_fitness), fresh)


class TestRickerCrossBackend:
    """growth_mode=4 (ricker) must be real Ricker on every backend.

    Regression guard: the Python reference paths previously fell into the
    Beverton-Holt `else` arm (age + discrete standard) or skipped regulation
    entirely (WF fused tick), while the Rust kernel ran true Ricker.
    """

    def _reference_trajectory(
        self, species_name: str, mode: int, ticks: int
    ) -> list[float]:
        import natal as nt

        sp = nt.Species.from_dict(
            name=species_name,
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(sp, stochastic=False)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0.0, 4500.0]},
                    "male": {"A|A": [0.0, 4500.0]},
                }
            )
            .reproduction(eggs_per_female=12.0)
            .competition(
                juvenile_growth_mode=mode,
                carrying_capacity=900.0,
                low_density_growth_rate=3.5,
            )
            .build()
        )
        pop.run(ticks, record_every=0)
        state = pop.state
        return [float(state.individual_count.sum())]

    def test_reference_discrete_ricker_diverges_from_beverton_holt(self) -> None:
        """mode 4 and mode 3 must produce different dynamics (not silent BH)."""
        ricker_total = self._reference_trajectory("slice3_ricker_a", 4, 3)
        bh_total = self._reference_trajectory("slice3_ricker_b", 3, 3)
        # Overcompensation crashes harder than the hyperbolic curve.
        assert ricker_total[-1] < bh_total[-1] * 0.5, (
            f"mode 4 ({ricker_total[-1]}) should crash much harder than "
            f"mode 3 ({bh_total[-1]}) — silent-BH fallback regression"
        )

    def test_reference_wf_ricker_regulates(self) -> None:
        """WF fused tick must apply ricker regulation, not skip it."""
        regulated = self._reference_trajectory("slice3_ricker_c", 4, 3)
        unregulated = self._reference_trajectory("slice3_ricker_d", 0, 3)
        # No-regulation grows ~r^3 (population multiplies every generation);
        # regulated stays bounded.
        assert regulated[-1] < unregulated[-1] * 0.5, (
            f"WF mode 4 ({regulated[-1]}) looks unregulated vs mode 0 "
            f"({unregulated[-1]})"
        )

    def _age_trajectory(self, species_name: str, mode: int) -> list[float]:
        import natal as nt

        sp = nt.Species.from_dict(
            name=species_name,
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )
        builder = (
            nt.AgeStructuredPopulation.setup(sp, stochastic=False)
            .initial_state(
                individual_count={
                    "female": {"A|A": [50.0, 4500.0]},
                    "male": {"A|A": [50.0, 4500.0]},
                }
            )
            .reproduction(eggs_per_female=12.0)
            .survival(female_age_based_survival=0.9, male_age_based_survival=0.8)
            .competition(
                juvenile_growth_mode=mode,
                carrying_capacity=900.0,
                low_density_growth_rate=3.5,
            )
        )
        pop = builder.build()
        pop.run(3, record_every=0)
        return [float(pop.state.individual_count.sum())]

    def test_age_ricker_diverges_from_beverton_holt(self) -> None:
        """Age path: mode 4 must differ from mode 3 in the engine.

        Regression guard for the silent-BH fallback in the age-structured
        lifecycle scaling dispatch.
        """
        ricker = self._age_trajectory("slice3_age_rick", 4)
        bh = self._age_trajectory("slice3_age_bh", 3)
        assert ricker[-1] != bh[-1], (
            f"mode 4 trajectory identical to mode 3 "
            f"({ricker[-1]}) — silent-BH fallback regression"
        )

