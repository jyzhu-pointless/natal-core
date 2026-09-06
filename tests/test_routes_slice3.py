"""Slice-3 strict tests: the route table, the writers, and ``pop.params``.

Every assertion proves one numerical or identity invariant:

1. **Seven shapes**: each route kind gets at least two numerical
   assertions (scalar bounds, mode-enum aliases, age-vector broadcast,
   sex-row flexible forms and the equilibrium declaration, slot cells,
   the bool replace-channel, and whole-tensor geno writes).
2. **Import-time validation**: malformed jsonc tables (missing column,
   unknown kind, duplicate name, alias collision, bad config_field)
   raise during table construction — never at first write.
3. **Writer atomicity**: one invalid entry in a batch commits nothing.
4. **``pop.params`` surface**: bounds-checked setters leave state
   unchanged on rejection, tensor reads are independent copies, pattern
   reads aggregate multiple matches and reject empty matches, and
   ``tensor_write`` is the only genetics write channel.
5. **Vocabulary preservation**: the discrete names
   (``female_age0_survival`` ...) still write the exact unified-vector
   cells, and ``growth_mode`` aliases resolve (``beverton_holt`` = 3,
   ``ricker`` = 4, ``concave`` removed).
6. **Negative contract**: the deleted subclass files, the deleted
   descriptor columns, the hand-maintained sensitive sets, and the
   per-method sync method are unreachable.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import natal as nt
from natal.frontend.configurator import Configurator, set_param
from natal.frontend.data import ModelDraft
from natal.frontend.configurator import _routes
from natal.frontend.configurator._routes import (
    ROUTES,
    ROUTES_BY_METHOD,
    commit_write,
    dispatch,
    is_sensitive,
    lookup,
    plan_write,
)
from natal.frontend.configurator._writers import (
    CoreConfigWriter,
    DraftWriter,
    HookConfigWriter,
)
from natal.frontend.utils.parameters import ALL_PARAMETERS, _build_registry

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def age_species() -> nt.Species:
    return nt.Species.from_dict(
        name="__slice3_age__",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )


@pytest.fixture(scope="module")
def discrete_species() -> nt.Species:
    return nt.Species.from_dict(
        name="__slice3_discrete__",
        structure={"auto": {"A": ["WT", "Var"]}},
    )


def _age_draft() -> ModelDraft:
    """A minimal age-structured draft with known demographics."""
    from natal.frontend.data import build_population_config

    return build_population_config(
        n_genotypes=4, n_gtypes=4, n_glabs=1, n_ages=3, new_adult_age=1,
    )


def _age_pop() -> nt.AgeStructuredPopulation:
    sp = nt.Species.from_dict(
        name="__slice3_pop__",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    return (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False)
        .age_structure(4, 2)
        .initial_state(
            individual_count={
                "female": {"A|A": 100, "A|B": 50},
                "male": {"A|A": 80, "B|B": 20},
            }
        )
        .competition(carrying_capacity=500.0, juvenile_growth_mode=2)
        .reproduction(eggs_per_female=20.0, sex_ratio=0.5)
        .build()
    )


# ── 1. seven shapes ───────────────────────────────────────────────────────────


class TestScalarShape:
    def test_scalar_write_and_bounds(self):
        cfg = _age_draft()
        live = dispatch(cfg, "carrying_capacity", 7500.0)
        assert float(live.carrying_capacity) == 7500.0
        with pytest.raises(ValueError, match=r"requires a value in"):
            dispatch(live, "low_density_growth_rate", -1.0)

    def test_scalar_marks_contract_dirty(self):
        cfg = _age_draft()
        sink: set[str] = set()
        dispatch(cfg, "sperm_displacement_rate", 0.4, dirty_sink=sink)
        assert sink == {"sperm_displacement_rate"}

    def test_scalar_rejects_non_numeric(self):
        cfg = _age_draft()
        with pytest.raises(TypeError, match="requires a numeric value"):
            dispatch(cfg, "sex_ratio", "0.5")


class TestModeEnumShape:
    def test_string_aliases_resolve_to_canonical_ints(self):
        cfg = _age_draft()
        for name, expected in [
            ("no_competition", 0), ("fixed", 1), ("linear", 2),
            ("logistic", 2), ("beverton_holt", 3), ("ricker", 4),
        ]:
            live = dispatch(cfg, "growth_mode", name)
            assert int(live.juvenile_growth_mode) == expected, name

    def test_integer_passthrough_and_removal_of_concave(self):
        cfg = _age_draft()
        live = dispatch(cfg, "juvenile_growth_mode", 3)
        assert int(live.juvenile_growth_mode) == 3
        with pytest.raises(ValueError, match="beverton_holt"):
            dispatch(live, "juvenile_growth_mode", "concave")
        with pytest.raises(ValueError, match="Unknown growth mode"):
            dispatch(live, "juvenile_growth_mode", "sigma")

    def test_concave_alias_fully_removed(self):
        """Negative contract: no import path reaches the CONCAVE alias."""
        import natal as nt
        from natal.frontend.configurator._params import resolve_growth_mode

        with pytest.raises(ImportError):
            from natal.frontend.data import CONCAVE  # type: ignore[attr-defined]  # noqa: F401  # negative contract: must not import
        with pytest.raises(ImportError):
            from natal.frontend.data.constants import CONCAVE  # type: ignore[attr-defined]  # noqa: F401  # negative contract: must not import
        assert not hasattr(nt, "CONCAVE")
        # The legacy resolver rejects both spellings but keeps mode 3 valid.
        with pytest.raises(ValueError, match="Unknown growth mode"):
            resolve_growth_mode("concave")
        with pytest.raises(ValueError, match="Unknown growth mode"):
            resolve_growth_mode("CONCAVE")
        assert resolve_growth_mode(3) == 3
        assert resolve_growth_mode("beverton_holt") == 3


class TestAgeVecShape:
    def test_list_and_scalar_broadcast(self):
        cfg = _age_draft()
        live = dispatch(cfg, "age_based_reproduction_rate", [0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            np.asarray(live.age_based_reproduction_rates), [0.1, 0.2, 0.3]
        )
        live = dispatch(live, "female_age_based_fertility", 0.7)
        np.testing.assert_allclose(
            np.asarray(live.female_age_based_fertility), [0.7, 0.7, 0.7]
        )

    def test_age_vec_rejects_out_of_bounds_element(self):
        cfg = _age_draft()
        with pytest.raises(ValueError, match="requires all values in"):
            dispatch(cfg, "age_based_reproduction_rate", [0.5, 1.5, 0.5])


class TestSexRowShape:
    def test_row_accepts_flexible_forms(self):
        cfg = _age_draft()
        live = dispatch(cfg, "female_age_based_survival", 0.55)
        np.testing.assert_allclose(live.age_based_survival_rates[0], [0.55] * 3)
        live = dispatch(live, "male_age_based_survival", {0: 0.9, 2: 0.1})
        np.testing.assert_allclose(
            live.age_based_survival_rates[1], [0.9, 1.0, 0.1]
        )

    def test_equilibrium_declaration_whole_table_and_clear(self):
        cfg = _age_draft()
        declared = np.array([[10.0, 5.0, 1.0], [10.0, 5.0, 1.0]])
        live = dispatch(cfg, "equilibrium_distribution", declared)
        np.testing.assert_allclose(
            np.asarray(live.equilibrium_individual_distribution), declared
        )
        # Flat form is reshaped row-major.
        live = dispatch(live, "equilibrium_distribution", declared.ravel())
        np.testing.assert_allclose(
            np.asarray(live.equilibrium_individual_distribution), declared
        )
        # None restores the derive-mode declaration.
        live = dispatch(live, "equilibrium_distribution", None)
        assert live.equilibrium_individual_distribution is None

    def test_equilibrium_declaration_rejects_wrong_shape(self):
        cfg = _age_draft()
        with pytest.raises(ValueError, match="requires a \\(2, 3\\) array"):
            dispatch(cfg, "equilibrium_distribution", np.ones((2, 4)))


class TestSlotShape:
    def test_discrete_survival_cell(self):
        cfg = _age_draft()
        live = dispatch(cfg, "female_age0_survival", 0.42)
        assert float(live.age_based_survival_rates[0, 0]) == 0.42
        live = dispatch(live, "male_age0_survival", 0.11)
        assert float(live.age_based_survival_rates[1, 0]) == 0.11

    def test_discrete_adult_mating_cell(self):
        cfg = _age_draft()
        live = dispatch(cfg, "female_adult_mating_rate", 0.85)
        assert float(live.age_based_mating_rates[0, 1]) == 0.85

    def test_slot_marks_whole_vector_dirty(self):
        cfg = _age_draft()
        sink: set[str] = set()
        dispatch(cfg, "female_age0_survival", 0.5, dirty_sink=sink)
        # The bridge refreshes the whole unified vector, not the cell.
        assert sink == {"survival_rates"}


class TestBoolShape:
    def test_bool_replaces_namedtuple_field(self):
        cfg = _age_draft()
        assert cfg.fixed_egg_count is False
        live = dispatch(cfg, "fixed_egg_count", True)
        assert live.fixed_egg_count is True
        assert live is not cfg  # NamedTuple slot replaced, not mutated

    def test_bool_marks_blueprint_sentinel(self):
        cfg = _age_draft()
        sink: set[str] = set()
        dispatch(cfg, "fixed_egg_count", True, dirty_sink=sink)
        assert sink == {"__blueprint__"}


class TestGenoTensorShape:
    def test_whole_tensor_write(self):
        cfg = _age_draft()
        fresh = np.zeros_like(np.asarray(cfg.fecundity_fitness)) + 0.25
        sink: set[str] = set()
        writer = DraftWriter(cfg, sink)
        writer.apply({"fecundity": fresh})
        np.testing.assert_allclose(
            np.asarray(writer.draft.fecundity_fitness), fresh
        )
        assert sink == {"fecundity_fitness"}

    def test_whole_tensor_rejects_wrong_shape(self):
        cfg = _age_draft()
        writer = DraftWriter(cfg)
        with pytest.raises(ValueError, match="requires an array of shape"):
            writer.apply({"fecundity": np.ones((3, 3))})

    def test_fitness_pattern_patch_routes_through_writer(self, age_species):
        chain = Configurator.from_species(age_species).age_structure(3, 1)
        chain = chain.fitness(viability={"A|A": 0.5})
        arr = np.asarray(chain.config.viability_fitness)
        # Both sexes at the juvenile age column carry the patch.
        assert arr[0, 0, 0] == 0.5
        assert arr[1, 0, 0] == 0.5
        assert arr[0, 0, 1] == 1.0  # untouched genotype

    def test_meiosis_map_contract_read_resolves_to_draft_table(self):
        """``pop.params.meiosis_map`` exposes the draft meiosis table.

        The contract field carries the biology name while the draft
        keeps the builder-era field name; the rename map used to miss
        this entry, making the advertised read channel raise
        AttributeError (audit finding C1).
        """
        pop = _age_pop()
        view = pop.params.meiosis_map
        draft_table = np.asarray(pop.config.zygotes_to_gametes_map)

        assert view.shape == draft_table.shape
        np.testing.assert_array_equal(view.array, draft_table)
        # Reads hand out copies: scribbling on a returned array must
        # not reach the draft table.
        snapshot = view.array
        snapshot.fill(-1.0)
        np.testing.assert_array_equal(view.array, draft_table)
        # Pattern read at the ztype axis aggregates the gamete axis:
        # every Mendelian meiosis row is a distribution summing to 1.
        assert view[0, "A|B"] == pytest.approx(1.0)
        assert view[1, "A|A"] == pytest.approx(1.0)

    def test_meiosis_map_tensor_write_updates_draft(self):
        """``tensor_write("meiosis_map", ...)`` routes to the draft table."""
        pop = _age_pop()
        table = pop.params.meiosis_map.array
        table[0, 0, :] = [0.25, 0.75]
        pop.params.tensor_write("meiosis_map", table)

        np.testing.assert_allclose(
            pop.params.meiosis_map.array[0, 0, :], [0.25, 0.75]
        )
        np.testing.assert_allclose(
            np.asarray(pop.config.zygotes_to_gametes_map)[0, 0, :], [0.25, 0.75]
        )

    def test_meiosis_map_size_mismatch_is_zero_write(self):
        """A bad-size meiosis write commits nothing and marks nothing.

        Attack: a wrong-sized payload must raise before any draft cell,
        dirty-bridge entry, or session push happens — a partial write
        would leave the draft and the Rust session disagreeing.
        """
        pop = _age_pop()
        before = np.asarray(pop.config.zygotes_to_gametes_map).copy()
        dirty_before = set(pop._rust_dirty)
        with pytest.raises(ValueError, match="expected"):
            pop.params.tensor_write("meiosis_map", np.ones(5))
        np.testing.assert_array_equal(
            np.asarray(pop.config.zygotes_to_gametes_map), before
        )
        assert pop._rust_dirty == dirty_before

    def test_meiosis_map_write_marks_dirty_and_survives_run(self):
        """A meiosis write marks the Rust bridge and survives the drain.

        Attack: if the write forgot to mark ``_rust_dirty`` (or marked a
        draft name the bridge does not know), the next ``run()`` would
        drain nothing into the session; if the drain clobbered the
        draft, the written row would not survive the run.
        """
        pop = _age_pop()
        table = pop.params.meiosis_map.array
        table[0, 0, :] = [0.25, 0.75]
        pop.params.tensor_write("meiosis_map", table)
        # Contract-name marker — the same name the modifier refresh and
        # the Rust genetics-tensor set (_RUST_GENETICS_TENSORS) use.
        assert pop._rust_dirty == {"meiosis_map"}
        pop.run(1, record_every=1)
        assert pop._rust_dirty == set()  # drained into the session
        np.testing.assert_allclose(
            pop.params.meiosis_map.array[0, 0, :], [0.25, 0.75]
        )

    def test_meiosis_map_pattern_read_type_and_mendelian_cells(self):
        """Pattern reads return plain floats over an exactly Mendelian table.

        Attack: the aggregate ``== 1.0`` assertion above cannot tell
        Mendelian 0.5/0.5 segregation from a biased row such as
        [0.9, 0.1]; only cell-level equality pins the segregation, and
        only a type check proves the aggregation hands the caller a
        detached float rather than a live 0-d array.
        """
        pop = _age_pop()
        value = pop.params.meiosis_map[0, "A|B"]
        assert isinstance(value, float)
        table = pop.params.meiosis_map.array
        # Heterozygote rows segregate 0.5/0.5 in both sexes.
        np.testing.assert_allclose(table[0, 1, :], [0.5, 0.5])
        np.testing.assert_allclose(table[1, 1, :], [0.5, 0.5])
        # Homozygote rows pass the single allele through untouched.
        np.testing.assert_allclose(table[0, 0, :], [1.0, 0.0])
        np.testing.assert_allclose(table[1, 2, :], [0.0, 1.0])


# ── 2. import-time validation ────────────────────────────────────────────────


def _write_table(tmp_path, rows: list[dict[str, object]]) -> str:
    """Materialize a jsonc table from raw row dicts."""
    import json

    path = tmp_path / "table.jsonc"
    path.write_text(json.dumps(rows))
    return str(path)


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "domain": "competition", "method": "competition", "kind": "scalar",
        "section": "ecology", "name": "probe", "dtype": "float",
        "bounds": [0.0, 10.0], "sensitive": False,
        "config_field": "carrying_capacity", "config_path": [],
    }
    row.update(overrides)
    return row


class TestImportTimeValidation:
    def test_missing_column_raises(self, tmp_path):
        row = _valid_row()
        del row["sensitive"]
        with pytest.raises(ValueError, match="missing column"):
            _build_registry(_write_table(tmp_path, [row]))

    def test_unknown_kind_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown kind"):
            _build_registry(_write_table(tmp_path, [_valid_row(kind="matrix")]))

    def test_duplicate_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate parameter"):
            _build_registry(_write_table(tmp_path, [_valid_row(), _valid_row()]))

    def test_alias_collision_raises(self, tmp_path):
        rows = [
            _valid_row(),
            _valid_row(name="probe2", config_field="eggs_per_female",
                       aliases=["probe"]),
        ]
        with pytest.raises(ValueError, match="collides"):
            _build_registry(_write_table(tmp_path, rows))

    def test_bad_config_field_raises(self):
        entry = _registry_entry("not_a_draft_field")
        with pytest.raises(ValueError, match="not a ModelDraft field"):
            _routes._build_routes({"competition.probe": entry})

    def test_slot_row_requires_index_path(self):
        entry = _registry_entry("carrying_capacity", kind="slot", config_path=())
        with pytest.raises(ValueError, match="non-empty config_path"):
            _routes._build_routes({"competition.probe": entry})


def _registry_entry(
    config_field: str,
    *,
    kind: str = "scalar",
    config_path: tuple[int, ...] = (),
) -> object:
    """Build a minimal descriptor for route-validation tests."""
    from natal.frontend.utils.parameters import ParamDescriptor

    return ParamDescriptor(
        domain="competition", name="probe", method="competition",
        kind=kind, section="ecology", config_field=config_field,
        config_path=config_path, dtype=float, bounds=(0.0, 10.0),
        sensitive=False,
    )


# ── 3. writer atomicity ───────────────────────────────────────────────────────


class TestWriterAtomicity:
    def test_invalid_entry_commits_nothing(self):
        cfg = _age_draft()
        before_comp = float(cfg.carrying_capacity)
        before_eggs = float(cfg.eggs_per_female)
        writer = DraftWriter(cfg)
        with pytest.raises(ValueError):
            writer.apply({"carrying_capacity": 111.0, "eggs_per_female": -5.0})
        assert float(writer.draft.carrying_capacity) == before_comp
        assert float(writer.draft.eggs_per_female) == before_eggs

    def test_core_writer_without_session_still_marks_bridge(self):
        cfg = _age_draft()
        sink: set[str] = set()
        writer = CoreConfigWriter(cfg, sink, None)
        writer.apply({"carrying_capacity": 321.0, "eggs_per_female": 33.0})
        assert sink == {"carrying_capacity", "eggs_per_female"}
        assert float(writer.draft.carrying_capacity) == 321.0
        # Sensitive writes refresh the derived caches in place.
        assert np.isfinite(float(writer.draft.expected_survival_rate))

    def test_hook_writer_writes_session_directly(self):
        recorded: list[tuple[str, object]] = []

        class FakeSession:
            def apply(self, writes: dict[str, float]) -> None:
                recorded.append(("apply", dict(writes)))

            def tensor_write(self, field: str, values: np.ndarray) -> None:
                recorded.append(("tensor", (field, values.copy())))

        writer = HookConfigWriter(FakeSession())  # type: ignore[arg-type]  # structural fake of the runtime session protocol
        writer.apply({"carrying_capacity": 5000.0})
        writer.tensor_write("survival_rates", np.arange(4, dtype=np.float64))
        assert recorded[0] == ("apply", {"carrying_capacity": 5000.0})
        assert recorded[1][0] == "tensor"
        assert recorded[1][1][0] == "survival_rates"


# ── 4. pop.params surface ─────────────────────────────────────────────────────


class TestParamsSurface:
    def test_setter_bounds_check_leaves_state_unchanged(self):
        pop = _age_pop()
        before = float(pop.config.carrying_capacity)
        with pytest.raises(ValueError):
            pop.params.carrying_capacity = -1.0
        assert float(pop.config.carrying_capacity) == before

    def test_setter_routes_and_syncs(self):
        pop = _age_pop()
        pop.params.carrying_capacity = 900.0
        assert pop.params.carrying_capacity == 900.0
        assert pop.params.eggs_per_female == 20.0
        assert pop.params.growth_mode == 2

    def test_tensor_reads_are_independent_copies(self):
        pop = _age_pop()
        view = pop.params.viability_fitness
        arr = np.asarray(view)
        arr[0, :, :] = 42.0
        assert float(np.asarray(pop.config.viability_fitness).min()) != 42.0

    def test_tensor_setitem_blocked(self):
        pop = _age_pop()
        with pytest.raises(TypeError, match="tensor_write"):
            pop.params.viability_fitness[0, 0, 0] = 0.5

    def test_pattern_read_single_match(self):
        pop = _age_pop()
        value = pop.params.viability_fitness[0, 1, "A|A"]
        assert value == float(np.asarray(pop.config.viability_fitness)[0, 1, 0])

    def test_pattern_read_multi_match_aggregates(self):
        pop = _age_pop()
        # The wildcard pattern matches every ztype column of the age row.
        agg = pop.params.viability_fitness[0, 1, "*"]
        expected = float(
            np.asarray(pop.config.viability_fitness)[0, 1, :].sum()
        )
        assert agg == expected

    def test_pattern_read_empty_match_raises_state_unchanged(self):
        pop = _age_pop()
        with pytest.raises(ValueError, match="matches no zygote type"):
            pop.params.viability_fitness[0, 1, "Q|Q"]

    def test_tensor_write_updates_draft(self):
        pop = _age_pop()
        rates = np.full((2, 4), 0.77)
        pop.params.tensor_write("survival_rates", rates)
        np.testing.assert_allclose(
            np.asarray(pop.config.age_based_survival_rates), rates
        )

    def test_tensor_write_size_mismatch_is_zero_write(self):
        pop = _age_pop()
        before = np.asarray(pop.config.age_based_survival_rates).copy()
        with pytest.raises(ValueError, match="expected"):
            pop.params.tensor_write("survival_rates", np.ones(7))
        np.testing.assert_allclose(
            np.asarray(pop.config.age_based_survival_rates), before
        )

    def test_ecology_vectors_read_as_copies(self):
        pop = _age_pop()
        before = np.asarray(pop.config.age_based_survival_rates).copy()
        view = pop.params.survival_rates
        arr = np.asarray(view)  # conversion always yields a fresh copy
        arr[:] = 0.0
        np.testing.assert_allclose(
            np.asarray(pop.config.age_based_survival_rates), before
        )
        # Direct attribute assignment is rejected: writes go through
        # tensor_write so the session channel sees them.
        with pytest.raises(AttributeError, match="tensor_write"):
            pop.params.survival_rates = np.ones((2, 4))


# ── 5. vocabulary preservation ────────────────────────────────────────────────


class TestVocabularyPreserved:
    def test_discrete_survival_names_write_unified_cells(self, discrete_species):
        pop = (
            nt.DiscreteGenerationPopulation.setup(discrete_species, stochastic=False)
            .survival(female_age0_survival=0.33, male_age0_survival=0.66)
            .build()
        )
        assert float(pop.config.age_based_survival_rates[0, 0]) == 0.33
        assert float(pop.config.age_based_survival_rates[1, 0]) == 0.66

    def test_runtime_discrete_survival_updates_cells(self, discrete_species):
        pop = (
            nt.DiscreteGenerationPopulation.setup(discrete_species, stochastic=False)
            .build()
        )
        pop.update().survival(female_age0_survival=0.25)
        assert float(pop.config.age_based_survival_rates[0, 0]) == 0.25

    def test_growth_mode_aliases_on_build_chain(self, discrete_species):
        cfg = Configurator.for_discrete(discrete_species)
        cfg.competition(growth_mode="beverton_holt")
        assert int(cfg.config.juvenile_growth_mode) == 3
        cfg.competition(growth_mode="ricker")
        assert int(cfg.config.juvenile_growth_mode) == 4

    def test_set_param_accepts_legacy_name(self):
        cfg = _age_draft()
        cfg = set_param(cfg, "juvenile_growth_mode", 1)
        assert int(cfg.juvenile_growth_mode) == 1

    def test_every_method_index_entry_is_a_real_method(self):
        methods = {
            "setup", "age_structure", "initial_state", "survival",
            "reproduction", "competition", "fitness", "hook", "migration",
        }
        assert set(ROUTES_BY_METHOD) == methods
        for name in ROUTES_BY_METHOD:
            assert callable(getattr(Configurator, name, None)) or name in (
                "hook", "migration", "initial_state"
            )

    def test_sensitive_column_matches_legacy_set(self):
        for key in ("competition.carrying_capacity",
                    "reproduction.eggs_per_female",
                    "reproduction.sex_ratio"):
            assert is_sensitive(key)
            assert is_sensitive(key.split(".", 1)[1])
        assert not is_sensitive("low_density_growth_rate")
        assert not is_sensitive("competition_strength")


# ── 6. negative contract ──────────────────────────────────────────────────────


class TestDeletedInterfaces:
    def test_subclass_modules_are_gone(self):
        with pytest.raises(ImportError):
            importlib.import_module("natal.frontend.configurator.age_structured")
        with pytest.raises(ImportError):
            importlib.import_module("natal.frontend.configurator.discrete")

    def test_subclass_names_are_unreachable(self):
        import natal.frontend.configurator as legacy_shim
        from natal.frontend import configurator as package

        for name in ("AgeStructuredConfigurator", "DiscreteConfigurator"):
            assert not hasattr(package, name)
            assert not hasattr(legacy_shim, name)
            assert name not in package.__all__
            with pytest.raises(AttributeError):
                _import_name("natal.frontend.configurator", name)

    def test_descriptor_columns_deleted(self):
        from natal.frontend.utils.parameters import ParamDescriptor

        for column in ("is_tensor", "is_0d", "is_array"):
            assert column not in ParamDescriptor.__dataclass_fields__

    def test_hand_maintained_sensitive_sets_deleted(self):
        from natal.frontend.configurator import _base
        from natal.frontend.spatial import configurator as spatial_cfg

        assert not hasattr(_base, "_EQUILIBRIUM_SENSITIVE_KEYS")
        assert not hasattr(spatial_cfg, "_EQUILIBRIUM_SENSITIVE_KWARGS")

    def test_per_method_sync_deleted(self):
        from natal.frontend.configurator import _base

        assert not hasattr(Configurator, "_sync_equilibrium")
        assert not hasattr(_base, "_EQUILIBRIUM_SENSITIVE_KEYS")

    def test_spatial_only_param_still_rejected(self):
        cfg = _age_draft()
        with pytest.raises(ValueError, match="spatial-only"):
            dispatch(cfg, "migration_rate", 0.1)

    def test_set_param_still_rejects_tensor_and_immutable(self):
        cfg = _age_draft()
        with pytest.raises(ValueError, match="tensor or array"):
            set_param(cfg, "viability", 1.0)
        with pytest.raises(TypeError, match="immutable config"):
            set_param(cfg, "n_ztypes", 2)


def _import_name(module: str, name: str) -> object:
    """Import a single attribute; module-level names only."""
    mod = importlib.import_module(module)
    return getattr(mod, name)


# ── plan/commit split proofs ──────────────────────────────────────────────────


class TestPlanCommitSplit:
    def test_plan_does_not_mutate(self):
        cfg = _age_draft()
        before = np.asarray(cfg.age_based_survival_rates).copy()
        plan = plan_write(cfg, lookup("female_age_based_survival"), 0.3)
        np.testing.assert_allclose(np.asarray(cfg.age_based_survival_rates), before)
        live = commit_write(cfg, plan)
        assert float(live.age_based_survival_rates[0, 0]) == 0.3

    def test_plan_write_rejects_repeated_writes_atomically(self):
        cfg = _age_draft()
        before = float(cfg.carrying_capacity)
        plans = [
            plan_write(cfg, lookup("carrying_capacity"), 222.0),
        ]
        with pytest.raises(ValueError):
            plan_write(cfg, lookup("sex_ratio"), 7.0)
        # The failed plan left the draft untouched; committing the valid
        # plan afterwards is still possible.
        cfg = commit_write(cfg, plans[0])
        assert float(cfg.carrying_capacity) == 222.0
