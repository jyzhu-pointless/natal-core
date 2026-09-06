"""Tests for the parameter descriptor registry (natal.parameters)."""

from __future__ import annotations

import builtins
import importlib.util
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TextIO

import pytest

from natal.frontend.data import ModelDraft
from natal.frontend.hooks.types import ECO_PARAM_NAMES
from natal.frontend.utils.parameters import (
    ALL_PARAMETERS,
    PARAM_IDS,
    PARAMETERS_BY_DOMAIN,
    ParamDescriptor,
    _build_registry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "generate_param_tables.py"
WIRE_TABLE = REPO_ROOT / "rust" / "src" / "eco_param_wire.rs"


class TestParamDescriptor:
    """Construction and immutability of ParamDescriptor."""

    def test_construction_defaults(self):
        """Defaults match documented behavior."""
        desc = ParamDescriptor(
            domain="test",
            name="foo",
            method="competition",
            kind="scalar",
            section="ecology",
            config_field="some_field",
            config_path=(),
            dtype=float,
            bounds=(0.0, 1.0),
            sensitive=False,
        )
        assert desc.target == "config"
        assert desc.doc == ""
        assert desc.aliases == ()
        assert desc.sensitive is False

    def test_frozen_prevents_mutation(self):
        """Assigning an attribute raises FrozenInstanceError."""
        desc = ParamDescriptor(
            domain="test",
            name="bar",
            method="setup",
            kind="bool",
            section="ecology",
            config_field="x",
            config_path=(),
            dtype=int,
            bounds=(0, 100),
            sensitive=False,
        )
        with pytest.raises(FrozenInstanceError):
            desc.domain = "other"  # type: ignore[misc]


class TestAllParameters:
    """Integrity of the ALL_PARAMETERS registry."""

    def test_non_empty(self):
        """At least 30 registered parameters."""
        assert len(ALL_PARAMETERS) >= 30

    def test_all_keys_have_domain_prefix(self):
        """Every key matches '{domain}.{name}'."""
        for key, desc in ALL_PARAMETERS.items():
            assert key == f"{desc.domain}.{desc.name}", key

    def test_all_values_are_param_descriptors(self):
        """Every value is a ParamDescriptor instance."""
        for desc in ALL_PARAMETERS.values():
            assert isinstance(desc, ParamDescriptor)

    def test_bare_names_unique_across_domains(self):
        """No two parameters share a bare name.

        ``scripts/generate_param_tables.py`` keys its ECO lookup by bare
        name, not by the ``{domain}.{name}`` registry key; a cross-domain
        collision would silently collapse to whichever entry iterated
        last, sourcing bounds from the wrong parameter.
        """
        names = [desc.name for desc in ALL_PARAMETERS.values()]
        assert len(names) == len(set(names)), "duplicate bare parameter names"

    @pytest.mark.parametrize(
        "key",
        [
            "competition.carrying_capacity",
            "reproduction.eggs_per_female",
            "setup.stochastic",
            "survival.female_age0_survival",
            "fitness.viability",
            "migration.migration_rate",
            "age_structure.n_ages",
            "initial_state.initial_individual_count",
            "hook.hook_slot",
        ],
    )
    def test_key_parameters_present(self, key: str):
        """Key parameters that must exist in the registry."""
        assert key in ALL_PARAMETERS, f"Missing expected parameter: {key}"


class TestParameterFieldMapping:
    """Verifying config_field mapping between parameters and ModelDraft."""

    def test_config_field_exists_on_config(self):
        """Every parameter with a config_field maps to an actual ModelDraft field."""
        for key, desc in ALL_PARAMETERS.items():
            if desc.config_field is None:
                # Spatial-only parameters have no config_field (e.g. migration_rate)
                continue
            assert hasattr(
                ModelDraft, desc.config_field
            ), f"{key}: ModelDraft has no field '{desc.config_field}'"

    def test_kinds_are_from_the_seven_shapes(self):
        """Every parameter carries one of the seven route kinds."""
        valid = {"scalar", "mode_enum", "age_vec", "sex_row", "slot", "bool", "geno_tensor"}
        for key, desc in ALL_PARAMETERS.items():
            assert desc.kind in valid, f"{key}: unknown kind {desc.kind!r}"

    def test_sections_are_ecology_or_genetics(self):
        """Ecology/genetics section tags partition the table."""
        for key, desc in ALL_PARAMETERS.items():
            assert desc.section in ("ecology", "genetics"), key
            if desc.kind == "geno_tensor" and desc.domain == "fitness":
                assert desc.section == "genetics", key

    def test_migration_rate_config_field_none(self):
        """migration.migration_rate has config_field=None (spatial only)."""
        desc = ALL_PARAMETERS["migration.migration_rate"]
        assert desc.config_field is None
        assert desc.target == "spatial"

    def test_carrying_capacity_is_scalar(self):
        """competition.carrying_capacity is a bounded scalar targeting 'config'."""
        desc = ALL_PARAMETERS["competition.carrying_capacity"]
        assert desc.kind == "scalar"
        assert desc.section == "ecology"
        assert desc.sensitive is True
        assert desc.target == "config"

    @pytest.mark.parametrize(
        "key",
        [
            "fitness.viability",
            "fitness.fecundity",
            "fitness.sexual_selection",
            "fitness.zygote_viability",
            "fitness.female_ztype_compatibility",
            "fitness.male_ztype_compatibility",
        ],
    )
    def test_fitness_is_geno_tensor(self, key: str):
        """Fitness-related parameters use the geno_tensor shape."""
        desc = ALL_PARAMETERS[key]
        assert desc.kind == "geno_tensor", f"{key}.kind should be geno_tensor"
        assert desc.section == "genetics"

    def test_aliases(self):
        """Known parameters carry their expected historical aliases."""

        cap_desc = ALL_PARAMETERS["competition.carrying_capacity"]
        assert "age_1_carrying_capacity" in cap_desc.aliases
        assert "old_juvenile_carrying_capacity" in cap_desc.aliases

        comp_desc = ALL_PARAMETERS["competition.competition_strength"]
        assert "relative_competition_factor" in comp_desc.aliases

        eggs_desc = ALL_PARAMETERS["reproduction.eggs_per_female"]
        assert "expected_eggs_per_female" in eggs_desc.aliases


class TestParametersByDomain:
    """Organization of parameters by domain."""

    def test_known_domains(self):
        """All expected domains are present."""
        expected = {
            "setup",
            "age_structure",
            "initial_state",
            "survival",
            "reproduction",
            "competition",
            "fitness",
            "hook",
            "migration",
        }
        assert set(PARAMETERS_BY_DOMAIN) == expected

    def test_competition_domain(self):
        """Competition domain contains expected parameters (growth_mode
        renamed from juvenile_growth_mode, which survives as an alias)."""
        comp = PARAMETERS_BY_DOMAIN["competition"]
        assert "carrying_capacity" in comp
        assert "low_density_growth_rate" in comp
        assert "growth_mode" in comp
        assert "juvenile_growth_mode" in comp["growth_mode"].aliases
        assert "competition_strength" in comp
        assert "expected_competition_strength" in comp
        assert "expected_survival_rate" in comp
        assert "external_expected_eggs" in comp
        assert "equilibrium_distribution" in comp

    def test_all_parameters_assigned_to_a_domain(self):
        """Every ALL_PARAMETERS entry appears in exactly one domain group."""
        total_in_domains = sum(len(v) for v in PARAMETERS_BY_DOMAIN.values())
        assert total_in_domains == len(ALL_PARAMETERS)


class TestParamIds:
    """Integrity of the PARAM_IDS mapping."""

    def test_all_parameters_have_ids(self):
        """Every ALL_PARAMETERS entry appears in PARAM_IDS."""
        for key in ALL_PARAMETERS:
            assert key in PARAM_IDS, f"Missing PARAM_IDS entry for {key}"

    def test_ids_are_contiguous(self):
        """PARAM_IDS values start at 0 and are contiguous."""
        ids = list(PARAM_IDS.values())
        assert ids == list(range(len(PARAM_IDS)))


class TestRegistryFileHandling:
    """Issue #40: the parameter-table read must not leak an open file handle."""

    def test_build_registry_closes_parameter_file(self, monkeypatch):
        """The handle ``_build_registry()`` opens for the table is closed by
        the time the call returns (``with open``), and the rebuilt table has
        the same keys as the import-time ALL_PARAMETERS table."""
        real_open = builtins.open
        opened_handles: list[TextIO] = []

        def tracking_open(path: str, mode: str = "r") -> TextIO:
            handle = real_open(path, mode)
            opened_handles.append(handle)
            return handle

        monkeypatch.setattr(builtins, "open", tracking_open)
        registry = _build_registry()
        monkeypatch.undo()

        # Exactly one file (the parameter table) was opened, and it is
        # deterministically closed after the call — no GC-timing reliance.
        assert len(opened_handles) == 1
        assert opened_handles[0].closed is True
        assert len(registry) == len(ALL_PARAMETERS)
        assert set(registry) == set(ALL_PARAMETERS)


def _load_generator() -> ModuleType:
    """Import ``scripts/generate_param_tables.py`` as an in-process module.

    The generator is a script, not a package member, so it is loaded by
    path (same pattern as ``tests/test_contract_ledger.py``).
    """
    spec = importlib.util.spec_from_file_location(
        "generate_param_tables_probe", GENERATOR_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def _columns_block(text: str) -> tuple[int, list[str]]:
    """Return (declared_len, ordered names) from the generated columns array."""
    block = re.search(
        r"pub const ECO_PARAM_COLUMNS: \[&str; (\d+)\] = \[(.*?)\];",
        text,
        re.DOTALL,
    )
    assert block is not None, "ECO_PARAM_COLUMNS definition missing"
    return int(block.group(1)), re.findall(r'"([^"]+)"', block.group(2))


class TestRustWireTables:
    """The Rust wire tables are generated from this jsonc (plan 5.4)."""

    def test_rust_wire_tables_fresh_against_jsonc(self) -> None:
        """``rust/src/eco_param_wire.rs`` matches the jsonc exactly.

        The bounds/names table must never be hand-written twice: this
        runs the generator in ``--check`` mode, so editing the jsonc
        without regenerating fails the suite.
        """
        result = subprocess.run(
            [sys.executable, str(GENERATOR_SCRIPT), "--check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "rust/src/eco_param_wire.rs is stale against "
            f"src/natal/parameters.jsonc:\n{result.stderr}"
        )

    def test_generated_wire_order_matches_eco_param_names(self) -> None:
        """The generated column list equals ECO_PARAM_NAMES exactly, in order.

        Substring membership is not enough: a reordered or extended table
        still contains every name, so the columns array is parsed out and
        compared as an ordered list (including its declared length).
        """
        declared_len, names = _columns_block(WIRE_TABLE.read_text(encoding="utf-8"))
        assert names == list(ECO_PARAM_NAMES)
        assert declared_len == len(ECO_PARAM_NAMES)

    def test_generated_bounds_match_jsonc_exactly(self) -> None:
        """Every generated bounds row equals the jsonc bounds for its column.

        The freshness subprocess test proves file == render(rows); this
        proves render(rows) == jsonc values, so a renderer bug (swapped
        lo/hi, misaligned row) cannot hide behind a self-consistent file.
        Floats compare exactly: both sides are decimal round-trips of the
        same f64.
        """
        text = WIRE_TABLE.read_text(encoding="utf-8")
        block = re.search(
            r"pub const ECO_PARAM_BOUNDS: \[\(f64, f64\); N_ECO_PARAMS\] = \[(.*?)\];",
            text,
            re.DOTALL,
        )
        assert block is not None, "ECO_PARAM_BOUNDS definition missing"
        rows = re.findall(r"\(([\d.eE+-]+),\s*([\d.eE+-]+)\),", block.group(1))
        by_name = {d.name: d.bounds for d in ALL_PARAMETERS.values()}
        expected = [by_name[name] for name in ECO_PARAM_NAMES]
        assert len(rows) == len(expected)
        for row, (name, want) in zip(rows, zip(ECO_PARAM_NAMES, expected), strict=True):
            assert (float(row[0]), float(row[1])) == want, (
                f"bounds row for {name!r}: file has ({row[0]}, {row[1]}), "
                f"jsonc has {want}"
            )

    def test_wire_tables_have_no_handwritten_copy_in_rust_src(self) -> None:
        """Only the generated module may define the wire/layout tables.

        ``use crate::eco_param_wire::...`` imports are the sanctioned
        access path; any other ``const ECO_PARAM_COLUMNS/BOUNDS/N_ECO_PARAMS
        = ...`` or ``ECOLOGY_SCALAR_COLUMNS/ECOLOGY_SCALARS = ...``
        definition in rust/src would reintroduce the manual sync
        discipline that plan 5.4 removes.
        """
        offenders: list[str] = []
        for path in sorted((REPO_ROOT / "rust" / "src").rglob("*.rs")):
            if path.name == "eco_param_wire.rs":
                continue
            body = path.read_text(encoding="utf-8")
            if re.search(r"ECO_PARAM_(?:COLUMNS|BOUNDS)\s*:\s*\[", body) or re.search(
                r"N_ECO_PARAMS\s*:\s*usize", body
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))
            if re.search(r"ECOLOGY_SCALAR_COLUMNS\s*:\s*\[", body) or re.search(
                r"ECOLOGY_SCALARS\s*:\s*\[", body
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert offenders == [], f"hand-written wire table copies: {offenders}"

    def test_ecology_layout_lists_extend_wire_order(self) -> None:
        """The generated layout lists extend the wire order, nothing else.

        Contract channels are the wire order plus ``external_expected_eggs``
        last; the checkpoint list additionally carries ``growth_mode``
        before it.  Both appended names must exist in the jsonc.
        """
        wire = list(ECO_PARAM_NAMES)
        text = WIRE_TABLE.read_text(encoding="utf-8")

        contract = re.search(
            r"pub const ECOLOGY_SCALAR_COLUMNS: \[&str; (\d+)\] = \[(.*?)\];",
            text, re.DOTALL,
        )
        checkpoint = re.search(
            r"pub const ECOLOGY_SCALARS: \[&str; (\d+)\] = \[(.*?)\];",
            text, re.DOTALL,
        )
        assert contract is not None and checkpoint is not None

        contract_names = re.findall(r'"([^"]+)"', contract.group(2))
        checkpoint_names = re.findall(r'"([^"]+)"', checkpoint.group(2))
        assert int(contract.group(1)) == len(contract_names) == 6
        assert int(checkpoint.group(1)) == len(checkpoint_names) == 7
        assert contract_names == wire + ["external_expected_eggs"]
        assert checkpoint_names == wire + ["growth_mode", "external_expected_eggs"]
        by_name = {d.name for d in ALL_PARAMETERS.values()}
        assert {"external_expected_eggs", "growth_mode"} <= by_name


class TestParamTableGenerator:
    """Adversarial probes of scripts/generate_param_tables.py itself."""

    def test_wire_rows_order_follows_eco_param_names(self, monkeypatch) -> None:
        """Reordering ECO_PARAM_NAMES reorders the generated rows.

        Proves the generator has no hidden name hardcoding: rows are
        driven by the tuple, and each row's bounds stay aligned with its
        name (not with the old position).
        """
        import natal.frontend.hooks.types as hooks_types

        generator = _load_generator()
        original = tuple(hooks_types.ECO_PARAM_NAMES)
        monkeypatch.setattr(hooks_types, "ECO_PARAM_NAMES", tuple(reversed(original)))
        rows = generator._wire_rows()
        assert [row[0] for row in rows] == list(reversed(original))
        by_name = {d.name: d.bounds for d in ALL_PARAMETERS.values()}
        for name, lo, hi in rows:
            assert (lo, hi) == by_name[name]
        declared_len, rendered_names = _columns_block(
            generator._render(
                rows,
                ["carrying_capacity", "growth_mode"],
                ["carrying_capacity"],
            )
        )
        assert rendered_names == list(reversed(original))
        assert declared_len == len(original)

    def test_wire_rows_missing_bounds_raises_system_exit(self, monkeypatch) -> None:
        """A registry entry without bounds aborts the generator loudly.

        The shipped registry cannot produce this state (bounds are a
        required jsonc column), so this pins the generator's own guard
        against a future optional-bounds descriptor silently emitting a
        ``(nan, nan)``-style row.
        """
        import natal.frontend.utils.parameters as params_module

        generator = _load_generator()
        entries = [
            SimpleNamespace(
                name=name,
                bounds=None if name == "carrying_capacity" else (0.0, 1.0),
            )
            for name in ECO_PARAM_NAMES
        ]
        monkeypatch.setattr(
            params_module,
            "ALL_PARAMETERS",
            {f"probe.{entry.name}": entry for entry in entries},
        )
        with pytest.raises(SystemExit, match="carrying_capacity"):
            generator._wire_rows()

    def test_layout_names_missing_appended_name_aborts(self) -> None:
        """A layout-appended name missing from the jsonc aborts loudly.

        ``external_expected_eggs`` (both layout lists) and ``growth_mode``
        (checkpoint list only) are appendees that must exist in the
        registry; pruning either must abort the generator with a message
        naming it, before any layout list is produced — a silent skip
        would emit a checkpoint contract that drops the field.
        """
        generator = _load_generator()
        full = {d.name: d for d in ALL_PARAMETERS.values()}
        for missing in ("external_expected_eggs", "growth_mode"):
            pruned = {k: v for k, v in full.items() if k != missing}
            with pytest.raises(SystemExit, match=re.escape(missing)):
                generator._layout_names(pruned)

    def test_layout_names_lists_are_wire_plus_appended(self) -> None:
        """``_layout_names`` is the wire order plus the fixed appendees.

        The two returned lists must be array-equal to ``ECO_PARAM_NAMES``
        with ``external_expected_eggs`` (contract) and additionally
        ``growth_mode`` before it (checkpoint) — no sorting, dedup, or
        reordering may intervene between the wire tuple and the layout
        output.
        """
        generator = _load_generator()
        by_name = {d.name: d for d in ALL_PARAMETERS.values()}
        contract_columns, checkpoint_columns = generator._layout_names(by_name)
        assert contract_columns == list(ECO_PARAM_NAMES) + [
            "external_expected_eggs"
        ]
        assert checkpoint_columns == list(ECO_PARAM_NAMES) + [
            "growth_mode",
            "external_expected_eggs",
        ]

    def test_render_is_deterministic_and_matches_disk(self) -> None:
        """Rendering is a pure function of its inputs and equals the shipped file.

        Two consecutive renders must be byte-identical (no timestamps or
        environment-dependent content), and both must equal the committed
        rust/src/eco_param_wire.rs.
        """
        generator = _load_generator()
        by_name = {d.name: d for d in ALL_PARAMETERS.values()}
        layout = generator._layout_names(by_name)

        first = generator._render(generator._wire_rows(), *layout)
        second = generator._render(generator._wire_rows(), *layout)

        assert first == second
        assert first == WIRE_TABLE.read_text(encoding="utf-8")

    def test_main_check_mode_runs_in_process_fresh(self) -> None:
        """``main(["--check"])`` exercises the full assembly in-process.

        The subprocess freshness test measures behavior but not line
        coverage; calling ``main`` directly keeps the generator's
        assembly path inside the pytest coverage net and still proves
        the zero return on a fresh tree.
        """
        generator = _load_generator()
        assert generator.main(["--check"]) == 0

    def test_main_write_mode_is_idempotent_in_process(self) -> None:
        """``main([])`` rewrites byte-identical content and stays fresh."""
        generator = _load_generator()
        before = WIRE_TABLE.read_text(encoding="utf-8")
        assert generator.main([]) == 0
        assert WIRE_TABLE.read_text(encoding="utf-8") == before
        assert generator.main(["--check"]) == 0
