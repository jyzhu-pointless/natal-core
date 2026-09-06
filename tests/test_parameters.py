"""Tests for the parameter descriptor registry (natal.parameters)."""

from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
from typing import TextIO

import pytest

from natal.frontend.utils.parameters import (
    ALL_PARAMETERS,
    PARAM_IDS,
    PARAMETERS_BY_DOMAIN,
    ParamDescriptor,
    _build_registry,
)
from natal.frontend.data import ModelDraft


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
