"""Tests for the parameter descriptor registry (natal.parameters)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from natal.parameters import (
    ALL_PARAMETERS,
    PARAM_IDS,
    PARAMETERS_BY_DOMAIN,
    ParamDescriptor,
)
from natal.population_config import PopulationConfig


class TestParamDescriptor:
    """Construction and immutability of ParamDescriptor."""

    def test_construction_defaults(self):
        """Defaults match documented behaviour."""
        desc = ParamDescriptor(
            domain="test",
            name="foo",
            config_field="some_field",
            config_path=(),
            dtype=float,
            bounds=(0.0, 1.0),
        )
        assert desc.is_tensor is False
        assert desc.is_0d is False
        assert desc.is_array is False
        assert desc.target == "config"
        assert desc.doc == ""
        assert desc.aliases == ()

    def test_frozen_prevents_mutation(self):
        """Assigning an attribute raises FrozenInstanceError."""
        desc = ParamDescriptor(
            domain="test",
            name="bar",
            config_field="x",
            config_path=(),
            dtype=int,
            bounds=(0, 100),
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
    """Verifying config_field mapping between parameters and PopulationConfig."""

    def test_config_field_exists_on_config(self):
        """Every parameter with a config_field maps to an actual PopulationConfig field."""
        for key, desc in ALL_PARAMETERS.items():
            if desc.config_field is None:
                # Spatial-only parameters have no config_field (e.g. migration_rate)
                continue
            assert hasattr(
                PopulationConfig, desc.config_field
            ), f"{key}: PopulationConfig has no field '{desc.config_field}'"

    def test_migration_rate_config_field_none(self):
        """migration.migration_rate has config_field=None (spatial only)."""
        desc = ALL_PARAMETERS["migration.migration_rate"]
        assert desc.config_field is None
        assert desc.target == "spatial"

    def test_carrying_capacity_is_0d(self):
        """competition.carrying_capacity is a 0-d ndarray targeting 'config'."""
        desc = ALL_PARAMETERS["competition.carrying_capacity"]
        assert desc.is_0d is True
        assert desc.target == "config"

    @pytest.mark.parametrize(
        "key",
        [
            "fitness.viability",
            "fitness.fecundity",
            "fitness.sexual_selection",
            "fitness.zygote_viability",
            "fitness.female_genotype_compatibility",
            "fitness.male_genotype_compatibility",
        ],
    )
    def test_fitness_is_tensor(self, key: str):
        """Fitness-related parameters have is_tensor=True."""
        desc = ALL_PARAMETERS[key]
        assert desc.is_tensor is True, f"{key}.is_tensor should be True"

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
    """Organisation of parameters by domain."""

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
        """Competition domain contains expected parameters."""
        comp = PARAMETERS_BY_DOMAIN["competition"]
        assert "carrying_capacity" in comp
        assert "low_density_growth_rate" in comp
        assert "juvenile_growth_mode" in comp
        assert "competition_strength" in comp
        assert "expected_competition_strength" in comp
        assert "expected_survival_rate" in comp

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
