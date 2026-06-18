"""Unit tests for natal.hooks.declarative — Op factories, selectors, and compilation."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.configurator import Configurator
from natal.hooks.entry.declarative import (
    Op,
    _resolve_ages,
    _resolve_genotypes,
    _resolve_sex,
    compile_declarative_hook,
)
from natal.hooks.types import HookOp, OpType
from natal.index_registry import IndexRegistry

# ══════════════════════════════════════════════════════════════════════════
# TestOpFactories
# ══════════════════════════════════════════════════════════════════════════


class TestOpFactories:
    """Verify each Op.* factory returns a HookOp with correct default values."""

    def test_scale_defaults(self):
        op = Op.scale()
        assert op == HookOp(OpType.SCALE, "*", "*", "both", 1.0, None)

    def test_set_count_defaults(self):
        op = Op.set_count()
        assert op == HookOp(OpType.SET, "*", "*", "both", 0.0, None)

    def test_add_defaults(self):
        op = Op.add()
        assert op == HookOp(OpType.ADD, "*", "*", "both", 0.0, None)

    def test_subtract_defaults(self):
        op = Op.subtract()
        assert op == HookOp(OpType.SUBTRACT, "*", "*", "both", 0.0, None)

    def test_kill_defaults(self):
        op = Op.kill()
        assert op == HookOp(OpType.KILL, "*", "*", "both", 0.0, None)

    def test_kill_invalid_prob(self):
        with pytest.raises(ValueError, match="prob must be in"):
            Op.kill(prob=1.5)

    def test_kill_negative_prob(self):
        with pytest.raises(ValueError, match="prob must be in"):
            Op.kill(prob=-0.1)

    def test_sample_defaults(self):
        op = Op.sample()
        assert op == HookOp(OpType.SAMPLE, "*", "*", "both", 0.0, None)

    def test_stop_if_zero(self):
        op = Op.stop_if_zero()
        assert op.op_type == OpType.STOP_IF_ZERO
        assert op == HookOp(OpType.STOP_IF_ZERO, "*", "*", "both", 0.0, None)

    def test_stop_if_below(self):
        op = Op.stop_if_below(threshold=100)
        assert op.param == 100.0
        assert op.op_type == OpType.STOP_IF_BELOW

    def test_stop_if_above(self):
        op = Op.stop_if_above(threshold=1000)
        assert op.param == 1000.0
        assert op.op_type == OpType.STOP_IF_ABOVE

    def test_stop_if_extinction(self):
        op = Op.stop_if_extinction()
        assert op.op_type == OpType.STOP_IF_EXTINCTION
        assert op.genotypes == "*"
        assert op.ages == "*"
        assert op.sex == "both"
        assert op.param == 0.0

    def test_custom_params_passthrough(self):
        """Non-default parameters are correctly passed through."""
        op = Op.scale(
            genotypes="WT|WT",
            ages=0,
            sex="female",
            factor=0.75,
            when="tick >= 10",
        )
        assert op == HookOp(OpType.SCALE, "WT|WT", 0, "female", 0.75, "tick >= 10")

        op2 = Op.kill(prob=0.5, when="tick % 5 == 0")
        assert op2 == HookOp(OpType.KILL, "*", "*", "both", 0.5, "tick % 5 == 0")

        op3 = Op.set_count(
            genotypes=["WT|WT", "Dr|Dr"],
            ages=range(2),
            sex="male",
            value=100.0,
        )
        assert op3 == HookOp(OpType.SET, ["WT|WT", "Dr|Dr"], range(2), "male", 100.0, None)

        op4 = Op.sample(genotypes="Dr|Dr", ages=[0, 1], sex="female", size=50)
        assert op4 == HookOp(OpType.SAMPLE, "Dr|Dr", [0, 1], "female", 50.0, None)

        op5 = Op.stop_if_below(threshold=1, when="tick > 5")
        assert op5 == HookOp(OpType.STOP_IF_BELOW, "*", "*", "both", 1.0, "tick > 5")


# ══════════════════════════════════════════════════════════════════════════
# TestResolveGenotypes
# ══════════════════════════════════════════════════════════════════════════


class TestResolveGenotypes:
    """Test symbolic-to-integer genotype resolution via _resolve_genotypes."""

    @pytest.fixture
    def registry_with_genotypes(self):
        """Build an IndexRegistry populated with genotypes from a 3-allele species."""
        species = nt.Species.from_dict(
            name="TestResolveGenotypes_Species",
            structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
            gamete_labels=["default"],
        )
        reg = IndexRegistry()
        for g in species.get_all_genotypes():
            reg.register_genotype(g)
        return reg

    def test_star_returns_all(self, registry_with_genotypes):
        reg = registry_with_genotypes
        result = _resolve_genotypes("*", reg, reg.index_to_genotype, reg.num_genotypes())
        expected = np.arange(reg.num_genotypes(), dtype=np.int32)
        assert np.array_equal(result, expected)

    def test_string_label(self, registry_with_genotypes):
        reg = registry_with_genotypes
        result = _resolve_genotypes("WT|WT", reg, reg.index_to_genotype, reg.num_genotypes())
        assert list(result) == [0]

    def test_list_of_strings(self, registry_with_genotypes):
        reg = registry_with_genotypes
        result = _resolve_genotypes(
            ["WT|WT", "Dr|Dr"], reg, reg.index_to_genotype, reg.num_genotypes()
        )
        assert list(result) == [0, 4]

    def test_int_input(self, registry_with_genotypes):
        """Bare int is not a supported selector (not iterable), so TypeError."""
        reg = registry_with_genotypes
        with pytest.raises(TypeError):
            _resolve_genotypes(2, reg, reg.index_to_genotype, reg.num_genotypes())

    def test_unknown_string_raises(self, registry_with_genotypes):
        reg = registry_with_genotypes
        with pytest.raises((KeyError, ValueError)):
            _resolve_genotypes("UNKNOWN", reg, reg.index_to_genotype, reg.num_genotypes())


# ══════════════════════════════════════════════════════════════════════════
# TestResolveAges
# ══════════════════════════════════════════════════════════════════════════


class TestResolveAges:
    """Test age selector resolution — pure function, no fixtures needed."""

    def test_star_returns_all(self):
        result = _resolve_ages("*", 5)
        assert list(result) == [0, 1, 2, 3, 4]

    def test_single_int(self):
        result = _resolve_ages(3, 5)
        assert list(result) == [3]

    def test_range(self):
        result = _resolve_ages(range(1, 4), 5)
        assert list(result) == [1, 2, 3]

    def test_list(self):
        result = _resolve_ages([0, 2, 4], 5)
        assert list(result) == [0, 2, 4]

    def test_out_of_range_int(self):
        """Ages outside valid range are still stored (validated at execution)."""
        result = _resolve_ages(99, 5)
        assert list(result) == [99]

    def test_empty_list(self):
        result = _resolve_ages([], 5)
        assert list(result) == []


# ══════════════════════════════════════════════════════════════════════════
# TestResolveSex
# ══════════════════════════════════════════════════════════════════════════


class TestResolveSex:
    """Test sex selector resolution — pure function, no fixtures needed."""

    def test_female(self):
        result = _resolve_sex("female")
        assert list(result) == [True, False]

    def test_male(self):
        result = _resolve_sex("male")
        assert list(result) == [False, True]

    def test_both(self):
        result = _resolve_sex("both")
        assert list(result) == [True, True]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown sex selector"):
            _resolve_sex("unknown")


# ══════════════════════════════════════════════════════════════════════════
# TestCompileDeclarativeHook
# ══════════════════════════════════════════════════════════════════════════


class TestCompileDeclarativeHook:
    """Test compile_declarative_hook with a real population."""

    def _build_pop(self, species: nt.Species):
        """Build a minimal age-structured population for hook compilation tests."""
        return (
            Configurator.from_species(species)
            .setup(stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(carrying_capacity=10000)
            .build(name="test_hook_pop")
        )

    def test_single_scale_op(self, simple_species):
        pop = self._build_pop(simple_species)
        descriptor = compile_declarative_hook([Op.scale(factor=0.5)], pop, event="early")

        assert descriptor.plan.n_ops == 1
        assert list(descriptor.plan.op_types) == [int(OpType.SCALE)]
        assert list(descriptor.plan.params) == [0.5]
        # "*" genotype selector → all 9 genotype indices
        assert len(descriptor.plan.gidx_data) == 9
        # "*" age selector → both ages
        assert len(descriptor.plan.age_data) == pop.config.n_ages

    def test_multiple_ops(self, simple_species):
        pop = self._build_pop(simple_species)
        ops = [
            Op.scale(factor=0.5),
            Op.add(delta=10),
            Op.kill(prob=0.1),
        ]
        descriptor = compile_declarative_hook(ops, pop, event="early")

        assert descriptor.plan.n_ops == 3
        assert list(descriptor.plan.op_types) == [
            int(OpType.SCALE),
            int(OpType.ADD),
            int(OpType.KILL),
        ]
        assert list(descriptor.plan.params) == [0.5, 10.0, 0.1]

    def test_with_condition(self, simple_species):
        pop = self._build_pop(simple_species)
        op = Op.scale(factor=0.5, when="tick >= 100")
        descriptor = compile_declarative_hook([op], pop, event="early")

        # Condition should be compiled (not the default COND_ALWAYS = 0)
        assert int(descriptor.plan.condition_types[0]) != 0
        assert list(descriptor.plan.condition_types) == [3]  # COND_TICK_GE

    def test_name_metadata(self, simple_species):
        pop = self._build_pop(simple_species)
        descriptor = compile_declarative_hook(
            [Op.scale()],
            pop,
            event="late",
            priority=7,
            name="my_custom_hook",
        )
        assert descriptor.name == "my_custom_hook"
        assert descriptor.event == "late"
        assert descriptor.priority == 7

    def test_compile_error_no_population(self, simple_species):
        """Passing None as population raises an appropriate error."""
        with pytest.raises((AttributeError, TypeError)):
            compile_declarative_hook([Op.scale()], None, event="early")
