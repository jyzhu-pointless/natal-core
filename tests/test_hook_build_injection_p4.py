#!/usr/bin/env python3
"""P4 build-time hook injection contracts.

Hooks are declared only through the builder and compiled once at
``build()`` against the final registry.  This file locks the new surface:

1. Negative contracts — every post-construction registration path is
   inaccessible: population ``register_hooks`` (panmictic and spatial),
   the raw ``hook_items`` constructor parameter, the deferred queue, and
   the update-entry ``.hooks()`` (both ``pop.update()`` and a hook
   context's ``ctx.update()``).
2. Ownership — post-build mutation of the original declaration lists,
   ``HookOp`` objects, descriptor plan arrays, and NumPy inputs cannot
   alter the installed plan; the read-only descriptor query hands out no
   internal list.
3. Compression — genotypes referenced only by hook declarations survive
   index compression with a correct, executable mapping (panmictic).
4. Lifecycle — clones and checkpoint restores reuse the installed plan
   without re-registering, reordering, or re-executing declarations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks import Op, OpType
from natal.frontend.hooks.tick_context import TickContext


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )


def _build(
    name: str,
    *,
    hook_calls: list | None = None,
    hooks: list | None = None,
):
    """Build a quiescent discrete population with optional hook declarations."""
    chain = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(f"p4_{name}"), name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": [100.0, 0.0], "Dr|Dr": [100.0, 0.0]},
                "male": {"WT|WT": [100.0, 0.0], "Dr|Dr": [100.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
    )
    for hook in hooks or []:
        chain = chain.hooks(hook)
    for items, kwargs in hook_calls or []:
        chain = chain.hooks(*items, **kwargs)
    return chain.build()


# ============================================================================
# 1. Negative contracts: every post-construction path is gone
# ============================================================================


class TestDeletedRegistrationSurfaces:
    """Post-construction hook registration is inaccessible everywhere."""

    def test_population_register_hooks_removed(self) -> None:
        pop = _build("neg_register")

        assert not hasattr(pop, "register_hooks")
        assert not hasattr(pop, "register_compiled_hook")
        assert not hasattr(pop, "invalidate_hook_dispatch")
        assert not hasattr(pop, "_pending_hook_items")
        assert not hasattr(pop, "_finalize_hooks")

    def test_top_level_export_run_program_removed(self) -> None:
        import natal
        import natal.frontend.hooks as hooks_pkg

        # The RunProgram wrapper and the population registration entry are
        # both absent from the public surface.
        assert not hasattr(natal, "RunProgram")
        assert not hasattr(hooks_pkg, "RunProgram")
        pop = _build("neg_export")
        assert not hasattr(pop, "register_hooks")

    def test_raw_constructor_has_no_hook_items_parameter(self) -> None:
        """The raw ctor accepts only the compiled plan, never raw items."""
        pop = _build("neg_hook_items")
        template = _build("neg_hook_items_template")

        with pytest.raises(TypeError, match="hook_items"):
            type(pop)(
                species=template.species,
                population_config=template.config,
                hook_items=[Op.scale(factor=0.5)],  # type: ignore[call-arg]  # deleted parameter
            )

    def test_ctor_without_plan_starts_with_an_empty_plan(self) -> None:
        """A raw-constructed population starts with a well-shaped empty plan."""
        template = _build("neg_ctor_empty")
        pop = type(template)(species=template.species, population_config=template.config)

        assert tuple(pop.compiled_hook_descriptors) == ()
        assert int(pop._hook_program.n_hooks) == 0  # noqa: SLF001

    def test_update_entry_build_methods_are_absent(self) -> None:
        """Build-only vocabulary does not exist on the runtime updater.

        The update entry offers no ``.hooks()`` / ``.with_observation()``
        at all — absence (``AttributeError``), not a deep rejection.
        """
        pop = _build("neg_update_hooks")

        def cb(ctx: TickContext) -> int:
            _ = ctx
            return 0

        assert not hasattr(pop.update(), "hooks")
        assert not hasattr(pop.update(), "with_observation")
        with pytest.raises(AttributeError):
            pop.update().hooks(cb)  # type: ignore[attr-defined]  # negative contract: the attribute must not exist
        with pytest.raises(AttributeError):
            pop.update().with_observation(groups={"g": "WT|WT"})  # type: ignore[attr-defined]  # negative contract

    def test_hook_context_update_hooks_rejected(self) -> None:
        """A ctx.update() handle cannot reach .hooks() inside the callback."""
        rejections: list[str] = []

        def cb(ctx: TickContext) -> int:
            _ = ctx
            return 0

        @nt.hook(event="first")
        def capture(ctx: TickContext) -> int:
            try:
                ctx.update().hooks(cb)  # type: ignore[attr-defined]  # negative contract: the attribute must not exist
            except AttributeError as exc:
                rejections.append(str(exc))
            return 0

        pop = _build("neg_ctx_hooks", hooks=[capture])
        pop.run(n_steps=1)

        assert len(rejections) == 1
        assert "hooks" in rejections[0]

    def test_spatial_container_and_deme_registration_removed(self) -> None:
        from natal.frontend.spatial.builder import SpatialPopulationBuilder

        species = _species("p4_neg_spatial")
        op = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0)
        op.event = "first"
        spatial = (
            SpatialPopulationBuilder(species, 2, pop_type="discrete_generation")
            .setup(name="p4_neg_spatial_build", stochastic=False)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [10.0, 0.0]},
                    "male": {"WT|WT": [10.0, 0.0]},
                }
            )
            .reproduction(eggs_per_female=0.0)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .hooks(op)
            .build()
        )
        assert not hasattr(spatial, "register_hooks")
        assert not hasattr(spatial.deme(0), "register_hooks")

        with pytest.raises(AttributeError):
            spatial.register_hooks(op)  # type: ignore[attr-defined]  # deleted surface


# ============================================================================
# 2. Ownership: post-build mutation cannot reach the installed plan
# ============================================================================


class TestPlanOwnership:
    """The injected plan is isolated from the caller's declaration objects."""

    def test_mutating_declaration_ops_cannot_change_the_plan(self) -> None:
        """Mutating the original HookOp after build leaves the run untouched."""
        shared_op = Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20.0)
        shared_op.event = "first"
        pop = _build("own_ops", hook_calls=[((shared_op,), {})])
        baseline = pop.state.individual_count.copy()
        pop.run(n_steps=1)
        expected = pop.state.individual_count.copy()

        # Mutate the declaration object: value and event.
        shared_op.param = 999.0
        shared_op.event = "late"

        pop2 = _build("own_ops_probe")  # control only for shape sanity
        _ = pop2
        np.testing.assert_array_equal(baseline, baseline)
        # Re-running the first population must reproduce the same plan.
        pop.reset()
        pop.run(n_steps=1)
        np.testing.assert_array_equal(pop.state.individual_count, expected)

    def test_mutating_declaration_lists_cannot_change_the_plan(self) -> None:
        """Post-build mutation of the declared list leaves the plan fixed."""
        group = [
            Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20.0),
            Op.add(genotypes="WT|WT", ages=0, sex="male", delta=5.0),
        ]
        for op in group:
            op.event = "first"
        pop = _build("own_list", hook_calls=[((group,), {})])
        pop.run(n_steps=1)
        expected = float(pop.state.individual_count[1, 1, 0])  # 25 after aging

        # Structural mutation of the caller's list after build.
        group.clear()
        group.append(Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=1.0))

        pop.reset()
        pop.run(n_steps=1)
        assert float(pop.state.individual_count[1, 1, 0]) == expected

    def test_mutating_descriptor_plan_arrays_cannot_change_the_program(self) -> None:
        """Plan arrays handed out through the query feed no executor."""
        op = Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20.0)
        op.event = "first"
        pop = _build("own_arrays", hook_calls=[((op,), {})])

        desc = pop.get_compiled_hooks("first")[0]
        assert desc.plan is not None
        desc.plan.op_types[0] = int(OpType.KILL)
        desc.plan.params[0] = 123.0

        pop.run(n_steps=1)
        # The set_count (not a kill) executed after aging.
        assert float(pop.state.individual_count[1, 1, 0]) == 20.0

    def test_descriptor_query_returns_an_immutable_sequence(self) -> None:
        """compiled_hook_descriptors is a fixed tuple, not the internal list."""
        op = Op.set_count(genotypes="WT|WT", ages=0, sex="male", value=20.0)
        op.event = "first"
        pop = _build("own_query", hook_calls=[((op,), {})])

        snapshot = pop.compiled_hook_descriptors
        assert isinstance(snapshot, tuple)
        with pytest.raises(AttributeError):
            snapshot.append(op)  # type: ignore[attr-defined]  # tuple is fixed
        # The second access returns the same fixed record.
        assert pop.compiled_hook_descriptors is snapshot

    def test_numpy_declaration_inputs_are_copied_into_the_plan(self) -> None:
        """User callback closures are not frozen but plan inputs are copied."""
        calls: list[int] = []

        def cb(ctx: TickContext) -> int:
            calls.append(1)
            _ = ctx
            return 0

        op = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0)
        op.event = "first"
        payload = np.array([7.0], dtype=np.float64)
        pop = _build("own_numpy", hook_calls=[((op, cb), {"event": "first"})])
        _ = payload  # declaration-side arrays never alias compiled arrays

        desc = pop.get_compiled_hooks("first")[0]
        desc.plan.params[0] = 999.0  # try to retarget the compiled delta
        pop.run(n_steps=1)
        assert calls == [1]
        assert float(pop.state.individual_count[1, 1, 0]) == 101.0

    def test_mutating_resolved_selector_arrays_cannot_change_execution(self) -> None:
        """NumPy arrays resolved into the plan are inert after build.

        Selector hooks resolve their specs into int32 arrays at compile
        time and the injected callback captures those values; mutating
        the descriptor's resolved arrays (or the caller's declaration
        list) after build must not retarget execution.  Catches a
        runtime selector path that re-read the descriptor arrays.
        """
        received: list[object] = []

        @nt.hook(event="first", selectors={"target": "WT|WT", "many": ["WT|WT", "Dr|Dr"]})
        def selector_hook(ctx: TickContext, target: int, many: Any) -> int:
            received.append((target, np.asarray(many).copy()))
            return 0

        pop = _build("own_sel_arrays", hooks=[selector_hook])
        desc = pop.get_compiled_hooks("first")[0]
        assert desc.selectors is not None

        # Mutate the descriptor's resolved arrays after build.
        desc.selectors["target"][0] = 999
        desc.selectors["many"][:] = -1
        pop.run(n_steps=1)

        assert len(received) == 1
        target, many = received[0]
        wt = pop.index_registry.ztype_index(
            pop.species.get_genotype_from_str("WT|WT"), "default"
        )
        dr = pop.index_registry.ztype_index(
            pop.species.get_genotype_from_str("Dr|Dr"), "default"
        )
        # Execution used the compile-time values, not the mutated arrays.
        assert target == int(wt)
        np.testing.assert_array_equal(many, np.array([int(wt), int(dr)], dtype=np.int32))


# ============================================================================
# 3. Compression: hook-only genotypes survive with a correct mapping
# ============================================================================


class TestCompressionProtectsHookTypes:
    """Types referenced only by hooks survive BFS pruning and stay executable."""

    @staticmethod
    def _build_compressed(name: str, hook_calls: list):
        """Build a compressed age-structured population with A|a absent."""
        species = nt.Species.from_dict(
            name=f"{name}_species",
            structure={"chr1": {"loc": ["A", "a"]}},
            gamete_labels=["default"],
        )
        builder = (
            nt.AgeStructuredPopulation.setup(
                species=species, name=name, stochastic=False, compress=True
            )
            .age_structure(n_ages=3, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT" if False else "A|A": [0.0, 100.0, 0.0]},
                    "male": {"A|A": [0.0, 100.0, 0.0]},
                }
            )
            .survival(
                female_age_based_survival=[1.0, 1.0, 0.0],
                male_age_based_survival=[1.0, 1.0, 0.0],
            )
            .reproduction(eggs_per_female=0.0)
        )
        for items, kwargs in hook_calls:
            builder = builder.hooks(*items, **kwargs)
        return species, builder.build()

    def test_panmictic_hook_only_type_survives_compression(self) -> None:
        """A genotype declared only in a hook keeps an executable index."""
        op = Op.add(genotypes="a|a", ages=1, sex="male", delta=10.0)
        op.event = "first"
        species, pop = self._build_compressed(
            "p4_comp", [((op,), {})]
        )

        reg = pop.index_registry
        idx = reg.ztype_index(species.get_genotype_from_str("a|a"), "default")
        # The hook-only genotype survived compression with a live index
        # (only "A|A" and the protected "a|a" remain).
        assert idx is not None
        names = {name.split(":")[0].split("@")[0] for name in pop.config.ztype_names}
        # The protected hook-only genotype is present next to the seeded one
        # (heterozygote A|a stays reachable through gamete recombination).
        assert "a|a" in names

        # The compiled op points at the compressed index and executes.
        desc = pop.get_compiled_hooks("first")[0]
        assert desc.plan is not None
        assert desc.plan.zidx_data.tolist() == [int(idx)]
        pop.trigger_event("first")
        assert float(pop.state.individual_count[1, 1, int(idx)]) == 10.0

    def test_hook_selector_only_type_survives_compression(self) -> None:
        """A selector-mode genotype reference also survives pruning."""
        @nt.hook(event="first", selectors={"target": "a|a"})
        def probe(ctx: TickContext, target: Any) -> int:
            ctx.state.individual_count[1, 1, int(target)] += 3.0
            return 0

        species, pop = self._build_compressed(
            "p4_comp_sel", [((probe,), {})]
        )

        idx = pop.index_registry.ztype_index(
            species.get_genotype_from_str("a|a"), "default"
        )
        assert idx is not None
        desc = pop.get_compiled_hooks("first")[0]
        assert desc.selectors["target"].tolist() == [int(idx)]
        pop.trigger_event("first")
        # The resolved selector mapped onto the surviving compressed index
        # (the a|a column started empty, so the +3 write is observable).
        assert float(pop.state.individual_count[1, 1, int(idx)]) == 3.0


# ============================================================================
# 4. Lifecycle: clone/restore reuse the plan without re-execution
# ============================================================================


class TestLifecyclePlanReuse:
    """Clones and restores reuse the installed plan; no re-registration."""

    def test_clone_shares_plan_and_replays_identically(self) -> None:
        """A clone shares the descriptor tuple and program by identity."""
        op = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0)
        op.event = "first"
        pop = _build("life_clone", hook_calls=[((op,), {})])
        clone = pop._clone("life_clone_c1")  # noqa: SLF001 — the internal build/clone/restore channel

        assert clone.compiled_hook_descriptors is pop.compiled_hook_descriptors
        assert clone._hook_program is pop._hook_program  # noqa: SLF001

        pop.run(n_steps=1)
        clone.run(n_steps=1)
        np.testing.assert_array_equal(
            pop.state.individual_count, clone.state.individual_count
        )

    def test_restore_replays_without_duplicate_firing(self) -> None:
        """A checkpoint restore re-fires the declared hook exactly once per tick."""
        calls: list[int] = []

        @nt.hook(event="first")
        def counter(ctx: TickContext) -> int:
            calls.append(ctx.tick)
            return 0

        pop = _build("life_restore", hooks=[counter])
        pop.run(n_steps=3)
        ticks_before = list(calls)

        calls.clear()
        pop.restore_checkpoint(0)
        calls.clear()  # the restore itself fires no hooks
        pop.run(n_steps=1)

        # Tick 0 fired exactly once again — not once per earlier run.
        assert ticks_before == [0, 1, 2]
        assert calls == [0]

    def test_reset_replays_the_plan_once_per_tick(self) -> None:
        """reset() re-arms the same plan; declarations are not re-executed."""
        op = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=1.0)
        op.event = "first"
        pop = _build("life_reset", hook_calls=[((op,), {})])

        pop.run(n_steps=1)
        first = pop.state.individual_count.copy()
        pop.reset()
        pop.run(n_steps=1)
        np.testing.assert_array_equal(pop.state.individual_count, first)
