"""The declaration-backed initial distribution and the dependency graph.

FRONTEND_REFACTOR_PLAN.md §4.3/§4.5 (item 1).  The builder stores the
user's initial distribution as the authoritative declaration and derives
the engine arrays from it whenever dimensions change — declaring the
distribution before locking the final age structure no longer silently
zeroes the population (external plan review finding 1, reproduced for both
a single population and a two-deme spatial build).  The compilation order
itself comes from ``model/dependencies.jsonc``: unknown nodes, missing
compute implementations and cycles are load-time errors.

Scalar counts keep their documented meaning — one count per adult age —
so the acceptance here is order-equivalence (declare-then-structure equals
structure-then-declare), plus the per-age totals that semantics implies.
"""

from __future__ import annotations

import pytest

import natal as nt
from natal.frontend.model.dependency_graph import (
    DependencyGraph,
    DerivationPipeline,
    load_dependency_graph,
)


@pytest.fixture(scope="module")
def species() -> nt.Species:
    return nt.Species.from_dict("declaration_pipeline", {"c": {"l": ["WT"]}})


class TestInitialDistributionDeclaration:
    def test_declaration_survives_final_age_structure(self, species: nt.Species) -> None:
        """The plan-review repro: counts must not vanish behind age_structure."""
        forward = (
            nt.AgeStructuredPopulation.setup(species)
            .initial_state(individual_count={"female": {"WT|WT": 100}})
            .age_structure(5, 2)
        )
        reversed_order = (
            nt.AgeStructuredPopulation.setup(species)
            .age_structure(5, 2)
            .initial_state(individual_count={"female": {"WT|WT": 100}})
        )
        assert forward.build().state.individual_count.sum() == pytest.approx(300.0)
        assert reversed_order.build().state.individual_count.sum() == pytest.approx(300.0)

    def test_scalar_replicates_over_the_final_adult_ages(
        self, species: nt.Species
    ) -> None:
        """100 per adult age: ages [2, 5) carry 100 each, juveniles zero."""
        pop = (
            nt.AgeStructuredPopulation.setup(species)
            .initial_state(individual_count={"female": {"WT|WT": 100}})
            .age_structure(5, 2)
            .build()
        )
        female = pop.state.individual_count[0]
        assert female.sum(axis=1)[:2] == pytest.approx([0.0, 0.0])
        assert female.sum(axis=1)[2:] == pytest.approx([100.0, 100.0, 100.0])

    def test_age_outside_the_final_structure_fails_explicitly(
        self, species: nt.Species
    ) -> None:
        builder = nt.AgeStructuredPopulation.setup(species).initial_state(
            individual_count={"female": {"WT|WT": {4: 100}}}
        )
        with pytest.raises(ValueError, match="out of range"):
            builder.age_structure(3, 1)

    def test_spatial_two_deme_declaration_survives(self) -> None:
        """The plan-review spatial repro (per-deme totals, scalar semantics)."""
        sp = nt.Species.from_dict("declaration_pipeline_spatial", {"c": {"l": ["WT"]}})
        builder = (
            nt.SpatialPopulation.builder(sp, n_demes=2, pop_type="age_structured")
            .setup(stochastic=False)
            .initial_state(individual_count={"female": {"WT|WT": 100}})
            .age_structure(5, 2)
        )
        assert builder.build().get_total_count() == pytest.approx(600.0)

    def test_declaration_is_the_authority_at_build(
        self, species: nt.Species
    ) -> None:
        """Whatever sits in the draft, build derives from the declaration."""
        builder = (
            nt.AgeStructuredPopulation.setup(species)
            .age_structure(3, 1)
            .initial_state(individual_count={"female": {"WT|WT": 10}})
        )
        assert builder.build().state.individual_count.sum() == pytest.approx(20.0)
        # A later declaration replaces the earlier one wholesale.
        builder.initial_state(individual_count={"female": {"WT|WT": 3}})
        assert builder.build().state.individual_count.sum() == pytest.approx(6.0)

    def test_repeated_builds_are_isolated(self, species: nt.Species) -> None:
        builder = (
            nt.AgeStructuredPopulation.setup(species)
            .age_structure(3, 1)
            .initial_state(individual_count={"female": {"WT|WT": 7}})
        )
        first = builder.build()
        second = builder.build()
        assert first.state.individual_count.sum() == pytest.approx(14.0)
        assert second.state.individual_count.sum() == pytest.approx(14.0)
        # Mutating the builder after a build leaves the published model alone.
        builder.initial_state(individual_count={"female": {"WT|WT": 99}})
        assert first.state.individual_count.sum() == pytest.approx(14.0)

    def test_redeclaring_one_input_keeps_the_others(
        self, species: nt.Species
    ) -> None:
        """§4.5-2: one declaration change loses neither fitness nor genetics."""
        from natal.frontend.presets import PointMutation

        builder = (
            nt.AgeStructuredPopulation.setup(species)
            .age_structure(3, 1)
            .initial_state(individual_count={"female": {"WT|WT": 10}})
            .fitness(viability={"WT|WT": 0.5})
        )
        builder.initial_state(individual_count={"female": {"WT|WT": 4}})
        pop = builder.build()
        assert pop.state.individual_count.sum() == pytest.approx(8.0)
        assert pop.config.viability_fitness[0, 0].min() == pytest.approx(0.5)

    def test_declaration_joins_the_model_definition(self, species: nt.Species) -> None:
        builder = (
            nt.AgeStructuredPopulation.setup(species)
            .age_structure(3, 1)
            .initial_state(individual_count={"female": {"WT|WT": 5}})
        )
        definition = builder._definition_for_compile()
        assert definition.initial_distribution is not None
        assert definition.initial_distribution.individual_count == {
            "female": {"WT|WT": 5}
        }


class TestDependencyGraph:
    def test_packaged_graph_loads_and_orders(self) -> None:
        graph = load_dependency_graph()
        order = graph.order()
        # Derivation products land before the publish-side nodes they feed.
        assert order.index("type_names") < order.index("compression")
        assert order.index("initial_counts") < order.index("compression")
        assert order.index("genetic_products") < order.index("compression")
        assert order.index("compression") < order.index("hooks")
        assert order.index("compression") < order.index("observation")

    def test_unknown_dependency_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            DependencyGraph(
                inputs=frozenset({"a"}),
                dependencies={"n": ("missing",)},
            ).order()

    def test_cycle_rejected(self) -> None:
        with pytest.raises(ValueError, match="cycle"):
            DependencyGraph(
                inputs=frozenset(),
                dependencies={"a": ("b",), "b": ("a",)},
            ).order()

    def test_input_and_computed_may_not_overlap(self) -> None:
        with pytest.raises(ValueError, match="both inputs and computed"):
            load_dependency_graph(
                '{"inputs": ["a"], "nodes": {"a": {"depends": []}}}'
            )

    def test_malformed_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            load_dependency_graph('{"nodes": {}}')

    def test_pipeline_rejects_missing_compute(self) -> None:
        graph = load_dependency_graph()
        pipeline = DerivationPipeline(graph)
        pipeline.register("type_names", lambda: None)
        with pytest.raises(ValueError, match="missing compute"):
            pipeline.run(("type_names", "initial_counts"))

    def test_pipeline_rejects_unknown_node(self) -> None:
        pipeline = DerivationPipeline(load_dependency_graph())
        with pytest.raises(ValueError, match="declares no computed node"):
            pipeline.register("nonexistent", lambda: None)

    def test_pipeline_runs_in_graph_order(self) -> None:
        graph = DependencyGraph(
            inputs=frozenset({"in"}),
            dependencies={"second": ("first",), "first": ("in",)},
        )
        pipeline = DerivationPipeline(graph)
        seen: list[str] = []
        pipeline.register("first", lambda: seen.append("first"))
        pipeline.register("second", lambda: seen.append("second"))
        pipeline.run(("second", "first"))
        assert seen == ["first", "second"]
