"""Regression tests for declaration-local spatial batch resolution."""

from __future__ import annotations

import natal as nt


def _species(name: str, alleles: list[str] | None = None) -> nt.Species:
    return nt.Species.from_dict(
        name,
        {"c": {"l": alleles or ["WT", "Dr"]}},
        somatic_labels=["default"],
    )


def test_batch_presets_do_not_replace_a_later_hooks_declaration() -> None:
    species = _species("declaration_order_hooks")
    first = nt.PointMutation("first", "WT", target_allele="Dr", mutation_rate=0.1)
    second = nt.PointMutation("second", "WT", target_allele="Dr", mutation_rate=0.2)

    population = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
        .age_structure(3, 1)
        .presets(nt.batch_setting([first, second]))
        .hooks(nt.Op.add(delta=1), event="first")
        .build()
    )

    assert all(
        len(population._deme_object(i).get_compiled_hooks("first")) == 1
        for i in range(2)
    )


def test_ordinary_redeclaration_overrides_an_earlier_batch() -> None:
    species = _species("declaration_order_override", ["WT"])

    population = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
        .age_structure(3, 1)
        .reproduction(eggs_per_female=nt.batch_setting([10.0, 20.0]))
        .reproduction(eggs_per_female=30.0)
        .build()
    )

    assert [deme.config.eggs_per_female for deme in population.demes] == [30.0, 30.0]


def test_repeated_preset_calls_keep_their_declaration_local_positions() -> None:
    species = _species("declaration_order_repeated_presets")
    first_a = nt.PointMutation("first_a", "WT", target_allele="Dr", mutation_rate=0.1)
    first_b = nt.PointMutation("first_b", "WT", target_allele="Dr", mutation_rate=0.2)
    second_a = nt.PointMutation("second_a", "WT", target_allele="Dr", mutation_rate=0.3)
    second_b = nt.PointMutation("second_b", "WT", target_allele="Dr", mutation_rate=0.4)

    builder = nt.SpatialPopulation.builder(species, n_demes=2)
    builder.presets(nt.batch_setting([first_a, first_b]))
    builder.presets(nt.batch_setting([second_a, second_b]))

    resolved_zero = builder._resolved_group_journal(  # pyright: ignore[reportPrivateUsage]
        {"_preset_0": first_a, "_preset_0#1": second_a}
    )
    resolved_one = builder._resolved_group_journal(  # pyright: ignore[reportPrivateUsage]
        {"_preset_0": first_b, "_preset_0#1": second_b}
    )

    assert resolved_zero[0][1]["__args__"] == (first_a,)
    assert resolved_zero[1][1]["__args__"] == (second_a,)
    assert resolved_one[0][1]["__args__"] == (first_b,)
    assert resolved_one[1][1]["__args__"] == (second_b,)


def test_batch_setting_nested_values_are_snapshotted_at_declaration() -> None:
    species = _species("declaration_order_batch_snapshot", ["WT"])
    first = {"female": {"WT|WT": {1: 10}}}
    second = {"female": {"WT|WT": {1: 20}}}
    builder = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
        .age_structure(3, 1)
        .initial_state(
            individual_count=nt.batch_setting([first, second]),
        )
    )
    first["female"]["WT|WT"][1] = 99
    second["female"]["WT|WT"][1] = 88

    population = builder.build()

    assert [deme.get_total_count() for deme in population.demes] == [10, 20]
