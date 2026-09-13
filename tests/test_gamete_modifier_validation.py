"""Strict key validation for custom gamete modifiers (CR-2 regression).

A custom gamete modifier that names an unknown source ztype, an unknown
target gamete, or an out-of-range index used to be applied silently:
unresolvable keys were dropped, so a partial distribution (e.g. row sum
0.5) entered the simulation and scaled offspring output without any
error.  The contract is now validate-then-apply: the whole modifier
output resolves against the registry before anything is written, an
invalid declaration fails loudly (build fails; a runtime update keeps
the previous valid configuration), while legal all-zero rows stay legal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import numpy as np
import pytest

import natal as nt


def _build(mods: list[Callable[[], Mapping[object, object]]] | None = None) -> nt.DiscreteGenerationPopulation:
    """100 WT|WT females and males, one egg each, deterministic, no competition."""
    species = nt.Species.from_dict(
        name="gmval", structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    builder = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="gmval", stochastic=False
        )
        .initial_state(individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
    )
    if mods:
        builder = builder.modifiers(gamete_modifiers=mods)
    return builder.build()


def _ztype_totals(pop: nt.DiscreteGenerationPopulation) -> list[float]:
    counts = pop.state.individual_count
    return [float(counts[:, :, i].sum()) for i in range(counts.shape[2])]


def test_valid_modifier_targets_split_mendelian_exactly() -> None:
    """{WT: 0.5, Dr: 0.5} on both sexes yields the exact 25/50/25 split."""

    def half_split() -> Mapping[object, object]:
        return {"WT|WT": {"WT": 0.5, "Dr": 0.5}}

    pop = _build([half_split])
    pop.run(1)
    assert _ztype_totals(pop) == [25.0, 50.0, 25.0]


def test_empty_distribution_is_legal_all_zero_row() -> None:
    """An empty distribution clears the row: a legal zero-gamete model."""

    def zero_row() -> Mapping[object, object]:
        return {"WT|WT": {}}

    pop = _build([zero_row])
    pop.run(1)
    assert pop.state.individual_count.sum() == 0.0


def test_unknown_target_gamete_key_raises() -> None:
    """`{"WT|WT": {"NOT_A_GAMETE": 1.0}}` fails the build (was: silent 0)."""

    def bad_target() -> Mapping[object, object]:
        return {"WT|WT": {"NOT_A_GAMETE": 1.0}}

    with pytest.raises(ValueError, match="NOT_A_GAMETE"):
        _build([bad_target])


def test_mixed_valid_and_invalid_target_raises() -> None:
    """Half-valid distributions do not enter the simulation (was: 25)."""

    def mixed() -> Mapping[object, object]:
        return {"WT|WT": {"WT": 0.5, "NOT_A_GAMETE": 0.5}}

    with pytest.raises(ValueError, match="NOT_A_GAMETE"):
        _build([mixed])


def test_unknown_source_key_raises() -> None:
    """An unresolvable source ztype is an error, not a no-op (was: 100)."""

    def bad_source() -> Mapping[object, object]:
        return {"NOT_A_SOURCE": {"WT": 1.0}}

    with pytest.raises(ValueError, match="NOT_A_SOURCE"):
        _build([bad_source])


def test_out_of_range_sex_index_raises() -> None:
    """An out-of-range sex index in an (sex, ztype) key is rejected."""

    def bad_sex() -> Mapping[object, object]:
        return {(5, "WT|WT"): {"WT": 1.0}}

    with pytest.raises(ValueError, match="sex index 5"):
        _build([bad_sex])


def test_out_of_range_ztype_and_gtype_indices_raise() -> None:
    """Pre-resolved integer indices outside the active axes are rejected."""

    def bad_ztype() -> Mapping[object, object]:
        return {(0, 999): {"WT": 1.0}}

    def bad_gtype() -> Mapping[object, object]:
        return {"WT|WT": {777: 1.0}}

    with pytest.raises(ValueError, match="ztype axis"):
        _build([bad_ztype])
    with pytest.raises(ValueError, match="gtype axis"):
        _build([bad_gtype])


def test_failed_runtime_update_keeps_previous_config() -> None:
    """A runtime modifier update that fails leaves the population runnable."""

    def bad_target() -> Mapping[object, object]:
        return {"WT|WT": {"NOT_A_GAMETE": 1.0}}

    pop = _build()
    with pytest.raises(ValueError, match="NOT_A_GAMETE"):
        pop.update().modifiers(gamete_modifiers=[bad_target])
    pop.run(1)
    assert _ztype_totals(pop) == [100.0, 0.0, 0.0]


def test_input_tensor_untouched_on_failure() -> None:
    """A raising modifier never leaves a half-applied gamete map behind."""
    from natal.frontend.modifiers.module import wrap_gamete_modifier

    def bad_target() -> Mapping[object, object]:
        return {"WT|WT": {"NOT_A_GAMETE": 1.0}}

    pop = _build()
    tensor = pop.config.zygotes_to_gametes_map.copy()
    wrapped = wrap_gamete_modifier(bad_target, None, pop.registry, name="boom")
    with pytest.raises(ValueError, match="boom"):
        wrapped(tensor)
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, tensor)


def test_sex_name_source_key_applies_and_rejects_bad_payload() -> None:
    """A sex-name source key writes that sex's rows; non-mapping payload fails.

    Female-only {WT: 0.5, Dr: 0.5} with unmodified males (all WT) must give
    exactly 50 WT|WT + 50 WT|Dr offspring (deterministic, one egg each).
    """
    def female_split() -> Mapping[object, object]:
        return {"female": {"WT|WT": {"WT": 0.5, "Dr": 0.5}}}

    pop = _build([female_split])
    pop.run(1)
    assert _ztype_totals(pop) == [50.0, 50.0, 0.0]

    def bad_payload() -> Mapping[object, object]:
        return {"female": "not-a-mapping"}

    with pytest.raises((TypeError, ValueError), match="must be a mapping"):
        _build([bad_payload])


def test_pair_source_key_writes_exact_rows() -> None:
    """A valid (sex_idx, ztype_key) pair source writes exactly those rows."""
    def by_pairs() -> Mapping[object, object]:
        return {
            (0, "WT|WT"): {"WT": 1.0},  # female WT|WT keeps producing WT
            (1, "WT|WT"): {"Dr": 1.0},  # male WT|WT redirected to Dr
        }

    pop = _build([by_pairs])
    pop.run(1)
    # Maternal WT x paternal Dr -> all heterozygous (canonical WT|Dr).
    assert _ztype_totals(pop) == [0.0, 100.0, 0.0]


def test_modifier_returning_non_mapping_is_rejected() -> None:
    """A modifier whose bulk output is not a mapping fails explicitly."""
    from natal.frontend.modifiers.module import wrap_gamete_modifier

    def bad_output() -> object:
        return "not-a-mapping"

    pop = _build()
    wrapped = wrap_gamete_modifier(bad_output, None, pop.registry, name="junk")
    with pytest.raises(TypeError, match="must return a mapping"):
        wrapped(pop.config.zygotes_to_gametes_map.copy())


def test_genotype_source_outside_active_axis_matches_nothing() -> None:
    """A resolvable genotype absent from the compressed axis is an error.

    Build-time compression validates modifiers against the full candidate
    registry and prunes afterwards, so the compressed-axis mismatch is
    asserted on the compressed population's own registry (the runtime
    update path).
    """
    species = nt.Species.from_dict(
        name="gmval-compressed", structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    inactive = species.get_genotype_from_str("Dr|Dr")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="gmval-compressed", stochastic=False, compress=True
        )
        .initial_state(individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )
    assert pop.registry.ztype_indices_for(inactive) == []  # pruned from the axis

    from natal.frontend.modifiers.module import wrap_gamete_modifier

    def source_off_axis() -> Mapping[object, object]:
        return {"female": {inactive: {"WT": 1.0}}}

    wrapped = wrap_gamete_modifier(source_off_axis, None, pop.registry, name="offaxis")
    with pytest.raises(ValueError, match="matches no ztype"):
        wrapped(pop.config.zygotes_to_gametes_map.copy())


@pytest.mark.parametrize("target", [(-1, 0), (0, -1)])
def test_negative_target_component_indices_raise(target: tuple[int, int]) -> None:
    """Negative haplotype/label indices must not select the last registry entry."""
    def bad_target() -> Mapping[object, object]:
        return {"WT|WT": {target: 1.0}}

    with pytest.raises(ValueError):
        _build([bad_target])
