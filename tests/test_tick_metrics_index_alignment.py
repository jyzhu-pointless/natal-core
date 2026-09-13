"""TickMetrics catalog alignment and name format (CR-12 regression).

Allele frequencies used to resolve every catalog name back to a genotype
through string splitting and a linear registry scan; the name directory
itself rendered as ``"genotype:label"``.  The contract now is: allele
frequencies walk the state's ztype count axis through the registry's
``(Genotype, slab)`` catalog by index, the axis and the catalog must be
aligned (mismatch raises instead of silently truncating), and the
directory uses the unified ``"genotype@label"`` format with the explicit
``@default`` entry.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.contracts.blueprint import format_type_name
from natal.frontend.hooks.tick_context import TickMetrics


def _population(name: str) -> nt.DiscreteGenerationPopulation:
    species = nt.Species.from_dict(
        name=name, structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 600, "Dr|Dr": 400},
                "male": {"WT|WT": 600, "Dr|Dr": 400},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
    )


def _metrics(pop: nt.DiscreteGenerationPopulation) -> TickMetrics:
    from natal.frontend.hooks.tick_context import _build_blueprint  # pyright: ignore[reportPrivateUsage]  # blueprint view builder for direct TickMetrics construction

    return TickMetrics(
        pop.state,
        _build_blueprint(pop),
        pop.species,
        pop.index_registry,
        lambda: pop.config,
    )


def test_allele_frequencies_walk_the_registry_axis() -> None:
    """Allele decomposition matches the exact per-genotype arithmetic."""
    pop = _population("cr12_metric").build()
    metrics = _metrics(pop)
    af = metrics.allele_frequencies["loc"]
    # 1200 WT|WT and 800 Dr|Dr diploids: two gene copies each.
    assert af["WT"] == pytest.approx(2.0 * 1200.0 / (2.0 * 2000.0))
    assert af["Dr"] == pytest.approx(2.0 * 800.0 / (2.0 * 2000.0))
    assert af["WT"] + af["Dr"] == pytest.approx(1.0)


def test_genotype_counts_names_use_the_at_format() -> None:
    """The directory spells ``genotype@default`` — no colon compat."""
    pop = _population("cr12_names").build()
    metrics = _metrics(pop)
    counts = metrics.genotype_counts
    assert set(counts) == set(pop.config.ztype_names)
    assert all("@" in name for name in counts)
    assert "WT|WT@default" in counts
    assert not any(":" in name for name in counts)


def test_format_type_name_uses_at_separator() -> None:
    """format_type_name renders ``genotype@label`` (explicit default)."""
    assert format_type_name("WT|WT", "default") == "WT|WT@default"
    assert format_type_name("A", "tag") == "A@tag"


def test_axis_catalog_mismatch_raises_instead_of_truncating() -> None:
    """A state axis wider than the catalog is an explicit error."""
    pop = _population("cr12_mismatch").build()
    state = pop.state
    widened = np.zeros(
        (state.individual_count.shape[0], state.individual_count.shape[1],
         state.individual_count.shape[2] + 1),
        dtype=np.float64,
    )
    widened[:, :, :-1] = state.individual_count
    from natal.frontend.data import DiscretePopulationState
    from natal.frontend.hooks.tick_context import _build_blueprint  # pyright: ignore[reportPrivateUsage]  # blueprint view builder for direct TickMetrics construction

    broken_state = DiscretePopulationState(
        n_tick=state.n_tick, individual_count=widened
    )
    metrics = TickMetrics(
        broken_state,
        _build_blueprint(pop),
        pop.species,
        pop.index_registry,
        lambda: pop.config,
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        metrics.genotype_counts
    with pytest.raises(RuntimeError, match="not aligned"):
        metrics.allele_frequencies


def test_allele_frequencies_cover_every_slab_branch() -> None:
    """Slab-expanded ztype axes contribute each (genotype, slab) entry's mass."""
    species = nt.Species.from_dict(
        name="cr12_slabs",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        somatic_labels=["default", "E"],
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="cr12_slab_metrics", stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT@default": 300, "Dr|Dr@E": 200},
                "male": {"WT|WT@default": 300, "Dr|Dr@E": 200},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
        .build()
    )
    metrics = _metrics(pop)
    counts = metrics.genotype_counts
    assert set(counts) == set(pop.config.ztype_names)
    assert counts["WT|WT@default"] == pytest.approx(600.0)
    assert counts["Dr|Dr@E"] == pytest.approx(400.0)
    af = metrics.allele_frequencies["loc"]
    # Slab expansion must not duplicate or drop ztype mass.
    assert af["WT"] == pytest.approx(0.6)
    assert af["Dr"] == pytest.approx(0.4)


def test_allele_frequencies_align_on_compressed_axis() -> None:
    """A compressed ztype axis is walked through its own registry positions."""
    species = nt.Species.from_dict(
        name="cr12_compressed",
        structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
    )
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="cr12_compressed_metrics", stochastic=False,
            compress=True,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 600, "Dr|Dr": 400},
                "male": {"WT|WT": 600, "Dr|Dr": 400},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
        .build()
    )
    assert pop.config.n_ztypes < 9  # R2 genotypes were pruned by the BFS
    metrics = _metrics(pop)
    counts = metrics.genotype_counts
    assert set(counts) == set(pop.config.ztype_names)
    af = metrics.allele_frequencies["loc"]
    assert af["WT"] == pytest.approx(0.6)
    assert af["Dr"] == pytest.approx(0.4)
    assert "R2" not in af


def test_allele_frequencies_registry_mismatch_raises() -> None:
    """A registry catalog out of step with the state axis is an explicit error.

    The genotype_counts check (blueprint names vs counts) cannot fire here,
    so this isolates the registry-vs-counts guard inside allele_frequencies.
    """
    pop = _population("cr12_reg_mismatch").build()
    from natal.frontend.hooks.tick_context import _build_blueprint  # pyright: ignore[reportPrivateUsage]  # blueprint view builder for direct TickMetrics construction

    blueprint = _build_blueprint(pop)
    registry = pop.index_registry
    from types import SimpleNamespace

    short_registry = SimpleNamespace(
        index_to_ztype=registry.index_to_ztype[:-1]
    )
    metrics = TickMetrics(
        pop.state, blueprint, pop.species, short_registry, lambda: pop.config,
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        metrics.allele_frequencies
