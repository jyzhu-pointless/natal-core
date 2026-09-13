"""Species baseline content-snapshot invalidation (CR-7 regression).

The species config blueprint (Mendelian meiosis/fertilization maps and
derivatives) is the single baseline every population build starts from.
It used to be built once and never invalidated: recombination or label
changes were silently ignored, and gamete results were memoized on
Genotype.  The contract now is lazy content-based invalidation — the
acquisition entry snapshots the structure content and rebuilds whenever
that content changes — plus per-call recomputation of gametes, so no
manual cache clearing exists or is needed.
"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
import pytest

import natal as nt


def _double_heterozygote_species(name: str) -> nt.Species:
    """Two two-allele loci on one chromosome: recombination is observable."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
    )


def _recombinant_gamete_fraction(species: nt.Species) -> float:
    """Frequency of the A1B2 gamete from the A1/B1 x A2/B2 genotype."""
    chrom = species.chromosomes[0]
    loc_a, loc_b = chrom.loci[0], chrom.loci[1]

    from natal.frontend.genetics.entities.gene import Gene
    from natal.frontend.genetics.entities.haplotype import HaploidGenotype, Haplotype

    mat = HaploidGenotype(species=species, haplotypes=[
        Haplotype(chromosome=chrom, genes=[Gene("A1", locus=loc_a), Gene("B1", locus=loc_b)])
    ])
    pat = HaploidGenotype(species=species, haplotypes=[
        Haplotype(chromosome=chrom, genes=[Gene("A2", locus=loc_a), Gene("B2", locus=loc_b)])
    ])
    rec = HaploidGenotype(species=species, haplotypes=[
        Haplotype(chromosome=chrom, genes=[Gene("A1", locus=loc_a), Gene("B2", locus=loc_b)])
    ])
    genotype = nt.Genotype(species, mat, pat)
    return float(genotype.produce_gametes()[rec])


def test_unchanged_dependencies_reuse_cached_baseline() -> None:
    """Back-to-back acquisitions with no edits return the same object."""
    s = _double_heterozygote_species("cr7_hit")
    first = s.get_config_blueprint()
    assert s.get_config_blueprint() is first


def test_recombination_rate_change_is_picked_up() -> None:
    """Setter writes change the recombinant fraction on the next build.

    Numerical contract: double heterozygote with rate r produces each
    recombinant gamete at r/2 — 0.05 at r=0.1, 0.25 at r=0.5.
    """
    s = _double_heterozygote_species("cr7_recomb")
    chrom = s.chromosomes[0]
    loc_a, loc_b = chrom.loci[0], chrom.loci[1]
    chrom.set_recombination(loc_a, loc_b, 0.1)
    baseline_r01 = s.get_config_blueprint()
    assert _recombinant_gamete_fraction(s) == pytest.approx(0.05)

    chrom.set_recombination(loc_a, loc_b, 0.5)
    baseline_r05 = s.get_config_blueprint()
    # The rebuild replaced the cache entry instead of mutating it in place.
    assert baseline_r05 is not baseline_r01
    assert baseline_r05 is s.get_config_blueprint()
    assert _recombinant_gamete_fraction(s) == pytest.approx(0.25)


def test_recombination_view_and_bulk_writes_are_detected() -> None:
    """Direct writes into the rate storage invalidate the baseline too.

    Covers the map setter's storage array written through a plain view,
    np.asarray, np.copyto, and a ufunc out= — plus the bulk setter API.
    """

    base = _double_heterozygote_species("cr7_writes")
    reference = base.get_config_blueprint()

    write_styles = {
        "asarray_view": lambda rates: np.asarray(rates).fill(0.5),
        "copyto": lambda rates: np.copyto(rates, 0.5),
        "ufunc_out": lambda rates: np.multiply(rates, 1.0 + 0.0, out=rates),
        "slice_view": lambda rates: rates.__setitem__(slice(None), 0.5),
        "bulk_setter": None,  # handled separately below
    }

    for i, (label, mutate) in enumerate(write_styles.items()):
        s = _double_heterozygote_species(f"cr7_writes_{i}")
        chrom = s.chromosomes[0]
        if label == "bulk_setter":
            loc_a, loc_b = chrom.loci[0], chrom.loci[1]
            chrom.set_recombination_bulk({(loc_a, loc_b): 0.5})
        else:
            s.get_config_blueprint()
            mutate(chrom.recombination_map._rates)
        rebuilt = s.get_config_blueprint()
        assert rebuilt is not reference, label


def test_label_change_rebuilds_axis() -> None:
    """A gamete-label change must not leave the old single-glab axis."""
    s = _double_heterozygote_species("cr7_labels")
    baseline = s.get_config_blueprint()
    assert baseline["n_glabs"] == 1

    s.gamete_labels = ["default", "tagged"]
    rebuilt = s.get_config_blueprint()
    assert rebuilt is not baseline
    assert rebuilt["n_glabs"] == 2
    assert rebuilt["n_gtypes"] == 2 * baseline["n_gtypes"]


def test_structure_change_rebuilds() -> None:
    """Adding a locus or flipping unordered rebuilds the baseline."""
    s = _double_heterozygote_species("cr7_struct")
    first = s.get_config_blueprint()

    from natal.frontend.genetics.entities.gene import Gene

    s.get_chromosome("chr1").add_locus("locC")
    loc_c = s.get_locus("locC")
    assert loc_c is not None
    loc_c.add_alleles([Gene("C1", locus=loc_c), Gene("C2", locus=loc_c)])
    second = s.get_config_blueprint()
    assert second is not first

    third = second
    s._unordered = not s._unordered
    fourth = s.get_config_blueprint()
    assert fourth is not third


def test_corrupted_cache_rebuilds_instead_of_reusing_or_failing() -> None:
    """Illegal values written into the cached arrays trigger a rebuild."""
    s = _double_heterozygote_species("cr7_corrupt")
    first = s.get_config_blueprint()
    first["zygotes_to_gametes_map"].fill(np.nan)

    rebuilt = s.get_config_blueprint()
    assert rebuilt is not first
    assert bool(np.all(np.isfinite(rebuilt["zygotes_to_gametes_map"])))
    assert bool(np.all(np.isfinite(s.get_config_blueprint()["zygotes_to_gametes_map"])))


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (lambda bp: bp.__setitem__("offspring_tensor", np.zeros((2, 2, 1))), "has shape"),
        (lambda bp: bp["zygotes_to_gametes_map"].fill(-0.5), "negative values"),
        (lambda bp: bp["zygotes_to_gametes_map"].fill(1.5), "probabilities > 1"),
        (lambda bp: bp.__setitem__("n_ztypes", 99), "catalogs disagree"),
    ],
)
def test_each_corruption_guard_forces_rebuild(corrupt: Callable[[dict], None], match: str) -> None:
    """Every content-validation guard rejects a corrupted cache entry.

    A cache hit validates shapes, finiteness, probability ranges, and
    catalog agreement before reuse; each violated guard must trigger a
    clean rebuild rather than reuse or a permanent failure.
    """
    s = _double_heterozygote_species(f"cr7_guard_{match[:12].replace(' ', '_')}")
    first = s.get_config_blueprint()
    # The runtime object is a plain dict; the cast only silences the
    # TypedDict value types so the test can corrupt it like a hostile
    # writer would.
    corrupt(cast("dict", first))

    rebuilt = s.get_config_blueprint()
    assert rebuilt is not first, match
    # The rebuilt baseline is fully legal again and cached for reuse.
    assert s.get_config_blueprint() is rebuilt


def test_existing_population_is_isolated_from_baseline_rebuild() -> None:
    """A species edit never rewrites a population built from the old baseline."""
    species = _double_heterozygote_species("cr7_isolated")
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name="cr7_pop", stochastic=False
        )
        .initial_state(individual_count={"female": {"A1/B1|A1/B1": 50}, "male": {"A1/B1|A1/B1": 50}})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )
    before = pop.config.zygotes_to_gametes_map.copy()

    chrom = species.chromosomes[0]
    chrom.set_recombination(chrom.loci[0], chrom.loci[1], 0.5)
    species.get_config_blueprint()  # force a rebuild

    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, before)
