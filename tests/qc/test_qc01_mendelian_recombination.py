"""QC spot-checks 1-3: Mendelian segregation and recombination numerics.

Random quality-control tests (not part of the regular suite).  Each test
states the claim, the independent reference value, and the wrong result
it would reject.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _one_locus_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "a"]}},
        gamete_labels=["default"],
        unordered=False,
    )


def _build_neutral(pop_name: str, species: nt.Species, female_key: object, male_key: object):
    """Deterministic neutral discrete population, two eggs per female.

    Two eggs per female keep the total stationary: only females reproduce,
    so the replacement rate is eggs/2 per capita.
    """
    return (
        nt.DiscreteGenerationPopulation.setup(species=species, name=pop_name, stochastic=False)
        .initial_state(
            individual_count={"female": {female_key: 500}, "male": {male_key: 500}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                     growth_mode="no_competition")
        .build()
    )


class TestMendelianSegregation:
    def test_het_x_het_gives_exact_1_2_1(self) -> None:
        """Claim: deterministic het x het cross yields exact 1:2:1 per sex.

        Reference: Mendelian segregation with eggs_per_female=2 and total
        replacement (500 females -> 1000 zygotes, 500 per sex).  Rejects
        segregation distortion, phase loss, or off-by-one count books.
        """
        species = _one_locus_species("qc_mendel")
        pop = _build_neutral("qc_mendel_pop", species, "WT|a", "WT|a")
        pop.run(1)

        counts = pop.state.individual_count
        assert float(counts.sum()) == 1000.0
        assert float(counts[0].sum()) == 500.0
        assert float(counts[1].sum()) == 500.0

        registry = pop.registry
        per_phase: dict[str, float] = {}
        for genotype, slab in registry.index_to_ztype:
            idx = registry.ztype_index(genotype, slab)
            per_phase[genotype.to_string()] = float(counts[:, :, idx].sum())
        # Ordered species: both heterozygote phases must appear at 250.
        assert per_phase["WT|WT"] == 250.0
        assert per_phase["WT|a"] == 250.0
        assert per_phase["a|WT"] == 250.0
        assert per_phase["a|a"] == 250.0

    def test_totals_conserved_over_neutral_ticks(self) -> None:
        """Claim: neutral WF replacement keeps totals exactly constant."""
        species = _one_locus_species("qc_mendel2")
        pop = _build_neutral("qc_mendel2_pop", species, "WT|a", "a|WT")
        totals = []
        for _ in range(5):
            pop.run(1)
            totals.append(float(pop.state.individual_count.sum()))
        assert totals == [1000.0] * 5


class TestRecombinationGametes:
    def _two_locus_species(self, name: str) -> nt.Species:
        sp = nt.Species.from_dict(
            name=name,
            structure={"chr1": {"locA": ["A1", "A2"], "locB": ["B1", "B2"]}},
            gamete_labels=["default"],
        )
        chr1 = sp.get_chromosome("chr1")
        chr1.set_recombination_rate("locA", "locB", 0.1)
        return sp

    def test_coupling_double_het_gametes(self) -> None:
        """Claim: AB/ab heterozygote with r=0.1 gives parentals (1-r)/2 each.

        Reference: standard meiosis, no interference: AB=ab=0.45,
        Ab=aB=0.05.  Rejects a swapped (r vs 1-r) mapping and unnormalized
        or missing gamete classes (the issue-#43 single-start drop).
        """
        sp = self._two_locus_species("qc_rec_coupling")
        gt = sp.get_genotype_from_str("A1/B1|A2/B2")
        gametes = {str(k): v for k, v in gt.produce_gametes().items()}
        assert abs(sum(gametes.values()) - 1.0) < 1e-12
        assert gametes["A1/B1"] == pytest.approx(0.45, abs=1e-12)
        assert gametes["A2/B2"] == pytest.approx(0.45, abs=1e-12)
        assert gametes["A1/B2"] == pytest.approx(0.05, abs=1e-12)
        assert gametes["A2/B1"] == pytest.approx(0.05, abs=1e-12)

    def test_repulsion_double_het_gametes(self) -> None:
        """Claim: Ab/aB heterozygote mirrors the coupling case."""
        sp = self._two_locus_species("qc_repulsion")
        gt = sp.get_genotype_from_str("A1/B2|A2/B1")
        gametes = {str(k): v for k, v in gt.produce_gametes().items()}
        assert gametes["A1/B2"] == pytest.approx(0.45, abs=1e-12)
        assert gametes["A2/B1"] == pytest.approx(0.45, abs=1e-12)
        assert gametes["A1/B1"] == pytest.approx(0.05, abs=1e-12)
        assert gametes["A2/B2"] == pytest.approx(0.05, abs=1e-12)

    def test_zero_rate_freezes_parental_phase(self) -> None:
        """Claim: r=0 yields only the two parental gametes at 0.5."""
        sp = self._two_locus_species("qc_rec_zero")
        chr1 = sp.get_chromosome("chr1")
        chr1.set_recombination_rate("locA", "locB", 0.0)
        gt = sp.get_genotype_from_str("A1/B1|A2/B2")
        gametes = {str(k): v for k, v in gt.produce_gametes().items()}
        assert set(gametes) == {"A1/B1", "A2/B2"}
        assert gametes["A1/B1"] == pytest.approx(0.5, abs=1e-12)

    def test_free_recombination_uniform(self) -> None:
        """Claim: r=0.5 gives all four gametes at 0.25."""
        sp = self._two_locus_species("qc_rec_free")
        chr1 = sp.get_chromosome("chr1")
        chr1.set_recombination_rate("locA", "locB", 0.5)
        gt = sp.get_genotype_from_str("A1/B1|A2/B2")
        gametes = {str(k): v for k, v in gt.produce_gametes().items()}
        assert sorted(gametes.values()) == pytest.approx([0.25] * 4, abs=1e-12)


class TestRecombinationPatternEnumerator:
    def test_three_loci_matches_bruteforce(self) -> None:
        """Claim: pattern frequencies equal the product crossover reference.

        Reference: freq(pattern) = prod over intervals (r_i on crossover,
        1-r_i otherwise), single maternal-chain start.  Rejects index or
        factor swaps in the 2^(n-1) enumerator.
        """
        rates = np.array([0.2, 0.3])
        patterns, freqs = nt.compute_recombinant_haplotypes(3, rates)
        assert abs(freqs.sum() - 1.0) < 1e-12
        assert len(patterns) == 4
        for pattern, freq in zip(patterns, freqs, strict=True):
            expected = 1.0
            for i in range(2):
                expected *= rates[i] if pattern[i + 1] != pattern[i] else 1.0 - rates[i]
            assert freq == pytest.approx(expected, abs=1e-15), (pattern, freq, expected)
