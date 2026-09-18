"""Independent linked-phase identity and exact Mendelian map contracts.

An unordered chromosome pair has H(H+1)/2 states for H haplotypes.
Independent chromosomes multiply their state counts. Meiosis at two linked
loci gives each parental haplotype (1-r)/2 and each recombinant r/2.
"""

import itertools

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry


def _linked(name, *, unordered=True):
    return nt.Species.from_dict(
        name, {'linked': {'la': ['A', 'a'], 'lb': ['B', 'b']}},
        unordered=unordered,
    )


def test_first_reverse_construction_stores_canonical_phase_and_reuses_identity():
    """Cache identity and stored parents agree even before any enumeration."""
    species = _linked('phase_independent_reverse_first')
    ab = species.get_haploid_genotype_from_str('A/b')
    a_b = species.get_haploid_genotype_from_str('a/B')
    reverse = nt.Genotype(species, a_b, ab)
    assert reverse.to_string() == 'A/b|a/B'
    assert reverse.maternal is ab and reverse.paternal is a_b
    forward = nt.Genotype(species, ab, a_b)
    assert reverse is forward
    coupling = species.get_genotype_from_str('A/B|a/b')
    assert coupling is not reverse
    assert len(species.get_all_genotypes(unordered=True)) == 10


def test_independent_chromosomes_canonicalize_separately_and_keep_nine_states():
    """Three unordered states at each independent locus give 3*3 states."""
    species = nt.Species.from_dict(
        'phase_independent_unlinked', {'one': {'la': ['A', 'a']}, 'two': {'lb': ['B', 'b']}},
    )
    objects = [species.get_genotype_from_str(text) for text in (
        'A|a;B|b', 'a|A;B|b', 'A|a;b|B', 'a|A;b|B',
    )]
    assert all(genotype is objects[0] for genotype in objects)
    assert len(species.get_all_genotypes(unordered=True)) == 9


@pytest.mark.parametrize('compress', [False, True])
def test_initial_linked_phases_keep_separate_indices_and_exact_counts(compress):
    """Two explicitly seeded phases retain 100 each, including compression."""
    species = _linked(f'phase_independent_initial_{compress}')
    population = (nt.DiscreteGenerationPopulation.setup(species, stochastic=False, compress=compress)
                  .initial_state(individual_count={'female': {'A/B|a/b': 100, 'A/b|a/B': 100}})
                  .build())
    indices = [population.registry.ztype_index(species.get_genotype_from_str(text), 'default')
               for text in ('A/B|a/b', 'A/b|a/B')]
    assert indices[0] != indices[1]
    counts = population.state.individual_count
    assert [counts[0, :, index].sum() for index in indices] == [100, 100]
    assert counts.sum() == 200


def test_fertilization_enters_distinct_linked_phase_slots():
    """Each deterministic gamete pair makes its own phase, with unit mass."""
    species = _linked('phase_independent_fertilization')
    blueprint = species.get_config_blueprint()
    registry = build_registry(species)
    matrix = blueprint['gametes_to_zygotes_map']
    targets = []
    for maternal, paternal, target in (
        ('A/B', 'a/b', 'A/B|a/b'), ('A/b', 'a/B', 'A/b|a/B'),
    ):
        m = registry.gtype_index(species.get_haploid_genotype_from_str(maternal), 'default')
        p = registry.gtype_index(species.get_haploid_genotype_from_str(paternal), 'default')
        z = registry.ztype_index(species.get_genotype_from_str(target), 'default')
        targets.append(z)
        expected = np.zeros(registry.n_ztypes)
        expected[z] = 1.0
        np.testing.assert_array_equal(matrix[m, p], expected)
        np.testing.assert_array_equal(matrix[p, m], expected)
    assert targets[0] != targets[1]


@pytest.mark.parametrize('rate', [0., .3, .5])
def test_meiosis_retains_phase_and_analytical_recombination_probabilities(rate):
    """Four probabilities follow meiosis, even when r=.5 makes rows equal."""
    species = _linked(f'phase_independent_meiosis_{rate}')
    species.get_chromosome('linked').set_recombination_rate('la', 'lb', rate)
    coupling = species.get_genotype_from_str('A/B|a/b')
    repulsion = species.get_genotype_from_str('A/b|a/B')
    assert coupling is not repulsion
    assert len(species.get_all_genotypes(unordered=True)) == 10
    names = ('A/B', 'A/b', 'a/B', 'a/b')
    blueprint = species.get_config_blueprint()
    registry = build_registry(species)
    for genotype, expected in (
        (coupling, [(1-rate)/2, rate/2, rate/2, (1-rate)/2]),
        (repulsion, [rate/2, (1-rate)/2, (1-rate)/2, rate/2]),
    ):
        gametes = {str(key): value for key, value in genotype.produce_gametes().items()}
        # A handful of additions/multiplications on unit probabilities: 16 eps
        # comfortably bounds roundoff and cannot hide the r vs 1-r phase swap.
        np.testing.assert_allclose([gametes.get(name, 0.) for name in names], expected,
                                   rtol=0, atol=16*np.finfo(float).eps)
        z = registry.ztype_index(genotype, 'default')
        columns = [registry.gtype_index(species.get_haploid_genotype_from_str(name), 'default') for name in names]
        for sex in (0, 1):
            np.testing.assert_allclose(blueprint['zygotes_to_gametes_map'][sex, z, columns], expected,
                                       rtol=0, atol=16*np.finfo(float).eps)


def test_ordered_linked_species_keeps_all_sixteen_parental_pairs():
    """Ordered mode keeps 4*4 identities without any homolog exchange."""
    species = _linked('phase_independent_ordered', unordered=False)
    haplotypes = species.get_all_haploid_genotypes()
    genotypes = [nt.Genotype(species, maternal, paternal)
                 for maternal, paternal in itertools.product(haplotypes, repeat=2)]
    assert len({id(genotype) for genotype in genotypes}) == 16
    assert len(species.get_all_genotypes(unordered=False)) == 16
    for genotype, (maternal, paternal) in zip(genotypes, itertools.product(haplotypes, repeat=2)):
        assert genotype.maternal is maternal
        assert genotype.paternal is paternal


@pytest.mark.parametrize('same,other', [('X', 'Y'), ('Z', 'W')])
def test_sex_chromosomes_keep_linked_phase_and_heterotypic_direction(same, other):
    """Same-type homologs preserve phase; unlike sex chromosomes never swap."""
    species = nt.Species.from_dict(
        f'phase_independent_sex_{same}', {
            same: {'sex_type': same, 'loci': {'la': ['A', 'a'], 'lb': ['B', 'b']}},
            other: {'sex_type': other, 'loci': {'sex_marker': ['O']}},
            'autosome': {'lc': ['C', 'c']},
        },
    )
    coupling = species.get_genotype_from_str('C|C;A/B|a/b')
    repulsion = species.get_genotype_from_str('C|C;A/b|a/B')
    reverse = species.get_genotype_from_str('C|C;a/B|A/b')
    assert coupling is not repulsion
    assert reverse is repulsion
    maternal = species.get_haploid_genotype_from_str('c;A/B')
    paternal = species.get_haploid_genotype_from_str('C;O')
    forward = nt.Genotype(species, maternal, paternal)
    backward = nt.Genotype(species, paternal, maternal)
    assert forward is not backward
    assert str(forward.maternal) == 'C;A/B'
    assert str(forward.paternal) == 'c;O'
    assert str(backward.maternal) == 'C;O'
    assert str(backward.paternal) == 'c;A/B'


def test_mixed_chromosomes_have_ten_times_three_distinct_states():
    """Linked diplotypes multiply with an independent single-locus genotype."""
    species = nt.Species.from_dict(
        'phase_independent_mixed', {
            'linked': {'la': ['A', 'a'], 'lb': ['B', 'b']},
            'independent': {'lc': ['C', 'c']},
        },
    )
    genotypes = species.get_all_genotypes(unordered=True)
    assert len(genotypes) == len({id(genotype) for genotype in genotypes}) == 30
