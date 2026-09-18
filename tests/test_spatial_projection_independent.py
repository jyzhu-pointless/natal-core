"""Independent contracts for ordered spatial declarations."""

import itertools

import pytest

import natal as nt


def test_partial_uniform_override_keeps_other_batch_dimensions():
    """A surviving batch field must not replay a superseded sibling field."""
    species = nt.Species.from_dict('independent_spatial_partial_override', {'c': {'l': ['WT']}})
    population = (nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured')
                  .age_structure(3, 1)
                  .reproduction(
                      eggs_per_female=nt.batch_setting([10., 20.]),
                      female_age_based_mating_rate=nt.batch_setting([[0., .2, .2], [0., .4, .4]]),
                  )
                  .reproduction(eggs_per_female=30.)
                  .build())
    assert [float(deme.config.eggs_per_female) for deme in population.demes] == [30., 30.]
    assert [float(deme.config.age_based_mating_rates[0, 1]) for deme in population.demes] == [.2, .4]


def test_spatial_projection_owns_nested_initial_declaration():
    """Per-deme projection must use the accepted input, not later caller edits."""
    species = nt.Species.from_dict('independent_spatial_initial_ownership', {'c': {'l': ['WT']}})
    distribution = {'female': {'WT|WT': {1: 10}}}
    builder = (nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured')
               .age_structure(3, 1)
               .initial_state(individual_count=distribution)
               .reproduction(eggs_per_female=nt.batch_setting([10., 20.])))
    distribution['female']['WT|WT'][1] = 99
    population = builder.build()
    assert [deme.get_total_count() for deme in population.demes] == [10., 10.]


def test_uniform_override_through_another_method_preserves_call_order():
    """Two APIs writing the same field retain their declared last-write order."""
    species = nt.Species.from_dict('independent_spatial_cross_method', {'c': {'l': ['WT']}})
    population = (nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured')
                  .age_structure(3, 1)
                  .reproduction(fixed_egg_count=nt.batch_setting([True, False]))
                  .setup(fixed_egg_count=True)
                  .build())
    assert [deme.config.fixed_egg_count for deme in population.demes] == [True, True]


def test_spatial_projection_owns_nested_batched_initial_values():
    """Batch wrapping does not transfer ownership of caller count mappings."""
    species = nt.Species.from_dict('independent_spatial_batch_ownership', {'c': {'l': ['WT']}})
    first = {'female': {'WT|WT': {1: 10}}}
    second = {'female': {'WT|WT': {1: 20}}}
    builder = (nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured')
               .age_structure(3, 1)
               .initial_state(individual_count=nt.batch_setting([first, second])))
    first['female']['WT|WT'][1] = 99
    second['female']['WT|WT'][1] = 88
    assert [deme.get_total_count() for deme in builder.build().demes] == [10., 20.]


def test_spatial_resolves_initial_age_on_final_declared_dimensions():
    """Structural ordering uses final age axes without reordering user writes."""
    species = nt.Species.from_dict('independent_spatial_final_ages', {'c': {'l': ['WT']}})
    population = (nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured')
                  .initial_state(individual_count={'female': {'WT|WT': {4: 100}}})
                  .age_structure(5, 2)
                  .reproduction(eggs_per_female=nt.batch_setting([10., 20.]))
                  .build())
    assert [deme.get_total_count() for deme in population.demes] == [100., 100.]
    assert all(deme.config.n_ages == 5 for deme in population.demes)


def test_later_equal_batch_overrides_previous_varying_batch():
    """A later batch can converge deme values without reviving earlier writes."""
    species = nt.Species.from_dict('independent_spatial_batch_converges', {'c': {'l': ['WT']}})
    population = (nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured')
                  .age_structure(3, 1)
                  .reproduction(eggs_per_female=nt.batch_setting([10., 20.]))
                  .reproduction(eggs_per_female=nt.batch_setting([30., 30.]))
                  .build())
    assert [float(deme.config.eggs_per_female) for deme in population.demes] == [30., 30.]


def test_counts_override_preserves_differing_batched_sperm():
    """A count override cannot turn a remaining sperm declaration into None counts."""
    species = nt.Species.from_dict('independent_spatial_partial_initial', {'c': {'l': ['WT']}})
    def counts(value):
        return {'female': {'WT|WT': {1: value}}}
    def sperm(value):
        return {'WT|WT': {'WT|WT': {1: value}}}
    population = (nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured')
                  .age_structure(3, 1)
                  .initial_state(
                      individual_count=nt.batch_setting([counts(10), counts(20)]),
                      sperm_storage=nt.batch_setting([sperm(1), sperm(2)]),
                  )
                  .initial_state(individual_count=counts(30))
                  .build())
    for index, expected_sperm in enumerate((1, 2)):
        state = population._deme_object(index).state
        assert state.individual_count.sum() == 30
        assert state.sperm_storage.sum() == expected_sperm


_INITIAL_MODES = list(itertools.product(('ordinary', 'equal_batch', 'varying_batch'), ('omitted', 'ordinary', 'equal_batch', 'varying_batch')))


@pytest.mark.parametrize(
    'first,second', itertools.product(_INITIAL_MODES, repeat=2),
    ids=[f'{a[0]}-{a[1]}_then_{b[0]}-{b[1]}' for a, b in itertools.product(_INITIAL_MODES, repeat=2)],
)
def test_initial_declaration_combinations_preserve_last_explicit_values(first, second):
    """Every two-call count/sperm combination follows the same per-field order."""
    species = nt.Species.from_dict('independent_spatial_initial_matrix', {'c': {'l': ['WT']}})
    builder = nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured').age_structure(3, 1)
    expected_sperm = [0, 0]
    expected_counts = [0, 0]
    for stage, (count_mode, sperm_mode) in enumerate((first, second), start=1):
        count_base = stage * 10
        expected_counts = [count_base, count_base + (1 if count_mode == 'varying_batch' else 0)]
        count_values = [{'female': {'WT|WT': {1: number}}} for number in expected_counts]
        count_input = count_values[0] if count_mode == 'ordinary' else nt.batch_setting(count_values)
        if sperm_mode == 'omitted':
            builder.initial_state(individual_count=count_input)
        else:
            expected_sperm = [stage, stage + (1 if sperm_mode == 'varying_batch' else 0)]
            sperm_values = [{'WT|WT': {'WT|WT': {1: number}}} for number in expected_sperm]
            sperm_input = sperm_values[0] if sperm_mode == 'ordinary' else nt.batch_setting(sperm_values)
            builder.initial_state(individual_count=count_input, sperm_storage=sperm_input)
    population = builder.build()
    for index in range(2):
        state = population._deme_object(index).state
        assert state.individual_count.sum() == expected_counts[index]
        assert state.sperm_storage.sum() == expected_sperm[index]
