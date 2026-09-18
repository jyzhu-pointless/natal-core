"""Independent ownership, memoization, and dependency-phase contracts."""

import numpy as np
import pytest

import natal as nt
from natal.frontend.model import initial_state as initial_module
from natal.frontend.model.dependency_graph import DependencyGraph, DerivationPipeline


def test_initial_sperm_input_and_exported_containers_are_isolated():
    """Neither caller containers nor returned declaration copies alter a build."""
    species = nt.Species.from_dict('independent_initial_sperm', {'c': {'l': ['WT']}})
    counts = {'female': {'WT|WT': [0, 10, 0]}}
    sperm = {'WT|WT': {'WT|WT': {1: 4}}}
    builder = (nt.AgeStructuredPopulation.setup(species).age_structure(3, 1)
               .initial_state(individual_count=counts, sperm_storage=sperm))
    declaration = builder._definition_for_compile().initial_distribution
    counts['female']['WT|WT'][1] = 90
    sperm['WT|WT']['WT|WT'][1] = 80
    exported_counts = declaration.individual_count
    exported_sperm = declaration.sperm_storage
    exported_counts['female']['WT|WT'][1] = 70
    exported_sperm['WT|WT']['WT|WT'][1] = 60
    frozen_counts, frozen_sperm = declaration.resolve(
        species, discrete_generation=False, n_ages=3, new_adult_age=1,
    )
    assert frozen_counts.sum() == 10
    assert frozen_sperm.sum() == 4
    with pytest.raises(ValueError):
        frozen_sperm.setflags(write=True)
    population = builder.age_structure(4, 1).build()
    assert population.state.individual_count.sum() == 10
    assert population.state.sperm_storage.sum() == 4


def test_repeated_build_reuses_initial_resolution_without_sharing_live_state(monkeypatch):
    """Same dimensions resolve once while each built population owns its arrays."""
    species = nt.Species.from_dict('independent_initial_memo', {'c': {'l': ['WT']}})
    original = initial_module.resolve_age_structured_initial_individual_count
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(initial_module, 'resolve_age_structured_initial_individual_count', counted)
    builder = (nt.AgeStructuredPopulation.setup(species).age_structure(3, 1)
               .initial_state(individual_count={'female': {'WT|WT': {1: 10}}}))
    first = builder.build()
    second = builder.build()
    assert len(calls) == 1
    assert not np.shares_memory(first._live_state().individual_count, second._live_state().individual_count)
    first._live_state().individual_count.fill(0)
    assert second.state.individual_count.sum() == 10
    assert builder.build().state.individual_count.sum() == 10
    assert len(calls) == 1


def test_phase_prevalidates_late_dependencies_before_first_compute():
    """A late unmet dependency cannot leave earlier phase effects behind."""
    graph = DependencyGraph(frozenset(), {'first': (), 'external': (), 'last': ('external',)})
    pipeline = DerivationPipeline(graph)
    calls = []
    pipeline.register('first', lambda: calls.append('first'))
    pipeline.register('last', lambda: calls.append('last'))
    with pytest.raises(ValueError, match='unfinished products'):
        pipeline.run(('first', 'last'))
    assert calls == []
    pipeline.run(('first', 'last'), completed=('external',))
    assert calls == ['first', 'last']


def test_definition_spatial_copy_retains_owned_initial_declaration():
    """Copying spatial controls cannot drop or unfreeze initial inputs."""
    species = nt.Species.from_dict('independent_initial_spatial_copy', {'c': {'l': ['WT']}})
    raw = {'female': {'WT|WT': {1: 10}}}
    builder = (nt.AgeStructuredPopulation.setup(species).age_structure(3, 1)
               .initial_state(individual_count=raw))
    original = builder._definition_for_compile()
    copied = original.with_spatial(None)
    raw['female']['WT|WT'][1] = 99
    exported = copied.initial_distribution.individual_count
    exported['female']['WT|WT'][1] = 88
    for definition in (original, copied):
        assert definition.initial_distribution is not None
        counts, _ = definition.initial_distribution.resolve(
            species, discrete_generation=False, n_ages=3, new_adult_age=1,
        )
        assert counts.sum() == 10
        with pytest.raises(ValueError):
            counts.setflags(write=True)


def test_mapping_input_journal_cannot_restore_external_mutations():
    """The accepted Mapping contract includes read-only views of caller data."""
    from types import MappingProxyType

    species = nt.Species.from_dict('independent_initial_mapping_view', {'c': {'l': ['WT']}})
    raw = {'female': {'WT|WT': {1: 10}}}
    builder = (nt.AgeStructuredPopulation.setup(species).age_structure(3, 1)
               .initial_state(individual_count=MappingProxyType(raw)))
    raw['female']['WT|WT'][1] = 99
    assert builder.build().get_total_count() == 10
    assert builder.age_structure(4, 1).build().get_total_count() == 10


@pytest.mark.parametrize('spatial', [False, True])
@pytest.mark.parametrize('build_between', [False, True])
def test_counts_only_redeclaration_retains_sperm(spatial, build_between):
    """Omitted sperm input preserves the accepted distribution regardless of cache."""
    species = nt.Species.from_dict(
        f'independent_sperm_redeclare_{spatial}_{build_between}', {'c': {'l': ['WT']}},
    )
    if spatial:
        builder = nt.SpatialPopulation.builder(species, n_demes=2, pop_type='age_structured')
    else:
        builder = nt.AgeStructuredPopulation.setup(species)
    builder.age_structure(3, 1).initial_state(
        individual_count={'female': {'WT|WT': {1: 10}}},
        sperm_storage={'WT|WT': {'WT|WT': {1: 5}}},
    )
    if spatial:
        builder.reproduction(eggs_per_female=nt.batch_setting([10., 20.]))
    if build_between:
        builder.build()
    builder.initial_state(individual_count={'female': {'WT|WT': {1: 20}}})
    population = builder.build()
    demes = [population._deme_object(i) for i in range(2)] if spatial else [population]
    for deme in demes:
        assert deme.state.individual_count.sum() == 20
        assert deme.state.sperm_storage.sum() == 5


@pytest.mark.parametrize('container', [tuple, np.array])
def test_exported_numeric_initial_sequences_cannot_mutate_declaration(container):
    """Tuple and ndarray exports preserve values without exposing owned input."""
    species = nt.Species.from_dict(f'independent_numeric_export_{container.__name__}', {'c': {'l': ['WT']}})
    declaration = initial_module.InitialDistributionDeclaration.capture(
        {'female': {'WT|WT': container([0., 10., 0.])}}, None,
    )
    exported = declaration.individual_count
    if isinstance(exported['female']['WT|WT'], np.ndarray):
        exported['female']['WT|WT'][1] = 99
    exported['female']['WT|WT'] = (0., 88., 0.)
    counts, _ = declaration.resolve(species, discrete_generation=False, n_ages=3, new_adult_age=1)
    assert counts.sum() == 10
