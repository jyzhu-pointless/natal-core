"""Independent equivalent-input contracts for unified pattern entries."""

import pytest

import natal as nt
from natal.frontend.patterns.parser import PatternParseError


@pytest.mark.parametrize('kind,text', [('ztype', 'WT|WT'), ('gtype', 'WT')])
def test_structured_target_keeps_required_label_contract(kind, text):
    """A parsed object cannot bypass a label required of equivalent text."""
    species = nt.Species.from_dict(f'independent_target_label_{kind}', {'c': {'l': ['WT']}})
    pattern = nt.parse_selector(text, species=species, kind=kind)
    for target in (text, pattern):
        with pytest.raises(PatternParseError):
            nt.parse_target(target, species=species, haploid=kind == 'gtype', require_label=True)


def test_registry_subset_retains_species_label_validation_context():
    """A valid absent label in a negation still selects surviving slab rows."""
    from natal.frontend.registry.index import IndexRegistry

    species = nt.Species.from_dict(
        'independent_selector_label_catalog', {'c': {'l': ['WT']}},
        somatic_labels=['default', 'infected'],
    )
    registry = IndexRegistry()
    registry.register_ztype(species.get_genotype_from_str('WT|WT'), 'default')
    selector = nt.IndividualSelector(ztype='WT|WT@!infected')
    actual = selector.compile_coordinates(registry, n_sexes=2, n_ages=3)
    expected = nt.IndividualSelector(ztype='WT|WT@default').compile_coordinates(
        registry, n_sexes=2, n_ages=3,
    )
    import numpy as np

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('kind', ['ztype', 'gtype'])
@pytest.mark.parametrize('suffix', ['', '@marked'])
def test_structured_target_matches_text_replacement_and_preservation(kind, suffix):
    """Structured conversion preserves exactly the same source parts as text."""
    species = nt.Species.from_dict(
        f'independent_structured_target_{kind}_{bool(suffix)}',
        {'c': {'l': ['A', 'B']}}, unordered=False,
        somatic_labels=['default', 'marked'], gamete_labels=['default', 'marked'],
    )
    text = ('B|*' if kind == 'ztype' else 'B') + suffix
    pattern = nt.parse_selector(text, species=species, kind=kind)
    structured = nt.parse_target(pattern, species=species, haploid=kind == 'gtype', validate=True)
    direct = nt.parse_target(text, species=species, haploid=kind == 'gtype', validate=True)
    if kind == 'ztype':
        source = species.get_genotype_from_str('A|A')
        actual = structured.apply_zygote(source, 'default', species)
        expected = direct.apply_zygote(source, 'default', species)
        assert str(actual[0]) == 'B|A'
    else:
        source = species.get_haploid_genotype_from_str('A')
        actual = structured.apply_gamete(source, 'default', species)
        expected = direct.apply_gamete(source, 'default', species)
        assert str(actual[0]) == 'B'
    assert actual == expected
    assert actual[1] == ('marked' if suffix else 'default')
    with pytest.raises(TypeError):
        nt.parse_target(pattern, species=species, haploid=kind != 'gtype')
    if kind == 'ztype':
        unbound = type(pattern)(pattern.genotype, pattern.slab)
    else:
        unbound = type(pattern)(pattern.genome, pattern.glab)
    with pytest.raises(ValueError, match='source pattern text'):
        nt.parse_target(unbound, species=species, haploid=kind == 'gtype')


@pytest.mark.parametrize('kind,text', [('genotype', 'A|A'), ('haploid', 'A')])
def test_content_selector_strict_allele_validation(kind, text):
    """Strict binding accepts catalog content and rejects misspelled alleles."""
    species = nt.Species.from_dict(f'independent_strict_content_{kind}', {'c': {'l': ['A']}})
    assert nt.parse_selector(text, species=species, kind=kind, validate_alleles=True) is not None
    with pytest.raises(ValueError, match='unknown allele'):
        nt.parse_selector(text.replace('A', 'missing'), species=species, kind=kind, validate_alleles=True)


@pytest.mark.parametrize('label', ['', 'marked'])
def test_exact_tuple_selector_export_and_target_preserve_genotype_label(label):
    """Tuple identity input remains a usable exact target and readable export."""
    species = nt.Species.from_dict(
        f'independent_tuple_target_{label}', {'c': {'l': ['A', 'B']}},
        unordered=False, somatic_labels=['default', 'marked'],
    )
    genotype = species.get_genotype_from_str('B|A')
    selector = nt.IndividualSelector(ztype=(genotype, label), sex='female', age=1)
    expected = f'B|A@{label}' if label else 'B|A'
    assert selector.to_dict()['atoms'][0]['ztype'] == [expected]
    sex, age, target = selector.as_target_spec()
    assert (sex, age, target) == (0, 1, expected)
    compiled = nt.parse_target(target, species=species, validate=True)
    assert compiled.apply_zygote(species.get_genotype_from_str('A|A'), 'default', species) == (genotype, label or 'default')
    structured_selector = nt.IndividualSelector(ztype=nt.parse_selector(expected, species=species))
    assert isinstance(structured_selector.to_dict()['atoms'][0]['ztype'][0], str)
