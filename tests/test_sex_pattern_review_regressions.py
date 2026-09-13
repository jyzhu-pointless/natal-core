"""Independent sex-group matching regressions for CR-0."""

import pytest

import natal as nt


@pytest.mark.parametrize(
    ("types", "genotype_text"),
    [(('X', 'Y'), 'X|Y'), (('Z', 'W'), 'W|Z')],
)
def test_exact_sex_group_pattern_matches_its_genotype(
    types: tuple[str, str], genotype_text: str,
) -> None:
    """A precise XY/ZW pattern must match the same valid genotype."""
    species = nt.Species.from_dict(
        name=f"review_sex_pattern_{types[0]}",
        structure={
            kind: {"sex_type": kind, "loci": {f"L{kind}": [kind]}}
            for kind in types
        },
        unordered=False,
    )
    genotype = species.get_genotype_from_str(genotype_text)
    assert nt.GenotypePatternParser(species).parse(genotype_text).matches(genotype)
