"""Review regressions for CR-1 filter validation and XY gamete matching."""

from types import SimpleNamespace

import numpy as np
import pytest

import natal as nt
from natal.frontend.builder._registry_builder import build_registry
from natal.frontend.genetics.compile import project_mendelian_maps
from natal.frontend.modifiers.gamete_conversion import GameteConversionRuleSet
from natal.frontend.modifiers.zygote_conversion import ZygoteConversionRuleSet


@pytest.mark.parametrize(
    ("stage", "key", "pattern"),
    [
        ("gamete", "current", "A@typo"),
        ("gamete", "parent", "A|A@typo"),
        ("zygote", "current", "A|A@typo"),
        ("zygote", "maternal", "A@typo"),
        ("zygote", "paternal", "A@typo"),
        ("gamete", "current", "A@"),
        ("zygote", "maternal", "A@"),
        ("gamete", "current", "{A"),
    ],
)
def test_invalid_conversion_filter_is_rejected(
    stage: str, key: str, pattern: str,
) -> None:
    """Misspelled labels and malformed patterns must not become no-ops."""
    species = nt.Species.from_dict(
        name="review_filter_validation",
        structure={"c": {"L": ["A", "B"]}},
        gamete_labels=["default", "I"],
        somatic_labels=["default", "I"],
    )
    host = SimpleNamespace(species=species, registry=build_registry(species))
    if stage == "gamete":
        rules = GameteConversionRuleSet().add_gtype_convert(
            to="B@*", rate=1.0, filters={key: pattern},
        )
        with pytest.raises(ValueError):
            rules.to_gamete_modifier(host)
    else:
        zrules = ZygoteConversionRuleSet().add_ztype_convert(
            to="B|B@*", rate=1.0, filters={key: pattern},
        )
        with pytest.raises(ValueError):
            zrules.to_zygote_modifier(host)


@pytest.mark.parametrize("stage", ["gamete", "zygote"])
def test_xy_identity_conversion_with_x_filter_preserves_baseline(stage: str) -> None:
    """Y gametes must fail an X filter normally, not raise missing-X errors."""
    species = nt.Species.from_dict(
        name="review_xy_filter_identity",
        structure={
            "cx": {"sex_type": "X", "loci": {"LX": ["X"]}},
            "cy": {"sex_type": "Y", "loci": {"LY": ["Y"]}},
        },
        unordered=False,
    )
    registry = build_registry(species)
    host = SimpleNamespace(species=species, registry=registry)
    meiosis, fertilization = project_mendelian_maps(species, registry)
    if stage == "gamete":
        rules = GameteConversionRuleSet().add_gtype_convert(
            to="*@*", rate=1.0, filters={"current": "X@*"},
        )
        rows = rules.to_gamete_modifier(host)()
        expected = meiosis
    else:
        zrules = ZygoteConversionRuleSet().add_ztype_convert(
            to="*@*", rate=1.0, filters={"paternal": "X@*"},
        )
        rows = zrules.to_zygote_modifier(host)()
        expected = fertilization
    actual = np.zeros_like(expected)
    for (first, second), row in rows.items():
        for target, probability in row.items():
            actual[first, second, target] = probability
    np.testing.assert_array_equal(actual, expected)
