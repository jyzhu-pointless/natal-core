from __future__ import annotations

import unittest
import uuid

from natal.configurator._params import iter_sexual_selection_entries
from natal.genetics import Species
from natal.utils.helpers import resolve_sex_label


def _make_simple_species() -> Species:
    # Single chromosome, single locus, two alleles -> four diploid ordered genotypes
    return Species.from_dict(
        f"TestSpecies_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["A", "a"],
            }
        },
    )


class TestPopulationBuilderFitnessPatterns(unittest.TestCase):
    def setUp(self) -> None:
        self.simple_species = _make_simple_species()
        # Use unordered genotypes for an unordered species — the
        # resolve_genotype_selectors function auto-promotes | to ::
        # when species.unordered is True, and unordered genotypes
        # are the correct matching target.
        self.all_genotypes = self.simple_species.get_all_genotypes(
            unordered=self.simple_species.unordered
        )

    def test_resolve_genotype_selector_exact_string(self) -> None:
        self.simple_species.resolve_genotype_selectors(
            selector="A|a",
            all_genotypes=self.all_genotypes,
            context="viability",
        )

        matched = self.simple_species.resolve_genotype_selectors(
            selector="A|a",
            all_genotypes=self.all_genotypes,
            context="viability",
        )

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0], self.simple_species.get_genotype_from_str("A|a"))

    def test_resolve_genotype_selector_pattern_string(self) -> None:
        matched = self.simple_species.resolve_genotype_selectors(
            selector="A|*",
            all_genotypes=self.all_genotypes,
            context="fecundity",
        )

        # Maternal haplotype fixed to A, paternal can be A or a
        self.assertEqual(len(matched), 2)

    def test_resolve_genotype_selector_tuple_union(self) -> None:
        matched = self.simple_species.resolve_genotype_selectors(
            selector=("A|a", "a|A"),
            all_genotypes=self.all_genotypes,
            context="viability",
        )

        # A|a and a|A canonicalize to the same genotype → dedup yields 1.
        self.assertEqual(len(matched), 1)

    def test_iter_sexual_selection_entries_nested_and_flat(self) -> None:
        nested = {
            "A|*": {
                "a|*": 0.8,
            }
        }
        nested_entries = list(iter_sexual_selection_entries(nested))
        self.assertEqual(nested_entries, [("A|*", "a|*", 0.8)])

        flat = {
            "a|*": 0.7,
        }
        flat_entries = list(iter_sexual_selection_entries(flat))
        self.assertEqual(flat_entries, [("*", "a|*", 0.7)])

    def test_sex_label_to_index_mapping_for_viability(self) -> None:
        self.assertEqual(resolve_sex_label("female"), 0)
        self.assertEqual(resolve_sex_label("f"), 0)
        self.assertEqual(resolve_sex_label("male"), 1)
        self.assertEqual(resolve_sex_label("m"), 1)

        with self.assertRaises(ValueError):
            resolve_sex_label("unknown")

    def test_resolve_genotype_selector_invalid_or_empty_match_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.simple_species.resolve_genotype_selectors(
                selector="NotAValidPattern(",
                all_genotypes=self.all_genotypes,
                context="sexual_selection",
            )

        with self.assertRaises(ValueError):
            self.simple_species.resolve_genotype_selectors(
                selector="B|*",
                all_genotypes=self.all_genotypes,
                context="sexual_selection",
            )

if __name__ == "__main__":
    unittest.main()
