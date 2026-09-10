from __future__ import annotations

import unittest
import uuid

import numpy as np

from natal.frontend.builder._params import (
    resolve_age_param,
    resolve_age_structured_initial_individual_count,
    resolve_age_structured_initial_sperm_storage,
    resolve_discrete_initial_individual_count,
)
from natal.frontend.genetics import Species
from natal.frontend.utils.types import Sex


def _make_species(prefix: str = "BuilderInjectionSpecies") -> Species:
    return Species.from_dict(
        f"{prefix}_{uuid.uuid4().hex}",
        {
            "Chr1": {
                "L1": ["WT", "Drive"],
            }
        },
    )


class TestPopulationBuilderInitialInjection(unittest.TestCase):
    def setUp(self) -> None:
        self.species = _make_species()

    def test_survival_parser_supports_legacy_formats(self) -> None:
        seq_none = resolve_age_param([1.0, 0.5, None], 5, [0.0])
        self.assertTrue(np.allclose(seq_none, np.array([1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)))

        from_dict = resolve_age_param({1: 0.7, 3: 0.2}, 4, [0.0])
        self.assertTrue(np.allclose(from_dict, np.array([1.0, 0.7, 1.0, 0.2], dtype=np.float64)))

        from_callable = resolve_age_param(lambda age: 1.0 - 0.1 * age, 3, [0.0])
        self.assertTrue(np.allclose(from_callable, np.array([1.0, 0.9, 0.8], dtype=np.float64)))

        from_scalar = resolve_age_param(0.3, 4, [0.0])
        self.assertTrue(np.allclose(from_scalar, np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float64)))


class TestInitialCountResolverKeyForms(unittest.TestCase):
    """The age-structured resolver accepts Sex enums and every genotype-key form."""

    def setUp(self) -> None:
        self.species = _make_species("ResolverKeyForms")
        self.wt_wt = self.species.get_all_genotypes(unordered=True)[0]

    def test_sex_enum_keys_fill_the_matching_rows(self) -> None:
        dist = {
            Sex.FEMALE: {self.wt_wt: [0.0, 10.0]},
            Sex.MALE: {self.wt_wt: [0.0, 4.0]},
        }
        out = resolve_age_structured_initial_individual_count(
            self.species, dist, n_ages=3, new_adult_age=1,
        )
        self.assertEqual(float(out[0, 1, 0]), 10.0)
        self.assertEqual(float(out[1, 1, 0]), 4.0)

    def test_genotype_instance_and_tuple_keys(self) -> None:
        dist = {
            "female": {
                self.wt_wt: [0.0, 10.0],
                ("WT|Drive", "default"): [0.0, 6.0],
                ("Drive|WT", "default"): [0.0, 2.0],
            },
        }
        out = resolve_age_structured_initial_individual_count(
            self.species, dist, n_ages=3, new_adult_age=1,
        )
        # 2 genotypes x unordered canonicalization: WT|Drive and Drive|WT
        # map to the same canonical genotype cell, so the counts accumulate.
        self.assertEqual(float(out[0, 1].sum()), 18.0)

    def test_tuple_with_bad_first_element_is_rejected(self) -> None:
        dist = {"female": {(42, "default"): 1}}
        with self.assertRaises(TypeError):
            resolve_age_structured_initial_individual_count(
                self.species, dist, n_ages=3, new_adult_age=1,
            )

    def test_unsupported_key_type_is_rejected(self) -> None:
        dist = {"female": {42: 1}}
        with self.assertRaises(TypeError):
            resolve_age_structured_initial_individual_count(
                self.species, dist, n_ages=3, new_adult_age=1,
            )


class TestInitialResolverValidation(unittest.TestCase):
    """Validation branches of the initial-state resolver functions."""

    def setUp(self) -> None:
        self.species = _make_species("ResolverValidation")

    def test_age_structured_rejects_bad_age_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            resolve_age_structured_initial_individual_count(
                self.species,
                {"female": {"WT|WT": {5: 10}}},
                n_ages=3, new_adult_age=1,
            )

    def test_age_structured_rejects_negative_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolve_age_structured_initial_individual_count(
                self.species,
                {"female": {"WT|WT": {1: -10}}},
                n_ages=3, new_adult_age=1,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolve_age_structured_initial_individual_count(
                self.species,
                {"female": {"WT|WT": [1.0, -10.0]}},
                n_ages=3, new_adult_age=1,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolve_age_structured_initial_individual_count(
                self.species,
                {"female": {"WT|WT": -10}},
                n_ages=3, new_adult_age=1,
            )

    def test_age_structured_scalar_zero_places_no_individuals(self) -> None:
        out = resolve_age_structured_initial_individual_count(
            self.species,
            {"female": {"WT|WT": 0}},
            n_ages=3, new_adult_age=1,
        )
        self.assertEqual(float(out.sum()), 0.0)

    def test_sperm_storage_resolves_and_rejects_bad_keys(self) -> None:
        out = resolve_age_structured_initial_sperm_storage(
            self.species,
            {"WT|WT": {"Drive|WT": {1: 5.0}}},
            n_ages=3, new_adult_age=1,
        )
        self.assertEqual(float(out.sum()), 5.0)
        with self.assertRaises(TypeError):
            resolve_age_structured_initial_sperm_storage(
                self.species,
                {42: {"WT|WT": {1: 5.0}}},
                n_ages=3, new_adult_age=1,
            )

    def test_discrete_scalar_and_list_forms(self) -> None:
        out = resolve_discrete_initial_individual_count(
            self.species,
            {"female": {"WT|WT": 4}, "male": {"WT|WT": [1.0, 2.0]}},
        )
        self.assertEqual(float(out[0, 1, 0]), 4.0)
        self.assertEqual(float(out[1, 0, 0]), 1.0)
        self.assertEqual(float(out[1, 1, 0]), 2.0)
        # Empty and single-element list forms.
        out = resolve_discrete_initial_individual_count(
            self.species,
            {"female": {"WT|WT": []}, "male": {"WT|WT": [7.0]}},
        )
        self.assertEqual(float(out.sum()), 7.0)

    def test_discrete_rejects_invalid_inputs(self) -> None:
        resolver = resolve_discrete_initial_individual_count
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolver(self.species, {"female": {"WT|WT": -4}})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolver(self.species, {"female": {"WT|WT": [1.0, -2.0]}})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolver(self.species, {"female": {"WT|WT": [-4.0]}})
        with self.assertRaisesRegex(ValueError, "length <= 2"):
            resolver(self.species, {"female": {"WT|WT": [1.0, 2.0, 3.0]}})
        with self.assertRaisesRegex(ValueError, "age keys 0 and 1"):
            resolver(self.species, {"female": {"WT|WT": {5: 10.0}}})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolver(self.species, {"female": {"WT|WT": {0: -10.0}}})
        with self.assertRaises(TypeError):
            resolver(self.species, {"female": {"WT|WT": None}})


if __name__ == "__main__":
    unittest.main()
