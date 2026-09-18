"""Initial sperm-storage input validation must not depend on ``assert``.

Regression contract for plan item 7 (``FRONTEND_REFACTOR_PLAN.md`` §3.2).
The three input type checks in
``AgeStructuredPopulation._distribute_initial_sperm_storage`` used to be
``assert`` statements, so ``python -O`` stripped them and a malformed mapping
fell through to an unrelated later failure.  They are now explicit
``TypeError`` raises.

Reaching these checks requires the direct constructor entry
(``initial_sperm_storage=``); the ``initial_state()`` builder path validates
the same inputs earlier with its own errors, which is covered here too so the
two paths stay distinguishable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import natal as nt

SPECIES_NAME = "sperm_storage_validation"

_INVALID_STORAGE = {
    "female_key": {7: {"A|A": {3: 100}}},
    "male_key": {"A|A": {7: {3: 100}}},
    "age_data": {"A|A": {"A|A": "not-a-number"}},
}

_EXPECTED_MESSAGE = {
    "female_key": "Female genotype key must be Genotype or str",
    "male_key": "Male genotype key must be Genotype or str",
    "age_data": "Age data must be Dict, List, or numeric scalar",
}


@pytest.fixture(scope="module")
def species() -> nt.Species:
    return nt.Species.from_dict(SPECIES_NAME, {"c": {"l": ["A"]}})


@pytest.fixture(scope="module")
def population_config(species: nt.Species) -> object:
    """A built age-structured config the direct constructor can consume."""
    return (
        nt.AgeStructuredPopulation.setup(species)
        .age_structure(n_ages=5, new_adult_age=3)
        .competition(juvenile_growth_mode="no_competition")
        .initial_state(
            individual_count={"female": {"A|A": 10}, "male": {"A|A": 10}},
        )
        .build()
        ._config
    )


def _direct(species: nt.Species, config: object, storage: object) -> nt.AgeStructuredPopulation:
    return nt.AgeStructuredPopulation(species, config, initial_sperm_storage=storage)  # type: ignore[arg-type]


@pytest.mark.parametrize("case", sorted(_INVALID_STORAGE))
def test_direct_constructor_rejects_invalid_input_with_type_error(
    species: nt.Species,
    population_config: object,
    case: str,
) -> None:
    with pytest.raises(TypeError, match=_EXPECTED_MESSAGE[case]):
        _direct(species, population_config, _INVALID_STORAGE[case])


@pytest.mark.parametrize(
    "storage",
    [
        {"A|A": {"A|A": {3: 100}}},
        {"A|A": {"A|A": [0, 0, 0, 100, 0]}},
        {"A|A": {"A|A": (0, 0, 0, 100, 0)}},
        {"A|A": {"A|A": 100}},
        {"A|A": {"A|A": 0}},
    ],
    ids=["dict", "list", "tuple", "scalar", "zero-scalar"],
)
def test_direct_constructor_still_accepts_every_legal_format(
    species: nt.Species,
    population_config: object,
    storage: dict,
) -> None:
    """The check must not narrow the accepted formats or their semantics."""
    pop = _direct(species, population_config, storage)
    expected = 0.0 if storage["A|A"]["A|A"] == 0 else 100.0
    assert pop.state.sperm_storage[3, 0, 0] == expected


def test_direct_constructor_accepts_genotype_objects_as_keys(
    species: nt.Species,
    population_config: object,
) -> None:
    het = species.get_genotype_from_str("A|A")
    pop = _direct(species, population_config, {het: {het: {3: 100}}})
    assert pop.state.sperm_storage[3, 0, 0] == 100.0


def test_builder_path_still_accepts_legal_sperm_storage(species: nt.Species) -> None:
    pop = (
        nt.AgeStructuredPopulation.setup(species)
        .age_structure(n_ages=5, new_adult_age=3)
        .competition(juvenile_growth_mode="no_competition")
        .initial_state(
            individual_count={"female": {"A|A": 10}, "male": {"A|A": 10}},
            sperm_storage={"A|A": {"A|A": {3: 100}}},
        )
        .build()
    )
    assert pop.state.sperm_storage[3, 0, 0] == 100.0


def test_negative_count_is_still_a_value_error(
    species: nt.Species,
    population_config: object,
) -> None:
    """Numeric validation semantics are unchanged by the type-check fix."""
    with pytest.raises(ValueError, match="non-negative"):
        _direct(species, population_config, {"A|A": {"A|A": -1}})


def test_checks_survive_python_optimized_mode(species: nt.Species) -> None:
    """The three checks must still fire under ``python -O``.

    Run in a subprocess and decide by exit status, so the outcome does not
    depend on any ``assert`` in the child (an ``assert``-based guard would be
    stripped by ``-O`` and the child would report a different failure type or
    no failure at all).
    """
    script = textwrap.dedent(
        """
        import natal as nt

        CASES = {
            "female_key": {7: {"A|A": {3: 100}}},
            "male_key": {"A|A": {7: {3: 100}}},
            "age_data": {"A|A": {"A|A": "not-a-number"}},
        }
        EXPECTED = {
            "female_key": "Female genotype key must be Genotype or str",
            "male_key": "Male genotype key must be Genotype or str",
            "age_data": "Age data must be Dict, List, or numeric scalar",
        }

        species = nt.Species.from_dict(
            "sperm_storage_validation_optimized", {"c": {"l": ["A"]}}
        )
        config = (
            nt.AgeStructuredPopulation.setup(species)
            .age_structure(n_ages=5, new_adult_age=3)
            .competition(juvenile_growth_mode="no_competition")
            .initial_state(
                individual_count={"female": {"A|A": 10}, "male": {"A|A": 10}}
            )
            .build()
            ._config
        )

        problems = []
        for name, storage in CASES.items():
            try:
                nt.AgeStructuredPopulation(species, config, initial_sperm_storage=storage)
            except TypeError as exc:
                if EXPECTED[name] not in str(exc):
                    problems.append(f"{name}: unexpected message: {exc}")
            except BaseException as exc:
                problems.append(f"{name}: raised {type(exc).__name__}: {exc}")
            else:
                problems.append(f"{name}: accepted without error")

        if problems:
            print("; ".join(problems), flush=True)
            raise SystemExit(1)
        raise SystemExit(0)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
