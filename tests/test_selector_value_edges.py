"""Sex/age selector value-edge contracts.

Closes the P9-disclosed residuals around sex selector values:

- NumPy integer sex and age values are accepted like any other
  :class:`numbers.Integral` (the pre-fix behavior raised ``TypeError:
  'numpy.int64' object is not iterable`` — the same values already
  worked for age/genotype on the observation path).
- Booleans are rejected before the Integral narrowing (``True`` used to
  silently mean sex index 1; the age selector already rejected bools).
- An empty sex label raises with the same "use None for a wildcard"
  guidance as an empty container (it used to defer to a bare
  ``Unknown sex label: ''`` from the constructor).
"""

from __future__ import annotations

import numpy as np
import pytest

from natal.frontend.output.observation import ObservationFilter
from natal.frontend.patterns import IndividualSelector


class TestSexIntegralValues:
    """NumPy integers and mixed collections normalize like Python ints."""

    def test_numpy_int_sex_flattens_to_index(self) -> None:
        assert ObservationFilter._flatten_sex_values(np.int64(0)) == [0]
        assert ObservationFilter._flatten_sex_values(np.int32(1)) == [1]

    def test_numpy_int_sex_in_constructor(self) -> None:
        selector = IndividualSelector(sex=np.int64(0))
        assert repr(selector) == "IndividualSelector(sex=(0,))"

    def test_mixed_numpy_and_labels_normalize(self) -> None:
        selector = IndividualSelector(sex=[np.int64(1), "female"])
        assert repr(selector) == "IndividualSelector(sex=(0, 1))"

    def test_python_int_sex_still_works(self) -> None:
        assert ObservationFilter._flatten_sex_values(1) == [1]
        assert repr(IndividualSelector(sex=0)) == "IndividualSelector(sex=(0,))"


class TestSexBoolRejected:
    """Booleans are not sex indices."""

    @pytest.mark.parametrize("caller", [ObservationFilter._flatten_sex_values])
    def test_flatten_rejects_bool(self, caller) -> None:
        with pytest.raises(TypeError, match="not a sex index"):
            caller(True)

    def test_constructor_rejects_bool(self) -> None:
        with pytest.raises(TypeError, match="not a sex index"):
            IndividualSelector(sex=True)

    def test_constructor_rejects_bool_inside_collection(self) -> None:
        with pytest.raises(TypeError, match="not a sex index"):
            IndividualSelector(sex=[0, True])


class TestEmptySexLabel:
    """An empty label selects nothing and says how to select everything."""

    def test_flatten_empty_string_raises_with_guidance(self) -> None:
        with pytest.raises(ValueError, match="use None for a wildcard"):
            ObservationFilter._flatten_sex_values("")

    def test_constructor_empty_string_raises_with_guidance(self) -> None:
        with pytest.raises(ValueError, match="use None for a wildcard"):
            IndividualSelector(sex="")

    def test_empty_container_keeps_its_guidance(self) -> None:
        with pytest.raises(ValueError, match="use None for a wildcard"):
            ObservationFilter._flatten_sex_values([])


class TestAgeIntegralValues:
    """The selector-module age normalizer matches the Integral policy."""

    def test_numpy_int_age_in_constructor(self) -> None:
        assert repr(IndividualSelector(age=np.int32(2))) == "IndividualSelector(age=(2,))"

    def test_age_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="not an age index"):
            IndividualSelector(age=True)
