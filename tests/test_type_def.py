"""Unit tests for natal.type_def."""

import pytest  # type: ignore

from natal.type_def import Sex


class TestSex:
    def test_female_value(self):
        assert Sex.FEMALE == 0

    def test_male_value(self):
        assert Sex.MALE == 1

    def test_female_is_int_compatible(self):
        assert int(Sex.FEMALE) == 0

    def test_male_is_int_compatible(self):
        assert int(Sex.MALE) == 1

    def test_sex_from_int_0(self):
        assert Sex(0) is Sex.FEMALE

    def test_sex_from_int_1(self):
        assert Sex(1) is Sex.MALE

    def test_repr(self):
        assert repr(Sex.FEMALE) == "Sex.FEMALE"
        assert repr(Sex.MALE) == "Sex.MALE"
