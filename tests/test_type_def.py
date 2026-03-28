"""Unit tests for natal.type_def."""

import pytest  # type: ignore

from natal.type_def import (
    Sex,
    get_age,
    get_genotype_index,
    get_sex,
    make_gamete_type,
    make_individual_type,
)


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


class TestMakeIndividualType:
    def test_with_sex_enum(self):
        ind = make_individual_type(Sex.FEMALE, 3, 5)
        assert ind == (Sex.FEMALE, 3, 5)

    def test_with_sex_int_zero(self):
        ind = make_individual_type(0, 2, 1)
        assert ind[0] is Sex.FEMALE
        assert ind[1] == 2
        assert ind[2] == 1

    def test_with_sex_int_one(self):
        ind = make_individual_type(1, 0, 7)
        assert ind[0] is Sex.MALE

    def test_age_and_genotype_stored_as_int(self):
        ind = make_individual_type(Sex.MALE, 4, 9)
        assert isinstance(ind[1], int)
        assert isinstance(ind[2], int)

    def test_with_numpy_integer_sex(self):
        ind = make_individual_type(0, 1, 0)
        assert ind[0] is Sex.FEMALE

    def test_invalid_sex_string_raises(self):
        with pytest.raises(AssertionError, match="invalid sex value"):
            make_individual_type("female", 0, 0)

    def test_invalid_sex_float_raises(self):
        # A float that converts to an out-of-range int (e.g. 2.7 → 2) should raise.
        with pytest.raises(AssertionError, match="invalid sex value"):
            make_individual_type(2.7, 0, 0)


class TestMakeGameteType:
    def test_basic(self):
        gam = make_gamete_type(Sex.MALE, 2, 0)
        assert gam == (Sex.MALE, 2, 0)

    def test_with_int_sex(self):
        gam = make_gamete_type(0, 3, 1)
        assert gam[0] is Sex.FEMALE
        assert gam[1] == 3
        assert gam[2] == 1

    def test_invalid_sex_raises(self):
        with pytest.raises(AssertionError, match="invalid sex value"):
            make_gamete_type("M", 0, 0)


class TestIndividualTypeAccessors:
    def test_get_sex(self):
        ind = make_individual_type(Sex.FEMALE, 5, 2)
        assert get_sex(ind) is Sex.FEMALE

    def test_get_age(self):
        ind = make_individual_type(Sex.MALE, 7, 3)
        assert get_age(ind) == 7

    def test_get_genotype_index(self):
        ind = make_individual_type(Sex.FEMALE, 1, 4)
        assert get_genotype_index(ind) == 4
