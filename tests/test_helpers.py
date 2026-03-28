"""Unit tests for natal.helpers."""

import pytest  # type: ignore

from natal.helpers import validate_name, resolve_sex_label
from natal.type_def import Sex


class TestValidateName:
    def test_letters_only(self):
        assert validate_name("ValidName") is True

    def test_digits_only(self):
        assert validate_name("12345") is True

    def test_underscore_only(self):
        assert validate_name("___") is True

    def test_mixed_valid(self):
        assert validate_name("Valid_Name_123") is True

    def test_empty_string(self):
        assert validate_name("") is False

    def test_space_invalid(self):
        assert validate_name("with space") is False

    def test_hyphen_invalid(self):
        assert validate_name("with-dash") is False

    def test_dot_invalid(self):
        assert validate_name("with.dot") is False

    def test_slash_invalid(self):
        assert validate_name("with/slash") is False

    def test_pipe_invalid(self):
        assert validate_name("a|b") is False

    def test_unicode_letters_invalid(self):
        assert validate_name("café") is False


class TestResolveSexLabel:
    # --- string inputs ---
    def test_female_string(self):
        assert resolve_sex_label("female") == 0

    def test_male_string(self):
        assert resolve_sex_label("male") == 1

    def test_f_short(self):
        assert resolve_sex_label("f") == 0

    def test_m_short(self):
        assert resolve_sex_label("m") == 1

    def test_case_insensitive_female(self):
        assert resolve_sex_label("FEMALE") == 0

    def test_case_insensitive_male(self):
        assert resolve_sex_label("MALE") == 1

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid sex label"):
            resolve_sex_label("other")

    # --- integer inputs ---
    def test_int_zero(self):
        assert resolve_sex_label(0) == 0

    def test_int_one(self):
        assert resolve_sex_label(1) == 1

    def test_invalid_int_raises(self):
        with pytest.raises(ValueError, match="Invalid sex index"):
            resolve_sex_label(2)

    def test_negative_int_raises(self):
        with pytest.raises(ValueError):
            resolve_sex_label(-1)

    # --- Sex enum inputs ---
    def test_sex_enum_female(self):
        assert resolve_sex_label(Sex.FEMALE) == 0

    def test_sex_enum_male(self):
        assert resolve_sex_label(Sex.MALE) == 1

    # --- invalid types ---
    def test_float_raises(self):
        with pytest.raises(TypeError, match="Invalid sex label type"):
            resolve_sex_label(3.14)  # type: ignore

    def test_none_raises(self):
        with pytest.raises(TypeError):
            resolve_sex_label(None)  # type: ignore
