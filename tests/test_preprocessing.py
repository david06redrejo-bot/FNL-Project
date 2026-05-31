"""Tests for preprocessing design decisions."""

import pandas as pd

from src.preprocessing import (
    clean_remove_accents,
    clean_remove_punctuation,
    clean_required,
    extract_text_pattern_features,
)


def test_required_preserves_case():
    assert clean_required("  VHC y HTA  ") == "VHC y HTA"


def test_required_preserves_accents():
    assert clean_required("miocardiopatía dilatada") == "miocardiopatía dilatada"


def test_required_preserves_punctuation():
    assert clean_required("fractura radio-cúbito (izq.)") == "fractura radio-cúbito (izq.)"


def test_required_collapses_whitespace():
    assert clean_required("  HTA\t\tirc\n6  ") == "HTA irc 6"


def test_required_handles_non_string_input():
    assert clean_required(123) == "123"


def test_required_handles_empty_and_null_values():
    assert clean_required("") == ""
    assert clean_required(None) == ""
    assert clean_required(float("nan")) == ""
    assert clean_required(pd.NA) == ""


def test_ablation_remove_accents_is_not_required_behavior():
    assert clean_required("Hèrnia") == "Hèrnia"
    assert clean_remove_accents("Hèrnia") == "Hernia"


def test_ablation_remove_punctuation_is_not_required_behavior():
    assert clean_required("VHC/VHB") == "VHC/VHB"
    assert clean_remove_punctuation("VHC/VHB") == "VHC VHB"


def test_extract_text_pattern_features():
    features = extract_text_pattern_features("HTA 6 mg/día")
    assert features["has_digit"] is True
    assert features["is_all_upper"] is False
    assert features["has_slash"] is True
    assert features["has_accent"] is True
    assert features["has_measurement_like"] is True
