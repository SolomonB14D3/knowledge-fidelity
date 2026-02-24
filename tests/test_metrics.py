"""Tests for shared metrics utilities."""

import pytest


class TestCheckNumeric:
    """Test the numeric answer extraction function."""

    def test_gsm8k_format(self):
        from rho_eval.behaviors.metrics import check_numeric
        assert check_numeric("The answer is #### 42", "42") is True
        assert check_numeric("#### 100", "100") is True
        assert check_numeric("#### 42", "43") is False

    def test_last_number_fallback(self):
        from rho_eval.behaviors.metrics import check_numeric
        assert check_numeric("The answer is 42.", "42") is True
        assert check_numeric("First 10, then 20, finally 42.", "42") is True

    def test_negative_numbers(self):
        from rho_eval.behaviors.metrics import check_numeric
        assert check_numeric("#### -5", "-5") is True

    def test_commas_in_numbers(self):
        from rho_eval.behaviors.metrics import check_numeric
        assert check_numeric("#### 1,000", "1000") is True

    def test_no_number(self):
        from rho_eval.behaviors.metrics import check_numeric
        assert check_numeric("I don't know.", "42") is False

    def test_empty_string(self):
        from rho_eval.behaviors.metrics import check_numeric
        assert check_numeric("", "42") is False


class TestMannWhitneyAUC:
    """Test the AUC calculation."""

    def test_perfect_separation(self):
        from rho_eval.behaviors.metrics import mann_whitney_auc
        auc = mann_whitney_auc([10, 11, 12], [1, 2, 3])
        assert auc == 1.0

    def test_reversed_separation(self):
        from rho_eval.behaviors.metrics import mann_whitney_auc
        auc = mann_whitney_auc([1, 2, 3], [10, 11, 12])
        assert auc == 0.0

    def test_random_chance(self):
        from rho_eval.behaviors.metrics import mann_whitney_auc
        auc = mann_whitney_auc([1, 2], [1, 2])
        assert abs(auc - 0.5) < 0.01

    def test_empty_lists(self):
        from rho_eval.behaviors.metrics import mann_whitney_auc
        assert mann_whitney_auc([], [1, 2, 3]) == 0.5
        assert mann_whitney_auc([1, 2], []) == 0.5
