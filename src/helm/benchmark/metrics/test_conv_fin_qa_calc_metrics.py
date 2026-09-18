from unittest import mock

from helm.benchmark.metrics import conv_fin_qa_calc_metrics
from helm.benchmark.metrics.conv_fin_qa_calc_metrics import float_equiv


def test_float_equiv_equal_floats():
    assert float_equiv("1.5", "1.50") == 1.0


def test_float_equiv_different_floats():
    assert float_equiv("1.5", "2.5") == 0.0


def test_float_equiv_non_float_returns_zero():
    # The guard fires when either string cannot be parsed as a float, and the score is 0.0
    # in every such case, including when both values are the same non-numeric string.
    assert float_equiv("abc", "1.5") == 0.0
    assert float_equiv("1.5", "abc") == 0.0
    assert float_equiv("abc", "abc") == 0.0


def test_float_equiv_warning_states_returned_value():
    # The warning is what a reader of the logs sees, so it must name the value the metric receives.
    for str1, str2 in [("abc", "1.5"), ("1.5", "abc"), ("abc", "abc")]:
        with mock.patch.object(conv_fin_qa_calc_metrics, "hwarn") as mock_hwarn:
            result = float_equiv(str1, str2)
        mock_hwarn.assert_called_once()
        message = mock_hwarn.call_args[0][0]
        assert f"returning {result}" in message, message
