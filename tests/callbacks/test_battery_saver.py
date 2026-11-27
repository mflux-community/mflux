from unittest.mock import MagicMock, patch

import pytest
import subprocess

from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.utils.exceptions import StopImageGenerationException


@pytest.mark.fast
def test_get_battery_percentage_while_charging():
    with (
        patch.object(BatterySaver, "_is_machine_battery_powered", return_value=True),
        patch("subprocess.run") as mock_run,
    ):
        # Set up mock to return an output that doesn't match the expected pattern
        mock_result = MagicMock()
        mock_result.stdout = "Now drawing from 'AC Power'"
        mock_run.return_value = mock_result

        # Call the method
        battery_saver = BatterySaver()
        percentage = battery_saver._get_battery_percentage()

        # Assert the function returns None when no match is found
        assert percentage is None


@pytest.mark.fast
def test_battery_saver_below_limit():
    with patch.object(BatterySaver, "_get_battery_percentage", return_value=5):
        # Create a BatterySaver instance with a limit of 10%
        battery_saver = BatterySaver(battery_percentage_stop_limit=10)

        # Assert that calling before_loop raises StopImageGenerationException
        with pytest.raises(StopImageGenerationException) as excinfo:
            battery_saver.call_before_loop()

        # Check the exception message contains the correct limit and percentage
        assert "10%" in str(excinfo.value)
        assert "5%" in str(excinfo.value)


@pytest.mark.fast
def test_battery_saver_above_limit():
    with patch.object(BatterySaver, "_get_battery_percentage", return_value=20):
        # Create a BatterySaver instance with a limit of 10%
        battery_saver = BatterySaver(battery_percentage_stop_limit=10)

        # Assert that calling before_loop does not raise an exception
        battery_saver.call_before_loop()


@pytest.mark.fast
def test_battery_saver_none_percentage():
    with patch.object(BatterySaver, "_get_battery_percentage", return_value=None):
        # Create a BatterySaver instance
        battery_saver = BatterySaver()

        # Assert that calling before_loop does not raise an exception when percentage is None
        battery_saver.call_before_loop()


@pytest.mark.fast
def test_battery_saver_equal_to_limit():
    with patch.object(BatterySaver, "_get_battery_percentage", return_value=10):
        # Create a BatterySaver instance with a limit of 10%
        battery_saver = BatterySaver(battery_percentage_stop_limit=10)

        # Assert that calling before_loop raises StopImageGenerationException
        with pytest.raises(StopImageGenerationException) as excinfo:
            battery_saver.call_before_loop()

        # Check the exception message contains the correct limit and percentage
        assert "10%" in str(excinfo.value)


@pytest.mark.fast
def test_get_machine_model_handles_system_profiler_timeout():
    """system_profiler can hang on some Macs (e.g. Mac Studio). The timeout
    ensures mflux proceeds instead of blocking forever."""
    # Reset the cached class attribute so the method actually runs
    BatterySaver._machine_model = None
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="system_profiler", timeout=10)):
        model = BatterySaver._get_machine_model()
        assert model == "Unknown"
    BatterySaver._machine_model = None  # clean up for other tests


@pytest.mark.fast
def test_get_battery_percentage_handles_pmset_timeout():
    """pmset can also hang; the timeout ensures graceful degradation."""
    with (
        patch.object(BatterySaver, "_is_machine_battery_powered", return_value=True),
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pmset", timeout=5)),
    ):
        battery_saver = BatterySaver()
        percentage = battery_saver._get_battery_percentage()
        assert percentage is None
