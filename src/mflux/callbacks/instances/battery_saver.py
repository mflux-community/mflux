import json
import logging
import platform
import re
import subprocess

from mflux.callbacks.callback import BeforeLoopCallback
from mflux.utils.exceptions import StopImageGenerationException

logger = logging.getLogger(__name__)


class BatterySaver(BeforeLoopCallback):
    PMSET_AC_POWER_STATUS = "Now drawing from 'AC Power'"
    PMSET_BATT_STATUS_PATTERN = r"InternalBattery-.+?(\d+)%"

    _machine_model: str | None = None
    _is_battery_powered: bool | None = None

    def __init__(self, battery_percentage_stop_limit: int = 10):
        self.limit = battery_percentage_stop_limit

    def call_before_loop(self, **kwargs) -> None:  # type: ignore
        current_pct = self._get_battery_percentage()
        if current_pct is not None and current_pct <= self.limit:
            raise StopImageGenerationException(f"Battery below {self.limit}% threshold: {current_pct}%")

    def _get_battery_percentage(self) -> int | None:
        if platform.uname().system != "Darwin":
            return None

        if not self._is_machine_battery_powered():
            return None

        percentage = None
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            if self.PMSET_AC_POWER_STATUS not in result.stdout:
                if match := re.search(self.PMSET_BATT_STATUS_PATTERN, result.stdout):
                    percentage = int(match.group(1))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, TypeError) as e:
            logger.warning(
                f"Cannot read battery percentage via 'pmset -g batt': {e}. "
                f"Battery saver functionality is disabled and the program will continue running."
            )

        return percentage

    @classmethod
    def _is_machine_battery_powered(cls) -> bool:
        if cls._is_battery_powered is None:
            machine_model = cls._get_machine_model()
            cls._is_battery_powered = "MacBook" in machine_model
        return cls._is_battery_powered

    @classmethod
    def _get_machine_model(cls) -> str:
        if cls._machine_model is None:
            try:
                result = subprocess.run(
                    ["/usr/sbin/system_profiler", "-json", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                data = json.loads(result.stdout)
                cls._machine_model = data["SPHardwareDataType"][0]["machine_model"]
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                json.JSONDecodeError,
                FileNotFoundError,
                IndexError,
                KeyError,
                OSError,
            ) as e:
                logger.warning(f"Cannot determine machine model via 'system_profiler -json SPHardwareDataType': {e}")
                cls._machine_model = "Unknown"
        return cls._machine_model
