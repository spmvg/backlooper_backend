"""HD44780 16x2 LCD driver over PCF8574 I2C expander."""
import logging
import time

logger = logging.getLogger(__name__)

try:
    import smbus2
    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False
    logger.warning('smbus2 not available; LCD output will be logged instead')

# PCF8574 → HD44780 pin mapping (standard backpack wiring)
_RS = 0x01  # register select: 0 = command, 1 = data
_EN = 0x04  # enable pulse
_BL = 0x08  # backlight

# HD44780 commands
_CMD_CLEAR = 0x01
_CMD_ENTRY_MODE = 0x06   # increment cursor, no display shift
_CMD_DISPLAY_ON = 0x0C  # display on, cursor off, blink off
_CMD_4BIT_2LINE = 0x28  # 4-bit, 2 lines, 5x8 dots
_CMD_DDRAM = 0x80        # set DDRAM address

_ROW_OFFSETS = (0x00, 0x40)
LCD_WIDTH = 16


class LCDScreen:
    """16x2 LCD screen connected via I2C (HD44780 + PCF8574 backpack)."""

    def __init__(self, i2c_address: int = 0x27, i2c_bus: int = 1) -> None:
        self._address = i2c_address
        self._available = _SMBUS_AVAILABLE
        if self._available:
            try:
                self._bus = smbus2.SMBus(i2c_bus)
                self._init()
            except Exception as exc:
                logger.warning('LCD initialisation failed (%s); output will be logged', exc)
                self._available = False

    # --- low-level I2C / HD44780 protocol ---------------------------------

    def _write_byte(self, data: int) -> None:
        self._bus.write_byte(self._address, data)
        time.sleep(0.0001)

    def _pulse_enable(self, data: int) -> None:
        self._write_byte(data | _EN)
        self._write_byte(data & ~_EN)

    def _write4bits(self, data: int) -> None:
        self._write_byte(data | _BL)
        self._pulse_enable(data | _BL)

    def _send(self, data: int, mode: int) -> None:
        """Send a full byte in two 4-bit nibbles."""
        self._write4bits(mode | (data & 0xF0))
        self._write4bits(mode | ((data << 4) & 0xF0))

    def _command(self, cmd: int) -> None:
        self._send(cmd, 0x00)
        time.sleep(0.002)

    def _init(self) -> None:
        # Initialisation sequence per HD44780 datasheet section 4
        time.sleep(0.05)
        for _ in range(3):
            self._write4bits(0x30)
            time.sleep(0.005)
        self._write4bits(0x20)  # switch to 4-bit interface
        self._command(_CMD_4BIT_2LINE)
        self._command(_CMD_DISPLAY_ON)
        self._command(_CMD_CLEAR)
        self._command(_CMD_ENTRY_MODE)

    # --- public API --------------------------------------------------------

    def write_line(self, row: int, text: str) -> None:
        """Write *text* to *row* (0 or 1), truncating/padding to 16 characters."""
        padded = text[:LCD_WIDTH].ljust(LCD_WIDTH)
        logger.info('LCD row %d: %s', row, padded)
        if not self._available:
            return
        self._command(_CMD_DDRAM | _ROW_OFFSETS[row])
        for char in padded:
            self._send(ord(char), _RS)

    def close(self) -> None:
        if self._available:
            self._bus.close()
