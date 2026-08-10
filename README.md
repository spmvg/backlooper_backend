# Backlooper

Backlooper loops audio without having to trigger beforehand.
Audio is always being recorded.
The last few bars will be played back if you select a track at approximately the first beat of the next bar.

Status is shown on a 16×2 LCD screen connected via I2C (HD44780 + PCF8574 backpack).

## Development setup
Install the backend locally:

```commandline
python -m pip install -e .
```

Run the backend:

```commandline
python -m backlooper
```

The input and output sound device IDs are prompted on startup.
To skip the prompt, set the `INPUT_DEVICE_ID` and `OUTPUT_DEVICE_ID` environment variables to the desired integer device IDs.
Available device IDs are logged on startup.

On a machine without `smbus2` or without an I2C bus, LCD output falls back to log messages.

Generate developer documentation locally:

```commandline
sphinx-build -M html docs build
```

### Running on a Raspberry Pi
Enable I2C in `raspi-config` and wire the LCD backpack to the I2C bus (SDA/SCL + 5 V + GND).

Setup before running:
```bash
ulimit -n 1048576  # workaround for https://github.com/spmvg/backlooper_backend/issues/3
export INPUT_DEVICE_ID=0
export OUTPUT_DEVICE_ID=0
```