# Backlooper.app

[Backlooper.app](https://backlooper.app) loops audio without having to trigger beforehand.
Audio is always being recorded.
The last few bars will be played back if you select a track at approximately the first beat of the next bar.

For user documentation, please see [here](https://backlooper.app/docs/general).

For developer documentation, please continue reading.
More developer documentation is provided [here](https://backlooper.readthedocs.io/en/latest/).

The frontend code is provided [here](https://github.com/spmvg/backlooper_frontend).

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

Generate developer documentation locally:

```commandline
sphinx-build -M html docs build
```

### Running on a Raspberry Pi
- Setup script
    - Workaround for [too many open files](https://github.com/spmvg/backlooper_backend/issues/3), necessary after every reboot.
    - Configure the Focusrite device ID.
```
ulimit -n 999999999
export INPUT_DEVICE_ID=0
export OUTPUT_DEVICE_ID=0
```