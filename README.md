# Backlooper.app

[Backlooper.app](https://backlooper.app) is a looper that allows you to decide to loop a few bars after you played them.
It is not necessary to hit "record" before playing a loop.
Audio is always being recorded.
The last few bars will be looped if you select a track at approximately the first beat of the next bar.

For user documentation, please see [here](https://backlooper.app/docs/general).

For developer documentation, please continue reading.
More developer documentation is provided [here](TODO sphinx).


The frontend code is provided [here](TODO frontend code).

## Development setup
Install the backend locally:

```commandline
python -m pip install -e .
```

Run the backend:

```commandline
python -m backlooper
```

Generate developer documentation locally:

```commandline
sphinx-build -M html docs build
```
