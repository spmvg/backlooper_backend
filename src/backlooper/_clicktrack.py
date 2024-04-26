import logging
import os
from functools import lru_cache

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


@lru_cache
def clicktrack(volume_multiplier=1) -> np.array:
    sampling_rate, click = wavfile.read(
        os.path.join(
            os.path.split(__file__)[0],
            'samples',
            'hi_hat.wav',
        )
    )
    click = click * 0.0000020 * volume_multiplier
    logger.debug(
        'Clicktrack amplitudes with multiplier %s are between %s and %s',
        volume_multiplier,
        click.min(),
        click.max()
    )
    return click
