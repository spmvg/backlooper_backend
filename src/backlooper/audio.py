"""
``audio`` is the module responsible for controlling the audio interface settings and audio stream.
Real-time audio processing is done by splitting into two processes: one main process and one process handling the audio
stream using a callback function.
Information is shared by ``multiprocessing``.
The audio interface is controlled by ``sounddevice``.
"""

import logging
import time
import webbrowser
from dataclasses import dataclass
from multiprocessing import Process, Value
from multiprocessing.shared_memory import ShareableList
from typing import Optional
from uuid import uuid4

import numpy as np
from sounddevice import Stream, sleep, query_devices

from backlooper._clicktrack import clicktrack
from backlooper.config import LOGS_FORMAT, NUMBER_OF_TRACKS, BEATS_PER_BAR
from backlooper.striped_storage import StripedStorage

_DEFAULT_LOOPER_VALUE = float('inf')
_DEFAULT_ORIGIN_VALUE = float('nan')
_DEFAULT_BPM_VALUE = float('nan')
_SHARED_FLOAT_TYPE = 'd'
_SHARED_INT_TYPE = 'i'
_LOOPER_FIELDS_PER_TRACK = 3

logger = logging.getLogger(__name__)


@dataclass
class AudioStream:
    """
    ``AudioStream`` controls the audio interface settings and audio stream.
    The ``callback`` function is the real-time audio processing function, running in a separate process.
    """
    channels: int = 2
    """Number of channels per track (stereo)."""
    block_size: int = 1000
    """Number of samples that are handled at one time by the ``callback`` function."""
    sample_rate: Optional[int] = 44100
    """Sample rate of the audio interface."""
    log_level: int = logging.DEBUG
    """Log level"""
    input_device_id: Optional[int] = None
    """ TODO """
    output_device_id: Optional[int] = None
    """ TODO """

    def __post_init__(self):
        self._clicktrack_bpm = Value(_SHARED_FLOAT_TYPE, _DEFAULT_BPM_VALUE)
        self._clicktrack_origin = Value(_SHARED_FLOAT_TYPE, _DEFAULT_ORIGIN_VALUE)

        self._storage_identifier = uuid4()
        self._storage = StripedStorage(identifier=str(self._storage_identifier))
        self._time_between_blocks = self.block_size / self.sample_rate
        self._current_index = 0  # time when _samples_origin is initialized, has _current_index 0
        self._samples_origin = Value(_SHARED_FLOAT_TYPE, _DEFAULT_ORIGIN_VALUE)
        self._previous_dac_time = None
        self._loop_start_end_times = ShareableList([_DEFAULT_LOOPER_VALUE, _DEFAULT_LOOPER_VALUE, 0] * NUMBER_OF_TRACKS)
        self._latency_samples = Value(_SHARED_INT_TYPE, 0)
        self._using_automatic_latency_correction = Value(_SHARED_INT_TYPE, 0)
        self._clicktrack_volume = Value(_SHARED_FLOAT_TYPE, 1)
        self._clicktrack = clicktrack()
        self._clicktrack_first_beat = clicktrack(volume_multiplier=3)
        self._logger = None
        self._input_latency_from_device_seconds = None
        self._output_latency_from_device_seconds = None
        self._driver_warning_printed = False

    def callback(self, indata, outdata, frames, callback_time, status):
        """
        The ``callback`` function is the real-time audio processing function, running in a separate process.
        This function must be able to return before the following buffer is handled to prevent audio clicks.
        There are three main parts:

        - Saving the latest recorded audio ``indata``,
        - Mixing in the tracks that are currently playing into ``outdata``,
        - And mixing in the clicktrack into ``outdata``.
        """
        if self._samples_origin.value != self._samples_origin.value:
            self._samples_origin.value = time.time()

        output_time = callback_time.outputBufferDacTime
        input_time = callback_time.inputBufferAdcTime
        if (not input_time or not output_time) and not self._driver_warning_printed:
            self._logger.warning(
                'Your audio driver does not support inputBufferAdcTime and/or '
                'outputBufferDacTime. If you want automatic latency '
                'correction, please use an audio device '
                'with ASIO drivers. You must calibrate now to set the '
                'latency correctly.'
            )
            self._driver_warning_printed = True
        latency_update_threshold_seconds = 0.002
        if input_time and output_time and abs(
                self.latency_seconds - (output_time-input_time)
        ) > latency_update_threshold_seconds:
            self.latency_seconds = output_time-input_time
            self._using_automatic_latency_correction.value = 1

        start_of_callback = time.time()

        self._storage.write(
            self._current_index,
            (indata + np.fliplr(indata)) / 2  # mono
        )

        desired_samples = outdata.shape[0]
        outdata[:] = np.zeros(outdata.shape)
        for track_id in range(NUMBER_OF_TRACKS):
            loop_starttime = self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id]
            loop_endtime = self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id+1]
            offset = self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id+2]
            if not (loop_starttime < loop_endtime < _DEFAULT_LOOPER_VALUE):
                continue

            loop_starttime_index = round((loop_starttime - self._samples_origin.value) * self.sample_rate)
            loop_endtime_index = round((loop_endtime - self._samples_origin.value) * self.sample_rate)
            offset_in_samples = round(offset * self.sample_rate)
            looped_current_index = (
                self._current_index - offset_in_samples - loop_starttime_index + self._latency_samples.value
            ) % (
                loop_endtime_index - loop_starttime_index
            ) + loop_starttime_index
            samples_until_end = max(loop_endtime_index - looped_current_index, 0)
            if samples_until_end >= desired_samples:
                outdata[:] += self._storage.read(
                    looped_current_index,
                    desired_samples
                )
            else:  # loop around
                outdata[:] += np.concatenate([
                    self._storage.read(
                        looped_current_index,
                        samples_until_end
                    ),
                    self._storage.read(
                        loop_starttime_index,
                        desired_samples - samples_until_end
                    )
                ])

        # mixing in the click track
        samples_per_beat = 60 * self.sample_rate / self._clicktrack_bpm.value
        offset_from_clicktrack = self._current_index - (
                self._clicktrack_origin.value - self._samples_origin.value
        ) * self.sample_rate
        start_index_in_clicktrack = round(offset_from_clicktrack % samples_per_beat)
        end_index_in_clicktrack = round((offset_from_clicktrack + desired_samples) % samples_per_beat)
        click = self._clicktrack_first_beat if round(
            offset_from_clicktrack / samples_per_beat
        ) % BEATS_PER_BAR == 0 else self._clicktrack
        clicktrack_length = click.shape[0]
        if start_index_in_clicktrack < clicktrack_length and start_index_in_clicktrack < end_index_in_clicktrack:
            # when the start of the clicktrack has already been played (or is exactly at the start of outdata)
            samples_to_take_in_clicktrack = min(
                clicktrack_length - start_index_in_clicktrack,
                desired_samples
            )
            outdata[:samples_to_take_in_clicktrack] += self._clicktrack_volume.value * click[
                start_index_in_clicktrack:start_index_in_clicktrack+samples_to_take_in_clicktrack
            ]
        elif (
                clicktrack_length > end_index_in_clicktrack > 0
                and end_index_in_clicktrack < start_index_in_clicktrack
        ):
            # happens when the start of the clicktrack is in the middle of outdata
            outdata[-end_index_in_clicktrack:] += self._clicktrack_volume.value * click[
                :end_index_in_clicktrack
            ]

        self._current_index += desired_samples
        self._previous_dac_time = callback_time.outputBufferDacTime

        if (time.time() - start_of_callback) / self._time_between_blocks > 0.5:
            self._logger.warning(
                f'The callback function took relatively long to run: actual {time.time() - start_of_callback} '
                f'is close to the limit {self._time_between_blocks}. This can result in audio glitches.'
            )

    def run(self):
        """
        Starts the audio stream and waits indefinitely.
        """
        duration = 1  # seconds. Increasing this value causes delay on exit.
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)  # TODO: make INFO on release
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOGS_FORMAT))
        self._logger.addHandler(handler)

        self._logger.info('Available sound devices:\n\n%s\n', query_devices())

        with Stream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                callback=self.callback,
                device=(self.input_device_id, self.output_device_id),  # TODO: make configurable
                latency=('low', 'low'),
        ) as stream:
            input_device_id, output_device_id = stream.device

            device_name_field = 'name'
            default_low_input_latency_field = 'default_low_input_latency'
            default_low_output_latency_field = 'default_low_output_latency'
            input_device_dict = query_devices(input_device_id)
            self._logger.debug(
                'Input device: %s',
                input_device_dict,
            )
            output_device_dict = query_devices(output_device_id)
            self._logger.debug(
                'Output device: %s',
                output_device_dict,
            )
            input_device = input_device_dict[device_name_field]
            output_device = output_device_dict[device_name_field]
            self._input_latency_from_device_seconds = input_device_dict[default_low_input_latency_field]
            self._output_latency_from_device_seconds = output_device_dict[default_low_output_latency_field]
            self._logger.info(f'Using input device {self.input_device_id}: {input_device}')
            self._logger.debug(f'Input latency as declared by device: {self._input_latency_from_device_seconds:.3f} s')
            self._logger.info(f'Using output device {self.output_device_id}: {output_device}')
            self._logger.debug(f'Output latency as declared by device: {self._output_latency_from_device_seconds:.3f} s')
            self.latency_seconds = self._input_latency_from_device_seconds + self._output_latency_from_device_seconds

            # session is running, so now we can open the browser
            # webbrowser.open("https://www.backlooper.app/")  # TODO: remove again before release

            while True:
                sleep(int(duration * 1000))

    def set_start_end_loop(
            self,
            start_time_value: float,
            end_time_value: float,
            track_id: int,
            offset: float = 0,
    ):
        """
        Configures the audio stream to start looping an interval between ``start_time_value`` and ``end_time_value``
        for a track ``track_id``.
        The offset allows the start of the track to be adjusted: otherwise the track could start in the middle, because
        the callback function "wraps around" the track automatically (irrespectively of whether the track was playing
        or not).
        """
        self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id] = start_time_value
        self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id+1] = end_time_value
        self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id+2] = offset
        logger.debug('Set loop start and end for track %s', track_id)

    def reset_loop(
            self,
            track_id: int,
    ):
        """Stops the loop for track ``track_id``."""
        self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id] = _DEFAULT_LOOPER_VALUE
        self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id+1] = _DEFAULT_LOOPER_VALUE
        self._loop_start_end_times[_LOOPER_FIELDS_PER_TRACK*track_id+2] = 0
        logger.debug('Reset loop for track %s', track_id)

    @property
    def origin(self):
        """
        Wrapper around the ``multiprocessing.Value`` object containing the first execution time of the callback
        function."""
        return self._samples_origin.value

    @property
    def clicktrack_origin(self):
        """Wrapper around the ``multiprocessing.Value`` object containing the start time of the clicktrack."""
        return self._clicktrack_origin.value

    @clicktrack_origin.setter
    def clicktrack_origin(self, time_value):
        self._clicktrack_origin.value = time_value
        logger.debug('Updated clicktrack origin to %s seconds', time_value)

    @property
    def clicktrack_bpm(self):
        """Wrapper around the ``multiprocessing.Value`` object containing the tempo."""
        return self._clicktrack_bpm.value

    @clicktrack_bpm.setter
    def clicktrack_bpm(self, bpm):
        self._clicktrack_bpm.value = bpm
        logger.debug('Updated clicktrack BPM to %s BPM', bpm)

    @property
    def latency_seconds(self):
        """Wrapper around the ``multiprocessing.Value`` object containing the latency in seconds."""
        return self._latency_samples.value / self.sample_rate

    @latency_seconds.setter
    def latency_seconds(self, latency_seconds: float):
        self._latency_samples.value = round(latency_seconds * self.sample_rate)
        logger.debug('Updated latency to %s seconds', latency_seconds)

    @property
    def clicktrack_volume(self) -> float:
        """Wrapper around the ``multiprocessing.Value`` object containing the clicktrack volume."""
        return self._clicktrack_volume.value

    @clicktrack_volume.setter
    def clicktrack_volume(self, clicktrack_volume: float):
        self._clicktrack_volume.value = clicktrack_volume
        logger.debug('Set clicktrack volume to %s', clicktrack_volume)

    def play(
            self,
    ):
        """Starts the audio stream in a separate process."""
        _audio_process = Process(target=self.run)
        _audio_process.start()

    def read(
            self,
            start_timestamp: float,
            end_timestamp: float,
    ) -> np.array:
        """Reads a ``np.array`` with recorded audio between time interval ``start_timestamp`` and ``end_timestamp``."""
        start_index = round((start_timestamp - self.origin) * self.sample_rate)
        end_index = round((end_timestamp - self.origin) * self.sample_rate)
        samples_to_read = end_index - start_index
        logger.debug('Reading %s samples from timestamp %s', samples_to_read, start_index)
        if samples_to_read < 0:
            raise ValueError(
                'Cannot read from audio with start timestamp %s and end timestamp %s',
                start_timestamp,
                end_timestamp,
            )
        return self._storage.read(
            start_index=start_index,
            length=samples_to_read,
        )

    def write(
        self,
        start_timestamp: float,
        array_to_store: np.array,
        overwrite: bool = True,
    ):
        """Writes a ``np.array`` with recorded audio starting at time ``start_timestamp``."""
        start_index = round((start_timestamp - self.origin) * self.sample_rate)
        self._storage.write(
            start_index=start_index,
            array_to_store=array_to_store,
            overwrite=overwrite,
        )
