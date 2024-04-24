"""
This module is responsible for maintaining the current state of the loop, such as the current beat, bar and track
states.
Big parts use ``asyncio`` so that functions can be handled asynchronously.
"""
import asyncio
import json
import logging
import math
import time
from base64 import b64encode
from dataclasses import dataclass, asdict
from enum import Enum
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from websockets.exceptions import ConnectionClosedOK
from websockets.server import WebSocketServerProtocol

from backlooper.audio import AudioStream
from backlooper.config import EVENT_TYPE_KEY, BEAT_KEY, TEMPO_EVENT, TRACKS_EVENT, TRACKS_KEY, LATENCY_EVENT, LATENCY_KEY, \
    BEATS_PER_BAR, NUMBER_OF_TRACKS, CALIBRATION_RESULT_EVENT, FIGURE_KEY, MESSAGE_KEY

logger = logging.getLogger(__name__)


class TrackState(str, Enum):
    """
    ``TrackState`` contains all possible track states.
    Tracks start out empty, and proceed through further states during the session.
    """
    EMPTY = 'EMPTY'
    TRIGGERED = 'TRIGGERED'
    RECORDING = 'RECORDING'
    PLAYING = 'PLAYING'
    STOPPING = 'STOPPING'
    STOPPED = 'STOPPED'


@dataclass
class Track:
    """
    A track can contain looped audio.
    Tracks are identified by their ``track_id``.
    If the track is looping, the ``start_timestamp`` and ``end_timestamp`` will be filled.
    """
    track_id: int
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    state: TrackState = TrackState.EMPTY


@dataclass
class Session:
    """
    The ``Session`` object is responsible for maintaining the current state of the loop, such as the current beat, bar
    and track states.
    The underlying audio is delegated to the ``audio`` object.
    """
    bpm: float
    """Tempo of the session in beats per minute."""
    audio: AudioStream
    """Manages the audio stream and interface."""

    def __post_init__(self):
        self.origin: Optional[float] = None
        self.current_bar: Optional[int] = None

        self._initialize_tracks()

        self._websocket: Optional[WebSocketServerProtocol] = None

    def _initialize_tracks(self):
        self.tracks = {
            track_id: Track(track_id)
            for track_id in range(NUMBER_OF_TRACKS)
        }

    def run(self):
        """Main starting point of a session."""
        logger.debug('Start running')
        self.origin = time.time()

        loop = asyncio.get_event_loop()
        loop.create_task(self.click())

        self.audio.clicktrack_bpm = self.bpm
        self.audio.clicktrack_origin = self.origin
        self.audio.play()

    async def click(self):
        """Sends updates to the frontend about the current beat. Loops indefinitely."""
        while True:
            now = time.time()
            absolute_beat_number = round(self._absolute_beat_number(now))
            self.current_beat = (absolute_beat_number % BEATS_PER_BAR) + 1  # one-indexed
            self.current_bar = math.floor(absolute_beat_number / BEATS_PER_BAR)

            if self._websocket:
                try:
                    await self._websocket.send(json.dumps({
                        EVENT_TYPE_KEY: TEMPO_EVENT,
                        BEAT_KEY: self.current_beat,
                    }))
                except ConnectionClosedOK:
                    self._websocket = None

            next_time_diff = (
                self.origin + (absolute_beat_number + 1) * self._get_seconds_per_beat()
            ) - now
            await asyncio.sleep(next_time_diff)

    def _absolute_beat_number(self, now) -> float:
        """Returns the number of beats since the origin. Can be a fraction to indicate a part of a beat."""
        return (now - self.origin) / self._get_seconds_per_beat()

    def _get_seconds_per_beat(self):
        """Returns the duration of a single beat."""
        return 60 / self.bpm

    async def set_bpm(self, bpm: int):
        """Updates the tempo."""
        if not self._websocket:
            return
        self.bpm = bpm
        self.origin = (time.time() + 0.5 * self._get_seconds_per_beat())  # prevent scratch noise when sliding BPM
        self.audio.clicktrack_bpm = self.bpm
        self.audio.clicktrack_origin = self.origin

    def register_websocket(self, websocket: WebSocketServerProtocol):
        """Stores the websocket for communication towards the frontend."""
        self._websocket = websocket

    async def request_recording(
            self,
            track_id: int,
            bars_to_record: int,
    ):
        """Records the previous ``bars_to_record`` bars for track ``track_id`` and starts playing."""
        if not self.current_bar or not self._websocket:
            return

        track = self.tracks.get(track_id)
        if not track:
            logger.warning('Cannot request recording for unknown track: %s', track_id)
            return
        if track.state != TrackState.EMPTY:
            logger.warning('Cannot request recording for already recording track: %s', track_id)
            return

        now = time.time()
        absolute_beat_number = self._absolute_beat_number(now)
        unit_progression_in_bar = (absolute_beat_number % BEATS_PER_BAR) / BEATS_PER_BAR
        current_bar = math.floor(absolute_beat_number / BEATS_PER_BAR)
        backloop_bar_threshold_unit = 0.75  # fourth beat in 4/4, last quarter in other measures
        if unit_progression_in_bar > backloop_bar_threshold_unit:
            logger.debug('Backloop recording ends at the start of the next bar')
            number_of_bars_offset = -bars_to_record + 1
        else:
            logger.debug('Backloop recording ended at the start of the current bar')
            number_of_bars_offset = -bars_to_record

        bar_start_time, bar_end_time = self._get_bar_interval(
            current_bar=current_bar,
            number_of_bars=bars_to_record,
            number_of_bars_offset=number_of_bars_offset,
        )
        logger.debug('Recording starts at %s and ends at %s', bar_start_time, bar_end_time)

        track.state = TrackState.TRIGGERED
        track.start_timestamp = bar_start_time
        track.end_timestamp = bar_end_time
        track.state = TrackState.RECORDING
        await self.send_tracks_update()

        time_to_wait = bar_end_time - time.time()
        if time_to_wait > 0:
            logger.debug('Waiting for recording to finish')
            await asyncio.sleep(time_to_wait)

        track.state = TrackState.PLAYING
        self.audio.set_start_end_loop(
            float(track.start_timestamp + self.audio.latency_seconds),
            float(track.end_timestamp + self.audio.latency_seconds),
            track_id=track_id,
        )
        # TODO: there is no crossfading yet for the first end-start transition

        await self.send_tracks_update()
        logger.debug('Tracks state set to PLAYING.')

        await asyncio.sleep(max(2*self.audio.latency_seconds, 3*self.audio.block_size / self.audio.sample_rate))

        logger.debug('Applying crossfading')
        crossfading_time = 0.025
        original_at_beginning_of_recording = self.audio.read(
            track.start_timestamp + self.audio.latency_seconds - crossfading_time,
            track.start_timestamp + self.audio.latency_seconds
        )
        original_at_end_of_recording = self.audio.read(
            track.end_timestamp + self.audio.latency_seconds - crossfading_time,
            track.end_timestamp + self.audio.latency_seconds
        )
        shortest_length = min(
            original_at_beginning_of_recording.shape[0],
            original_at_end_of_recording.shape[0],
        )  # sizes could differ due to off-by-one error
        increasing_slope = np.repeat(
            np.arange(  # TODO: nonlinear slope might be nicer (continuous derivative)
                shortest_length
            ).reshape(
                (shortest_length, 1)
            ) / shortest_length,
            repeats=original_at_beginning_of_recording.shape[1],
            axis=1
        )
        crossfaded = (
            increasing_slope * original_at_beginning_of_recording[:shortest_length, :]
            + (1 - increasing_slope) * original_at_end_of_recording[:shortest_length, :]
        )
        self.audio.write(
            track.end_timestamp + self.audio.latency_seconds - crossfading_time,
            array_to_store=crossfaded,
        )

    def _get_bar_interval(
            self,
            current_bar: int,  # don't use self.current_bar to prevent race conditions
            number_of_bars: int = 0,
            number_of_bars_offset: int = 0,
    ):
        """Returns the start time and end time in seconds of a certain bar interval."""
        bar_start = current_bar + number_of_bars_offset
        bar_end = bar_start + number_of_bars
        bar_start_time = self.origin + bar_start * BEATS_PER_BAR * self._get_seconds_per_beat()
        bar_end_time = self.origin + bar_end * BEATS_PER_BAR * self._get_seconds_per_beat()
        return bar_start_time, bar_end_time

    async def start_playing(
        self,
        track_id: int,
    ):
        """Starts playing a track ``track_id`` for which recording already has taken place."""
        if not self.current_bar or not self._websocket:
            return

        track = self.tracks.get(track_id)
        if not track:
            logger.warning('Cannot stop playing for unknown track: %s', track_id)
            return
        if track.state != TrackState.STOPPED:
            logger.warning('Cannot start playing for a track in state %s: %s', track.state, track_id)
            return
        track.state = TrackState.TRIGGERED
        await self.send_tracks_update()

        bar_start_time, _ = self._get_bar_interval(
            current_bar=self.current_bar,
            number_of_bars_offset=1,
        )
        logger.debug('Playing starts at %s for track ID %s', bar_start_time, track_id)

        time_to_wait = bar_start_time - time.time()
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)

        track.state = TrackState.PLAYING
        start_timestamp = float(track.start_timestamp + self.audio.latency_seconds)
        end_timestamp = float(track.end_timestamp + self.audio.latency_seconds)
        offset = (bar_start_time - track.start_timestamp) % (track.end_timestamp - track.start_timestamp)
        self.audio.set_start_end_loop(
            start_timestamp,
            end_timestamp,
            track_id=track_id,
            offset=offset,
        )
        await self.send_tracks_update()
        logger.debug('Playing started from %s to %s with offset %s', start_timestamp, end_timestamp, offset)

    async def stop_playing(
        self,
        track_id: int,
    ):
        """Stops playing a track ``track_id``."""
        if not self.current_bar or not self._websocket:
            return

        track = self.tracks.get(track_id)
        if not track:
            logger.warning('Cannot stop playing for unknown track: %s', track_id)
            return
        if track.state != TrackState.PLAYING:
            logger.warning('Cannot stop playing for a track in state %s: %s', track.state, track_id)
            return
        track.state = TrackState.STOPPING
        await self.send_tracks_update()

        bar_start_time, _ = self._get_bar_interval(
            current_bar=self.current_bar,
            number_of_bars_offset=1,
        )
        logger.debug('Track ID %s stops at %s', track_id, bar_start_time)

        time_to_wait = bar_start_time - time.time()
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)

        track.state = TrackState.STOPPED
        self.audio.reset_loop(track.track_id)
        await self.send_tracks_update()
        logger.debug('Recording stopped')

    async def reset(self):
        """Resets all tracks to their starting state."""
        if not self.current_bar or not self._websocket:
            return

        for track_id in self.tracks.keys():
            self.audio.reset_loop(track_id)
        self._initialize_tracks()
        await self.send_tracks_update()
        # TODO: empty memory

    async def calibrate(self):
        """Executes the calibration routine to determine round-trip latency."""
        plt.rcParams.update({'font.size': 22})

        bpm_for_calibration = 30
        time_to_wait_while_calibrate_records = 9

        self.audio.clicktrack_bpm = bpm_for_calibration
        start_time = time.time()
        self.audio.clicktrack_origin = start_time

        logger.debug('Waiting %s seconds to record some loopback', time_to_wait_while_calibrate_records)
        await asyncio.sleep(time_to_wait_while_calibrate_records)
        logger.debug('Recording loopback complete')

        recording = self.audio.read(
            start_timestamp=start_time,
            end_timestamp=start_time + time_to_wait_while_calibrate_records,
        )
        mono_recording = np.abs((recording[:, 0] + recording[:, 1]) / 2)
        noise_level = np.percentile(mono_recording, 75)
        signal_threshold = 10
        threshold = noise_level * signal_threshold

        recording_column = 'Absolute recording'
        mono_recording_df = pd.DataFrame(
            {recording_column: mono_recording},
            index=np.arange(mono_recording.shape[0]) / self.audio.sample_rate
        )

        time_between_beats = (60 / bpm_for_calibration)
        beats_in_recording = math.floor(time_to_wait_while_calibrate_records / time_between_beats)

        fig = plt.figure(figsize=(10, 7))
        ax = plt.gca()
        ax.semilogy(
            mono_recording_df.index,
            mono_recording_df[recording_column],
            '.',
            alpha=.5,
            label=recording_column,
        )
        ax.axhline(
            threshold,
            color='r',
            linestyle='--',
            alpha=.8,
            label='Threshold'
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal (arbitrary unit)')
        ax.set_xlim(0, None)
        ax.legend(loc='upper right')
        ax.grid()
        ax.set_title(rf'Latency calibration finished with error')
        fig.tight_layout()

        measured_latencies = []
        for beat in range(1, beats_in_recording):  # skip first and last
            beat_content = mono_recording_df[beat * time_between_beats:(beat + 1) * time_between_beats]
            threshold_exceeded_time = beat_content[
                beat_content[recording_column] > threshold
                ].index.min()
            if pd.isnull(threshold_exceeded_time):
                logger.warning(
                    f'Missing threshold exceedance in beat {beat}. Check the latency calibration diagram for more '
                    f'information. Try increasing the signal-to-noise ratio.'
                )
                await self.send_calibration_diagram("FAIL")
                return
            measured_latencies.append(threshold_exceeded_time - beat * time_between_beats)

            ax.axvline(
                beat * time_between_beats,
                color='k',
                linestyle='-',
                alpha=.8,
                label='Expected beats' if beat == 1 else None
            )
            ax.axvline(
                threshold_exceeded_time,
                color='k',
                linestyle='--',
                alpha=.8,
                label='Actual beats' if beat == 1 else None
            )

        latency = np.average(measured_latencies)
        latency_error = np.std(measured_latencies)

        ax.legend(loc='upper right')
        ax.set_title(rf'Latency calibration: ${round(1000 * latency)} \pm {round(1000 * latency_error)}$ ms')

        self.audio.latency_seconds = latency
        self.audio.clicktrack_bpm = self.bpm
        self.audio.clicktrack_origin = self.origin

        await self.send_latency_update()
        await self.send_calibration_diagram("SUCCESS")

    async def send_calibration_diagram(
            self,
            message: str,
    ):
        """Sends the calibration diagram to the frontend."""
        if not self._websocket:
            logger.warning('Cannot send calibration diagram because there is no websocket.')
            return

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        await self._websocket.send(json.dumps({
            EVENT_TYPE_KEY: CALIBRATION_RESULT_EVENT,
            FIGURE_KEY: b64encode(buffer.read()).decode(),
            MESSAGE_KEY: message,
        }))

    async def send_tracks_update(self):
        """Sends the current state of all tracks to the frontend."""
        if not self._websocket:
            logger.warning('Cannot send tracks update because there is no websocket.')
            return
        await self._websocket.send(json.dumps({
            EVENT_TYPE_KEY: TRACKS_EVENT,
            TRACKS_KEY: [
                asdict(track) for track in self.tracks.values()
            ],
        }))

    async def send_latency_update(self):
        """Sends the currently configured latency to the frontend."""
        if not self._websocket:
            logger.warning('Cannot send tracks update because there is no websocket.')
            return
        await self._websocket.send(json.dumps({
            EVENT_TYPE_KEY: LATENCY_EVENT,
            LATENCY_KEY: self.audio.latency_seconds
        }))
