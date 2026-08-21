"""
This module is responsible for maintaining the current state of the loop, such as the current beat, bar and track
states.
Big parts use ``asyncio`` so that functions can be handled asynchronously.
"""
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from backlooper.audio import AudioStream
from backlooper.config import BEATS_PER_BAR, NUMBER_OF_TRACKS, VERSION
from backlooper.lcd import LCDScreen

_TRACK_CHARS = {
    'EMPTY': '_',
    'TRIGGERED': 'T',
    'RECORDING': 'O',
    'PLAYING': '>',
    'STOPPING': 'x',
    'STOPPED': 'X',
}

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
    screen: LCDScreen

    def __post_init__(self):
        self.origin: Optional[float] = None
        self.current_bar: Optional[int] = None
        self._last_desync_counter: int = 0
        self._initialize_tracks()

    def _initialize_tracks(self):
        self.tracks = {
            track_id: Track(track_id)
            for track_id in range(NUMBER_OF_TRACKS)
        }

    def run(self):
        """Main starting point of a session."""
        logger.debug('Start running')
        self.origin = time.monotonic()

        loop = asyncio.get_event_loop()
        loop.create_task(self.click())
        loop.create_task(self.send_tracks_update())
        loop.create_task(self._watch_desync())

        self.audio.clicktrack_bpm = self.bpm
        self.audio.clicktrack_origin = self.origin
        self.audio.play()
        self._send_status(f'Version {VERSION}')

    async def _watch_desync(self):
        """Resets track state whenever the audio process signals a desync."""
        while True:
            await asyncio.sleep(0.2)
            counter = self.audio.desync_counter
            if counter != self._last_desync_counter:
                self._last_desync_counter = counter
                logger.error('ERROR: desync signalled by audio process — resetting all tracks')
                for track_id in self.tracks.keys():
                    self.audio.reset_loop(track_id)
                self._initialize_tracks()
                self._send_status('ERROR: Desynced')
                await asyncio.sleep(2)
                await self.send_tracks_update()

    async def click(self):
        """Tracks the current beat and bar internally. Loops indefinitely."""
        while True:
            now = time.monotonic()
            absolute_beat_number = round(self._absolute_beat_number(now))
            self.current_beat = (absolute_beat_number % BEATS_PER_BAR) + 1  # one-indexed
            self.current_bar = math.floor(absolute_beat_number / BEATS_PER_BAR)

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
        self.bpm = bpm
        self.origin = (time.monotonic() + 0.5 * self._get_seconds_per_beat())  # prevent scratch noise when sliding BPM
        self.audio.clicktrack_bpm = self.bpm
        self.audio.clicktrack_origin = self.origin

    async def request_recording(
            self,
            track_id: int,
            bars_to_record: int,
    ):
        """Records the previous ``bars_to_record`` bars for track ``track_id`` and starts playing."""
        if not self.current_bar:
            return

        track = self.tracks.get(track_id)
        if not track:
            logger.warning('Cannot request recording for unknown track: %s', track_id)
            return
        if track.state != TrackState.EMPTY:
            logger.warning('Cannot request recording for already recording track: %s', track_id)
            return

        now = time.monotonic()
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
        audio_origin = self.audio.origin
        logger.info(
            'request_recording track=%d bars=%d bar_start=%.3f bar_end=%.3f '
            'audio_origin=%s session_origin=%.3f',
            track_id, bars_to_record, bar_start_time, bar_end_time,
            f'{audio_origin:.3f}' if audio_origin == audio_origin else 'NaN',
            self.origin,
        )
        if audio_origin == audio_origin and bar_start_time < audio_origin:
            logger.warning(
                'Track %d: bar_start_time (%.3f) is %.3f s before audio origin (%.3f) — '
                'those samples were never recorded; expect silence at loop start.',
                track_id, bar_start_time, audio_origin - bar_start_time, audio_origin,
            )

        track.state = TrackState.TRIGGERED
        track.start_timestamp = bar_start_time
        track.end_timestamp = bar_end_time
        track.state = TrackState.RECORDING
        self._send_status('Recording')
        await self.send_tracks_update()

        time_to_wait = bar_end_time - time.monotonic()
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

        self._send_status('Playing')
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
        if not self.current_bar:
            return

        track = self.tracks.get(track_id)
        if not track:
            logger.warning('Cannot stop playing for unknown track: %s', track_id)
            return
        if track.state != TrackState.STOPPED:
            logger.warning('Cannot start playing for a track in state %s: %s', track.state, track_id)
            return
        track.state = TrackState.TRIGGERED
        self._send_status('Triggered')
        await self.send_tracks_update()

        bar_start_time, _ = self._get_bar_interval(
            current_bar=self.current_bar,
            number_of_bars_offset=1,
        )
        logger.debug('Playing starts at %s for track ID %s', bar_start_time, track_id)

        time_to_wait = bar_start_time - time.monotonic()
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)

        track.state = TrackState.PLAYING
        self._send_status('Playing')
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
        if not self.current_bar:
            return

        track = self.tracks.get(track_id)
        if not track:
            logger.warning('Cannot stop playing for unknown track: %s', track_id)
            return
        if track.state != TrackState.PLAYING:
            logger.warning('Cannot stop playing for a track in state %s: %s', track.state, track_id)
            return
        track.state = TrackState.STOPPING
        self._send_status('Stopping')
        await self.send_tracks_update()

        bar_start_time, _ = self._get_bar_interval(
            current_bar=self.current_bar,
            number_of_bars_offset=1,
        )
        logger.debug('Track ID %s stops at %s', track_id, bar_start_time)

        time_to_wait = bar_start_time - time.monotonic()
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)

        track.state = TrackState.STOPPED
        self._send_status('Stopped')
        self.audio.reset_loop(track.track_id)
        await self.send_tracks_update()
        logger.debug('Recording stopped')

    async def reset(self):
        """Resets all tracks to their starting state."""
        if not self.current_bar:
            return

        for track_id in self.tracks.keys():
            self.audio.reset_loop(track_id)
        self._initialize_tracks()
        self._send_status('Reset')
        await self.send_tracks_update()
        # TODO: empty memory

    async def send_tracks_update(self) -> None:
        """Renders the 6-track state as two groups of three on LCD row 0."""
        chars = [_TRACK_CHARS[self.tracks[i].state] for i in range(NUMBER_OF_TRACKS)]
        self.screen.write_line(0, ''.join(chars[:3]) + ' ' + ''.join(chars[3:]))

    def _send_status(self, message: str) -> None:
        self.screen.write_line(1, message)
