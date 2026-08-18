"""
MIDI input: long-press any button for 5 s to enter mapping mode, then sequentially assign
controls to all slots.  In normal operation, mapped controls are dispatched to session actions.
All I/O is callback-driven (rtmidi thread); async session calls are posted to the event loop.
"""
import asyncio
import json
import logging
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import mido

from backlooper.config import BPM_MAX, BPM_MIN, DEFAULT_BARS_TO_RECORD

logger = logging.getLogger(__name__)

MAPPING_FILE = Path.home() / '.backlooper_midi_map.json'
LONG_PRESS_SECONDS = 5.0
MAPPING_TIMEOUT_SECONDS = 5.0
FIRST_MAPPING_TIMEOUT_SECONDS = MAPPING_TIMEOUT_SECONDS * 2
FADER_DEBOUNCE_SECONDS = 2.0


class ActionType(str, Enum):
    TRACK_TOGGLE = 'track_toggle'
    BARS_TO_RECORD = 'bars_to_record'
    CLICKTRACK_VOLUME = 'clicktrack_volume'
    TEMPO = 'tempo'
    RESET = 'reset'


# Ordered list of controls the user maps one by one.
# input_type 'button' only accepts note_on; 'fader' only accepts control_change.
_SLOTS: List[Dict[str, Any]] = [
    {'label': f'Track {i + 1}', 'action': ActionType.TRACK_TOGGLE, 'track_id': i, 'input_type': 'button'}
    for i in range(6)
] + [
    {'label': 'Reset tracks', 'action': ActionType.RESET, 'input_type': 'button'},
    {'label': 'Tempo',   'action': ActionType.TEMPO,             'input_type': 'fader'},
    {'label': 'Bars to record',    'action': ActionType.BARS_TO_RECORD,    'input_type': 'fader'},
    {'label': 'Click Volume',  'action': ActionType.CLICKTRACK_VOLUME, 'input_type': 'fader'},
]


def _midi_key(msg: mido.Message) -> str:
    """Stable JSON key for a control, ignoring velocity / CC value."""
    if msg.type in ('note_on', 'note_off'):
        return json.dumps(['note', msg.channel, msg.note])
    if msg.type == 'control_change':
        return json.dumps(['cc', msg.channel, msg.control])
    return json.dumps([msg.type, msg.channel])


def _load_map() -> Dict[str, Dict]:
    if not MAPPING_FILE.exists():
        logger.info('No MIDI map at %s, starting empty', MAPPING_FILE)
        return {}
    try:
        data = json.loads(MAPPING_FILE.read_text())
        logger.info('Loaded MIDI map: %d entries from %s', len(data), MAPPING_FILE)
        return data
    except Exception as exc:
        logger.warning('Could not load MIDI map (%s), starting empty', exc)
        return {}


def _save_map(midi_map: Dict[str, Dict]) -> None:
    try:
        MAPPING_FILE.write_text(json.dumps(midi_map, indent=2))
        logger.info('MIDI map saved: %d entries → %s', len(midi_map), MAPPING_FILE)
    except Exception as exc:
        logger.warning('Could not save MIDI map: %s', exc)


class MidiController:
    """
    Owns a MIDI input port and routes messages to session actions.

    Long-press (5 s) on any MIDI button enters mapping mode.
    In mapping mode, controls are assigned sequentially to the slots in ``_SLOTS``.
    After all slots are mapped, or after 5 s of inactivity, the partial map is persisted.
    """

    def __init__(self, session: Any, screen: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._session = session
        self._screen = screen
        self._loop = loop
        self._port: Optional[mido.ports.BaseInput] = None
        self._midi_map: Dict[str, Dict] = _load_map()
        self._bars_to_record = DEFAULT_BARS_TO_RECORD

        # long-press detection (normal mode)
        self._held_key: Optional[str] = None
        self._long_press_timer: Optional[threading.Timer] = None

        # mapping-mode state
        self._in_mapping = False
        self._slot_index = 0
        self._partial_map: Dict[str, Dict] = {}
        self._last_capture_time = 0.0
        self._fader_accept_after = 0.0
        self._fader_debounce_timer: Optional[threading.Timer] = None
        self._timeout_timer: Optional[threading.Timer] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def open(self, port_name: str) -> None:
        self._port = mido.open_input(port_name, callback=self._on_message)
        logger.info('MIDI port opened: %s', port_name)

    def close(self) -> None:
        self._cancel_all_timers()
        if self._port:
            self._port.close()

    def _cancel_all_timers(self) -> None:
        for attr in ('_long_press_timer', '_fader_debounce_timer', '_timeout_timer'):
            t = getattr(self, attr, None)
            if t:
                t.cancel()
        self._long_press_timer = None
        self._timeout_timer = None

    # ── main callback (rtmidi thread) ─────────────────────────────────────

    def _on_message(self, msg: mido.Message) -> None:
        # suppress high-frequency system messages
        if msg.type in ('clock', 'active_sensing', 'sysex'):
            return
        if self._in_mapping:
            self._on_mapping_message(msg)
        else:
            self._on_normal_message(msg)

    # ── normal mode ───────────────────────────────────────────────────────

    def _on_normal_message(self, msg: mido.Message) -> None:
        is_press   = msg.type == 'note_on' and msg.velocity > 0
        is_release = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)

        if is_press:
            self._arm_long_press(_midi_key(msg))
        elif is_release:
            key = _midi_key(msg)
            if key == self._held_key and self._long_press_timer is not None:
                # short press → cancel timer and dispatch
                self._long_press_timer.cancel()
                self._long_press_timer = None
                self._held_key = None
                self._dispatch_key(key, value=127)
            else:
                self._held_key = None
        elif msg.type == 'control_change':
            self._dispatch_key(_midi_key(msg), value=msg.value)

    def _arm_long_press(self, key: str) -> None:
        self._held_key = key
        if self._long_press_timer:
            self._long_press_timer.cancel()
        t = threading.Timer(LONG_PRESS_SECONDS, self._long_press_fired)
        t.daemon = True
        t.start()
        self._long_press_timer = t

    def _long_press_fired(self) -> None:
        self._long_press_timer = None
        self._held_key = None
        logger.info('Long press detected — entering MIDI mapping mode')
        self._screen.write_line(0, 'MAPPING MODE')
        self._screen.write_line(1, 'Hold to exit...')
        time.sleep(1.0)
        self._enter_mapping()

    # ── dispatch ──────────────────────────────────────────────────────────

    def _dispatch_key(self, key: str, value: int) -> None:
        entry = self._midi_map.get(key)
        if entry is None:
            return
        action = entry['action']
        if action == ActionType.TRACK_TOGGLE:
            track_id = entry.get('track_id')
            asyncio.run_coroutine_threadsafe(self._toggle_track(track_id), self._loop)
        elif action == ActionType.BARS_TO_RECORD:
            self._bars_to_record = (1, 2, 4, 8)[min(value * 4 // 128, 3)]
            logger.info('Bars to record set to %d', self._bars_to_record)
            self._screen.write_line(1, f'Bars: {self._bars_to_record}')
        elif action == ActionType.CLICKTRACK_VOLUME:
            self._session.audio.clicktrack_volume = value / 127.0
        elif action == ActionType.TEMPO:
            bpm = round(BPM_MIN + (value / 127.0) * (BPM_MAX - BPM_MIN))
            asyncio.run_coroutine_threadsafe(self._session.set_bpm(bpm), self._loop)
        elif action == ActionType.RESET:
            asyncio.run_coroutine_threadsafe(self._session.reset(), self._loop)

    async def _toggle_track(self, track_id: int) -> None:
        from backlooper.session import TrackState
        track = self._session.tracks.get(track_id)
        if track is None:
            return
        state = track.state
        if state == TrackState.EMPTY:
            await self._session.request_recording(track_id, self._bars_to_record)
        elif state == TrackState.PLAYING:
            await self._session.stop_playing(track_id)
        elif state == TrackState.STOPPED:
            await self._session.start_playing(track_id)
        # TRIGGERED / RECORDING / STOPPING → in progress, ignore

    # ── mapping mode ──────────────────────────────────────────────────────

    def _enter_mapping(self) -> None:
        self._in_mapping = True
        self._slot_index = 0
        self._partial_map = {}
        self._last_capture_time = 0.0
        self._fader_accept_after = 0.0
        self._fader_debounce_timer = None
        self._show_slot_prompt()

    def _show_slot_prompt(self) -> None:
        slot = _SLOTS[self._slot_index]
        n = len(_SLOTS)
        logger.info('Mapping %d/%d: %s (%s)', self._slot_index + 1, n, slot['label'], slot['input_type'])
        self._screen.write_line(0, f"{self._slot_index + 1}/{n} {slot['label']}")
        self._screen.write_line(1, 'Press a ' + slot['input_type'])
        self._reset_timeout()

    def _reset_timeout(self) -> None:
        if self._timeout_timer:
            self._timeout_timer.cancel()
        timeout = FIRST_MAPPING_TIMEOUT_SECONDS if self._slot_index == 0 else MAPPING_TIMEOUT_SECONDS
        t = threading.Timer(timeout, self._timeout_fired)
        t.daemon = True
        t.start()
        self._timeout_timer = t

    def _timeout_fired(self) -> None:
        logger.info('Mapping timeout — %d/%d slots captured', len(self._partial_map), len(_SLOTS))
        self._finish_mapping()

    def _on_mapping_message(self, msg: mido.Message) -> None:
        if self._fader_debounce_timer is not None:
            return

        # ignore releases (e.g. lifting the long-press key that triggered mapping)
        is_release = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
        if is_release:
            return

        slot = _SLOTS[self._slot_index]
        expected = slot['input_type']

        # only accept the expected input type for this slot
        is_button = msg.type == 'note_on' and msg.velocity > 0
        is_fader  = msg.type == 'control_change'
        if expected == 'button' and not is_button:
            return
        if expected == 'fader' and not is_fader:
            return

        now = time.monotonic()
        if is_fader and now < self._fader_accept_after:
            self._reset_timeout()
            return
        self._last_capture_time = now

        key = _midi_key(msg)
        entry: Dict[str, Any] = {'action': slot['action']}
        if 'track_id' in slot:
            entry['track_id'] = slot['track_id']
        self._partial_map[key] = entry
        logger.info('  mapped %s → %s', key, slot['label'])
        if is_fader:
            self._fader_accept_after = now + FADER_DEBOUNCE_SECONDS
            logger.info('Release the fader; advancing in %.1f seconds', FADER_DEBOUNCE_SECONDS)
            self._screen.write_line(0, 'Fader mapped')
            self._screen.write_line(1, 'Release fader')
            timer = threading.Timer(FADER_DEBOUNCE_SECONDS, self._advance_mapping_slot)
            timer.daemon = True
            timer.start()
            self._fader_debounce_timer = timer
            return

        self._advance_mapping_slot()

    def _advance_mapping_slot(self) -> None:
        self._fader_debounce_timer = None
        self._slot_index += 1
        if self._slot_index >= len(_SLOTS):
            logger.info('All %d slots mapped', len(_SLOTS))
            self._finish_mapping()
        else:
            self._show_slot_prompt()

    def _finish_mapping(self) -> None:
        if self._timeout_timer:
            self._timeout_timer.cancel()
            self._timeout_timer = None
        self._in_mapping = False
        self._midi_map.update(self._partial_map)
        _save_map(self._midi_map)
        count = len(self._partial_map)
        logger.info('Mapping done: %d slot(s) saved', count)
        self._screen.write_line(0, f'Saved {count}/{len(_SLOTS)} ctrl')
        self._screen.write_line(1, 'Mapping done')
        t = threading.Timer(2.0, self._restore_display)
        t.daemon = True
        t.start()

    def _restore_display(self) -> None:
        asyncio.run_coroutine_threadsafe(self._session.send_tracks_update(), self._loop)
        self._session._send_status('Ready')
