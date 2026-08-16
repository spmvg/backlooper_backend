"""
This is the main entrypoint for the backlooper backend.
It is started by running: ``python -m backlooper``.
"""
import argparse
import asyncio
import logging
import multiprocessing
import os

from sounddevice import query_devices

import mido

from backlooper.audio import AudioStream
from backlooper.config import DEFAULT_BPM, LOGS_FORMAT
from backlooper.lcd import LCDScreen
from backlooper.midi import MidiController
from backlooper.session import Session

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Backlooper')
    parser.add_argument('--debug', help='Enable debug logging', action='store_true')
    args = parser.parse_args()
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG

    logger = logging.getLogger('backlooper')
    logger.setLevel(log_level)
    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(logging.Formatter(LOGS_FORMAT))
    logger.addHandler(screen_handler)

    logger.info('Available sound devices:\n\n%s\n', query_devices())

    _input_device_id_env = os.environ.get('INPUT_DEVICE_ID')
    if _input_device_id_env is not None:
        logger.info('Taking input device ID from INPUT_DEVICE_ID env var: %s', _input_device_id_env)
        input_device_id = int(_input_device_id_env)
    else:
        try:
            input_device_id = int(input('Enter the input device ID (integer): '))
        except ValueError:
            logger.error('Invalid input device ID. Please enter an integer.')
            exit(1)

    _output_device_id_env = os.environ.get('OUTPUT_DEVICE_ID')
    if _output_device_id_env is not None:
        logger.info('Taking output device ID from OUTPUT_DEVICE_ID env var: %s', _output_device_id_env)
        output_device_id = int(_output_device_id_env)
    else:
        try:
            output_device_id = int(input('Enter the output device ID (integer): '))
        except ValueError:
            logger.error('Invalid output device ID. Please enter an integer.')
            exit(1)

    audio = AudioStream(
        log_level=log_level,
        input_device_id=input_device_id,
        output_device_id=output_device_id,
    )

    _midi_port_env = os.environ.get('MIDI_PORT_NAME')
    midi_port_name = None
    _midi_ports = mido.get_input_names()
    if not _midi_ports:
        logger.warning('No MIDI input ports found; MIDI disabled')
    elif _midi_port_env is not None:
        logger.info('Taking MIDI port from MIDI_PORT_NAME env var: %s', _midi_port_env)
        midi_port_name = _midi_port_env
    else:
        logger.info('MIDI_PORT_NAME not set; MIDI disabled. Available ports:\n%s',
                    '\n'.join(f'  {i}: {name}' for i, name in enumerate(_midi_ports)))

    screen = LCDScreen()
    session = Session(
        bpm=DEFAULT_BPM,
        audio=audio,
        screen=screen,
    )

    async def main():
        loop = asyncio.get_running_loop()
        midi_ctrl = MidiController(session=session, screen=screen, loop=loop)
        if midi_port_name:
            midi_ctrl.open(midi_port_name)
        session.run()
        try:
            await asyncio.Future()  # run forever
        finally:
            midi_ctrl.close()
            screen.write_line(0, '')
            screen.write_line(1, 'Backlooper off')
            screen.close()

    asyncio.run(main())
