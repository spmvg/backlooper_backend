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

from backlooper.audio import AudioStream
from backlooper.config import DEFAULT_BPM, LOGS_FORMAT
from backlooper.lcd import LCDScreen
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

    session = Session(
        bpm=DEFAULT_BPM,
        audio=audio,
        screen=LCDScreen(),
    )

    async def main():
        session.run()
        await asyncio.Future()  # run forever

    asyncio.run(main())
