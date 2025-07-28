"""
This is the main entrypoint for the backlooper.app backend.
It is started by running: ``python -m backlooper``.

``__main__`` starts a websockets endpoint on the local machine for the frontend to connect to.
The frontend and backend communicate by sending JSON messages back and forth.
The implementation is based on ``asyncio``.
More documentation is provided in the underlying modules, such as ``audio`` and ``session``.
"""
import argparse
import asyncio
import json
import logging
import multiprocessing
from json import JSONDecodeError

import websockets

from backlooper.audio import AudioStream
from backlooper.config import EVENT_TYPE_KEY, INITIALIZE_EVENT, ERROR_EVENT, \
    MESSAGE_KEY, SET_BPM_EVENT, \
    BPM_EVENT, BPM_KEY, STOP_EVENT, CALIBRATE_EVENT, LOGS_FORMAT, VOLUME_KEY, \
    CLICKTRACK_VOLUME_EVENT, LATENCY_EVENT, LATENCY_KEY, START_EVENT, \
    BACKLOOP_EVENT, RESET_EVENT, TRACK_KEY, \
    BARS_TO_RECORD_KEY, DEFAULT_BPM, MAJOR_VERSION_EVENT, MAJOR_VERSION, \
    USING_AUTOMATIC_LATENCY_CORRECTION_KEY
from backlooper.session import Session

if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='Backlooper')
    parser.add_argument('--input-device-id', help='ID (integer) of the input device to use. Run once without this argument and check the logs to see which IDs are available.', type=int)
    parser.add_argument('--output-device-id', help='ID (integer) of the output device to use. Run once without this argument and check the logs to see which IDs are available.', type=int)
    parser.add_argument('--debug', help='Enable debug logging', action='store_true')
    args = parser.parse_args()
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG

    logger = logging.getLogger('backlooper')  # logger of backlooper calls itself backlooper
    logger.setLevel(log_level)
    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(logging.Formatter(LOGS_FORMAT))
    logger.addHandler(screen_handler)

    audio = AudioStream(
        log_level=log_level,
        input_device_id=args.input_device_id,
        output_device_id=args.output_device_id,
    )

    session = Session(
        bpm=DEFAULT_BPM,
        audio=audio,
    )

    async def handle_initialize(websocket, event):
        session.register_websocket(websocket)
        await websocket.send(json.dumps({
            EVENT_TYPE_KEY: BPM_EVENT,
            BPM_KEY: session.bpm
        }))
        await websocket.send(json.dumps({
            EVENT_TYPE_KEY: CLICKTRACK_VOLUME_EVENT,
            VOLUME_KEY: session.audio.clicktrack_volume
        }))
        await session.send_latency_update()
        await session.send_tracks_update()
        await websocket.send(json.dumps({
            EVENT_TYPE_KEY: MAJOR_VERSION_EVENT,
            MESSAGE_KEY: MAJOR_VERSION,
            USING_AUTOMATIC_LATENCY_CORRECTION_KEY: bool(audio.using_automatic_latency_correction.value),
        }))
        logger.debug('GUI connected')

    async def handle_set_bpm(websocket, event):
        loop = asyncio.get_event_loop()
        loop.create_task(session.set_bpm(event[BPM_KEY]))
        logger.debug('Set BPM to %s', event[BPM_KEY])

    async def handle_backloop(websocket, event):
        loop = asyncio.get_event_loop()
        loop.create_task(session.request_recording(
            track_id=event[TRACK_KEY],
            bars_to_record=event[BARS_TO_RECORD_KEY],
        ))

    async def handle_start(websocket, event):
        loop = asyncio.get_event_loop()
        loop.create_task(session.start_playing(event[TRACK_KEY]))

    async def handle_stop(websocket, event):
        loop = asyncio.get_event_loop()
        loop.create_task(session.stop_playing(event[TRACK_KEY]))

    async def handle_calibrate(websocket, event):
        loop = asyncio.get_event_loop()
        loop.create_task(session.calibrate())

    async def handle_reset(websocket, event):
        loop = asyncio.get_event_loop()
        loop.create_task(session.reset())

    async def handle_unknown(websocket, event):
        await websocket.send(json.dumps({
            EVENT_TYPE_KEY: ERROR_EVENT,
            MESSAGE_KEY: 'Unknown event type'
        }))
        logger.warning('Unknown event type in event: %s', event)

    async def handle_clicktrack_volume(websocket, event):
        session.audio.clicktrack_volume = max(min(event[VOLUME_KEY], 1), 0)

    async def handle_latency(websocket, event):
        session.audio.latency_seconds = event[LATENCY_KEY]

    message_handler = {
        INITIALIZE_EVENT: handle_initialize,
        SET_BPM_EVENT: handle_set_bpm,
        BACKLOOP_EVENT: handle_backloop,
        START_EVENT: handle_start,
        STOP_EVENT: handle_stop,
        CALIBRATE_EVENT: handle_calibrate,
        CLICKTRACK_VOLUME_EVENT: handle_clicktrack_volume,
        LATENCY_EVENT: handle_latency,
        RESET_EVENT: handle_reset,
    }

    async def handler(websocket):
        async for message in websocket:
            try:
                message_dict = json.loads(message)
            except JSONDecodeError:
                await websocket.send(json.dumps({
                    EVENT_TYPE_KEY: ERROR_EVENT,
                    MESSAGE_KEY: 'Cannot decode JSON'
                }))
                logger.warning('Cannot decode JSON in event: %s', message)
                continue

            await message_handler.get(message_dict[EVENT_TYPE_KEY], handle_unknown)(websocket, message_dict)

    # TODO: HTTPS (wss)
    async def main():
        session.run()
        async with websockets.serve(handler, "", 8001):
            await asyncio.Future()  # run forever

    asyncio.run(main())
