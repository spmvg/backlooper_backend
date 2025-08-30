from unittest import TestCase
from uuid import uuid4

import numpy as np

from backlooper.striped_storage import StripedStorage, _dtype


class TestStripedStorage(TestCase):
    def test_write_striped_storage(self):
        array_size = 22
        audio = np.arange(array_size).reshape((int(array_size / 2), 2))

        prefix = str(uuid4())
        storage = StripedStorage(
            stripe_size=3,
            identifier=prefix
        )
        storage.write(
            start_index=99,
            array_to_store=audio
        )
        storage.write(  # check overwrite
            start_index=99,
            array_to_store=audio
        )
        desired_storage = {
            prefix + '33': np.array([
                [0., 1.],
                [2., 3.],
                [4., 5.]
            ]),
            prefix + '34': np.array([
                [6., 7.],
                [8., 9.],
                [10., 11.]
            ]),
            prefix + '35': np.array([
                [12., 13.],
                [14., 15.],
                [16., 17.]
            ]),
            prefix + '36': np.array([
                [18., 19.],
                [20., 21.],
                [0., 0.]
            ])
        }
        for key, desired_value in desired_storage.items():
            memory = storage._get_existing_memory(key)
            storage_array = np.ndarray(
                (storage.stripe_size, storage.channels),
                dtype=_dtype,
                buffer=memory.buf
            )
            np.testing.assert_equal(
                storage_array,
                desired_value,
            )

    def test_write_striped_storage_no_overwrite(self):
        array_size = 22
        audio = np.arange(array_size).reshape((int(array_size / 2), 2))

        prefix = str(uuid4())
        storage = StripedStorage(
            stripe_size=3,
            identifier=prefix
        )
        storage.write(
            start_index=99,
            array_to_store=audio,
            overwrite=False,
        )
        storage.write(
            start_index=99,
            array_to_store=audio,
            overwrite=False,
        )
        desired_storage = {
            prefix + '33': 2*np.array([
                [0., 1.],
                [2., 3.],
                [4., 5.]
            ]),
            prefix + '34': 2*np.array([
                [6., 7.],
                [8., 9.],
                [10., 11.]
            ]),
            prefix + '35': 2*np.array([
                [12., 13.],
                [14., 15.],
                [16., 17.]
            ]),
            prefix + '36': 2*np.array([
                [18., 19.],
                [20., 21.],
                [0., 0.]
            ])
        }
        for key, desired_value in desired_storage.items():
            memory = storage._get_existing_memory(key)
            storage_array = np.ndarray(
                (storage.stripe_size, storage.channels),
                dtype=_dtype,
                buffer=memory.buf
            )
            np.testing.assert_equal(
                storage_array,
                desired_value,
            )

    def test_write_striped_storage_on_stripe_end(self):
        array_size = 14
        audio = np.arange(array_size).reshape((int(array_size / 2), 2))

        prefix = str(uuid4())
        storage = StripedStorage(
            stripe_size=3,
            identifier=prefix
        )
        storage.write(
            start_index=101,
            array_to_store=audio
        )
        desired_storage = {
            prefix + '33': np.array([
                [0., 0.],
                [0., 0.],
                [0., 1.]
            ]),
            prefix + '34': np.array([
                [2., 3.],
                [4., 5.],
                [6., 7.]
            ]),
            prefix + '35': np.array([
                [8., 9.],
                [10., 11.],
                [12., 13.]
            ]),
        }
        for key, desired_value in desired_storage.items():
            memory = storage._get_existing_memory(key)
            storage_array = np.ndarray(
                (storage.stripe_size, storage.channels),
                dtype=_dtype,
                buffer=memory.buf
            )
            np.testing.assert_equal(
                storage_array,
                desired_value,
            )

    def test_write_striped_to_single_stripe(self):
        audio = np.array([[1, 2]])

        prefix = str(uuid4())
        storage = StripedStorage(
            stripe_size=3,
            identifier=prefix
        )
        storage.write(
            start_index=100,
            array_to_store=audio
        )
        desired_storage = {
            prefix + '33': np.array([
                [0., 0.],
                [1., 2.],
                [0., 0.]
            ]),
        }
        for key, desired_value in desired_storage.items():
            memory = storage._get_existing_memory(key)
            storage_array = np.ndarray(
                (storage.stripe_size, storage.channels),
                dtype=_dtype,
                buffer=memory.buf
            )
            np.testing.assert_equal(
                storage_array,
                desired_value,
            )

    def test_read_striped_storage(self):
        array_size = 20
        audio = np.arange(array_size).reshape((int(array_size / 2), 2))

        prefix = str(uuid4())
        storage = StripedStorage(
            stripe_size=3,
            identifier=prefix
        )
        storage.write(
            start_index=100,
            array_to_store=audio
        )
        result = storage.read(
            start_index=100,
            length=10,
        )
        np.testing.assert_equal(
            result,
            audio
        )

    def test_read_nothing_shape(self):
        array_size = 20
        audio = np.arange(array_size).reshape((int(array_size / 2), 2))

        prefix = str(uuid4())
        storage = StripedStorage(
            stripe_size=3,
            identifier=prefix
        )
        storage.write(
            start_index=100,
            array_to_store=audio
        )
        result = storage.read(
            start_index=100,
            length=0,
        )
        np.testing.assert_equal(
            np.zeros((0, 2)),
            result
        )

    def test_read_missing_partition(self):
        prefix = str(uuid4())
        storage = StripedStorage(
            stripe_size=3,
            identifier=prefix
        )
        result = storage.read(
            start_index=100,
            length=10,
        )
        np.testing.assert_equal(
            np.zeros((10, 2)),
            result
        )

    def test_index_to_key(self):
        prefix = str(uuid4())
        storage = StripedStorage(identifier=prefix, stripe_size=100)
        self.assertEqual(
            (prefix+'11', 50),
            storage._index_to_key_offset(1150)
        )

    def test_key_to_index(self):
        prefix = str(uuid4())
        storage = StripedStorage(identifier=prefix, stripe_size=100)
        self.assertEqual(
            1150,
            storage._key_offset_to_index((prefix+'11', 50))
        )
