"""
This module contains the implementation of storage of audio for recording and playback.
"""
from dataclasses import dataclass
from math import floor
from multiprocessing import shared_memory
from typing import Tuple

import numpy as np


KeyOffset = Tuple[str, int]
"""
``KeyOffset`` is a pointer towards audio data in the striped storage.
It contains of two parts:

- The key of the striped storage
- The offset in the striped storage

To allow multiple processes to read and write the same storage, ``multiprocessing.shared_memory`` is used.
"""

_dtype = float


@dataclass
class StripedStorage:
    """
    ``StripedStorage`` stores audio data and can be thought of as a tape storage with a start and end.
    In order to prevent having to allocate a big amount of memory, the data is partitioned in chunks, called "stripes".

    The underlying data has a dictionary structure with the following keys and values:

    - `key_0`: ``np.array`` of size (``stripe_size``, ``channels``),
    - `key_1`: ``np.array`` of size (``stripe_size``, ``channels``),
    - `key_2`: ``np.array`` of size (``stripe_size``, ``channels``),
    - ...

    Data in ``StripedStorage`` can be referenced in two ways:

    - Either by an absolute index, starting from zero (the first datapoint),
    - Or by a ``KeyOffset``, which includes the stripe key and offset within the stripe.
    """
    identifier: str
    """
    Unique identifier of the shared memory which stores the audio data.
    Different processes can use this identifier to access the shared memory.
    """
    stripe_size: int = 10000
    """Size in samples of a single stripe."""
    channels: int = 2
    """Number of channels per data point (stereo)."""

    def __post_init__(self):
        self._stripe_size_bytes = self._empty_storage().nbytes
        self._memory_handles = {}  # TODO: garbage collecting
        self._unique_prefix = self.identifier + '-'

    def _empty_storage(self):
        """Returns a stripe without any data."""
        return np.zeros((self.stripe_size, self.channels), dtype=_dtype)

    def _index_to_key_offset(self, start_index: int) -> KeyOffset:
        """
        Returns a ``KeyOffset`` for a certain start index ``start_index``, determining which stripe contains the
        requested data.
        ``start_index`` can be bigger than the stripe size.
        """
        return self._unique_prefix + str(floor(start_index / self.stripe_size)), start_index % self.stripe_size

    def _key_offset_to_index(self, key: KeyOffset) -> int:
        """Inverse of ``_index_to_key_offset``."""
        return int(key[0][len(self._unique_prefix):]) * self.stripe_size + key[1]

    def read(
            self,
            start_index: int,
            length: int,
    ) -> np.array:
        """Reads ``length`` datapoints starting at index ``start_index``."""
        read_arrays = []
        read_data = 0
        while True:
            data_left_to_read = length - read_data
            if data_left_to_read == 0:
                break

            key, offset = self._index_to_key_offset(start_index + read_data)

            try:
                memory = self._get_existing_memory(key)
            except FileNotFoundError:
                memory = None

            storage_array = np.ndarray(
                (self.stripe_size, self.channels),
                dtype=_dtype,
                buffer=memory.buf
            ) if memory else None

            if offset + data_left_to_read <= self.stripe_size:
                read_arrays.append(
                    storage_array[
                        offset:offset + data_left_to_read,
                        :
                    ].copy() if memory else np.zeros((data_left_to_read, self.channels))
                )
                break

            read_arrays.append(
                storage_array[
                    offset:self.stripe_size,
                    :
                ].copy() if memory else np.zeros((self.stripe_size - offset, self.channels))
            )
            read_data += self.stripe_size - offset
        if not read_arrays:
            return np.zeros((0, self.channels))
        return np.concatenate(read_arrays)

    def write(
            self,
            start_index: int,
            array_to_store: np.array,
            overwrite: bool = True,
    ):
        """
        Writes array ``array_to_store`` starting at index ``start_index``.
        If ``overwrite`` is set, existing data is overwritten.
        If ``overwrite`` is not set, ``array_to_store`` is added to the existing data.
        """
        if array_to_store.shape[1] != self.channels:
            raise ValueError(f'array_to_store should have {self.channels} channels, not {array_to_store.shape[1]}')

        data_stored = 0
        while True:
            data_left_to_write = array_to_store.shape[0] - data_stored
            if data_left_to_write == 0:
                break

            key, offset = self._index_to_key_offset(start_index + data_stored)

            try:
                memory = self._get_existing_memory(key)
            except FileNotFoundError:
                memory = shared_memory.SharedMemory(
                    name=key,
                    create=True,
                    size=self._stripe_size_bytes
                )
                storage_array = np.ndarray(
                    (self.stripe_size, self.channels),
                    dtype=_dtype,
                    buffer=memory.buf
                )
                storage_array[:] = 0
                self._memory_handles[key] = memory  # prevent the memory from being freed by the garbage collector

            storage_array = np.ndarray(
                (self.stripe_size, self.channels),
                dtype=_dtype,
                buffer=memory.buf
            )

            if offset + data_left_to_write <= self.stripe_size:
                if overwrite:
                    storage_array[
                        offset:offset + data_left_to_write,
                        :
                    ] = array_to_store[
                        data_stored:data_stored + data_left_to_write,
                        :
                    ]
                else:
                    storage_array[
                        offset:offset + data_left_to_write,
                        :
                    ] += array_to_store[
                        data_stored:data_stored + data_left_to_write,
                        :
                    ]
                break

            if overwrite:
                storage_array[
                    offset:self.stripe_size,
                    :
                ] = array_to_store[
                    data_stored:data_stored + self.stripe_size - offset,
                    :
                ]
            else:
                storage_array[
                    offset:self.stripe_size,
                    :
                ] += array_to_store[
                    data_stored:data_stored + self.stripe_size - offset,
                    :
                ]
            data_stored += self.stripe_size - offset

    @staticmethod
    def _get_existing_memory(key: str):
        """Returns the ``SharedMemory`` with key ``key``."""
        return shared_memory.SharedMemory(
            name=key,
            create=False
        )
