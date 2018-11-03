# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools
import pickle
import random
import tempfile
import zlib
from itertools import count
from typing import Iterable, Generator

import numpy as np

from src.utils.common_utils import GlobalNames
from .dataset import Record, zip_records

__all__ = [
    'DataIterator'
]

random.seed(GlobalNames.SEED)

DEFAULT_BUFFER_SIZE_FACTOR = 20


class Batch(object):
    """
    'Batch' is a list of 'Record's which can coalesce into one batch.

    'content' is a list of records which will be packed into one batch
    """

    def __init__(self, *records):

        self.content = list(records)

    def unpack(self):
        """ Unpack a 'Batch' instance into batched data.

        records in a batch will be split into several list according to the number of
        fields. For example, if a batch has three records R1, R2, R3. Ri has two fields, the
        the value of which are [a, b], then the result of unpack will be two lists, i.e.
        [a1, a2, a3], [b1, b2, b3]
        """
        n_fields = self.content[0].n_fields  # all the records must have the same field

        outs = tuple([r.fields[ii] for r in self.content] for ii in range(n_fields))

        if n_fields == 1:
            return outs[0]
        else:
            return outs

    @classmethod
    def pack(cls, *records: Record) -> 'Batch':
        """
        Pack a list of records into a batch.
        """

        return cls(*records)


def fill_buffer(data_iter, stop, key):
    records = []

    n_samples = 0
    key_values = 0

    while True:
        try:
            record = next(data_iter)
        except StopIteration:
            break

        records.append(record)

        n_samples += 1
        key_values += record.index

        if key == "samples":
            if n_samples >= stop:
                break
        else:
            if key_values >= stop:
                break

    return records


def add_noise_to_length(lengths, noise=1.0):
    """Add noise to the length of sequences.

    Args:
        lengths: The length of sequences.
        noise_ratio: The ratio to add noise to the lengths.
    """

    noisy_lengths = [l + np.random.uniform(- noise, noise) for l in lengths]

    return noisy_lengths


def numbering_records_iterator(record_iter: Iterable[Record]):
    """Numbering iterator from dataset.
    """
    for ii in count():
        try:
            record = next(record_iter)
        except StopIteration:
            break

        yield zip_records(Record(ii, index=-float('inf')), record)


def shuffle_iterator(iterator: Iterable[Record]) -> Generator[Record, None, None]:
    buffer = []

    for item in iterator:
        buffer.append(item)

    random.shuffle(buffer)
    buffer = [zlib.compress(pickle.dumps(obj)) for obj in buffer]
    tmp_handle = tempfile.TemporaryFile(mode="a+b")
    tmp_handle.writelines(buffer)
    del buffer

    tmp_handle.seek(0)

    for item in tmp_handle:
        yield pickle.loads(zlib.decompress(item))


def split_shards_iterator(iterator: Iterable[Record], number_shards, n_shard) -> Generator[Record, None, None]:
    for item in itertools.islice(iterator, n_shard, None, number_shards):
        yield item


def bucket_iterator(iterator: Iterable[Record], buffer_size, batching_key) -> Generator[Record, None, None]:
    buffer = []

    while True:

        # 1. fill buffer
        if len(buffer) == 0:
            _inc_buffer = fill_buffer(iterator, buffer_size, key=batching_key)

            if len(_inc_buffer) == 0:
                break
            else:
                buffer = buffer + _inc_buffer

            # 2. Sorting buffer
            scores = np.array([record.index for record in buffer])
            noisy_scores = add_noise_to_length(scores)
            sorted_indices = np.argsort(noisy_scores).tolist()
            buffer = [buffer[i] for i in sorted_indices]

        yield buffer.pop(0)


def batching_iterator(iterator: Iterable[Record], batch_size, batching_key) -> Generator[Batch, None, None]:
    batch_buffer = []

    num_samples = 0
    max_tokens_per_sample = 0

    for item in iterator:
        batch_buffer.append(item)
        num_samples += 1

        if batching_key == "samples":

            if num_samples >= batch_size:
                yield Batch.pack(*batch_buffer)

                num_samples = 0
                batch_buffer = []
        else:
            max_tokens_per_sample = max(max_tokens_per_sample, item.index)

            if max_tokens_per_sample * num_samples >= batch_size:
                yield Batch.pack(*batch_buffer)

                num_samples = 0
                max_tokens_per_sample = 0
                batch_buffer = []

    if len(batch_buffer) > 0:
        yield Batch.pack(*batch_buffer)


class DataIterator(object):
    """
    ```DataIterator``` defines the way to group your data into a batch. You can choose the way to batchify your data.
    In current implementation, we only provide "samples" and "tokens", which are the two main methods in machine
    translation.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 buffer_size=None,
                 use_bucket=True,
                 batching_func="samples",
                 numbering=False,
                 shuffle=False,
                 world_size=1,
                 rank=0
                 ):

        """ Build data iterator given a dataset

        Args:
            dataset: An Dataset Object
            batch_size: Integer. Size of a batch. When batching_key is "samples", it represents the
                the number of samples. When batching_key is "tokens", it represents the tokens in a batch.
            use_bucket: Boolean value. Whether to use bucket.
            batching_key: Criterion to allocate a batch. Can only be "samples" or "tokens"
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Batching Key
        #
        # We have two kinds of batching key, ```tokens``` and ```samples```.
        # For tokens, we allocate a batch according to the number of tokens in it. For example,
        # in machine translation, if we use "tokens" as the key and set the batch_size as 4096,
        # we allocate a batch when the number of tokens at source or target side reach 4096.
        # For samples, we allocate a batch according to the number of samples in it. In machine
        # translation, 50 batch size with "samples" as key means 50 bi-text sentences.

        if batching_func not in {"samples", "tokens"}:
            raise ValueError("Unknown batching key {0}".format(batching_func))
        self._batching_key = batching_func

        # buffer size for bucketing
        # buffer size is the max number of batches in a buffer
        # if batching key is 'samples', buffer size is 100 times of batch size,
        # else we suppose that their are 50 tokens in one sample and then estimate
        # the number of samples in one batch as self.batch_size // 50

        if buffer_size is None:
            buffer_size = self.batch_size * DEFAULT_BUFFER_SIZE_FACTOR

        self._buffer_size = buffer_size
        self.use_bucket = use_bucket
        self.numbering = numbering

        # For distributed learning
        self.world_size = world_size
        self.rank = rank

        self.reset()

    def __len__(self):
        return len(self.dataset)

    @property
    def n_datasets(self):
        return self.dataset.n_fields

    @property
    def is_end(self):
        return self._end

    def reset(self):

        self.buffer = []

        # 1. build data_iterator from dataset
        data_iter = self.dataset.read()

        # 2. numbering (optional)
        if self.numbering:
            data_iter = numbering_records_iterator(data_iter)

        # 3. shuffle (optional)
        if self.shuffle:
            data_iter = shuffle_iterator(data_iter)

        # 4. distributed(optional)
        if self.world_size > 1:
            data_iter = split_shards_iterator(data_iter, number_shards=self.world_size, n_shard=self.rank)

        # 5. bucketing (optional)
        if self.use_bucket:
            data_iter = bucket_iterator(data_iter, buffer_size=self._buffer_size, batching_key=self._batching_key)

        # 5. batching
        data_iter = batching_iterator(data_iter, batch_size=self.batch_size, batching_key=self._batching_key)

        self.data_iter = data_iter

        self._end = False

    def build_generator(self, batch_size=None):

        while True:

            # Accumulated batches until reach the batch_size
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.reset()
                break

            yield batch.unpack()
