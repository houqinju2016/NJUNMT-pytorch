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

import os
import pickle
import uuid

import horovod.torch as hvd
import torch

__all__ = [
    'synchronize_all_processes',
    'all_gather_with_shared_fs',
    'all_gather_list'
]


def gen_random_name():
    """Return a random name for temp file"""
    return uuid.UUID(bytes=os.urandom(16), version=4).hex


class SharedFSTransferProtocol(object):
    """
    Protocol for transfering data between processes by shared filesystem.

    This is useful when you want to transfer some relative big data.
    """

    def __init__(self, prefix="/tmp", name=None):

        self.prefix = prefix

        if name is None:
            name = gen_random_name()

        self.name = name

        self.path = None

    def __getstate__(self):

        return {"prefix": self.prefix, "name": self.name, "path": self.path}

    def __setstate__(self, state):

        self.prefix = state['prefix']
        self.name = state['name']
        self.path = state['path']

    def read(self):

        if self.path is None:
            raise ValueError

        with open(self.path, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def _write(self, obj):

        self.path = os.path.join(self.prefix, self.name) + ".pkl"

        with open(self.path, "wb") as f:
            pickle.dump(obj, f)

    def close(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            # file has been removed by another process
            pass

    @classmethod
    def write(cls, obj, shared_fs_root="/tmp"):

        protoc = cls(prefix=shared_fs_root)
        protoc._write(obj)

        return protoc


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def synchronize_all_processes():
    """Synchronize all processes by reducing a null tensor"""
    null_tensor = torch.zeros(1)

    _ = hvd.allreduce(null_tensor)


def all_gather_with_shared_fs(data, shared_fs_root="/tmp"):
    tmp_protoc = SharedFSTransferProtocol.write(data, shared_fs_root=shared_fs_root)

    gathered_tmp_protoc = all_gather_list(tmp_protoc)

    gathered_data = [protoc.read() for protoc in gathered_tmp_protoc]

    synchronize_all_processes()

    if hvd.rank() == 0:
        for protoc in gathered_tmp_protoc:
            protoc.close()

    return gathered_data


def all_gather_list(data, max_size=65000):
    """ Gathers arbitrary data from all nodes into a list.

    This function is heavily borrowed from fairseq (https://github.com/pytorch/fairseq)

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """

    world_size = hvd.size()

    buffer_size = max_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)

    buffer = all_gather_list._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))

    buffer_rank = buffer
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2:enc_size + 2] = torch.ByteTensor(list(enc))

    buffer_gathered = hvd.allgather(buffer)

    result = []
    for i in range(world_size):
        out_buffer = buffer_gathered[i * max_size: (i + 1) * max_size]
        size = (255 * item(out_buffer[0])) + item(out_buffer[1])
        if size > 0:
            result.append(
                pickle.loads(bytes(out_buffer[2:size + 2].tolist()))
            )
    return result
