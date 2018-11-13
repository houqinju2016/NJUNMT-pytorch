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
from collections import namedtuple

import torch
import torch.nn as nn

__all__ = [
    'init',
    'get_world_size',
    'get_rank',
    'all_gather',
    'all_reduce',
    'all_gather_object',
    'all_gather_object_with_shared_fs'
]

_use_c10d = [True]

C10dStatus = namedtuple('C10dStatus', ['has_c10d', 'is_default'])

if hasattr(nn.parallel, 'deprecated'):
    c10d_status = C10dStatus(has_c10d=True, is_default=True)
elif hasattr(torch.distributed, 'c10d') and hasattr(torch.distributed.c10d, 'init_process_group'):
    c10d_status = C10dStatus(has_c10d=True, is_default=False)
else:
    c10d_status = C10dStatus(has_c10d=False, is_default=False)

if c10d_status.is_default:
    import torch.distributed as dist_c10d
    import torch.distributed.deprecated as dist_no_c10d
elif c10d_status.has_c10d:
    import torch.distributed.c10d as dist_c10d
    import torch.distributed as dist_no_c10d
else:
    import torch.distributed as dist_no_c10d


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


def init():
    current_env = os.environ.copy()
    world_size = int(current_env["WORLD_SIZE"])
    rank = int(current_env["RANK"])

    if not c10d_status.has_c10d:
        _use_c10d[0] = False

    if _use_c10d[0]:
        init_fn = dist_c10d.init_process_group
    else:
        init_fn = dist_no_c10d.init_process_group

    init_fn(backend="nccl",
            world_size=world_size,
            rank=rank)


def get_rank():
    if _use_c10d[0]:
        return dist_c10d.get_rank()
    else:
        return dist_no_c10d.get_rank()


def get_world_size():
    if _use_c10d[0]:
        return dist_c10d.get_world_size()
    else:
        return dist_no_c10d.get_world_size()


def get_default_group():
    if _use_c10d[0]:
        return dist_c10d.group.WORLD
    else:
        return dist_no_c10d.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    if _use_c10d[0]:
        return dist_c10d.all_reduce(tensor, group=group)
    else:
        return dist_no_c10d.all_reduce(tensor, group=group)


def all_gather(tensor, group=None):
    world_size = get_world_size()
    output_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

    if group is None:
        group = get_default_group()
    if _use_c10d[0]:
        dist_c10d.all_gather(output_tensors, tensor, group=group)
    else:
        dist_no_c10d.all_gather(output_tensors, tensor, group=group)

    return output_tensors


def synchronize_all_processes():
    """Synchronize all processes by reducing a null tensor"""
    null_tensor = torch.zeros(1).cuda()

    _ = all_reduce(null_tensor)


def all_gather_object(data, group=None, max_size=16384):
    """all_gather operation applied to any python object"""
    world_size = get_world_size()

    buffer_size = max_size
    if not hasattr(all_gather, '_buffer') or \
            all_gather_object._buffer.numel() < buffer_size:
        all_gather_object._buffer = torch.cuda.ByteTensor(buffer_size)

    buffer = all_gather_object._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))

    buffer_rank = buffer
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2:enc_size + 2] = torch.ByteTensor(list(enc))

    tensor_list = all_gather(buffer, group=group)

    result = []
    for i in range(world_size):
        out_buffer = tensor_list[i]
        size = (255 * item(out_buffer[0])) + item(out_buffer[1])
        if size > 0:
            result.append(
                pickle.loads(bytes(out_buffer[2:size + 2].tolist()))
            )
    return result


def all_gather_object_with_shared_fs(data, shared_fs_root="/tmp"):
    """ all_gather operation applied to any python object with shared file system

    Different with ```all_gather_object```, which is limited at size of object, this
    function can apply to any size of object with the help of a shared filesystem.

    This function will firstly save object shards into pickle files under a directory which can be visited by
    all the nodes. After that, all the file names will be all gathered by all_gather_object. At last, object
    will be combined into a list and return.
    """
    tmp_protoc = SharedFSTransferProtocol.write(data, shared_fs_root=shared_fs_root)

    gathered_tmp_protoc = all_gather_object(tmp_protoc)

    gathered_data = [protoc.read() for protoc in gathered_tmp_protoc]

    synchronize_all_processes()

    if get_rank() == 0:
        for protoc in gathered_tmp_protoc:
            protoc.close()

    return gathered_data
