from pequegrad.tensor import Tensor
from typing import List
from pequegrad import ops


class ShardedTensor:
    def __init__(self, shards: List[Tensor]):
        self.shards = shards

    def __add__(self, other):
        return ShardedTensor([a + b for a, b in zip(self.shards, other.shards)])

    def __sub__(self, other):
        return ShardedTensor([a - b for a, b in zip(self.shards, other.shards)])

    def __mul__(self, other):
        return ShardedTensor([a * b for a, b in zip(self.shards, other.shards)])

    def __truediv__(self, other):
        return ShardedTensor([a / b for a, b in zip(self.shards, other.shards)])

    def __neg__(self):
        return ShardedTensor([-a for a in self.shards])

    def __repr__(self):
        # nice format
        strs = [str(shard) for shard in self.shards]

        return "ShardedTensor(\n" + ",\n".join(strs) + "\n)"


def shard_tensor(
    x: Tensor, num_shards: int, dim: int = 0, devices=None
) -> ShardedTensor:
    shards = ops.chunk(x, num_shards, dim)
    if devices is not None:
        shards = [shard.eval().to(device) for shard, device in zip(shards, devices)]
    return ShardedTensor(shards)


def unshard_tensor(x: ShardedTensor, dim: int = 0, device=None) -> Tensor:
    device = device or x.shards[0].device
    if device is not None:
        shards = [shard.eval().to(device) for shard in x.shards]
    else:
        shards = x.shards
    return ops.cat(shards, dim)
