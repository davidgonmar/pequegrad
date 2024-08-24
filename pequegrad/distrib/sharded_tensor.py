from pequegrad.tensor import Tensor
from typing import List
from pequegrad import ops
from pequegrad.backend.c import device


class ShardedTensor:
    def __init__(self, shards: List[Tensor], dim: List[int] | None = None):
        self.shards = shards
        self.dim = dim

        if self.dim is not None:
            # compute the shape as the sum of the shapes of the shards in the dim
            self.shape = [0] * shards[0].ndim

            for i in range(len(self.shape)):
                if i in dim:
                    self.shape[i] = sum(shard.shape[i] for shard in shards)
                else:
                    self.shape[i] = shards[0].shape[i]
        else:
            # compute the shape as the shape of the first shard
            self.shape = shards[0].shape

    def is_replicated(self):
        return self.dim is None

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

        return "ShardedTensor(\n" + ",\n".join(strs) + "\n, dim=" + str(self.dim) + ")"

    def __matmul__(self, other):
        a, b = self, other
        # both need to be 2D and sharded
        assert (
            len(a.shape) == 2
            and len(b.shape) == 2
            and isinstance(a, ShardedTensor)
            and isinstance(b, ShardedTensor)
        ), "got {}, {}".format(a, b)
        # m, n, k requirements
        assert a.shape[1] == b.shape[0], "got {}, {}".format(a, b)
        # a must be replicated
        assert (
            a.is_replicated() or b.is_replicated()
        ), "one of the tensors must be replicated"
        # b must be sharded along the second dimensio
        res = [a_shard @ b_shard for a_shard, b_shard in zip(a.shards, b.shards)]

        return ShardedTensor(res, b.dim if a.is_replicated() else a.dim)

    def numpy(self):
        assert len(self.dim) == 1, "only works for 1D sharded tensors"
        return unshard_tensor(self, self.dim[0]).numpy()


def shard_tensor(
    x: Tensor, num_shards: int, dim: List[int] | int = 0, devices=None
) -> ShardedTensor:
    assert num_shards == len(devices), "num_shards must match the number of devices"
    if dim is None:
        return ShardedTensor([x.to(device) for device in devices], None)
    shards = ops.chunk(x, num_shards, dim)
    if devices is not None:
        shards = [
            ops.as_contiguous(shard).eval().to(device)
            for shard, device in zip(shards, devices)
        ]
    return ShardedTensor(shards, [dim])


def unshard_tensor(x: ShardedTensor, dim: int = 0, dev=None) -> Tensor:
    dev = dev or x.shards[0].device
    if device is not None:
        shards = [shard.eval().to(dev) for shard in x.shards]
    else:
        shards = x.shards
    return ops.cat(shards, dim)


def topology_diagram(t):
    names = {
        device.cpu: "CPU",
        device.cuda: "CUDA",
    }

    grid_size = int(len(t.shards) * 10)
    grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
    # first and last row filled with --
    for i in range(grid_size):
        grid[0][i] = "_"
        grid[-1][i] = "_"

    # firt and last column filled with |
    for i in range(grid_size):
        grid[i][0] = "|"
        grid[i][-1] = "|"

    # divide the grid into equal parts in the axis
    is_horizontal = t.dim and t.dim[0] == 0
    is_vertical = not is_horizontal

    if is_horizontal:
        for i in range(grid_size):
            for s in range(len(t.shards)):
                grid[i][s * 9] = "|"

    if is_vertical:
        for i in range(grid_size):
            for s in range(len(t.shards)):
                grid[s * 9][i] = "_"

    # print device names
    if is_horizontal:
        for i, shard in enumerate(t.shards):
            for j in range(len(names[shard.device])):
                grid[1][i * 10 + j + 1] = names[shard.device][j]

    if is_vertical:
        for i, shard in enumerate(t.shards):
            for j in range(len(names[shard.device])):
                grid[i * 10 + 1][j + 1] = names[shard.device][j]

    for row in grid:
        print("".join(row))
