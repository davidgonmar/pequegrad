from pequegrad.transforms import *  # noqa
from pequegrad.backend.c import (
    dt,
    device as device_kind,
    custom_prim as _custom_prim,
    sync_cuda_device,
    Device,
    from_str,
)  # noqa
from pequegrad.tensor import *  # noqa
from pequegrad.optim import *  # noqa
from pequegrad.modules import *  # noqa
from pequegrad.einsum import *  # noqa
from pequegrad.transforms.compile import *  # noqa

from pequegrad.ops import *  # noqa
from pequegrad.distrib.sharded_tensor import *  # noqa
from pequegrad.viz import *  # noqa


class DeviceModule:
    @staticmethod
    def cuda(idx: int = 0) -> Device:
        return from_str(f"cuda:{idx}")

    @staticmethod
    def cpu(idx: int = 0) -> Device:
        return from_str(f"cpu:{idx}")


device = DeviceModule
