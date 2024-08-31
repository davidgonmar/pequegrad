from pequegrad.backend.c import Tensor
from typing import Any, Tuple


def _cache_individual_item(item: Any):
    if isinstance(item, Tensor):
        return tuple((tuple(item.shape), tuple(item.strides), item.dtype, item.device))
    return item


def get_cache_key(flattened_args: Tuple[Any]) -> int:
    tup = tuple(_cache_individual_item(item) for item in flattened_args)
    hashed = hash(tup)
    return hashed


def extract_input_tensors(flattened_args: Tuple[Any]) -> Tuple[Tensor]:
    return tuple(item for item in flattened_args if isinstance(item, Tensor))


def bridge_args_to_lazy_fn(inps: Tuple[Tensor], args: Tuple[Tensor]) -> None:
    for inp, arg in zip(inps, args):
        inp._inplace_as_copy(arg)
