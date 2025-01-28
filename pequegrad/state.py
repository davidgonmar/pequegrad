from pequegrad.backend.c import (
    set_global_state_cuda_allocator,  # noqa
    reset_global_allocator_memory as reset_custom_allocator,  # noqa
    get_custom_allocator_alloc_history,  # noqa
    get_global_state_cuda_stream,  # noqa
    set_global_state_cuda_stream,  # noqa
)  # noqa
import contextlib


@contextlib.contextmanager
def cuda_allocator(allocator):
    try:
        set_global_state_cuda_allocator(allocator)
        yield
    finally:
        set_global_state_cuda_allocator("default")


@contextlib.contextmanager
def stream(stream):
    try:
        set_global_state_cuda_stream(stream)
        yield
    finally:
        set_global_state_cuda_stream(None)
