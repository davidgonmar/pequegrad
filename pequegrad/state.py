from pequegrad.backend.c import (
    set_global_state_cuda_allocator,
    reset_global_allocator_memory as reset_custom_allocator,
    get_custom_allocator_alloc_history,
)  # noqa
import contextlib


@contextlib.contextmanager
def cuda_allocator(allocator):
    try:
        set_global_state_cuda_allocator(allocator)
        yield
    finally:
        set_global_state_cuda_allocator("default")
