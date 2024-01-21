from contextlib import contextmanager


# Use singleton pattern to avoid having to have just one context
class _PequegradContext:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_PequegradContext, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.grad_enabled = True


pequegrad_context = _PequegradContext()


@contextmanager
def no_grad():
    try:
        pequegrad_context.grad_enabled = False
        yield
    finally:
        pequegrad_context.grad_enabled = True
