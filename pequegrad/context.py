from contextlib import contextmanager
from typing import Optional


# Use singleton pattern to avoid having to have just one context
class _PequegradContext:
    _instance = None
    grad_enabled: bool
    force_training: Optional[bool]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_PequegradContext, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.grad_enabled = True
        self.force_training = None


pequegrad_context = _PequegradContext()


@contextmanager
def no_grad():
    try:
        pequegrad_context.grad_enabled = False
        yield
    finally:
        pequegrad_context.grad_enabled = True


@contextmanager
def eval():
    prev = pequegrad_context.force_training
    pequegrad_context.force_training = False
    try:
        yield
    finally:
        pequegrad_context.force_training = prev


@contextmanager
def train():
    prev = pequegrad_context.force_training
    pequegrad_context.force_training = True
    try:
        yield
    finally:
        pequegrad_context.force_training = prev
