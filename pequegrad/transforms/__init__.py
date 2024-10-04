from .autodiff import *  # noqa
from .compile import *  # noqa
from .extra import *  # noqa
from .automatic_mixed_precision import *  # noqa
from .distrib import *  # noqa


def maybe(transform, condition):
    def transform_fn(f):
        return transform(f) if condition else f

    return transform_fn
