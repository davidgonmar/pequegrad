from pequegrad.modules import NonStatefulModule
import PIL
import numpy as np
from pequegrad.tensor import Tensor

Mod = NonStatefulModule


class Resize(Mod):
    def __init__(self, size):
        self.size = size

    def forward(self, x: PIL.Image.Image):
        assert isinstance(x, PIL.Image.Image), "Input must be a PIL image"

        return x.resize(self.size)


class ToTensor(Mod):
    def forward(self, x: PIL.Image.Image):
        assert isinstance(x, PIL.Image.Image), "Input must be a PIL image"

        return np.array(x) / 255.0


class Normalize(Mod):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, x: PIL.Image.Image | Tensor):
        if isinstance(x, PIL.Image.Image):
            x = np.array(x)
            x = x / 255.0
            x = Tensor(x)

        x = x - self.mean
        x = x / self.std

        return x


class EvalAndDetach(Mod):
    def forward(self, x: Tensor):
        return x.eval().detach()
