from pequegrad.modules import NonStatefulModule
import numpy as np
from pequegrad.tensor import Tensor
from PIL import Image

Mod = NonStatefulModule


class Resize(Mod):
    def __init__(self, size):
        self.size = size

    def forward(self, x: Image.Image):
        if isinstance(x, np.ndarray):
            # if ndim == 4, then it is a batch of images
            x = (x * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            if x.ndim == 4:
                # put channels first
                return np.array(
                    [np.array(Image.fromarray(img).resize(self.size)) for img in x]
                ).transpose(0, 3, 1, 2)

            return np.array(Image.fromarray(x).resize(self.size)).transpose(2, 0, 1)
        if isinstance(x, Tensor):
            x = (x.eval().detach().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            if x.ndim == 4:
                return (
                    Tensor(
                        [np.array(Image.fromarray(img).resize(self.size)) for img in x]
                    ).permute(0, 3, 1, 2)
                    / 255.0
                )
            return (
                Tensor(np.array(Image.fromarray(x).resize(self.size))).permute(2, 0, 1)
                / 255.0
            )


class ToTensor(Mod):
    def forward(self, x: Image):
        if isinstance(x, Tensor):
            return x

        if isinstance(x, np.ndarray):
            return Tensor(x)

        assert isinstance(x, Image.Image), "Input must be a PIL image, got {}".format(
            type(x)
        )

        return (np.array(x) / 255.0).transpose(2, 0, 1)


class Normalize(Mod):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, x: Image.Image or np.ndarray or Tensor):
        if isinstance(x, Image.Image):
            x = np.array(x)
            x = Tensor(x)
        elif isinstance(x, np.ndarray):
            x = Tensor(x)

        ndim = x.ndim if hasattr(x, "ndim") else x.dim
        if ndim == 3:
            x = x - Tensor(self.mean).reshape((3, 1, 1)).astype(x.dtype)
            x = x / Tensor(self.std).reshape((3, 1, 1)).astype(x.dtype)

            return x / 255.0
        elif ndim == 4:
            x = x - Tensor(self.mean).reshape((1, 3, 1, 1)).astype(x.dtype)
            x = x / Tensor(self.std).reshape((1, 3, 1, 1)).astype(x.dtype)

            return x / 255.0


class EvalAndDetach(Mod):
    def forward(self, x: Tensor):
        return x.eval().detach()


class Compose(Mod):
    def __init__(self, transforms):
        self.transforms = transforms

    def forward(self, x):
        for t in self.transforms:
            x = t(x)
        return x
