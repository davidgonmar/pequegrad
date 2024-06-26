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
        else:
            assert isinstance(
                x, Image.Image
            ), "Input must be a PIL image, got {}".format(type(x))

            return x.resize(self.size)


class ToTensor(Mod):
    def forward(self, x: Image):
        if isinstance(x, Tensor):
            return x

        if isinstance(x, np.ndarray):
            return Tensor(x)

        assert isinstance(x, Image.Image), "Input must be a PIL image, got {}".format(
            type(x)
        )

        return Tensor((np.array(x) / 255.0))


class PermuteFromTo(Mod):
    def __init__(self, from_, to):
        self.from_ = from_
        self.to = to

    def forward(self, x: Tensor):
        return x.permute(*self.to)


class Normalize(Mod):
    def __init__(self, mean, std):
        # stats per channel
        self.mean = mean
        self.std = std

    def forward(self, x: Image.Image or np.ndarray or Tensor):
        # mean -> np array or tensor depending on the input
        self.mean = (
            np.array(self.mean)
            if isinstance(x, np.ndarray)
            else Tensor(self.mean).astype(x.dtype).eval().detach()
            if isinstance(x, Tensor) and not isinstance(self.mean, Tensor)
            else self.mean
        )

        self.std = (
            np.array(self.std)
            if isinstance(x, np.ndarray)
            else Tensor(self.std).astype(x.dtype).eval().detach()
            if isinstance(x, Tensor) and not isinstance(self.std, Tensor)
            else self.std
        )

        if isinstance(x, Image.Image):
            # perform the normalization in numpy
            x = np.array(x).transpose(2, 0, 1)
            norm = (x - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)
            # return image
            return Image.fromarray(norm.transpose(1, 2, 0))

        if isinstance(x, np.ndarray):
            if x.ndim == 4:
                x = x.transpose(0, 2, 3, 1)
                return np.array(
                    [
                        (img - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)
                        for img in x
                    ]
                ).transpose(0, 3, 1, 2)

            return np.array(
                (x - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)
            ).transpose(2, 0, 1)

        if isinstance(x, Tensor):
            if x.ndim == 4:
                a = (x - self.mean.reshape((1, 3, 1, 1))) / self.std.reshape(
                    (1, 3, 1, 1)
                )
                return a
            return (x - self.mean.reshape((3, 1, 1))) / self.std.reshape((3, 1, 1))


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
