import pytest
import numpy as np
from PIL import Image
from pequegrad.tensor import Tensor
from pequegrad.ds_transforms import Resize, ToTensor, Normalize, EvalAndDetach


@pytest.fixture
def sample_image():
    # Create a simple image with known values
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[50:70, 50:70] = [255, 0, 0]  # red square
    return Image.fromarray(data)


@pytest.mark.parametrize(
    "size, expected_size", [((50, 50), (50, 50)), ((200, 200), (200, 200))]
)
def test_resize(sample_image, size, expected_size):
    resize = Resize(size)
    resized_image = resize(sample_image)
    assert (
        resized_image.size == expected_size
    ), f"Expected size {expected_size}, got {resized_image.size}"


def test_to_tensor(sample_image):
    to_tensor = ToTensor()
    tensor = to_tensor(sample_image)
    assert isinstance(tensor, Tensor), "Output should be a Tensor"
    assert tuple(tensor.shape) == (
        3,
        100,
        100,
    ), f"Expected shape (3, 100, 100), got {tensor.shape}"
    assert np.all(tensor.numpy() <= 1) and np.all(
        tensor.numpy() >= 0
    ), "Tensor values should be in range [0, 1]"


@pytest.mark.parametrize(
    "mean, std", [([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ([0, 0, 0], [1, 1, 1])]
)
def test_normalize(sample_image, mean, std):
    normalize = Normalize(mean, std)
    mean, std = np.array(mean), np.array(std)
    mean, std = mean.reshape((3, 1, 1)), std.reshape(
        (3, 1, 1)
    )  # reshape for broadcasting
    tensor = ToTensor()(sample_image)
    normalized_tensor = normalize.forward(tensor)
    assert isinstance(
        normalized_tensor, Tensor
    ), "Output should be a Tensor, got {}".format(type(normalized_tensor))
    expected_tensor = (tensor.numpy() - np.array(mean)) / np.array(std)
    assert np.allclose(
        normalized_tensor.numpy(), expected_tensor
    ), "Normalized tensor values are incorrect"


def test_eval_and_detach():
    data = np.random.rand(100, 100, 3)
    tensor = Tensor(data)
    eval_and_detach = EvalAndDetach()
    result = eval_and_detach.forward(tensor)
    assert isinstance(result, Tensor), "Output should be a Tensor"
    assert np.allclose(result.numpy(), data), "Tensor values should match input data"


if __name__ == "__main__":
    pytest.main()
