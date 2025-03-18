from pequegrad import Tensor
import pequegrad.ops as ops
import pequegrad.image as image
import cv2
import numpy as np
import tempfile

def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.numpy()

def numpy_to_tensor(array: np.ndarray) -> Tensor:
    return Tensor(array)

def imread(path: str, flags=cv2.IMREAD_COLOR) -> Tensor:
    """Read an image from a file and return it as a torch tensor."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img_path = tmp_file.name
    img = cv2.imread(img_path, flags)
    if img is None:
        raise FileNotFoundError(f"Image at path '{path}' not found.")
    return numpy_to_tensor(img)

def imwrite(path: str, img: Tensor) -> bool:
    """Write a torch tensor image to a file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img_path = tmp_file.name
    img_np = tensor_to_numpy(img)
    return cv2.imwrite(img_path, img_np)

def resize(img: Tensor, size: tuple, interpolation=cv2.INTER_LINEAR) -> Tensor:
    """Resize an image using OpenCV while keeping it in torch tensor format."""
    # only downsample is supported
    if size[0] > img.shape[0] or size[1] > img.shape[1]:
        raise ValueError("Only downsample is supported.")
    if interpolation not in [cv2.INTER_NEAREST, cv2.INTER_LINEAR]:
        raise ValueError("Interpolation method not supported.")
    assert img.shape[0] / size[0] == img.shape[1] / size[1], "Aspect ratio must be preserved."
    k = img.shape[0] // size[0]
    # HWC -> CHW
    img = img.permute(2, 0, 1)
    if interpolation == cv2.INTER_NEAREST:
        resized_img = image.downsample_nearest(img, k)
    elif interpolation == cv2.INTER_LINEAR:
        resized_img = image.downsample_bilinear(img, k)
    resized_img = resized_img.permute(1, 2, 0) # CHW -> HWC
    return resized_img


def cvtColor(img: Tensor, code: int) -> Tensor:
    """Convert an image color space using OpenCV."""
    if code not in [cv2.COLOR_BGR2GRAY, cv2.COLOR_GRAY2BGR, cv2.COLOR_BGR2RGB]:
        raise ValueError("Color conversion code not supported.")
    assert img.shape[2] == 3, "Only 3-channel images are supported."

    
    if code == cv2.COLOR_BGR2GRAY:
        # use the same coefficients as OpenCV
        gray = img @ Tensor([0.299, 0.587, 0.114]).reshape((3, 1))
        return gray.astype("int32")
    
    elif code == cv2.COLOR_GRAY2BGR:
        return img.repeat(1, 1, 3)
    elif code == cv2.COLOR_BGR2RGB:
        return img.permute(2, 0, 1)

    
def gaussian_kernel(ksize: int, sigma: float):
    """Generate a 2D Gaussian kernel manually."""
    ax = ops.arange(0, ksize).astype("float32") - ksize // 2
    xx, yy = ops.meshgrid(ax, ax)
    kernel = ops.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def GaussianBlur(img: Tensor, ksize: tuple, sigmaX: float) -> Tensor:
    """Apply Gaussian Blur using OpenCV."""
    filter = gaussian_kernel(ksize[0], sigmaX)
    filter = filter @ filter.T
    filter = filter / filter.sum()
    filter = filter.astype("float32")
    filter = filter.reshape((1, 1, ksize[0], ksize[0]))
    img = img.permute(2, 0, 1) # HWC -> CHW
    img = img.unsqueeze(0) # CHW -> BCHW
    
    filter = ops.broadcast_to(filter, (3, 1, ksize[0], ksize[0]))
    #raise ValueError
    re = ops.conv2d(img, filter, padding=ksize[0]//2, groups=3) # apply the filter to each channel

    return re.squeeze(0).permute(1, 2, 0) # BCHW -> CHW -> HWC


# Aliasing OpenCV constants
INTER_LINEAR = cv2.INTER_LINEAR
INTER_NEAREST = cv2.INTER_NEAREST
INTER_CUBIC = cv2.INTER_CUBIC
INTER_LANCZOS4 = cv2.INTER_LANCZOS4
COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
COLOR_GRAY2BGR = cv2.COLOR_GRAY2BGR