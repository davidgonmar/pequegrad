import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from pequegrad.tensor import Tensor
from pequegrad.image import (
    downsample_nearest,
    downsample_bilinear,
    downsample_avgpooling,
)

response = requests.get("https://cataas.com/cat")
img = Image.open(BytesIO(response.content)).convert("RGB")

width, height = img.size
k = 8
new_width = width - (width % k)
new_height = height - (height % k)

img = img.resize((new_width, new_height))

print(f"Original Image Size: {img.size}")
print(f"Downsampled Image Size: {img.size[0] // k} x {img.size[1] // k}")
img_np = np.array(img)
img_tensor = Tensor(img_np.transpose(2, 0, 1).astype(np.float32))

downsampled_nearest = downsample_nearest(img_tensor, k)
downsampled_bilinear = downsample_bilinear(img_tensor, k)
downsampled_avgpool = downsample_avgpooling(img_tensor, k)

downsampled_nearest_np = downsampled_nearest.numpy().transpose(1, 2, 0).astype(np.uint8)
downsampled_bilinear_np = (
    downsampled_bilinear.numpy().transpose(1, 2, 0).astype(np.uint8)
)
downsampled_avgpool_np = downsampled_avgpool.numpy().transpose(1, 2, 0).astype(np.uint8)

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(downsampled_nearest_np)
plt.title("Nearest Neighbor Downsampling")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(downsampled_bilinear_np)
plt.title("Bilinear Downsampling")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(downsampled_avgpool_np)
plt.title("Average Pooling Downsampling")
plt.axis("off")

plt.tight_layout()
plt.show()
