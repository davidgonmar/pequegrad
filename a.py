from pequegrad.tensor import Tensor
import numpy as np

# Define the input tensor with random values
# For example, a batch of 1, with 1 input channel, height of 10, and width of 5
input_shape = (1, 1, 1000, 2, 500)
input_tensor = Tensor(np.random.rand(*input_shape), requires_grad=True).to("cuda")

# Define the kernel tensor with random values
# For example, 1 output channel, 1 input channel, kernel height of 3, and kernel width of 3
kernel_shape = (5, 1, 3, 3)
kernel_tensor = Tensor(np.random.rand(*kernel_shape), requires_grad=True).to("cuda")

# Perform the convolution operation
output_tensor = input_tensor.conv2d(kernel_tensor)

print(output_tensor.numpy())
