## Pequegrad

Pequegrad is a simple deep learning framework for Python. It works with tensors, and provides automatic differentiation. It's API is similar to PyTorch's.
It has strong GPU acceleration through CUDA.

### Requirements (might work with other versions, but these are the ones that I tested)
```bash
- General
numpy==1.26.2

- Testing
pytest==7.4.0
torch==2.1.1 (it is used to check the correctness of the results)
```
### Examples
There are some examples in the examples directory. You can run them with the following commands:
(in case you don't want to use the GPU, just remove the --cuda flag from the commands below)

- MLP
```python -m examples.mlp_mnist --cuda```

- CNN
```python -m examples.conv_mnist --cuda```

The CNN example trains a simple CNN on the MNIST dataset. It reaches 90%+ accuracy in about 1.3 seconds on a RTX 4090 GPU, and about 3.6 seconds on a RTX 2070 Mobile laptop GPU, as per my tests.
The MLP example trains a simple MLP on the MNIST dataset. It reaches 90%+ accuracy in about 1 second on a RTX 2070 Mobile laptop GPU, as per my tests.

### Getting started
The tensor functionality is very similar to PyTorch's. You can create tensors, perform operations on them, and use the .backward() method to compute the gradients. The gradients are stored in the .grad attribute of the tensor.

```python
from pequegrad.tensor import Tensor

t1 = Tensor([1, 2, 3, 4, 5], requires_grad=True)
t2 = Tensor([5, 4, 3, 2, 1], requires_grad=True)
t3 = t1 + t2
t4 = t3.sum()
t4.backward()
print(t1.grad) # [1, 1, 1, 1, 1]
print(t2.grad) # [1, 1, 1, 1, 1]
print(t3) # [6, 6, 6, 6, 6]
```
The same example in PyTorch would be:
```python
import torch

t1 = torch.tensor([1, 2, 3, 4, 5], requires_grad=True)
t2 = torch.tensor([5, 4, 3, 2, 1], requires_grad=True)
t3 = t1 + t2
t4 = t3.sum()
t4.backward()
print(t1.grad) # tensor([1, 1, 1, 1, 1])
print(t2.grad) # tensor([1, 1, 1, 1, 1])
print(t3) # tensor([6, 6, 6, 6, 6], grad_fn=<AddBackward0>)
```

### GPU acceleration
In order to use the GPU acceleration, you need to be able to compile CUDA programs (so you need to have the CUDA toolkit installed). A CMakeLists.txt file is provided, so you can use cmake to compile the CUDA code. You must also have the pybind11 library installed and searcheable.
Once the csrc directory is compiled, the library should be left in the ./build directory, and you can use the Pequegrad library as usual.
You can still use Pequegrad without the GPU acceleration, and will use Numpy under the hood, but you wont be able to use things like .cuda() or .to("cuda") on tensors.


### Note
This is a toy project, and my main concern was to make it work, and then refactor it. Don't expect it to be readable at all at the moment.
