## Pequegrad

Pequegrad is a simple deep learning framework for Python. It works with tensors, and provides automatic differentiation. It supports CUDA and CPU computation.

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
  `python -m examples.mlp_mnist --cuda`

- CNN
  `python -m examples.conv_mnist --cuda`

### Automatic differentiation

You can perform simple autodifferentiation with Pequegrad like this:

```python
from pequegrad.tensor import Tensor
from pequegrad.autodiff import grads

t1 = Tensor([1, 2, 3, 4, 5])
t2 = Tensor([5, 4, 3, 2, 1])
t3 = t1 + t2
t4 = t3.sum()
dt4_dt1, dt4_dt2 = grads([t1, t2], t4)

print(dt4_dt1.numpy()) # [1, 1, 1, 1, 1]
print(dt4_dt2.numpy()) # [1, 1, 1, 1, 1]
```

The same example in PyTorch would be:

```python
import torch

t1 = torch.tensor([1, 2, 3, 4, 5], requires_grad=True)
t2 = torch.tensor([5, 4, 3, 2, 1], requires_grad=True)
t3 = t1 + t2
t4 = t3.sum()
t4.backward()
dt4_dt1 = t1.grad
dt4_dt2 = t2.grad

print(dt4_dt1.detach().numpy()) # [1, 1, 1, 1, 1]
print(dt4_dt2.detach().numpy()) # [1, 1, 1, 1, 1]
```

### Training a simple neural network!

(please go to the examples directory to see the full example, like loading the data and saving/loading a model)

```python
class MLP(StatefulModule):
    def __init__(self):
        self.fc1 = Linear(784, 200)
        self.fc2 = Linear(200, 10)

    def forward(self, input):
        input = self.fc1.forward(input).relu()
        return self.fc2.forward(input)

def train(model, X_train, Y_train, epochs=13, batch_size=4096):
    optim = Adam(model.parameters(), lr=0.021)

    def train_step(batch_X, batch_Y):
        prediction = model.forward(batch_X)
        loss = prediction.cross_entropy_loss_indices(batch_Y)
        g = grads(model.parameters(), loss)
        return [loss] + g

    for epoch in range(epochs):
        indices = np.random.choice(len(X_train), batch_size)
        batch_X = X_train[indices]
        batch_Y = Y_train[indices]
        outs = train_step(batch_X, batch_Y)
        loss = outs[0]
        g = outs[1:]
        optim.step(g)
        print(f"Epoch {epoch} | Loss {loss.numpy()}")

    return model

model = MLP()
model = train(model, X_train, Y_train)
```

### Laziness

The whole Tensor class is lazy. This means that shapes and dtypes are inferred when you write a tensor computation like

```python
from pequegrad.tensor import Tensor, dt

t1 = Tensor([1, 2, 3, 4, 5]).astype(dt.float32)
t2 = Tensor([5, 4, 3, 2, 1]).astype(dt.float32)
t3 = t1 + t2
print(t3.shape) # [5]
print(t3.dtype) # dt.float32
print(t3) # Tensor(shape=[5], dtype=float32, device=CPU, evaled=0, primitive=Add, id=34)
```

But t3 is not actually computed until .eval() is called, or until you try to access the .numpy() attribute.
This is useful since useless computations are not done, and allows for graph optimizations in the future.

You can even extend the DAG with the tensor gradients, without anything being actually computed!

```python
t1 = Tensor([1, 2, 3, 4, 5])
t2 = Tensor([5, 4, 3, 2, 1])
t3 = t1 + t2
t4 = t3.sum()
dt4_dt1, dt4_dt2 = grads([t1, t2], t4)
```

Here, nothing is actually computed!

### GPU support

All operations must have the same device for all inputs. To use the GPU, simply pass the tensor to the device you want to use:

```python
from pequegrad.tensor import Tensor, dt, device

t1 = Tensor([1, 2, 3, 4, 5]).to(device.cuda)
t2 = Tensor([5, 4, 3, 2, 1]).to(device.cuda)
t3 = t1 + t2 # fine!!!!
t2_cpu = t2.to(device.cpu)
t3 = t1 + t2_cpu # this will raise an error
```

To pass an entire model to the GPU, you can use the .to(device) method:

```python
model = MLP()
model = model.to(device.cuda) # now, it computes everything in the GPU, and expects all inputs to be in the GPU!
```

### More operations

Pequegrad has support for most core operations like:

- Binary operations: +, -, \*, /, @
- Unary operations: relu, sigmoid, tanh
- Reduction operations: sum, mean, max
- Indexing operations: getitem, setitem
- Folding operations: im2col (fold), col2im (unfold)
- Padding
- Convolutions (regular and transposed)

And higher level operations like tensordot (tensor contractions), pooling, norms, naive einsum, dropoutm, statistical operations, dropout, etc.

### Building the library

Most of the computation (include graphs and automatic differentiation) is done in C++. You'll need to compile the CPP code to use the library. A CMakeLists.txt file is provided. Once it is compiled, the built files should be in the build directory.
At the moment, the library can only be built with support for different technologies like CUDA, OpenMP and OpenBLAS. I'm working on making it more flexible and able to be built without these dependencies.
