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

### Laziness and just-in-time compilation

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

Moreover, you can also just-in-time compile the computation to a function that can be called with the actual values of the tensors.
To use it, just decorate a function with the @jit decorator (must be a pure function and have no python things (only pequegrad ops)):

```python
from pequegrad.compile import jit
import time
from pequegrad.tensor import Tensor, dt, device

dev = device.cuda
def test_some_fn():
    @jit
    def some_function(x, y, z):
        return x.log() + y + z.exp().log().exp()

    def non_jitted(x, y, z):
        return x.log() + y + z.exp().log().exp()

    for i in range(10):
        x = Tensor(np.random.randn(10000, 1000), device=dev).astype(dt.float32).eval()
        y = Tensor(np.random.randn(10000, 1000), device=dev).astype(dt.float32).eval()
        z = Tensor(np.random.randn(10000, 1000), device=dev).astype(dt.float32).eval()

        start = time.time()
        j = some_function(x, y, z).eval()
        jittedtime = time.time() - start

        start = time.time()
        nj = non_jitted(x, y, z).eval()
        nonjittedtime = time.time() - start

        print(f"Jitted time: {jittedtime}, Non-jitted time: {nonjittedtime}")
        np.testing.assert_allclose(j.numpy(), nj.numpy(), atol=1e-3)
```

Use PG_KERNEL_DB to print generated kernels

```bash
PG_KERNEL_DB=true python -m examples.jit --cuda
file:
extern "C" {
    __global__ void kernel_name(float *arg0, float *arg1, float *arg2, float *arg3)
    {
        int bidx = blockIdx.x;
        int bdim = blockDim.x;
        int tidx = threadIdx.x;
        int global_idx = (bidx * bdim) + tidx;
        if ((10000000 < global_idx)) {
            return;
        }
        float load0 = arg0[global_idx];
        float load1 = arg1[global_idx];
        float load2 = arg2[global_idx];
        arg3[global_idx] = ((log(load0) + load1) + exp(log(exp(load2))));
    }
}
Jitted time: 0.10751175880432129, Non-jitted time: 0.017236709594726562
Jitted time: 0.002004384994506836, Non-jitted time: 0.016346454620361328
Jitted time: 0.003206491470336914, Non-jitted time: 0.017115354537963867
Jitted time: 0.0030007362365722656, Non-jitted time: 0.017028331756591797
Jitted time: 0.0030069351196289062, Non-jitted time: 0.017102718353271484
Jitted time: 0.0030078887939453125, Non-jitted time: 0.015584945678710938
Jitted time: 0.008017539978027344, Non-jitted time: 0.03783774375915527
Jitted time: 0.01613640785217285, Non-jitted time: 0.0693202018737793
Jitted time: 0.003019571304321289, Non-jitted time: 0.016047954559326172
Jitted time: 0.0077784061431884766, Non-jitted time: 0.0374603271484375
```

As you can see, the jitted function is slower the first time it is called, but then it is a lot faster than the non-jitted function.
All the operations in the function were compiled to a single CUDA kernel, so no intermediate memory is used and the overhead of the kernel launch is minimal.
The only downside is that gradient computation must be done before the jitting, for example:

```python
@partial(jit, externals=model.parameters())
def train_step(batch_X, batch_Y):
        prediction = model.forward(batch_X)
        loss = prediction.cross_entropy_loss_probs(batch_Y)
        g = grads(model.parameters(), loss)
        return [loss] + g
```

The `externals` parameter is used to tell the jitting function which tensors are external to the function (the function depends on them, but they are not arguments of the function). As you can see, we can compute the gradients inside a jitted function, and then use them to update the model parameters.

However, we cannot do this:

```python
@partial(jit, externals=model.parameters())
def train_step(batch_X, batch_Y):
        prediction = model.forward(batch_X)
        loss = prediction.cross_entropy_loss_probs(batch_Y)
        return loss


loss = train_step(batch_X, batch_Y)
g = grads(model.parameters(), loss) # Error: cannot differentiate through a compiled function!!!

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
