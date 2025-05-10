# pequegrad

**pequegrad** is a toy‑deep learning framework for Python.  
The main actors are **Tensors**—multi‑dimensional arrays that live on the CPU or GPU and carry elements of certain data‑type (`dtype`).

pequegrad features:

* A **tracing mechanism** that turns a subset of Python functions into a custom graph structure (`GraphTrace`).  
  This powers automatic differentiation (AD) and other advanced features.
* A concise **core operation set**, used to implement the other operations.
* A comprehensive **neural‑network module** that lets you build and train models with ease.

---

## Automatic Differentiation

`fngrad` transforms a pure Python function into one that returns gradients (and optionally the original outputs).

```python
from pequegrad.tensor import Tensor
from pequegrad import fngrad
import pequegrad as pg

def f(a, b, c):
    x = a * b
    return x * c

f_and_grad = fngrad(f, wrt=[0, 1, 2], return_outs=True)

a, b, c = (
    pg.rand((2, 3), dtype=pg.dt.float32, device=pg.device.cuda(0)),
    pg.rand((2, 3), dtype=pg.dt.float32, device=pg.device.cuda(0)),
    pg.rand((2, 3), dtype=pg.dt.float32, device=pg.device.cuda(0)),
)

outs, grads = f_and_grad(a, b, c)
```

Beyond simple gradients, pequegrad can compute Jacobians, Hessians, JVPs, VJPs, and even Taylor expansions.

---

## JIT Compilation

The tracing mechanism feeds a graph compiler with several optimisation passes:

1. **Remove redundant ops** e.g. reshapes with no effect.
2. **Detect common patterns** On CUDA, it looks for convolution/normalisation patterns and substitutes sub-graphs with cuDNN kernels.
3. **Operator fusion** Element‑wise and reduction ops are merged; broadcasts may be hoisted to enable deeper fusion.


Let's see a simple example. Set `PG_KERNEL_DB=true` to print generated CUDA kernels:

```python
import pequegrad as pg
import numpy as np
import time

x = pg.Tensor(np.random.rand(1000, 20, 100)).to("cuda").astype("float32")
y = pg.Tensor(np.random.rand(1000, 100, 20)).to("cuda").astype("float32")

def f(x, y):
    return ((pg.permute(x, (0, 2, 1)) + 2) + y).sum(2) * 3

fjit = pg.jit(f)

# Warmup
for _ in range(10):
    f(x, y).eval()
    pg.sync_cuda_device()
    fjit(x, y).eval()
    pg.sync_cuda_device()

# Test without JIT
start_time = time.time()
for _ in range(1000):
    f(x, y).eval()
    pg.sync_cuda_device()
end_time = time.time()
print(f"Time without JIT: {end_time - start_time} seconds")

# Test with JIT
start_time = time.time()
for _ in range(1000):
    fjit(x, y).eval()
    pg.sync_cuda_device()
end_time = time.time()
print(f"Time with JIT: {end_time - start_time} seconds")
```
```bash
PG_KERNEL_DB=true python example.py
...
__global__ void __launch_bounds__(1024, 1) reduce_kernel(const float * __restrict__ arg0, const float * __restrict__ arg1, float * __restrict__ arg2
){
  int bidx = blockIdx.x;
  int bdim = blockDim.x;
  int tidx = threadIdx.x;
  int global_idx = (bidx * bdim) + tidx;
  if ((100000 <= global_idx)) {
    return;
  }
  int arg_0_idx_1 = (global_idx / (1 * 1)) % 100;
  int arg_0_idx_0 = (global_idx / 100) % 1000;
  int arg_2_idx_1 = (global_idx / (1 * 1)) % 100;
  int arg_2_idx_0 = (global_idx / 100) % 1000;
  int arg_4_idx_1 = (global_idx / (1 * 1)) % 100;
  int arg_4_idx_0 = (global_idx / 100) % 1000;
  float acc0 = 0.000000000f;
  int const23 = 0;
  #pragma unroll
  for (const23; const23 < 20; const23+=1) {
    int arg_0_idx_2 = (const23 / (1 * 1)) % 20;
    int arg_2_idx_2 = (const23 / (1 * 1)) % 20;
    float load0 = __ldg(arg0 + (((arg_0_idx_0 * 2000) + (arg_0_idx_1 * 1)) + (arg_0_idx_2 * 100)));
    float load1 = __ldg(arg1 + (((arg_2_idx_0 * 2000) + (arg_2_idx_1 * 20)) + (arg_2_idx_2 * 1)));
    acc0 += ((load0 + 2.000000000f) + load1);
  }
  arg2[((arg_4_idx_0 * 100) + (arg_4_idx_1 * 1))] = (acc0 * 3.000000000f);
}

Time without JIT: 2.6615936756134033 seconds
Time with JIT: 0.27284669876098633 seconds
```

The fuser was able to combine the whole function into a single kernel, including folding the constants into the kernel and the sum reduction. Note that the permute is done through stride reordering, and is implicit in the kernel (look at how the loads are done).

---

## Training a Neural Network

The training utilities are fairly complete. Below is a condensed AlexNet‑on‑CIFAR‑10 example:

```python
class AlexNet(nn.StatefulModule):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   96, kernel_size=11, stride=4, padding=2),
            ...
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            ...
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.reshape((x.shape[0], 256 * 6 * 6))
        return self.classifier(x)
```

Data pipeline:

```python
transform = transforms.Compose([
    transforms.ToTensor(device="cuda"),
    transforms.JitCompose([
        transforms.PermuteFromTo((0, 1, 2, 3), (0, 3, 1, 2)),  # NHWC → NCHW
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        transforms.Resize((224, 224)),
    ]),
    transforms.EvalAndDetach(),
])
```

Training step:

```python
@pg.jit.withargs(allocator="custom") # uses a custom CUDA allocator, about 8% faster on a 2070 laptop
def update_step(state, params_dict, x, y):
    x, y = x.to("cuda"), y.to("cuda")
    y = Tensor.one_hot(100, y)

    def loss_fn(x, y, params):
        preds = nn.apply_to_module(model, params, x)
        return preds.cross_entropy_loss_probs(y)

    loss, (grads,) = fngrad(loss_fn, wrt=[2], return_outs=True)(x, y, params_dict)
    state, params_dict = sgd(params_dict, grads, state)
    return state, params_dict, loss
```



## Other Cool Things

* **JAX frontend** (early‑stage) – train a simple MLP using familiar JAX APIs.
* **ONNX importer/exporter** – move models in and out of pequegrad.

See the *examples* directory for more.


## Disclaimer

This project is a hobby. Expect outdated code, TODOs, and the occasional bug.  
I focus on building fun stuff rather than maintaining a stable codebase.



## Core Requirements

```text
numpy == 1.26.2

# For tests (used to validate results)
pytest == 7.4.0
torch  == 2.1.1
```

## Building the Library

Most heavy lifting (graphs and AD) is in C++. You can build it with CMake.
Currently, CUDA, OpenMP, and OpenBLAS are required. I’m working on relaxing those dependencies.
The resulting binaries land in `build/`, and Python will pick them up automatically.
Right now, everything is set up for my system and it is difficult to build it on other systems. I'm working on it.

---
