import pequegrad as pg
import time


size = 8192
num_iterations = 20


def linear_relu(x, w, b):
    def _f(x, w, b):
        l1 = pg.relu(x @ w + b)
        l2 = pg.relu(l1 @ w + b)
        return pg.relu(l2 @ w + b)

    ret, grads = pg.fngrad(_f, wrt=[0, 1, 2], return_outs=True)(x, w, b)
    return ret, grads


amp_linear_relu = pg.amp(linear_relu)
amp_linear_relu = pg.jit(amp_linear_relu)

linear_relu = pg.jit(
    linear_relu, opts={"fused_linear_relu": False, "experimental_toposort_optim": False}
)

x = pg.Tensor(pg.np.random.randn(size, size) / 100).to("cuda").astype("float32")
w = pg.Tensor(pg.np.random.randn(size, size) / 100).to("cuda").astype("float32")

b = pg.Tensor(pg.np.zeros(size)).to("cuda").astype("float32")

_ = amp_linear_relu(x, w, b)
_ = linear_relu(x, w, b)

print("Warming up done")
total_time_amp = 0
total_time_regular = 0

for _ in range(num_iterations):
    start_time = time.time()
    res_amp = amp_linear_relu(x, w, b)
    pg.sync_cuda_device()
    end_time = time.time()
    total_time_amp += end_time - start_time

    start_time = time.time()
    res_regular = linear_relu(x, w, b)
    pg.sync_cuda_device()
    end_time = time.time()
    total_time_regular += end_time - start_time

average_time_amp = total_time_amp / num_iterations
average_time_regular = total_time_regular / num_iterations

print(
    f"Average time taken for amp_linear_relu over {num_iterations} iterations: {average_time_amp:.6f} seconds"
)

print(
    f"Average time taken for regular linear_relu over {num_iterations} iterations: {average_time_regular:.6f} seconds"
)

amp_linear_relu.print_trace()

print("Result of amp_linear_relu:", res_amp[0].numpy())

print("Result of regular linear_relu:", res_regular[0].numpy())
