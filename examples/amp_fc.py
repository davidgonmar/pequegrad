import pequegrad as pg
import time


size = 8192
num_iterations = 20


@pg.jit
@pg.amp
def amp_linear_relu(x, w, b):
    return pg.relu(x @ w + b)


@pg.jit
def linear_relu(x, w, b):
    return pg.relu(x @ w + b)


x = pg.Tensor(pg.np.random.rand(size, size)).to("cuda").astype("float32")
w = pg.Tensor(pg.np.random.rand(size, size)).to("cuda").astype("float32")

b = pg.Tensor(pg.np.random.rand(size)).to("cuda").astype("float32")

_ = amp_linear_relu(x, w, b).eval()
_ = linear_relu(x, w, b).eval()

print("Warming up done")
total_time_amp = 0
total_time_regular = 0

for _ in range(num_iterations):
    start_time = time.time()
    res_amp = amp_linear_relu(x, w, b).eval()
    pg.sync_cuda_device()
    end_time = time.time()
    total_time_amp += end_time - start_time

    start_time = time.time()
    res_regular = linear_relu(x, w, b).eval()
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

print("Result of amp_linear_relu:", res_amp.numpy())

print("Result of regular linear_relu:", res_regular.numpy())
