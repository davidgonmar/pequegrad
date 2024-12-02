import pequegrad as pg
import time


size = 16384
num_iterations = 20


matA = pg.Tensor(pg.np.random.rand(size, size)).to("cuda").astype("float32")
matB = pg.Tensor(pg.np.random.rand(size, size)).to("cuda").astype("float32")


@pg.jit
@pg.amp
def amp_matmul(a, b):
    return a @ b


@pg.jit
def matmul(a, b):
    return a @ b


_ = amp_matmul(matA, matB).eval()
_ = matmul(matA, matB).eval()

print("Warming up done")
total_time_amp = 0
total_time_regular = 0

for _ in range(num_iterations):
    start_time = time.time()
    matC_amp = amp_matmul(matA, matB).eval()
    pg.sync_cuda_device()
    end_time = time.time()
    total_time_amp += end_time - start_time

    start_time = time.time()
    matC_regular = matmul(matA, matB).eval()
    pg.sync_cuda_device()
    end_time = time.time()
    total_time_regular += end_time - start_time

average_time_amp = total_time_amp / num_iterations
average_time_regular = total_time_regular / num_iterations

print(
    f"Average time taken for amp_matmul over {num_iterations} iterations: {average_time_amp:.6f} seconds"
)
print(
    f"Average time taken for regular matmul over {num_iterations} iterations: {average_time_regular:.6f} seconds"
)

amp_matmul.print_trace()

print("Result of amp_matmul:", matC_amp.numpy())
print("Result of regular matmul:", matC_regular.numpy())
