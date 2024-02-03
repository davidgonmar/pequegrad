from pequegrad.cuda import CudaArray
import numpy as np  # noqa


nparr = np.random.rand(3, 4).astype(np.float32)
cudaarr = CudaArray.from_numpy(nparr)


nparrbroadcasted = np.broadcast_to(nparr, (3, 3, 4))
cudaarrbroadcasted = cudaarr.broadcast_to((1, 1, 1, 4))

print(nparrbroadcasted.shape)
print(cudaarrbroadcasted.shape)
print(nparrbroadcasted.strides)
print(cudaarrbroadcasted.strides)


for i in range(3):
    for j in range(3):
        for k in range(4):
            assert (
                nparrbroadcasted[i, j, k] == cudaarrbroadcasted[i, j, k]
            ), "expected equal elements in broadcasted arrays, got {} and {}".format(
                nparrbroadcasted[i, j, k], cudaarrbroadcasted[i, j, k]
            )

print("All tests passed!")
