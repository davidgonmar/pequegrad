import argparse
import pequegrad.onnx as pgonnx
import pequegrad as pg
import numpy as np
import onnxruntime as ort
import torch
import time


# I tested:  ["efficientnet_em_Opset17.onnx", "alexnet_Opset16.onnx"]
parser = argparse.ArgumentParser(description="Load ONNX model for inference.")
parser.add_argument(
    "--model_path",
    type=str,
    default="alexnet_Opset16.onnx",
    help="Path to the ONNX model.",
)
args = parser.parse_args()

providers = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": torch.cuda.current_device(),
            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
        },
    )
]

sess_options = ort.SessionOptions()

model = pgonnx.from_onnx_path(args.model_path).to("cuda")

input_spec = model.get_input_shapes()["x"]


input_array = np.random.randn(*input_spec).astype(np.float32)
input_tensor = pg.Tensor(input_array).to("cuda")
params = model.tree_flatten()


@pg.jit.withargs(eval_outs=False)
def model_run(params_dict, x):
    return pg.apply_to_module(model, params_dict, x)


pgouts = pg.tree_flatten(model_run(params, {"x": input_tensor}))[0]

pg.viz(pgouts)
ort_session = ort.InferenceSession(
    args.model_path, sess_options=sess_options, providers=providers
)

ortouts = pg.tree_flatten(ort_session.run(None, {"x": input_tensor.numpy()}))[0]

for pg_out, ort_out in zip(pgouts, ortouts):
    np.testing.assert_allclose(pg_out.numpy(), ort_out, atol=2e-3)

print("All tests passed!")

# Efficiency test
start = time.time()
for _ in range(100):
    res = model_run(params, {"x": input_tensor})
    pg.tree_map(lambda x: x.eval() if isinstance(x, pg.Tensor) else None, res)
end = time.time()
print(f"Time taken for Pequegrad: {end - start:.2f}s")

start = time.time()
for _ in range(100):
    res = ort_session.run(None, {"x": input_tensor.numpy()})
end = time.time()
print(f"Time taken for ONNX Runtime: {end - start:.2f}s")
