import pequegrad as pg
from pequegrad.extra.mnist import MNISTDataset
from pequegrad.onnx import from_onnx_path, export_to_onnx, from_onnx_model

# Train the model in examples/helper/train_onnx_mlp.py
model = from_onnx_path("mnist_mlp.onnx")
model.to("cuda")
mnist = MNISTDataset(device="cuda")
inputs = {"input": pg.Tensor(mnist[0][0].reshape((1, 28, 28))).to("cuda")}

outputs = model.forward(inputs)
print(outputs["output"].numpy().argmax() == mnist[0][1].numpy())


# save
onnx_model = export_to_onnx(model, inputs, out_names=["output"])


# RERUN
model_new = from_onnx_model(onnx_model)

model_new.to("cuda")


@pg.jit.withargs(eval_outs=False, opts={"fuser": False, "common_subexpr_elim": False})
def apply(params, inputs):
    return pg.apply_to_module(model_new, params, inputs)


outputs = apply(model_new.tree_flatten(), inputs)


print(outputs["output"].numpy().argmax() == mnist[0][1].numpy())
