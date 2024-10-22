import pequegrad as pg
from pequegrad.extra.mnist import MNISTDataset
from pequegrad.onnx import from_onnx_path

# Train the model in examples/helper/train_onnx_mlp.py
model = from_onnx_path("mnist_mlp.onnx")
mnist = MNISTDataset(device="cpu")
inputs = {"input": pg.Tensor(mnist[0][0].reshape((1, 28, 28)))}
outputs = model.forward(inputs)
print(outputs["output"].numpy().argmax() == mnist[0][1].numpy())
